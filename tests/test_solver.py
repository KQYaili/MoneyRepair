import numpy as np

from moneyrepair.compat import PackedCompatibilityMatrix, compute_compatibility
from moneyrepair.solver import solve_covering_sets
from moneyrepair.types import Fragment


def test_solver_finds_high_coverage_compatible_set():
    masks = []
    for x0, x1 in [(0, 3), (3, 6), (6, 10)]:
        mask = np.zeros((4, 10), dtype=bool)
        mask[:, x0:x1] = True
        masks.append(mask)

    overlapping = np.zeros((4, 10), dtype=bool)
    overlapping[:, 2:7] = True

    fragments = [
        Fragment("left", masks[0]),
        Fragment("middle", masks[1]),
        Fragment("right", masks[2]),
        Fragment("bad_overlap", overlapping),
    ]
    matrix = compute_compatibility(fragments)
    solutions = solve_covering_sets(fragments, matrix, target_coverage=1.0, max_solutions=3)

    assert solutions
    assert solutions[0].coverage == 1.0
    assert set(solutions[0].fragment_ids) == {"left", "middle", "right"}


def test_solver_respects_start_id():
    mask_a = np.zeros((2, 4), dtype=bool)
    mask_b = np.zeros((2, 4), dtype=bool)
    mask_a[:, :2] = True
    mask_b[:, 2:] = True
    fragments = [Fragment("a", mask_a), Fragment("b", mask_b)]
    matrix = compute_compatibility(fragments)

    solutions = solve_covering_sets(fragments, matrix, target_coverage=1.0, start_id="b")

    assert solutions[0].fragment_ids == ("a", "b")


def test_solver_accepts_packed_compatibility_matrix():
    mask_a = np.zeros((2, 4), dtype=bool)
    mask_b = np.zeros((2, 4), dtype=bool)
    mask_a[:, :2] = True
    mask_b[:, 2:] = True
    fragments = [Fragment("a", mask_a), Fragment("b", mask_b)]
    dense = compute_compatibility(fragments)
    packed = PackedCompatibilityMatrix.from_dense(dense)

    solutions = solve_covering_sets(fragments, packed, target_coverage=1.0)

    assert solutions[0].coverage == 1.0


def test_solver_allowed_ids_excludes_confirmed_fragments():
    masks = []
    for x0, x1 in [(0, 2), (2, 4), (4, 6)]:
        mask = np.zeros((2, 6), dtype=bool)
        mask[:, x0:x1] = True
        masks.append(mask)
    fragments = [Fragment("a", masks[0]), Fragment("b", masks[1]), Fragment("c", masks[2])]
    matrix = compute_compatibility(fragments)

    solutions = solve_covering_sets(fragments, matrix, target_coverage=2 / 3, allowed_ids={"b", "c"})

    assert solutions
    assert set(solutions[0].fragment_ids) == {"b", "c"}


def test_solver_supports_order_strategies():
    masks = []
    for x0, x1 in [(0, 2), (2, 4), (4, 6)]:
        mask = np.zeros((2, 6), dtype=bool)
        mask[:, x0:x1] = True
        masks.append(mask)
    fragments = [Fragment("a", masks[0]), Fragment("b", masks[1]), Fragment("c", masks[2])]
    matrix = compute_compatibility(fragments)

    for strategy in ("area", "degree", "area_degree"):
        solutions = solve_covering_sets(fragments, matrix, target_coverage=1.0, order_strategy=strategy)
        assert solutions[0].coverage == 1.0
