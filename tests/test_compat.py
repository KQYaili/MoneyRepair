import numpy as np

from moneyrepair.compat import (
    CompatibilityMatrix,
    PackedCompatibilityMatrix,
    compatibility_from_pair_records,
    compute_compatibility,
    filter_compatibility_to_ids,
    load_pair_records,
)
from moneyrepair.types import Fragment


def test_compute_compatibility_marks_overlaps_incompatible(tmp_path):
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    c = np.zeros((8, 8), dtype=bool)
    a[1:4, 1:4] = True
    b[3:6, 3:6] = True
    c[5:7, 5:7] = True

    matrix = compute_compatibility(
        [
            Fragment("a", a),
            Fragment("b", b),
            Fragment("c", c),
        ]
    )

    assert not matrix.is_compatible("a", "b")
    assert matrix.is_compatible("a", "c")
    assert not matrix.is_compatible("b", "c")

    path = tmp_path / "matrix.npz"
    matrix.save(path)
    loaded = CompatibilityMatrix.load(path)
    assert loaded.ids == ("a", "b", "c")
    np.testing.assert_array_equal(loaded.compatible, matrix.compatible)


def test_filter_compatibility_to_allowed_ids():
    mask_a = np.zeros((4, 6), dtype=bool)
    mask_b = np.zeros((4, 6), dtype=bool)
    mask_c = np.zeros((4, 6), dtype=bool)
    mask_a[:, :2] = True
    mask_b[:, 2:4] = True
    mask_c[:, 4:] = True
    matrix = compute_compatibility([Fragment("a", mask_a), Fragment("b", mask_b), Fragment("c", mask_c)])

    filtered = filter_compatibility_to_ids(matrix, {"a", "b"})

    assert filtered.is_compatible("a", "b")
    assert not filtered.is_compatible("a", "c")
    assert not filtered.is_compatible("b", "c")


def test_packed_compatibility_roundtrip_and_indices(tmp_path):
    dense = np.array(
        [
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ],
        dtype=bool,
    )
    matrix = PackedCompatibilityMatrix.from_dense(CompatibilityMatrix(("a", "b", "c"), dense))

    assert matrix.is_compatible("a", "b")
    assert not matrix.is_compatible("a", "c")
    assert matrix.compatible_indices(1, (0, 2)) == (0, 2)

    path = tmp_path / "packed.npz"
    matrix.save(path)
    loaded = PackedCompatibilityMatrix.load(path)
    np.testing.assert_array_equal(loaded.to_dense().compatible, dense)


def test_compatibility_from_pair_records_supports_precomputed_incompatible_pairs(tmp_path):
    pairs_path = tmp_path / "pairs.csv"
    pairs_path.write_text("\ufeffleft,right\na,c\n", encoding="utf-8")
    pairs = load_pair_records(pairs_path)

    matrix = compatibility_from_pair_records(["a", "b", "c"], pairs, relation="incompatible")

    assert matrix.is_compatible("a", "b")
    assert not matrix.is_compatible("a", "c")
    assert matrix.is_compatible("b", "c")


def test_compatibility_from_pair_records_supports_precomputed_compatible_pairs():
    matrix = compatibility_from_pair_records(["a", "b", "c"], [("a", "b")], relation="compatible")

    assert matrix.is_compatible("a", "b")
    assert not matrix.is_compatible("a", "c")
