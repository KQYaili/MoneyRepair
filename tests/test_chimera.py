from moneyrepair.compat import compute_compatibility_fast
from moneyrepair.diagnostics import diagnose_solutions
from moneyrepair.fingerprint import cluster_fragments_by_appearance, discriminative_compatibility
from moneyrepair.simulate import load_dataset, make_multi_note_fragments, save_dataset
from moneyrepair.solver import solve_covering_sets


def _pool(notes=3, pieces_per_note=8, seed=7):
    return make_multi_note_fragments(notes=notes, pieces_per_note=pieces_per_note, width=160, height=90, seed=seed)


def test_meta_note_id_round_trips_through_disk(tmp_path):
    template, fragments = _pool()
    assert all(fragment.meta.get("note_id") for fragment in fragments)

    path = tmp_path / "multi.npz"
    save_dataset(path, template, fragments)
    _, reloaded = load_dataset(path)

    assert reloaded[0].meta.get("note_id") == fragments[0].meta.get("note_id")
    assert {fragment.meta["note_id"] for fragment in reloaded} == {fragment.meta["note_id"] for fragment in fragments}


def test_appearance_clustering_separates_notes():
    template, fragments = _pool(notes=4)
    groups = cluster_fragments_by_appearance(fragments, template, tolerance=0.05)

    # one cluster per note
    assert len(set(groups.values())) == 4
    # every fragment of a note lands in the same cluster
    by_note: dict[str, set[int]] = {}
    for fragment in fragments:
        by_note.setdefault(fragment.meta["note_id"], set()).add(groups[fragment.id])
    assert all(len(clusters) == 1 for clusters in by_note.values())


def test_overlap_only_matrix_produces_chimeras():
    template, fragments = _pool(notes=3)
    matrix = compute_compatibility_fast(fragments)
    solutions = solve_covering_sets(
        fragments, matrix, target_coverage=0.95, max_solutions=20, time_limit_seconds=20, order_strategy="area_degree"
    )
    diagnosis = diagnose_solutions(solutions, fragments)

    # the trap: with identical denominations, overlap-only stitches across notes
    assert diagnosis["chimeras"] > 0


def test_discriminative_matrix_eliminates_chimeras():
    template, fragments = _pool(notes=3)
    matrix = discriminative_compatibility(fragments, template, mode="appearance", tolerance=0.05)
    solutions = solve_covering_sets(
        fragments, matrix, target_coverage=0.95, max_solutions=20, time_limit_seconds=20, order_strategy="area_degree"
    )
    diagnosis = diagnose_solutions(solutions, fragments)

    assert diagnosis["chimeras"] == 0
    assert len(diagnosis["pure_notes_found"]) == 3


def test_dbscan_clustering_order_independent_and_no_drift():
    from moneyrepair.types import Fragment
    from moneyrepair.fingerprint import cluster_fragments_by_appearance
    import numpy as np
    import moneyrepair.fingerprint as fp

    orig_appearances = fp.fragment_appearances
    try:
        frags = [Fragment(id=f"f{i}", mask=np.ones((10, 10), dtype=bool)) for i in range(5)]
        mock_appearances = {
            "f0": np.array([1.0, 1.0, 1.0]),
            "f1": np.array([1.01, 1.01, 1.01]),
            "f2": np.array([1.1, 1.1, 1.1]),
            "f3": np.array([1.11, 1.11, 1.11]),
            "f4": np.array([1.3, 1.3, 1.3]),
        }
        fp.fragment_appearances = lambda frags, temp: mock_appearances

        # Order 1
        groups1 = cluster_fragments_by_appearance(frags, np.zeros((10, 10, 3)), tolerance=0.03, min_samples=2)

        # Order 2: shuffled frags list
        shuffled_frags = [frags[2], frags[4], frags[0], frags[3], frags[1]]
        groups2 = cluster_fragments_by_appearance(shuffled_frags, np.zeros((10, 10, 3)), tolerance=0.03, min_samples=2)

        # Group by group ID to compare partitions
        part1 = {}
        for fid, gid in groups1.items():
            part1.setdefault(gid, set()).add(fid)
        sets1 = sorted([sorted(list(s)) for s in part1.values()])

        part2 = {}
        for fid, gid in groups2.items():
            part2.setdefault(gid, set()).add(fid)
        sets2 = sorted([sorted(list(s)) for s in part2.values()])

        assert sets1 == sets2

        # Expected groups: f4 (noise), {f0, f1}, {f2, f3}
        expected = [["f4"], ["f0", "f1"], ["f2", "f3"]]
        expected_sets = sorted([sorted(list(s)) for s in expected])
        assert sets1 == expected_sets

    finally:
        fp.fragment_appearances = orig_appearances
