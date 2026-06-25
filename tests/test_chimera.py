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
