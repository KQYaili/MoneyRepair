from __future__ import annotations

from moneyrepair.tearfit import (
    AssemblyCandidate,
    FractalTearConfig,
    diagnose_confirmed_candidates,
    make_fractal_tear_fragments,
    run_tearfit_trial,
    score_absolute_tear_pairs,
    select_exact_cover_candidates,
)


def test_fractal_tears_have_serial_anchor_per_note():
    _template, fragments = make_fractal_tear_fragments(
        FractalTearConfig(notes=4, pieces_per_note=5, width=96, height=54, seed=3)
    )

    labels_by_note = {}
    for fragment in fragments:
        labels_by_note.setdefault(fragment.meta["note_id"], set())
        if fragment.label:
            labels_by_note[fragment.meta["note_id"]].add(fragment.label)

    assert len(labels_by_note) == 4
    assert all(len(labels) == 1 for labels in labels_by_note.values())


def test_absolute_tear_overlap_separates_some_true_edges_from_false_edges():
    _template, fragments = make_fractal_tear_fragments(
        FractalTearConfig(notes=3, pieces_per_note=5, width=90, height=48, seed=11)
    )
    _all_scores, edges = score_absolute_tear_pairs(
        fragments,
        tolerance=2,
        min_overlap_pixels=6,
        use_labels=False,
    )

    true_edges = [
        edge
        for edge in edges
        if fragments[edge.left].meta["note_id"] == fragments[edge.right].meta["note_id"]
    ]
    false_edges = [
        edge
        for edge in edges
        if fragments[edge.left].meta["note_id"] != fragments[edge.right].meta["note_id"]
    ]

    assert true_edges
    assert len(false_edges) < len(true_edges)
    assert max(edge.overlap_pixels for edge in true_edges) > max((edge.overlap_pixels for edge in false_edges), default=0)


def test_exact_cover_selection_reuses_no_fragment_or_serial():
    candidates = [
        AssemblyCandidate(("a", "b"), coverage=0.98, raw_coverage=0.94, score=10.0, support_pixels=10, labels=("S1",)),
        AssemblyCandidate(("b", "c"), coverage=0.99, raw_coverage=0.95, score=30.0, support_pixels=30, labels=("S2",)),
        AssemblyCandidate(("d", "e"), coverage=0.99, raw_coverage=0.95, score=20.0, support_pixels=20, labels=("S1",)),
        AssemblyCandidate(("f", "g"), coverage=0.99, raw_coverage=0.95, score=15.0, support_pixels=15, labels=("S3",)),
    ]

    selected = select_exact_cover_candidates(candidates)
    used_fragments = [fragment_id for candidate in selected for fragment_id in candidate.fragment_ids]
    used_labels = [label for candidate in selected for label in candidate.labels]

    assert len(used_fragments) == len(set(used_fragments))
    assert len(used_labels) == len(set(used_labels))


def test_labelled_tearfit_trial_confirms_pure_candidates():
    result = run_tearfit_trial(
        FractalTearConfig(notes=3, pieces_per_note=5, width=90, height=48, seed=11),
        min_overlap_pixels=6,
        beam_width=24,
    )

    assert result.accepted_edges > 0
    assert result.diagnostics.confirmed > 0
    assert result.diagnostics.chimeras == 0
    assert result.diagnostics.pure_precision == 1.0


def test_diagnosis_counts_exact_and_chimera_candidates():
    _template, fragments = make_fractal_tear_fragments(
        FractalTearConfig(notes=2, pieces_per_note=4, width=80, height=44, seed=17)
    )
    note_sets = {}
    for fragment in fragments:
        note_sets.setdefault(fragment.meta["note_id"], []).append(fragment.id)
    exact_ids = tuple(sorted(note_sets["note-000"]))
    chimera_ids = tuple(sorted((note_sets["note-000"][0], note_sets["note-001"][0])))

    diag = diagnose_confirmed_candidates(
        [
            AssemblyCandidate(exact_ids, coverage=1.0, raw_coverage=0.98, score=1.0, support_pixels=1),
            AssemblyCandidate(chimera_ids, coverage=1.0, raw_coverage=0.98, score=1.0, support_pixels=1),
        ],
        fragments,
    )

    assert diag.exact_confirmed == 1
    assert diag.chimeras == 1
