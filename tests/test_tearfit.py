from __future__ import annotations

import numpy as np

from moneyrepair.tearfit import (
    AssemblyCandidate,
    FractalTearConfig,
    augment_candidates_with_group_gap,
    diagnose_confirmed_candidates,
    generate_assembly_candidates,
    make_fractal_tear_fragments,
    run_tearfit_trial,
    run_tearfit_strategy_comparison,
    run_tearfit_v43_ablation,
    score_absolute_tear_pairs,
    tear_match_effectiveness,
    select_exact_cover_candidates,
    TearFitEdge,
)
from moneyrepair.types import Fragment


def test_fractal_tears_have_serial_anchor_per_note():
    _template, fragments = make_fractal_tear_fragments(
        FractalTearConfig(notes=4, pieces_per_note=5, width=96, height=54, seed=3, ensure_serial_anchor=True, serial_ocr_rate=1.0)
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


def test_exact_cover_can_use_weighted_score_objective():
    candidates = [
        AssemblyCandidate(("a", "b"), coverage=0.99, raw_coverage=0.95, score=10.0, support_pixels=10),
        AssemblyCandidate(("c", "d"), coverage=0.99, raw_coverage=0.95, score=10.0, support_pixels=10),
        AssemblyCandidate(("a", "c"), coverage=0.99, raw_coverage=0.95, score=30.0, support_pixels=30),
    ]

    count_first = select_exact_cover_candidates(candidates, objective="count_then_score")
    score_first = select_exact_cover_candidates(candidates, objective="score_then_count")

    assert {item.fragment_ids for item in count_first} == {("a", "b"), ("c", "d")}
    assert [item.fragment_ids for item in score_first] == [("a", "c")]


def test_exact_cover_respects_locked_candidates():
    candidates = [
        AssemblyCandidate(("a", "b"), coverage=0.99, raw_coverage=0.95, score=10.0, support_pixels=10),
        AssemblyCandidate(("c", "d"), coverage=0.99, raw_coverage=0.95, score=10.0, support_pixels=10),
        AssemblyCandidate(("a", "c"), coverage=0.99, raw_coverage=0.95, score=30.0, support_pixels=30),
    ]

    selected = select_exact_cover_candidates(
        candidates,
        objective="score_then_count",
        locked_candidates={("a", "b")},
    )

    assert ("a", "b") in {item.fragment_ids for item in selected}
    assert ("a", "c") not in {item.fragment_ids for item in selected}


def test_candidate_generation_respects_seed_labels_and_forbidden_pairs():
    top = np.array(
        [[True, True, True, True], [True, True, True, True], [False, False, False, False], [False, False, False, False]]
    )
    bottom = np.array(
        [[False, False, False, False], [False, False, False, False], [True, True, True, True], [True, True, True, True]]
    )
    fragments = [
        Fragment("a", mask=top, label="S1"),
        Fragment("b", mask=bottom, label="S1"),
        Fragment("c", mask=top, label="S2"),
        Fragment("d", mask=bottom, label="S2"),
    ]
    edges = [
        TearFitEdge(left=0, right=1, overlap_pixels=8, left_hits=8, right_hits=8, overlap_ratio=1.0),
        TearFitEdge(left=2, right=3, overlap_pixels=8, left_hits=8, right_hits=8, overlap_ratio=1.0),
    ]

    seeded = generate_assembly_candidates(
        fragments,
        edges,
        coverage_threshold=0.9,
        max_pieces=2,
        seed_labels={"S2"},
    )
    blocked = generate_assembly_candidates(
        fragments,
        edges,
        coverage_threshold=0.9,
        max_pieces=2,
        forbidden_pairs={("c", "d")},
    )

    assert [candidate.fragment_ids for candidate in seeded] == [("c", "d")]
    assert ("c", "d") not in {candidate.fragment_ids for candidate in blocked}


def test_labelled_tearfit_trial_confirms_pure_candidates():
    result = run_tearfit_trial(
        FractalTearConfig(notes=3, pieces_per_note=5, width=90, height=48, seed=11, ensure_serial_anchor=True, serial_ocr_rate=1.0),
        min_overlap_pixels=6,
        beam_width=24,
    )

    assert result.accepted_edges > 0
    assert result.diagnostics.confirmed > 0
    assert result.diagnostics.chimeras == 0
    assert result.diagnostics.pure_precision == 1.0


def test_anchor_priority_does_not_make_ocr_coverage_a_hard_ceiling():
    config = FractalTearConfig(
        notes=3,
        pieces_per_note=5,
        width=90,
        height=48,
        seed=11,
        serial_ocr_rate=0.0,
    )

    old_anchor_only = run_tearfit_trial(
        config,
        min_overlap_pixels=6,
        beam_width=24,
        seed_strategy="anchor_only",
    )
    anchor_priority = run_tearfit_trial(
        config,
        min_overlap_pixels=6,
        beam_width=24,
        seed_strategy="anchor_priority",
    )

    assert old_anchor_only.diagnostics.confirmed == 0
    assert anchor_priority.diagnostics.confirmed > 0
    assert anchor_priority.diagnostics.exact_precision == 1.0


def test_strategy_comparison_reports_best_seed_strategy():
    payload = run_tearfit_strategy_comparison(
        profile="smoke",
        seed_strategies=("anchor_only", "anchor_priority"),
        serial_ocr_rates=(0.0,),
        width=90,
        height=48,
        min_overlap_pixels=6,
        beam_width=24,
        candidate_time_limit_seconds=5.0,
        cover_time_limit_seconds=2.0,
    )

    assert payload["best_seed_strategy"]["seed_strategy"] == "anchor_priority"
    assert payload["best_seed_strategy"]["cover_objective"] in {"count_then_score", "score_then_count"}
    assert len(payload["rows"]) == 2 * 2
    assert payload["summary"][0]["mean_exact_yield"] > payload["summary"][1]["mean_exact_yield"]


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


def test_exact_cover_scales_past_recursion_limit_without_crashing():
    # Regression: the set-packing search used to recurse once per candidate, so
    # a large pool (N=200-scale runs generate well over 1000 candidates) blew
    # Python's recursion limit and crashed mid-search. The iterative form must
    # return the full disjoint packing instead.
    count = 2500
    candidates = [
        AssemblyCandidate(
            fragment_ids=(f"f{index:05d}",),
            coverage=0.99,
            raw_coverage=0.99,
            score=1.0,
            support_pixels=1,
        )
        for index in range(count)
    ]
    selected = select_exact_cover_candidates(candidates, time_limit_seconds=None, objective="score_then_count")
    # every candidate is disjoint, so the optimal packing keeps all of them.
    assert len(selected) == count
    assert len({candidate.fragment_ids for candidate in selected}) == count

def test_effectiveness_rewards_short_coherent_support_over_scattered_hits():
    coherent = tear_match_effectiveness(
        contiguous_pixels=10,
        overlap_pixels=10,
        left_hits=10,
        right_hits=10,
        overlap_ratio=0.15,
        normal_opposition=1.0,
        curvature_entropy=0.8,
        expected_accidental_hits=1.0,
    )
    scattered = tear_match_effectiveness(
        contiguous_pixels=2,
        overlap_pixels=14,
        left_hits=14,
        right_hits=14,
        overlap_ratio=0.20,
        normal_opposition=0.8,
        curvature_entropy=0.8,
        expected_accidental_hits=1.0,
    )

    assert coherent > 1.0
    assert coherent > scattered * 10.0



def test_effectiveness_separates_complementary_jagged_seam_from_sparse_crossings():
    height, width = 30, 30
    y = np.arange(height)
    seam = 13 + ((y % 7) >= 3).astype(int) + ((y % 11) >= 7).astype(int)
    left = np.zeros((height, width), dtype=bool)
    true_right = np.zeros_like(left)
    false_right = np.zeros_like(left)
    for row, split in enumerate(seam):
        left[row, :split] = True
        true_right[row, split:] = True
        false_split = split if row % 6 == 0 else min(width - 1, split + 4)
        false_right[row, false_split:] = True
    fragments = [
        Fragment("left", left),
        Fragment("true", true_right),
        Fragment("false", false_right),
    ]
    scores, _accepted = score_absolute_tear_pairs(
        fragments,
        tolerance=1,
        use_labels=False,
        scoring="effectiveness",
        min_effectiveness=0.0,
        automatic_effectiveness=0.0,
        min_contiguous_pixels=1,
        automatic_contiguous_pixels=1,
    )
    by_pair = {(edge.left, edge.right): edge for edge in scores}

    assert by_pair[(0, 1)].contiguous_pixels > by_pair[(0, 2)].contiguous_pixels
    assert by_pair[(0, 1)].effectiveness > by_pair[(0, 2)].effectiveness



def test_pose_uncertainty_downgrades_tear_effectiveness():
    left = np.zeros((12, 12), dtype=bool)
    left[:, :6] = True
    right = np.zeros((12, 12), dtype=bool)
    right[:, 6:] = True
    certain = [Fragment("a", left), Fragment("b", right)]
    uncertain = [
        Fragment("a", left),
        Fragment("b", right, meta={"pose_sigma_x": 4.0, "pose_sigma_y": 4.0}),
    ]

    certain_scores, _ = score_absolute_tear_pairs(
        certain,
        tolerance=1,
        scoring="effectiveness",
        min_effectiveness=0.0,
        automatic_effectiveness=0.0,
        min_contiguous_pixels=1,
        automatic_contiguous_pixels=1,
    )
    uncertain_scores, _ = score_absolute_tear_pairs(
        uncertain,
        tolerance=1,
        scoring="effectiveness",
        min_effectiveness=0.0,
        automatic_effectiveness=0.0,
        min_contiguous_pixels=1,
        automatic_contiguous_pixels=1,
    )

    assert uncertain_scores[0].pose_uncertainty > 0.0
    assert uncertain_scores[0].effectiveness < certain_scores[0].effectiveness



def test_group_gap_can_add_fragment_supported_by_two_weak_seams():
    top_left = np.zeros((8, 8), dtype=bool)
    top_left[:4, :4] = True
    bottom_left = np.zeros((8, 8), dtype=bool)
    bottom_left[4:, :4] = True
    right = np.zeros((8, 8), dtype=bool)
    right[:, 4:] = True
    fragments = [
        Fragment("a", top_left),
        Fragment("b", bottom_left),
        Fragment("c", right),
    ]
    weak = dict(
        overlap_pixels=4,
        left_hits=4,
        right_hits=4,
        overlap_ratio=0.5,
        contiguous_pixels=4,
        continuity_ratio=1.0,
        bidirectional_balance=1.0,
        normal_opposition=1.0,
        curvature_entropy=0.5,
        expected_accidental_hits=0.2,
        effectiveness=0.8,
        evidence_level="review",
    )
    gap_edges = [
        TearFitEdge(left=0, right=2, **weak),
        TearFitEdge(left=1, right=2, **weak),
    ]
    base = AssemblyCandidate(
        ("a", "b"),
        coverage=0.5,
        raw_coverage=0.5,
        score=5_000.0,
        support_pixels=4,
    )

    augmented = augment_candidates_with_group_gap(
        fragments,
        [base],
        gap_edges,
        tolerance=1,
        coverage_threshold=0.95,
        gap_fill_radius=0,
        max_pieces=3,
        time_limit_seconds=None,
    )

    exact = next(
        candidate
        for candidate in augmented
        if candidate.fragment_ids == ("a", "b", "c")
    )
    assert exact.coverage == 1.0
    assert exact.gap_steps == 1
    assert exact.evidence_level == "automatic"



def test_candidate_generation_can_retain_incomplete_core_for_gap_stage():
    top_left = np.zeros((8, 8), dtype=bool)
    top_left[:4, :4] = True
    bottom_left = np.zeros((8, 8), dtype=bool)
    bottom_left[4:, :4] = True
    fragments = [Fragment("a", top_left), Fragment("b", bottom_left)]
    edge = TearFitEdge(
        left=0,
        right=1,
        overlap_pixels=4,
        left_hits=4,
        right_hits=4,
        overlap_ratio=0.5,
        contiguous_pixels=4,
        effectiveness=3.0,
        evidence_level="automatic",
    )

    cores = generate_assembly_candidates(
        fragments,
        [edge],
        coverage_threshold=0.95,
        minimum_candidate_raw_coverage=0.5,
        gap_fill_radius=0,
        max_pieces=2,
        time_limit_seconds=None,
    )

    core = next(
        candidate for candidate in cores if candidate.fragment_ids == ("a", "b")
    )
    assert core.raw_coverage == 0.5
    assert core.coverage < 0.95



def test_candidate_generation_reports_deterministic_state_budget():
    left = np.zeros((8, 8), dtype=bool)
    left[:, :4] = True
    right = np.zeros((8, 8), dtype=bool)
    right[:, 4:] = True
    fragments = [Fragment("a", left), Fragment("b", right)]
    edge = TearFitEdge(0, 1, 4, 4, 4, 0.5)
    stats: dict[str, int | bool] = {}

    generate_assembly_candidates(
        fragments,
        [edge],
        coverage_threshold=0.95,
        max_pieces=2,
        time_limit_seconds=None,
        max_expanded_states=1,
        search_stats=stats,
    )

    assert stats["expanded_states"] == 1
    assert stats["state_limit_reached"] is True
    assert stats["time_limit_reached"] is False



def test_v43_ablation_keeps_same_seed_and_reports_baseline_deltas():
    payload = run_tearfit_v43_ablation(
        notes=2,
        pieces_list=(4,),
        seeds=(13,),
        algorithms=("baseline", "effectiveness"),
        width=72,
        height=40,
        min_overlap_pixels=5,
        beam_width=12,
        candidate_time_limit_seconds=2.0,
        cover_time_limit_seconds=1.0,
    )

    assert {(row["algorithm"], row["seed"]) for row in payload["rows"]} == {
        ("baseline", 13),
        ("effectiveness", 13),
    }
    assert all("delta_exact_yield_vs_baseline" in row for row in payload["summary"])



def test_v43_ablation_reuses_the_identical_routed_trial():
    payload = run_tearfit_v43_ablation(
        notes=2,
        pieces_list=(4,),
        seeds=(23,),
        algorithms=("baseline", "v43_routed"),
        width=72,
        height=40,
        min_overlap_pixels=4,
        beam_width=8,
        candidate_time_limit_seconds=1.0,
        partial_gap_time_limit_seconds=0.2,
        cover_time_limit_seconds=1.0,
    )

    routed = next(row for row in payload["rows"] if row["algorithm"] == "v43_routed")
    assert routed["resolved_algorithm"] == "baseline"
    assert routed["reused_from_algorithm"] == "baseline"



def test_review_candidate_does_not_reduce_automatic_manual_queue():
    _template, fragments = make_fractal_tear_fragments(
        FractalTearConfig(notes=1, pieces_per_note=4, width=80, height=44, seed=19)
    )
    exact_ids = tuple(sorted(fragment.id for fragment in fragments))
    diag = diagnose_confirmed_candidates(
        [
            AssemblyCandidate(
                exact_ids,
                coverage=1.0,
                raw_coverage=0.98,
                score=1.0,
                support_pixels=1,
                evidence_level="review",
            )
        ],
        fragments,
    )

    assert diag.exact_yield == 1.0
    assert diag.automatic_exact_yield == 0.0
    assert diag.manual_notes_remaining == 1



def test_exact_cover_reports_deterministic_node_budget():
    candidates = [
        AssemblyCandidate((f"f{index}",), 0.99, 0.99, 1.0, 1) for index in range(20)
    ]
    stats: dict[str, int | bool] = {}

    select_exact_cover_candidates(
        candidates,
        time_limit_seconds=None,
        max_search_nodes=3,
        search_stats=stats,
    )

    assert stats["search_nodes"] == 3
    assert stats["node_limit_reached"] is True
    assert stats["time_limit_reached"] is False


def test_v43_ablation_p8_p16_p24_regime_comparison():
    """Verify same-seed A/B ablation across p=8, 16, 24 regimes.

    Tests that baseline vs effectiveness vs effectiveness_gap vs v43_routed produce
    consistent diagnostic metrics and triage evidence classifications (automatic,
    review, insufficient-evidence) on identical fractal tear inputs.
    """
    payload = run_tearfit_v43_ablation(
        notes=2,
        pieces_list=(8, 16, 24),
        seeds=(42,),
        algorithms=("baseline", "effectiveness", "effectiveness_gap", "v43_routed"),
        width=72,
        height=40,
        min_overlap_pixels=4,
        beam_width=8,
        candidate_time_limit_seconds=2.0,
        partial_gap_time_limit_seconds=0.5,
        cover_time_limit_seconds=1.0,
    )

    # 3 pieces x 4 algorithms = 12 total trial rows
    assert len(payload["rows"]) == 12
    pieces_covered = {row["pieces_per_note"] for row in payload["rows"]}
    assert pieces_covered == {8, 16, 24}

    for row in payload["rows"]:
        assert "exact_yield" in row
        assert "exact_precision" in row
        assert "automatic_exact_yield" in row
        assert "automatic_exact_precision" in row
        assert "manual_notes_remaining" in row
        assert "edge_decisions" in row
        assert "candidate_decisions" in row
        assert set(row["edge_decisions"].keys()) == {"automatic", "review", "insufficient-evidence"}
        assert set(row["candidate_decisions"].keys()) == {"automatic", "review"}
        assert row["true_accepted_edges"] + row["false_accepted_edges"] == row["accepted_edges"]
        assert row["selected_core_candidates"] + row["selected_gap_candidates"] == row["confirmed"]
        assert (
            row["selected_complete_gap_candidates"]
            + row["selected_partial_gap_candidates"]
            == row["selected_gap_candidates"]
        )

    # Verify summary has baseline deltas for all pieces
    summary_by_piece = {}
    for item in payload["summary"]:
        summary_by_piece.setdefault(item["pieces_per_note"], []).append(item["algorithm"])
    assert set(summary_by_piece.keys()) == {8, 16, 24}
    for piece, algos in summary_by_piece.items():
        assert set(algos) == {"baseline", "effectiveness", "effectiveness_gap", "v43_routed"}

    for item in payload["summary"]:
        assert "mean_accepted_edges" in item
        assert "mean_false_accepted_edges" in item
        assert "mean_candidates" in item
        assert "mean_selected_gap_candidates" in item

    comparisons = payload["mechanism_comparisons"]
    assert len(comparisons) == 9
    assert {item["stage"] for item in comparisons} == {
        "adaptive_edge",
        "group_gap",
        "routing",
    }
    routed = [item for item in comparisons if item["stage"] == "routing"]
    assert all(item["reused_identical_trials"] for item in routed)
