from __future__ import annotations

import numpy as np
import pytest

from moneyrepair.scale import (
    assess_v432_bottleneck,
    assess_v433_oracle_false_edge_deletion,
    assess_v44_boundary_contact_proposal,
    run_v432_scale_protocol,
    run_v433_oracle_false_edge_diagnostic,
    run_v44_boundary_contact_proposal_diagnostic,
    run_v44_candidate_funnel_diagnostic,
    run_v44_core_connectivity_diagnostic,
)
from moneyrepair.tearfit import (
    AssemblyCandidate,
    FractalTearConfig,
    GapProposalEvaluation,
    ResidualGapRegion,
    augment_candidates_with_group_gap,
    classify_gap_complexity,
    compute_residual_gap_components,
    diagnose_confirmed_candidates,
    evaluate_gap_proposal,
    generate_assembly_candidates,
    make_fractal_tear_fragments,
    run_tearfit_trial,
    run_tearfit_strategy_comparison,
    run_tearfit_v43_ablation,
    score_absolute_tear_pairs,
    tear_boundary_evidence,
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


def test_boundary_contact_pool_exposes_fragment_without_weak_pair_evidence():
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
    contact_only_edges = [
        TearFitEdge(0, 2, 4, 4, 4, 0.5),
        TearFitEdge(1, 2, 4, 4, 4, 0.5),
    ]
    base = AssemblyCandidate(
        ("a", "b"),
        coverage=0.5,
        raw_coverage=0.5,
        score=5_000.0,
        support_pixels=4,
    )
    weak_stats: dict[str, int | bool] = {}
    contact_stats: dict[str, int | bool] = {}

    weak_only = augment_candidates_with_group_gap(
        fragments,
        [base],
        contact_only_edges,
        tolerance=1,
        coverage_threshold=0.95,
        gap_fill_radius=0,
        max_pieces=3,
        proposal_pool="weak_pair",
        time_limit_seconds=None,
        search_stats=weak_stats,
    )
    contact_pool = augment_candidates_with_group_gap(
        fragments,
        [base],
        contact_only_edges,
        tolerance=1,
        coverage_threshold=0.95,
        gap_fill_radius=0,
        max_pieces=3,
        proposal_pool="boundary_contact",
        time_limit_seconds=None,
        search_stats=contact_stats,
    )

    assert not any(item.fragment_ids == ("a", "b", "c") for item in weak_only)
    exact = next(
        item for item in contact_pool if item.fragment_ids == ("a", "b", "c")
    )
    assert exact.evidence_level == "review"
    assert weak_stats["proposal_edges"] == 0
    assert contact_stats["proposal_edges"] == 2
    assert contact_stats["accepted_no_weak_edge_proposals"] == 1



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


def test_trial_reports_oracle_edges_provenance_and_stage_timings():
    result = run_tearfit_trial(
        FractalTearConfig(
            notes=2,
            pieces_per_note=4,
            width=72,
            height=40,
            seed=29,
            serial_ocr_rate=0.0,
        ),
        algorithm="effectiveness_gap",
        use_labels=False,
        min_overlap_pixels=4,
        beam_width=8,
        candidate_time_limit_seconds=None,
        candidate_state_limit=20_000,
        partial_gap_time_limit_seconds=None,
        gap_state_limit=4_000,
        partial_gap_state_limit=1_000,
        cover_time_limit_seconds=None,
        cover_node_limit=20_000,
    )

    assert result.true_possible_edges >= result.true_accepted_edges
    assert result.true_edge_recall == pytest.approx(
        result.true_accepted_edges / max(1, result.true_possible_edges)
    )
    assert result.true_gap_candidates + result.false_gap_candidates == result.gap_candidates
    assert (
        result.selected_true_gap_candidates + result.selected_false_gap_candidates
        == result.selected_gap_candidates
    )
    assert 0.0 <= result.oracle_candidate_recall <= 1.0
    assert len(result.selected_solution_fingerprint) == 64
    assert len(result.candidate_provenance_fingerprint) == 64
    assert set(result.stage_timings) == {
        "simulation",
        "pair_scoring",
        "core_search",
        "complete_gap_search",
        "partial_gap_search",
        "exact_cover",
        "diagnostics",
        "total",
    }


def test_truth_only_candidate_funnel_accounts_for_every_simulated_note():
    result = run_tearfit_trial(
        FractalTearConfig(
            notes=3,
            pieces_per_note=5,
            width=72,
            height=40,
            seed=30,
            serial_ocr_rate=0.0,
        ),
        algorithm="effectiveness_gap",
        use_labels=False,
        min_overlap_pixels=4,
        beam_width=8,
        candidate_time_limit_seconds=None,
        candidate_state_limit=20_000,
        partial_gap_time_limit_seconds=None,
        gap_state_limit=4_000,
        partial_gap_state_limit=1_000,
        cover_time_limit_seconds=None,
        cover_node_limit=20_000,
        diagnostic_candidate_funnel=True,
    )

    funnel = result.candidate_funnel
    assert funnel["simulation_truth_only"] is True
    assert len(funnel["notes"]) == 3
    assert sum(funnel["category_counts"].values()) == 3
    assert all("category" in row for row in funnel["notes"])


def test_workload_normalized_budget_rejects_an_absolute_limit():
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_tearfit_trial(
            FractalTearConfig(
                notes=1,
                pieces_per_note=3,
                width=60,
                height=36,
                seed=31,
            ),
            candidate_state_limit=100,
            candidate_states_per_pair_score=2.0,
        )


def test_v432_scale_protocol_calibrates_anchor_and_emits_two_tracks():
    checkpointed = []
    payload = run_v432_scale_protocol(
        notes_list=(2, 3),
        seeds=(37,),
        pieces_per_note=4,
        anchor_notes=2,
        anchor_budget_factors=(1, 2),
        full_mechanism_through=3,
        width=72,
        height=40,
        min_overlap_pixels=4,
        beam_width=8,
        candidate_state_limit=20_000,
        gap_state_limit=4_000,
        partial_gap_state_limit=1_000,
        cover_node_limit=20_000,
        case_sink=checkpointed.append,
    )

    assert payload["anchor_calibration"]["stable"] is True
    assert payload["anchor_calibration"]["selected_factor"] == 1
    assert {row["track"] for row in payload["rows"]} == {"fixed", "normalized"}
    assert checkpointed
    normalized = next(
        row
        for row in payload["rows"]
        if row["track"] == "normalized"
        and row["notes"] == 3
        and row["algorithm"] == "baseline"
    )
    rates = payload["anchor_calibration"]["normalization_rates_by_seed"][37]
    assert normalized["budgets"]["candidate_state_limit"] == pytest.approx(
        np.ceil(rates["candidate_states_per_pair_score"] * normalized["pair_scores"])
    )
    assert normalized["budgets"]["cover_node_limit"] == pytest.approx(
        np.ceil(rates["cover_nodes_per_note"] * 3)
    )
    assert payload["quality_assessment"]
    assert payload["mechanism_assessment"]
    assert payload["bottleneck_assessment"]["status"] == "not_reached"


def test_v432_bottleneck_excludes_cover_when_oracle_equals_yield():
    stage_seconds = {
        "simulation": 1.0,
        "pair_scoring": 10.0,
        "core_search": 30.0,
        "complete_gap_search": 50.0,
        "partial_gap_search": 20.0,
        "exact_cover": 1.0,
        "diagnostics": 0.0,
        "total": 112.0,
    }
    summary = [
        {
            "track": "normalized",
            "notes": 20,
            "algorithm": "v43_routed",
            "mean_exact_yield": 0.92,
            "mean_oracle_candidate_recall": 0.92,
            "mean_true_edge_recall": 0.38,
            "mean_false_edge_rate": 0.04,
            "core_state_saturation_rate": 0.0,
            "complete_gap_state_saturation_rate": 0.0,
            "partial_gap_state_saturation_rate": 0.0,
            "mean_stage_seconds": stage_seconds,
        },
        {
            "track": "normalized",
            "notes": 100,
            "algorithm": "v43_routed",
            "mean_exact_yield": 0.84,
            "mean_oracle_candidate_recall": 0.84,
            "mean_true_edge_recall": 0.37,
            "mean_false_edge_rate": 0.14,
            "core_state_saturation_rate": 0.0,
            "complete_gap_state_saturation_rate": 0.0,
            "partial_gap_state_saturation_rate": 0.0,
            "mean_stage_seconds": stage_seconds,
        },
    ]
    assessment = assess_v432_bottleneck(
        summary,
        [
            {"notes": 20, "quality_scale_stable": True},
            {"notes": 100, "quality_scale_stable": False},
        ],
    )

    assert assessment["status"] == "candidate_evidence_wall"
    assert assessment["first_failed_notes"] == 100
    assert assessment["dominant_runtime_stage"] == "complete_gap_search"
    assert assessment["exact_cover_excluded_as_quality_limiter"] is True


def test_oracle_false_edge_deletion_preserves_true_core_edges():
    config = FractalTearConfig(
        notes=3,
        pieces_per_note=4,
        width=72,
        height=40,
        seed=41,
        serial_ocr_rate=0.0,
    )
    common = {
        "algorithm": "effectiveness",
        "use_labels": False,
        "min_overlap_pixels": 4,
        "beam_width": 8,
        "candidate_time_limit_seconds": None,
        "candidate_state_limit": 20_000,
        "cover_time_limit_seconds": None,
        "cover_node_limit": 20_000,
    }
    control = run_tearfit_trial(config, **common)
    intervention = run_tearfit_trial(
        config,
        **common,
        diagnostic_oracle_drop_false_accepted_edges=True,
    )

    assert intervention.true_accepted_edges == control.true_accepted_edges
    assert intervention.false_accepted_edges == 0
    assert (
        intervention.diagnostic_removed_false_core_edges
        == control.false_accepted_edges
    )
    assert intervention.accepted_edges == control.true_accepted_edges


def test_v433_oracle_gate_selects_one_causal_branch():
    control = {"oracle_candidate_recall": 0.84}
    rescued = assess_v433_oracle_false_edge_deletion(
        control,
        {"oracle_candidate_recall": 0.90},
    )
    not_rescued = assess_v433_oracle_false_edge_deletion(
        control,
        {"oracle_candidate_recall": 0.88},
    )

    assert rescued["status"] == "false_edge_contamination_limiter"
    assert rescued["falsification_gate_passed"] is True
    assert not_rescued["status"] == "gap_proposal_candidate_construction_wall"
    assert not_rescued["falsification_gate_passed"] is False


def test_v433_oracle_diagnostic_emits_control_and_intervention():
    payload = run_v433_oracle_false_edge_diagnostic(
        notes=2,
        pieces_per_note=4,
        seed=43,
        width=72,
        height=40,
        min_overlap_pixels=4,
        beam_width=8,
        candidate_states_per_pair_score=100.0,
        gap_states_per_fragment=100.0,
        partial_gap_states_per_fragment=25.0,
        cover_nodes_per_note=10_000.0,
    )

    assert payload["config"]["schema_version"] == "4.3.3"
    assert payload["control"]["false_accepted_edges"] >= 0
    assert payload["oracle_false_edge_deleted"]["false_accepted_edges"] == 0
    assert payload["assessment"]["status"] in {
        "false_edge_contamination_limiter",
        "gap_proposal_candidate_construction_wall",
    }


def test_v44_proposal_gate_requires_candidate_and_quality_rescue():
    control = {
        "oracle_candidate_recall": 0.84,
        "exact_yield": 0.84,
        "exact_precision": 0.98,
    }
    rescued = assess_v44_boundary_contact_proposal(
        control,
        {
            "oracle_candidate_recall": 0.90,
            "exact_yield": 0.90,
            "exact_precision": 0.97,
        },
    )
    not_rescued = assess_v44_boundary_contact_proposal(
        control,
        {
            "oracle_candidate_recall": 0.86,
            "exact_yield": 0.86,
            "exact_precision": 0.99,
            "complete_gap_saturated": False,
            "partial_gap_saturated": False,
        },
    )

    assert rescued["status"] == "weak_pair_proposal_gate_limiter"
    assert rescued["oracle_rescue_gate_passed"] is True
    assert rescued["quality_gate_passed"] is True
    assert not_rescued["status"] == "deeper_candidate_construction_wall"
    assert not_rescued["oracle_rescue_gate_passed"] is False


def test_v44_proposal_diagnostic_emits_single_variable_pair():
    payload = run_v44_boundary_contact_proposal_diagnostic(
        notes=2,
        pieces_per_note=4,
        seed=47,
        width=72,
        height=40,
        min_overlap_pixels=4,
        beam_width=8,
        candidate_states_per_pair_score=100.0,
        gap_states_per_fragment=100.0,
        partial_gap_states_per_fragment=25.0,
        cover_nodes_per_note=10_000.0,
    )

    assert payload["config"]["schema_version"] == "4.4.0-diagnostic"
    assert payload["weak_pair_control"]["gap_proposal_pool"] == "weak_pair"
    assert (
        payload["boundary_contact_intervention"]["gap_proposal_pool"]
        == "boundary_contact"
    )
    assert payload["assessment"]["status"] in {
        "weak_pair_proposal_gate_limiter",
        "proposal_recall_rescued_without_quality_rescue",
        "inconclusive_gap_budget_saturation",
        "deeper_candidate_construction_wall",
    }


def test_v44_candidate_funnel_diagnostic_reports_dominant_category():
    payload = run_v44_candidate_funnel_diagnostic(
        notes=2,
        pieces_per_note=4,
        seed=49,
        width=72,
        height=40,
        min_overlap_pixels=4,
        beam_width=8,
        candidate_states_per_pair_score=100.0,
        gap_states_per_fragment=100.0,
        partial_gap_states_per_fragment=25.0,
        cover_nodes_per_note=10_000.0,
    )

    assert payload["config"]["diagnostic"] == "truth_restricted_candidate_funnel"
    counts = payload["assessment"]["category_counts"]
    assert sum(counts.values()) == 2
    assert payload["trial"]["candidate_funnel"]["simulation_truth_only"] is True


def test_v44_core_connectivity_diagnostic_accounts_for_notes():
    payload = run_v44_core_connectivity_diagnostic(
        notes=3,
        pieces_per_note=5,
        seed=51,
        width=72,
        height=40,
        min_contiguous_pixels=2,
        automatic_contiguous_pixels=3,
        core_raw_coverage_threshold=0.6,
    )

    connectivity = payload["connectivity"]
    assert payload["config"]["diagnostic"] == "true_core_connectivity"
    assert connectivity["recordable_notes"] + connectivity["unrecordable_notes"] == 3
    assert len(connectivity["notes"]) == 3



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


def test_compute_residual_gap_components_finds_known_hole():
    """The complement of a partial core exposes exactly the missing region."""
    frame = np.ones((8, 8), dtype=bool)
    frame[3:5, 3:5] = False  # a known 2x2 interior hole
    cols = np.arange(8)
    left_mask = frame & (cols[None, :] < 4)
    right_mask = frame & (cols[None, :] >= 4)
    fragments = [
        Fragment(id="A", mask=left_mask),
        Fragment(id="B", mask=right_mask),
    ]
    core = AssemblyCandidate(
        fragment_ids=("A", "B"),
        coverage=0.95,
        raw_coverage=0.9375,
        score=1.0,
        support_pixels=0,
    )
    regions = compute_residual_gap_components(
        fragments, [core], coverage_threshold=0.93, gap_fill_radius=1
    )
    assert len(regions) == 1
    region = regions[0]
    assert region.area == 4
    assert int(region.mask.sum()) == 4
    assert set(region.adjacent_fragment_indices) == {0, 1}
    assert region.routing_class in {"simple", "moderate", "complex"}


def test_classify_gap_complexity_boundaries():
    """Routing thresholds separate simple, moderate, and complex gaps."""
    dummy = np.zeros((2, 2), dtype=bool)

    def region(area, perimeter, neighbours):
        return ResidualGapRegion(
            mask=dummy,
            area=area,
            perimeter_pixels=perimeter,
            component_id=0,
            adjacent_fragment_indices=tuple(range(neighbours)),
        )

    assert classify_gap_complexity(region(10, 12, 1)) == "simple"
    assert classify_gap_complexity(region(300, 80, 1)) == "complex"
    assert classify_gap_complexity(region(10, 12, 5)) == "complex"
    assert classify_gap_complexity(region(100, 120, 3)) == "moderate"


def test_gap_proposal_rejects_sliver_and_accepts_closed_gap():
    """E_proposal accepts a well-closed gap and rejects a sub-tolerance sliver."""
    frame_shape = (12, 12)
    full = np.ones(frame_shape, dtype=bool)
    gap = np.zeros(frame_shape, dtype=bool)
    gap[4:8, 4:8] = True  # 4x4 interior gap
    base_mask = full & ~gap
    filler_mask = gap.copy()
    sliver_mask = np.zeros(frame_shape, dtype=bool)
    sliver_mask[5, 5] = True
    fragments = [
        Fragment(id="base", mask=base_mask),
        Fragment(id="filler", mask=filler_mask),
        Fragment(id="sliver", mask=sliver_mask),
    ]
    boundary_evidence = tear_boundary_evidence(fragments, tolerance=2)
    region = ResidualGapRegion(
        mask=gap,
        area=int(gap.sum()),
        perimeter_pixels=int(gap.sum()),
        component_id=0,
        adjacent_fragment_indices=(0, 1, 2),
    )
    base_state = frozenset({0})
    base_union = base_mask.copy()

    accepted = evaluate_gap_proposal(
        fragments,
        base_state,
        base_union,
        (1,),
        region,
        gap_lookup={},
        boundary_evidence=boundary_evidence,
        tolerance=2,
        min_informative_scale=4,
    )
    rejected = evaluate_gap_proposal(
        fragments,
        base_state,
        base_union,
        (2,),
        region,
        gap_lookup={},
        boundary_evidence=boundary_evidence,
        tolerance=2,
        min_informative_scale=4,
    )

    assert isinstance(accepted, GapProposalEvaluation)
    assert accepted.accepted is True
    assert accepted.net_new_area == 16
    assert accepted.closed_perimeter_delta > 0.0
    assert accepted.effectiveness > 0.0
    assert rejected.accepted is False
    assert rejected.reason == "sub_tolerance_sliver"


def _v44_trial_kwargs():
    return dict(
        use_labels=False,
        min_overlap_pixels=4,
        beam_width=8,
        candidate_time_limit_seconds=None,
        candidate_state_limit=20_000,
        partial_gap_time_limit_seconds=None,
        gap_state_limit=4_000,
        partial_gap_state_limit=1_000,
        cover_time_limit_seconds=None,
        cover_node_limit=20_000,
    )


def test_v44_gap_first_valid_cover_and_v43_routed_fingerprint_unchanged():
    """v44_gap_first yields a valid disjoint cover; v43_routed stays bit-identical."""
    config = FractalTearConfig(
        notes=3,
        pieces_per_note=5,
        width=72,
        height=40,
        seed=30,
        serial_ocr_rate=0.0,
    )

    v44 = run_tearfit_trial(config, algorithm="v44_gap_first", **_v44_trial_kwargs())
    assert v44.candidates > 0
    assert 0.0 <= v44.oracle_candidate_recall <= 1.0
    assert v44.proposal_efficiency >= 0.0
    # The selected notes must be a disjoint (valid) exact cover.
    seen_fragment_ids: set[str] = set()
    for candidate in v44.diagnostics.confirmed_candidates:
        ids = set(candidate.fragment_ids)
        assert seen_fragment_ids.isdisjoint(ids)
        seen_fragment_ids |= ids
    # v44 stage timing is present and additive; group-gap stages remain.
    assert "gap_first" in v44.stage_timings

    # v43_routed must be identical to the algorithm it resolves to, proving the
    # additive v44 code path did not perturb the existing v4.3 pipeline.
    routed = run_tearfit_trial(config, algorithm="v43_routed", **_v44_trial_kwargs())
    resolved = routed.config["resolved_algorithm"]
    direct = run_tearfit_trial(config, algorithm=resolved, **_v44_trial_kwargs())
    assert routed.selected_solution_fingerprint == direct.selected_solution_fingerprint
    assert (
        routed.candidate_provenance_fingerprint
        == direct.candidate_provenance_fingerprint
    )
    assert "gap_first" not in direct.stage_timings
