"""The measurement-only core tools report what the core did, and nothing else."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from moneyrepair.tearfit import AssemblyCandidate

EXPERIMENTS = Path(__file__).resolve().parents[1] / "docs" / "experiments"
EQUIVALENCE = runpy.run_path(str(EXPERIMENTS / "core_equivalence.py"))
PROFILE = runpy.run_path(str(EXPERIMENTS / "core_profile.py"))
SCALING = runpy.run_path(str(EXPERIMENTS / "core_scaling.py"))
AUDIT = runpy.run_path(str(EXPERIMENTS / "cover_optimality_audit.py"))


def _dump(**items):
    base = {
        "scored_pairs": [{"pair": ("a", "b"), "evidence_level": "review", "effectiveness": 1.5}],
        "objective": {"count": 1, "score": 10.0},
        "search_counters": {"core.expanded_states": 7},
    }
    base.update(items)
    return {
        "case": {"notes": 1, "pieces": 2, "seed": 3, "strategy": "global"},
        "tree": "t",
        "machine": "m",
        "numpy": "n",
        "wall_seconds": 1.0,
        "peak_rss_mb": 1.0,
        "stage_timings": {},
        "items": base,
    }


def test_equivalence_compares_every_item_exactly_unless_a_tolerance_is_declared():
    compare = EQUIVALENCE["compare_dumps"]
    assert compare(_dump(), _dump(), 0.0)["equivalent"]

    signed_zero = _dump(objective={"count": 1, "score": -0.0})
    assert not compare(_dump(objective={"count": 1, "score": 0.0}), signed_zero, 0.0)["equivalent"]

    nudged = _dump(scored_pairs=[{"pair": ("a", "b"), "evidence_level": "review", "effectiveness": 1.5 + 1e-12}])
    assert not compare(_dump(), nudged, 0.0)["equivalent"]
    report = compare(_dump(), nudged, 1e-9)
    assert report["equivalent"]
    assert 0.0 < report["categories"]["scored_pairs"]["max_float_difference"] < 1e-9

    relabelled = _dump(scored_pairs=[{"pair": ("a", "b"), "evidence_level": "automatic", "effectiveness": 1.5}])
    report = compare(_dump(), relabelled, 1.0)
    assert not report["equivalent"]
    assert report["categories"]["scored_pairs"]["differences"] == 1

    shorter = _dump(scored_pairs=[])
    assert compare(_dump(), shorter, 0.0)["categories"]["scored_pairs"]["candidate_count"] == 0
    assert not compare(_dump(), shorter, 0.0)["equivalent"]


def test_equivalence_case_round_trip():
    case = EQUIVALENCE["parse_case"]("100:24:7:disjoint_round_robin")
    assert case == {"notes": 100, "pieces": 24, "seed": 7, "strategy": "disjoint_round_robin"}
    assert EQUIVALENCE["case_name"](case) == "n100_p24_s7_disjoint_round_robin"


def test_profile_counts_reconcile_with_the_core_counters():
    report = PROFILE["run"](4, 24, 3, "disjoint_round_robin", False)
    assert report["resolved_algorithm"] == "effectiveness_gap"
    funnel = report["pair_funnel"]
    assert funnel["scored_pairs_consistent"]
    assert all(size > 0 for size in report["retained_memory"].values())
    assert funnel["bbox_candidate_pairs"] - funnel["mask_overlap_dropped"] == funnel["scored_pairs"]
    evaluated = {
        stage: stats.get("proposal_evaluations", 0)
        for stage, stats in report["gap_search_stats"].items()
        if stats.get("proposal_evaluations", 0)
    }
    assert evaluated, "the case must exercise the gap search"
    assert set(report["gap_proposals"]) == set(evaluated)
    for stage, outcomes in report["gap_proposals"].items():
        stats = report["gap_search_stats"][stage]
        assert sum(outcomes.values()) == stats["proposal_evaluations"]
        accepted = outcomes.get("accepted_and_kept", 0) + outcomes.get(
            "accepted_then_pruned_for_union_overlap", 0
        )
        assert accepted == stats["accepted_proposals"]


def test_profile_leaves_the_core_unpatched():
    from moneyrepair import tearfit

    before = (
        tearfit.reconstruct_placed_fragments,
        tearfit.augment_candidates_with_group_gap,
        tearfit._AssemblyGeometry.score_against,
        tearfit._AssemblyGeometry.score_proposals,
    )
    PROFILE["run"](3, 8, 2, "global", False)
    after = (
        tearfit.reconstruct_placed_fragments,
        tearfit.augment_candidates_with_group_gap,
        tearfit._AssemblyGeometry.score_against,
        tearfit._AssemblyGeometry.score_proposals,
    )
    assert before == after


def test_scaling_row_reports_budgets_truncation_and_outcome():
    row = SCALING["run_case"](3, 8, 2, "disjoint_round_robin")
    assert row["case"]["notes"] == 3
    assert row["budgets"]["resolved_cover_node_limit"] == 75_000
    assert set(row["truncated"]) == {"core", "complete_gap", "partial_gap", "exact_cover", "any_time_limit"}
    assert row["truncated"]["any_time_limit"] is False
    assert 0.0 <= row["outcome"]["oracle_candidate_recall"] <= 1.0
    assert row["peak_rss_mb"] > 0


def _candidate(ids, score, labels=()):
    return AssemblyCandidate(
        fragment_ids=tuple(ids), coverage=1.0, raw_coverage=1.0, score=score,
        support_pixels=0, labels=tuple(labels),
    )


def test_packing_audit_is_lexicographic_and_respects_labels():
    pytest.importorskip("scipy")
    solve = AUDIT["solve_packing"]
    pool = [
        _candidate("ab", 3.0),
        _candidate("bc", 3.0),
        _candidate("c", 1.0),
        _candidate("a", 1.0),
        _candidate("d", 0.5),
    ]
    result = solve(pool, time_limit=60)
    assert result["score_optimal"]
    assert result["score"] == pytest.approx(4.5)
    assert result["count"] == 3

    labelled = [_candidate("a", 2.0, ["S1"]), _candidate("b", 2.0, ["S1"]), _candidate("c", 1.0)]
    result = solve(labelled, time_limit=60)
    assert result["score"] == pytest.approx(3.0)
    assert result["count"] == 2

    tie = [_candidate("ab", 2.0), _candidate("a", 1.0), _candidate("b", 1.0)]
    result = solve(tie, time_limit=60)
    assert result["score"] == pytest.approx(2.0)
    assert result["count"] == 2
