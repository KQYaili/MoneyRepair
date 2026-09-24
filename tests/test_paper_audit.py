from __future__ import annotations

from math import pi

import numpy as np
import pytest

from moneyrepair.diagnostics import (
    diagnose_pair_retrieval,
    mask_geometry_features,
    summarize_fragment_geometry,
)
from moneyrepair.paper_audit import run_paper_transfer_audit
from moneyrepair.types import Fragment


def _fragment(fragment_id: str, note_id: str, mask: np.ndarray | None = None) -> Fragment:
    if mask is None:
        mask = np.ones((2, 2), dtype=bool)
    return Fragment(fragment_id, mask, meta={"note_id": note_id})


def test_pair_retrieval_reports_perfect_same_note_ranking() -> None:
    fragments = [
        _fragment("a0", "a"),
        _fragment("a1", "a"),
        _fragment("b0", "b"),
        _fragment("b1", "b"),
    ]
    metrics = diagnose_pair_retrieval(
        fragments,
        [
            (0, 1, 0.90),
            (0, 2, 0.30),
            (0, 3, 0.20),
            (1, 2, 0.10),
            (1, 3, 0.40),
            (2, 3, 0.95),
        ],
    )

    assert metrics["candidate_graph_recall"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["recall_at_k"]["1"] == 1.0
    assert metrics["reciprocal_best_buddy_pairs"] == 2
    assert metrics["reciprocal_best_buddy_precision"] == 1.0


def test_pair_retrieval_charges_missing_positive_candidates() -> None:
    fragments = [
        _fragment("a0", "a"),
        _fragment("a1", "a"),
        _fragment("b0", "b"),
        _fragment("b1", "b"),
    ]
    metrics = diagnose_pair_retrieval(
        fragments,
        [(0, 2, 0.90), (1, 3, 0.80)],
    )

    assert metrics["queries_with_candidates"] == 4
    assert metrics["queries_with_positive_candidates"] == 0
    assert metrics["candidate_graph_recall"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["rank_histogram"]["missing"] == 4
    assert metrics["reciprocal_best_buddy_precision"] == 0.0


def test_rbb_predictions_do_not_depend_on_truth_query_eligibility() -> None:
    fragments = [
        _fragment("a0", "a"),
        _fragment("a1", "a"),
        _fragment("b0", "b"),
        _fragment("b1", "b"),
        _fragment("c0", "c"),
        _fragment("d0", "d"),
    ]
    metrics = diagnose_pair_retrieval(
        fragments,
        [
            (0, 1, 0.90),
            (2, 3, 0.80),
            (4, 5, 0.70),
        ],
    )

    assert metrics["eligible_queries"] == 4
    assert metrics["mrr"] == 1.0
    assert metrics["reciprocal_best_buddy_pairs"] == 3
    assert metrics["evaluable_reciprocal_best_buddy_pairs"] == 3
    assert metrics["true_reciprocal_best_buddy_pairs"] == 2
    assert metrics["false_reciprocal_best_buddy_pairs"] == 1
    assert metrics["reciprocal_best_buddy_precision"] == pytest.approx(2.0 / 3.0)


def test_rbb_precision_reports_unknown_truth_pairs_separately() -> None:
    fragments = [
        _fragment("a0", "a"),
        _fragment("a1", "a"),
        Fragment("unknown", np.ones((2, 2), dtype=bool)),
        _fragment("b0", "b"),
    ]
    metrics = diagnose_pair_retrieval(
        fragments,
        [(0, 1, 0.90), (2, 3, 0.80)],
    )

    assert metrics["reciprocal_best_buddy_pairs"] == 2
    assert metrics["evaluable_reciprocal_best_buddy_pairs"] == 1
    assert metrics["unknown_truth_reciprocal_best_buddy_pairs"] == 1
    assert metrics["reciprocal_best_buddy_precision"] == 1.0


def test_mask_geometry_uses_pixel_cells_and_detects_concavity() -> None:
    square = np.ones((2, 2), dtype=bool)
    square_features = mask_geometry_features(square)
    assert square_features["area_pixels"] == 4.0
    assert square_features["perimeter_pixels"] == 8.0
    assert square_features["convex_hull_area_pixels"] == 4.0
    assert square_features["convex_solidity"] == 1.0
    assert square_features["circularity"] == pytest.approx(pi / 4.0)

    concave = np.array([[1, 1], [1, 0]], dtype=bool)
    concave_features = mask_geometry_features(concave)
    assert concave_features["convex_solidity"] < 1.0
    assert concave_features["concavity_ratio"] > 0.0


def test_geometry_summary_and_paper_audit_are_measurement_only() -> None:
    fragments = [
        _fragment("a0", "a", np.ones((2, 2), dtype=bool)),
        _fragment("a1", "a", np.array([[1, 1], [1, 0]], dtype=bool)),
    ]
    summary = summarize_fragment_geometry(fragments)
    assert summary["measurement_only"] is True
    assert summary["fragments"] == 2
    assert summary["mask_shapes"] == {"2x2": 2}
    assert summary["features"]["area_pixels"]["median"] == 3.5

    audit = run_paper_transfer_audit(
        notes=2,
        pieces_per_note=4,
        width=48,
        height=24,
        seed=7,
    )
    assert audit["production_behavior_changed"] is False
    assert audit["pair_retrieval"]["all_scored"]["evaluation_truth_only"] is True
    assert audit["pair_retrieval"]["automatic_only"]["evaluation_truth_only"] is True
    assert audit["fragment_geometry"]["fragments"] == 8
    assert audit["component_oracle"]["simulation_truth_only"] is True
    assert audit["decision_boundary"]["next_stage"] == "physical pilot acquisition"
