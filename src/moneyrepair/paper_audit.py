from __future__ import annotations

from collections import Counter
from time import monotonic

from moneyrepair.diagnostics import (
    diagnose_pair_retrieval,
    summarize_fragment_geometry,
)
from moneyrepair.tearfit import (
    FractalTearConfig,
    diagnose_true_core_connectivity,
    make_fractal_tear_fragments,
    score_absolute_tear_pairs,
)


def run_paper_transfer_audit(
    *,
    notes: int = 20,
    pieces_per_note: int = 24,
    seed: int = 7,
    width: int = 180,
    height: int = 90,
    roughness: float = 4.0,
    fray_layers: int = 2,
    fray_probability: float = 0.18,
    tolerance: int = 2,
    min_effectiveness: float = 1.0,
    automatic_effectiveness: float = 2.0,
    min_contiguous_pixels: int = 3,
    automatic_contiguous_pixels: int = 5,
    core_raw_coverage_threshold: float = 0.78,
) -> dict:
    """Run literature-derived diagnostics without changing reconstruction.

    The audit transfers three measurement ideas only: ranked pair retrieval,
    explicit mask-shape distributions, and component-level oracle coverage.
    Simulator truth is read after scoring and never changes an edge or candidate.
    """

    started = monotonic()
    config = FractalTearConfig(
        notes=notes,
        pieces_per_note=pieces_per_note,
        width=width,
        height=height,
        seed=seed,
        roughness=roughness,
        fray_layers=fray_layers,
        fray_probability=fray_probability,
        serial_ocr_rate=0.0,
    )
    simulation_started = monotonic()
    _template, fragments = make_fractal_tear_fragments(config)
    simulation_seconds = monotonic() - simulation_started

    geometry_started = monotonic()
    geometry = summarize_fragment_geometry(fragments)
    geometry_seconds = monotonic() - geometry_started

    scoring_started = monotonic()
    scores, _accepted = score_absolute_tear_pairs(
        fragments,
        tolerance=tolerance,
        use_labels=False,
        scoring="effectiveness",
        min_effectiveness=min_effectiveness,
        automatic_effectiveness=automatic_effectiveness,
        min_contiguous_pixels=min_contiguous_pixels,
        automatic_contiguous_pixels=automatic_contiguous_pixels,
    )
    scoring_seconds = monotonic() - scoring_started

    diagnostics_started = monotonic()
    retrieval = {
        "all_scored": diagnose_pair_retrieval(
            fragments,
            ((edge.left, edge.right, edge.effectiveness) for edge in scores),
        ),
        "review_or_better": diagnose_pair_retrieval(
            fragments,
            (
                (edge.left, edge.right, edge.effectiveness)
                for edge in scores
                if edge.evidence_level != "insufficient-evidence"
            ),
        ),
        "automatic_only": diagnose_pair_retrieval(
            fragments,
            (
                (edge.left, edge.right, edge.effectiveness)
                for edge in scores
                if edge.evidence_level == "automatic"
            ),
        ),
    }
    connectivity = diagnose_true_core_connectivity(
        fragments,
        scores,
        minimum_raw_coverage=core_raw_coverage_threshold,
    )
    evidence_levels = Counter(edge.evidence_level for edge in scores)
    diagnostics_seconds = monotonic() - diagnostics_started

    return {
        "schema_version": "1.0",
        "mode": "measurement-only literature transfer audit",
        "production_behavior_changed": False,
        "config": {
            **config.__dict__,
            "tolerance": tolerance,
            "min_effectiveness": min_effectiveness,
            "automatic_effectiveness": automatic_effectiveness,
            "min_contiguous_pixels": min_contiguous_pixels,
            "automatic_contiguous_pixels": automatic_contiguous_pixels,
            "core_raw_coverage_threshold": core_raw_coverage_threshold,
        },
        "transfers": {
            "generic_hybrid_framework": (
                "Top-k/MRR and reciprocal-best-buddy diagnostics; no GA and no "
                "whole-fragment appearance identity"
            ),
            "missing_gap": (
                "explicit mask-shape distribution audit; no claim that generated "
                "single-fragment shapes reproduce paired paper tears"
            ),
            "erl_mpp": (
                "component-level oracle accounting; no RL policy and no grid-puzzlet "
                "assumption"
            ),
        },
        "pair_scores": len(scores),
        "evidence_levels": dict(sorted(evidence_levels.items())),
        "pair_retrieval": retrieval,
        "fragment_geometry": geometry,
        "component_oracle": connectivity,
        "decision_boundary": {
            "may_unfreeze_production_core": False,
            "reason": (
                "STATUS.md requires reliable real-mask and true-pose handoff before "
                "reconsidering component bridges or learned tear descriptors"
            ),
            "next_stage": "physical pilot acquisition",
        },
        "timings_seconds": {
            "simulation": simulation_seconds,
            "geometry": geometry_seconds,
            "pair_scoring": scoring_seconds,
            "diagnostics": diagnostics_seconds,
            "total": monotonic() - started,
        },
    }
