from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from statistics import fmean
from time import monotonic
from typing import Callable, Iterable

from moneyrepair.tearfit import (
    FractalTearConfig,
    TEARFIT_ALGORITHMS,
    TEARFIT_V43_FINE_FRACTION,
    TearFitTrialResult,
    diagnose_true_core_connectivity,
    make_fractal_tear_fragments,
    run_tearfit_trial,
    run_tearfit_v43_ablation,
    score_absolute_tear_pairs,
)


V432_QUALITY_THRESHOLDS = {
    "minimum_precision": 0.97,
    "maximum_precision_drop": 0.02,
    "minimum_yield": 0.80,
    "maximum_yield_drop": 0.10,
    "maximum_manual_fraction": 0.20,
}

V433_ORACLE_RESCUE_THRESHOLD = 0.05
V44_PROPOSAL_RESCUE_THRESHOLD = 0.05
V44_MAX_PRECISION_DROP = 0.02
V433_SEED7_NORMALIZED_RATES = {
    "candidate_states_per_pair_score": 12.691965985531159,
    "gap_states_per_fragment": 83.33333333333333,
    "partial_gap_states_per_fragment": 20.833333333333332,
    "cover_nodes_per_note": 25_000.0,
}


def _case_key(
    phase: str,
    *,
    notes: int,
    seed: int,
    budget_factor: int | None = None,
) -> str:
    suffix = f":factor={budget_factor}" if budget_factor is not None else ""
    return f"{phase}:notes={notes}:seed={seed}{suffix}"


def _row_stable(previous: dict, current: dict) -> bool:
    return (
        previous["selected_solution_fingerprint"]
        == current["selected_solution_fingerprint"]
        and previous["candidate_provenance_fingerprint"]
        == current["candidate_provenance_fingerprint"]
        and previous["exact_yield"] == current["exact_yield"]
        and previous["exact_precision"] == current["exact_precision"]
        and abs(
            float(previous["selected_score_total"])
            - float(current["selected_score_total"])
        )
        <= 1e-9
    )


def _mean(rows: list[dict], field: str) -> float:
    return float(fmean(float(row[field]) for row in rows))


def _saturation_rate(rows: list[dict], stage: str, field: str) -> float:
    return float(
        fmean(bool(row["search_stats"][stage].get(field, False)) for row in rows)
    )


def _summarize_scale_rows(rows: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    keys = sorted(
        {
            (str(row["track"]), int(row["notes"]), str(row["algorithm"]))
            for row in rows
            if row["track"] in {"fixed", "normalized"}
        }
    )
    for track, notes, algorithm in keys:
        selected = [
            row
            for row in rows
            if row["track"] == track
            and row["notes"] == notes
            and row["algorithm"] == algorithm
        ]
        stage_names = (
            "simulation",
            "pair_scoring",
            "core_search",
            "complete_gap_search",
            "partial_gap_search",
            "exact_cover",
            "diagnostics",
            "total",
        )
        summaries.append(
            {
                "track": track,
                "notes": notes,
                "algorithm": algorithm,
                "runs": len(selected),
                "mean_exact_yield": _mean(selected, "exact_yield"),
                "mean_exact_precision": _mean(selected, "exact_precision"),
                "mean_automatic_exact_yield": _mean(
                    selected, "automatic_exact_yield"
                ),
                "mean_automatic_exact_precision": _mean(
                    selected, "automatic_exact_precision"
                ),
                "mean_manual_notes_remaining": _mean(
                    selected, "manual_notes_remaining"
                ),
                "mean_manual_fraction": _mean(
                    [
                        {
                            "value": row["manual_notes_remaining"]
                            / max(1, row["notes"])
                        }
                        for row in selected
                    ],
                    "value",
                ),
                "mean_pair_scores": _mean(selected, "pair_scores"),
                "mean_accepted_edges": _mean(selected, "accepted_edges"),
                "mean_true_possible_edges": _mean(
                    selected, "true_possible_edges"
                ),
                "mean_true_accepted_edges": _mean(
                    selected, "true_accepted_edges"
                ),
                "mean_true_edge_recall": _mean(selected, "true_edge_recall"),
                "mean_false_accepted_edges": _mean(
                    selected, "false_accepted_edges"
                ),
                "mean_false_edge_rate": _mean(selected, "false_edge_rate"),
                "mean_candidates": _mean(selected, "candidates"),
                "mean_gap_candidates": _mean(selected, "gap_candidates"),
                "mean_true_gap_candidates": _mean(
                    selected, "true_gap_candidates"
                ),
                "mean_false_gap_candidates": _mean(
                    selected, "false_gap_candidates"
                ),
                "mean_selected_gap_candidates": _mean(
                    selected, "selected_gap_candidates"
                ),
                "mean_selected_true_gap_candidates": _mean(
                    selected, "selected_true_gap_candidates"
                ),
                "mean_selected_false_gap_candidates": _mean(
                    selected, "selected_false_gap_candidates"
                ),
                "mean_oracle_candidate_recall": _mean(
                    selected, "oracle_candidate_recall"
                ),
                "core_state_saturation_rate": _saturation_rate(
                    selected, "core", "state_limit_reached"
                ),
                "complete_gap_state_saturation_rate": _saturation_rate(
                    selected, "complete_gap", "state_limit_reached"
                ),
                "partial_gap_state_saturation_rate": _saturation_rate(
                    selected, "partial_gap", "state_limit_reached"
                ),
                "exact_cover_saturation_rate": _saturation_rate(
                    selected, "exact_cover", "node_limit_reached"
                ),
                "mean_stage_seconds": {
                    stage: float(
                        fmean(float(row["stage_timings"].get(stage, 0.0)) for row in selected)
                    )
                    for stage in stage_names
                },
            }
        )
    return summaries


def _quality_assessment(summary: list[dict]) -> list[dict]:
    normalized_routed = sorted(
        (
            row
            for row in summary
            if row["track"] == "normalized" and row["algorithm"] == "v43_routed"
        ),
        key=lambda row: row["notes"],
    )
    if not normalized_routed:
        return []
    anchor = normalized_routed[0]
    assessments = []
    for row in normalized_routed:
        precision_drop = anchor["mean_exact_precision"] - row["mean_exact_precision"]
        yield_drop = anchor["mean_exact_yield"] - row["mean_exact_yield"]
        checks = {
            "minimum_precision": row["mean_exact_precision"]
            >= V432_QUALITY_THRESHOLDS["minimum_precision"],
            "maximum_precision_drop": precision_drop
            <= V432_QUALITY_THRESHOLDS["maximum_precision_drop"],
            "minimum_yield": row["mean_exact_yield"]
            >= V432_QUALITY_THRESHOLDS["minimum_yield"],
            "maximum_yield_drop": yield_drop
            <= V432_QUALITY_THRESHOLDS["maximum_yield_drop"],
            "maximum_manual_fraction": row["mean_manual_fraction"]
            <= V432_QUALITY_THRESHOLDS["maximum_manual_fraction"],
        }
        assessments.append(
            {
                "notes": row["notes"],
                "precision_drop_from_anchor": precision_drop,
                "yield_drop_from_anchor": yield_drop,
                "checks": checks,
                "quality_scale_stable": all(checks.values()),
            }
        )
    return assessments


def _mechanism_assessment(summary: list[dict]) -> list[dict]:
    by_key = {
        (row["track"], row["notes"], row["algorithm"]): row for row in summary
    }
    assessments = []
    for track, notes, algorithm in sorted(by_key):
        if algorithm != "baseline":
            continue
        baseline = by_key[(track, notes, "baseline")]
        effectiveness = by_key.get((track, notes, "effectiveness"))
        gap = by_key.get((track, notes, "effectiveness_gap"))
        if effectiveness is None:
            continue
        baseline_false_rate = float(baseline["mean_false_edge_rate"])
        false_rate_reduction = (
            1.0
            - float(effectiveness["mean_false_edge_rate"]) / baseline_false_rate
            if baseline_false_rate > 0.0
            else 0.0
        )
        item = {
            "track": track,
            "notes": notes,
            "false_edge_rate_reduction": false_rate_reduction,
            "edge_cleanup_retained": false_rate_reduction >= 0.40,
        }
        if gap is not None:
            candidate_base = max(1.0, float(effectiveness["mean_candidates"]))
            candidate_inflation = (
                float(gap["mean_candidates"])
                - float(effectiveness["mean_candidates"])
            ) / candidate_base
            yield_gain = (
                float(gap["mean_exact_yield"])
                - float(effectiveness["mean_exact_yield"])
            )
            precision_gain = (
                float(gap["mean_exact_precision"])
                - float(effectiveness["mean_exact_precision"])
            )
            item.update(
                gap_candidate_inflation=candidate_inflation,
                gap_yield_gain=yield_gain,
                gap_precision_gain=precision_gain,
                gap_mechanism_retained=(
                    candidate_inflation <= 0.05
                    and (yield_gain >= 0.05 or precision_gain >= 0.02)
                ),
            )
        assessments.append(item)
    return assessments


def assess_v432_bottleneck(
    summary: list[dict], quality_assessment: list[dict]
) -> dict:
    routed = {
        int(row["notes"]): row
        for row in summary
        if row["track"] == "normalized" and row["algorithm"] == "v43_routed"
    }
    failed = next(
        (
            item
            for item in sorted(quality_assessment, key=lambda value: value["notes"])
            if not item["quality_scale_stable"]
        ),
        None,
    )
    if failed is None:
        return {
            "status": "not_reached",
            "first_failed_notes": None,
            "statement": "No measured normalized-compute point crossed the preregistered quality boundary.",
        }
    notes = int(failed["notes"])
    row = routed[notes]
    anchor = routed[min(routed)]
    timings = {
        stage: float(value)
        for stage, value in row["mean_stage_seconds"].items()
        if stage not in {"total", "simulation", "diagnostics"}
    }
    dominant_stage = max(timings, key=lambda stage: timings[stage]) if timings else None
    oracle_selection_gap = float(row["mean_oracle_candidate_recall"]) - float(
        row["mean_exact_yield"]
    )
    candidate_search_unsaturated = (
        float(row["core_state_saturation_rate"]) == 0.0
        and float(row["complete_gap_state_saturation_rate"]) == 0.0
        and float(row["partial_gap_state_saturation_rate"]) == 0.0
    )
    missing_candidate_fraction = 1.0 - float(row["mean_oracle_candidate_recall"])
    false_edge_rate_increase = float(row["mean_false_edge_rate"]) - float(
        anchor["mean_false_edge_rate"]
    )
    exact_cover_excluded = (
        abs(oracle_selection_gap) <= 0.01
        and float(row["mean_stage_seconds"]["exact_cover"])
        <= 0.10 * max(1e-9, float(row["mean_stage_seconds"]["total"]))
    )
    candidate_evidence_wall = (
        candidate_search_unsaturated
        and missing_candidate_fraction >= 0.05
        and exact_cover_excluded
    )
    status = (
        "candidate_evidence_wall"
        if candidate_evidence_wall
        else "unresolved_stage_failure"
    )
    return {
        "status": status,
        "first_failed_notes": notes,
        "dominant_runtime_stage": dominant_stage,
        "candidate_search_unsaturated": candidate_search_unsaturated,
        "missing_candidate_fraction": missing_candidate_fraction,
        "oracle_selection_gap": oracle_selection_gap,
        "false_edge_rate_increase_from_anchor": false_edge_rate_increase,
        "true_edge_recall_drop_from_anchor": float(anchor["mean_true_edge_recall"])
        - float(row["mean_true_edge_recall"]),
        "exact_cover_excluded_as_quality_limiter": exact_cover_excluded,
        "statement": (
            "Correct assemblies are absent before exact cover despite unsaturated candidate/gap search; "
            "the measured wall is in edge discrimination or gap proposal, not final selection."
            if candidate_evidence_wall
            else "The first failed point needs a single-stage rescue before assigning a dominant wall."
        ),
    }


def _tag_rows(
    rows: list[dict],
    *,
    track: str,
    case_key: str,
    budget_factor: int | None = None,
) -> list[dict]:
    tagged = deepcopy(rows)
    for row in tagged:
        row["track"] = track
        row["case_key"] = case_key
        row["budget_factor"] = budget_factor
    return tagged


def run_v432_scale_protocol(
    *,
    notes_list: Iterable[int] = (20, 50, 100, 200),
    seeds: Iterable[int] = (7, 8, 9),
    algorithms: Iterable[str] = TEARFIT_ALGORITHMS,
    pieces_per_note: int = 24,
    anchor_notes: int = 20,
    anchor_budget_factors: Iterable[int] = (1, 2, 4, 8),
    full_mechanism_through: int = 100,
    endpoint_algorithms: Iterable[str] = ("baseline", "v43_routed"),
    route_fragment_fraction_threshold: float = TEARFIT_V43_FINE_FRACTION,
    width: int = 180,
    height: int = 90,
    tolerance: int = 2,
    min_overlap_pixels: int = 14,
    min_effectiveness: float = 1.0,
    automatic_effectiveness: float = 2.0,
    min_contiguous_pixels: int = 3,
    automatic_contiguous_pixels: int = 5,
    coverage_threshold: float = 0.93,
    core_raw_coverage_threshold: float | None = None,
    gap_fill_radius: int = 2,
    beam_width: int = 32,
    max_partial_core_candidates: int = 128,
    candidate_state_limit: int = 100_000,
    gap_state_limit: int = 20_000,
    partial_gap_state_limit: int = 5_000,
    cover_node_limit: int = 250_000,
    resume_cases: Iterable[dict] = (),
    case_sink: Callable[[dict], None] | None = None,
) -> dict:
    """Run the preregistered v4.3.2 anchor and scale-fineness protocol.

    The fixed track preserves one absolute compute contract. The normalized
    track is emitted only after adjacent anchor staircase levels produce the
    same solution and candidate-provenance fingerprints for every case.
    """

    notes_values = tuple(dict.fromkeys(int(value) for value in notes_list))
    seed_values = tuple(dict.fromkeys(int(value) for value in seeds))
    algorithm_values = tuple(dict.fromkeys(str(value) for value in algorithms))
    factor_values = tuple(dict.fromkeys(int(value) for value in anchor_budget_factors))
    endpoint_values = tuple(dict.fromkeys(str(value) for value in endpoint_algorithms))
    if anchor_notes not in notes_values:
        raise ValueError("anchor_notes must be present in notes_list")
    if not seed_values or not algorithm_values or not factor_values:
        raise ValueError("seeds, algorithms, and anchor_budget_factors must be non-empty")
    if any(value <= 0 for value in (*notes_values, *factor_values)):
        raise ValueError("notes and anchor budget factors must be positive")
    if factor_values != tuple(sorted(factor_values)):
        raise ValueError("anchor budget factors must be increasing")
    unknown = set(algorithm_values) - set(TEARFIT_ALGORITHMS)
    if unknown:
        raise ValueError(f"unknown algorithms: {sorted(unknown)}")
    unknown_endpoint = set(endpoint_values) - set(algorithm_values)
    if unknown_endpoint:
        raise ValueError(
            f"endpoint algorithms are not enabled: {sorted(unknown_endpoint)}"
        )

    protocol_identity = {
        "schema_version": "4.3.2",
        "algorithms": algorithm_values,
        "pieces_per_note": pieces_per_note,
        "anchor_notes": anchor_notes,
        "full_mechanism_through": full_mechanism_through,
        "endpoint_algorithms": endpoint_values,
        "route_fragment_fraction_threshold": route_fragment_fraction_threshold,
        "width": width,
        "height": height,
        "tolerance": tolerance,
        "min_overlap_pixels": min_overlap_pixels,
        "min_effectiveness": min_effectiveness,
        "automatic_effectiveness": automatic_effectiveness,
        "min_contiguous_pixels": min_contiguous_pixels,
        "automatic_contiguous_pixels": automatic_contiguous_pixels,
        "coverage_threshold": coverage_threshold,
        "core_raw_coverage_threshold": core_raw_coverage_threshold,
        "gap_fill_radius": gap_fill_radius,
        "beam_width": beam_width,
        "max_partial_core_candidates": max_partial_core_candidates,
        "candidate_state_limit": candidate_state_limit,
        "gap_state_limit": gap_state_limit,
        "partial_gap_state_limit": partial_gap_state_limit,
        "cover_node_limit": cover_node_limit,
    }
    protocol_id = sha256(
        json.dumps(protocol_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    cases: dict[str, dict] = {}
    case_order: list[str] = []
    for item in resume_cases:
        if item.get("protocol_id") != protocol_id:
            raise ValueError(
                "checkpoint protocol does not match the current v4.3.2 arguments"
            )
        key = str(item["case_key"])
        if key not in cases:
            cases[key] = deepcopy(item)
            case_order.append(key)

    def emit(record: dict) -> dict:
        key = str(record["case_key"])
        if key not in cases:
            cases[key] = record
            case_order.append(key)
            if case_sink is not None:
                case_sink(deepcopy(record))
        return cases[key]

    def run_case(
        *,
        phase: str,
        notes: int,
        seed: int,
        selected_algorithms: tuple[str, ...],
        budget_factor: int | None = None,
        normalized_rates: dict[str, float] | None = None,
        reused_rows: list[dict] | None = None,
    ) -> dict:
        key = _case_key(
            phase,
            notes=notes,
            seed=seed,
            budget_factor=budget_factor,
        )
        if key in cases:
            return cases[key]
        if reused_rows is None:
            if normalized_rates is None:
                factor = budget_factor or 1
                candidate_limit: int | None = candidate_state_limit * factor
                candidate_per_pair: float | None = None
                gap_limit: int | None = gap_state_limit * factor
                gap_per_fragment: float | None = None
                partial_gap_limit: int | None = partial_gap_state_limit * factor
                partial_gap_per_fragment: float | None = None
                cover_limit: int | None = cover_node_limit * factor
                cover_per_note: float | None = None
            else:
                candidate_limit = None
                candidate_per_pair = normalized_rates[
                    "candidate_states_per_pair_score"
                ]
                gap_limit = None
                gap_per_fragment = normalized_rates["gap_states_per_fragment"]
                partial_gap_limit = None
                partial_gap_per_fragment = normalized_rates[
                    "partial_gap_states_per_fragment"
                ]
                cover_limit = None
                cover_per_note = normalized_rates["cover_nodes_per_note"]
            payload = run_tearfit_v43_ablation(
                notes=notes,
                pieces_list=(pieces_per_note,),
                seeds=(seed,),
                algorithms=selected_algorithms,
                route_fragment_fraction_threshold=route_fragment_fraction_threshold,
                width=width,
                height=height,
                tolerance=tolerance,
                min_overlap_pixels=min_overlap_pixels,
                min_effectiveness=min_effectiveness,
                automatic_effectiveness=automatic_effectiveness,
                min_contiguous_pixels=min_contiguous_pixels,
                automatic_contiguous_pixels=automatic_contiguous_pixels,
                coverage_threshold=coverage_threshold,
                core_raw_coverage_threshold=core_raw_coverage_threshold,
                gap_fill_radius=gap_fill_radius,
                beam_width=beam_width,
                max_partial_core_candidates=max_partial_core_candidates,
                candidate_time_limit_seconds=None,
                candidate_state_limit=candidate_limit,
                candidate_states_per_pair_score=candidate_per_pair,
                partial_gap_time_limit_seconds=None,
                gap_state_limit=gap_limit,
                gap_states_per_fragment=gap_per_fragment,
                partial_gap_state_limit=partial_gap_limit,
                partial_gap_states_per_fragment=partial_gap_per_fragment,
                cover_time_limit_seconds=None,
                cover_node_limit=cover_limit,
                cover_nodes_per_note=cover_per_note,
            )
            source_rows = payload["rows"]
        else:
            source_rows = [
                row for row in reused_rows if row["algorithm"] in selected_algorithms
            ]
        rows = _tag_rows(
            source_rows,
            track=phase,
            case_key=key,
            budget_factor=budget_factor,
        )
        return emit(
            {
                "case_key": key,
                "protocol_id": protocol_id,
                "phase": phase,
                "notes": notes,
                "seed": seed,
                "budget_factor": budget_factor,
                "algorithms": selected_algorithms,
                "normalized_rates": normalized_rates,
                "rows": rows,
            }
        )

    for factor in factor_values:
        for seed in seed_values:
            run_case(
                phase="anchor",
                notes=anchor_notes,
                seed=seed,
                selected_algorithms=algorithm_values,
                budget_factor=factor,
            )

    anchor_rows_by_factor: dict[int, dict[tuple[int, str], dict]] = {}
    for factor in factor_values:
        indexed: dict[tuple[int, str], dict] = {}
        for seed in seed_values:
            record = cases[
                _case_key(
                    "anchor",
                    notes=anchor_notes,
                    seed=seed,
                    budget_factor=factor,
                )
            ]
            for row in record["rows"]:
                indexed[(seed, row["algorithm"])] = row
        anchor_rows_by_factor[factor] = indexed

    stability_checks = []
    selected_anchor_factor: int | None = None
    for previous_factor, current_factor in zip(factor_values, factor_values[1:]):
        previous = anchor_rows_by_factor[previous_factor]
        current = anchor_rows_by_factor[current_factor]
        keys_match = set(previous) == set(current)
        stable_cases = {
            f"seed={seed}:algorithm={algorithm}": (
                keys_match and _row_stable(previous[(seed, algorithm)], current[(seed, algorithm)])
            )
            for seed, algorithm in sorted(set(previous) | set(current))
            if (seed, algorithm) in previous and (seed, algorithm) in current
        }
        stable = keys_match and bool(stable_cases) and all(stable_cases.values())
        stability_checks.append(
            {
                "lower_factor": previous_factor,
                "upper_factor": current_factor,
                "stable": stable,
                "case_stability": stable_cases,
            }
        )
        if stable and selected_anchor_factor is None:
            selected_anchor_factor = previous_factor

    active_scale_case_keys: list[str] = []
    factor_one_rows = anchor_rows_by_factor[factor_values[0]]
    for notes in notes_values:
        for seed_index, seed in enumerate(seed_values):
            selected_algorithms = (
                algorithm_values
                if notes <= full_mechanism_through or seed_index == 0
                else tuple(
                    algorithm
                    for algorithm in algorithm_values
                    if algorithm in endpoint_values
                )
            )
            reused = None
            if notes == anchor_notes:
                reused = [
                    factor_one_rows[(seed, algorithm)]
                    for algorithm in selected_algorithms
                ]
            record = run_case(
                phase="fixed",
                notes=notes,
                seed=seed,
                selected_algorithms=selected_algorithms,
                reused_rows=reused,
            )
            active_scale_case_keys.append(record["case_key"])

    normalization_rates: dict[int, dict[str, float]] = {}
    if selected_anchor_factor is not None:
        anchor_rows = anchor_rows_by_factor[selected_anchor_factor]
        selected_candidate_limit = candidate_state_limit * selected_anchor_factor
        selected_gap_limit = gap_state_limit * selected_anchor_factor
        selected_partial_gap_limit = partial_gap_state_limit * selected_anchor_factor
        selected_cover_limit = cover_node_limit * selected_anchor_factor
        for seed in seed_values:
            reference = anchor_rows[(seed, algorithm_values[0])]
            normalization_rates[seed] = {
                "candidate_states_per_pair_score": selected_candidate_limit
                / max(1, reference["pair_scores"]),
                "gap_states_per_fragment": selected_gap_limit
                / max(1, reference["fragments"]),
                "partial_gap_states_per_fragment": selected_partial_gap_limit
                / max(1, reference["fragments"]),
                "cover_nodes_per_note": selected_cover_limit / anchor_notes,
            }
        for notes in notes_values:
            for seed_index, seed in enumerate(seed_values):
                selected_algorithms = (
                    algorithm_values
                    if notes <= full_mechanism_through or seed_index == 0
                    else tuple(
                        algorithm
                        for algorithm in algorithm_values
                        if algorithm in endpoint_values
                    )
                )
                reused = None
                if notes == anchor_notes:
                    reused = [
                        anchor_rows[(seed, algorithm)]
                        for algorithm in selected_algorithms
                    ]
                record = run_case(
                    phase="normalized",
                    notes=notes,
                    seed=seed,
                    selected_algorithms=selected_algorithms,
                    budget_factor=selected_anchor_factor,
                    normalized_rates=normalization_rates[seed],
                    reused_rows=reused,
                )
                active_scale_case_keys.append(record["case_key"])

    rows = [
        row
        for key in active_scale_case_keys
        for row in cases[key]["rows"]
    ]
    summary = _summarize_scale_rows(rows)
    quality_assessment = _quality_assessment(summary)
    return {
        "config": {
            "schema_version": "4.3.2",
            "protocol_id": protocol_id,
            "notes_list": notes_values,
            "seeds": seed_values,
            "algorithms": algorithm_values,
            "pieces_per_note": pieces_per_note,
            "anchor_notes": anchor_notes,
            "anchor_budget_factors": factor_values,
            "full_mechanism_through": full_mechanism_through,
            "endpoint_algorithms": endpoint_values,
            "base_budgets": {
                "candidate_state_limit": candidate_state_limit,
                "gap_state_limit": gap_state_limit,
                "partial_gap_state_limit": partial_gap_state_limit,
                "cover_node_limit": cover_node_limit,
            },
            "quality_thresholds": V432_QUALITY_THRESHOLDS,
            "time_limits": None,
            "simulation_boundary": "placed fragments; no locator or OCR errors",
        },
        "anchor_calibration": {
            "stable": selected_anchor_factor is not None,
            "selected_factor": selected_anchor_factor,
            "stability_checks": stability_checks,
            "normalization_rates_by_seed": normalization_rates,
            "failure_statement": (
                None
                if selected_anchor_factor is not None
                else "N=anchor remains computationally truncated; normalized scaling is not identifiable."
            ),
        },
        "cases": [cases[key] for key in case_order],
        "rows": rows,
        "summary": summary,
        "quality_assessment": quality_assessment,
        "mechanism_assessment": _mechanism_assessment(summary),
        "bottleneck_assessment": assess_v432_bottleneck(
            summary, quality_assessment
        ),
    }


def _v433_trial_snapshot(result: TearFitTrialResult) -> dict:
    diagnostics = result.diagnostics
    return {
        "oracle_candidate_recall": result.oracle_candidate_recall,
        "exact_yield": diagnostics.exact_yield,
        "exact_precision": diagnostics.exact_precision,
        "manual_notes_remaining": diagnostics.manual_notes_remaining,
        "accepted_edges": result.accepted_edges,
        "true_accepted_edges": result.true_accepted_edges,
        "false_accepted_edges": result.false_accepted_edges,
        "false_edge_rate": result.false_edge_rate,
        "candidates": result.candidates,
        "core_candidates": result.core_candidates,
        "gap_candidates": result.gap_candidates,
        "true_gap_candidates": result.true_gap_candidates,
        "false_gap_candidates": result.false_gap_candidates,
        "selected_true_gap_candidates": result.selected_true_gap_candidates,
        "selected_false_gap_candidates": result.selected_false_gap_candidates,
        "diagnostic_removed_false_accepted_edges": (
            result.diagnostic_removed_false_accepted_edges
        ),
        "diagnostic_removed_false_core_edges": (
            result.diagnostic_removed_false_core_edges
        ),
        "search_stats": deepcopy(result.search_stats),
        "stage_timings": dict(result.stage_timings),
        "selected_solution_fingerprint": result.selected_solution_fingerprint,
        "candidate_provenance_fingerprint": (
            result.candidate_provenance_fingerprint
        ),
    }


def assess_v433_oracle_false_edge_deletion(
    control: dict,
    intervention: dict,
    *,
    minimum_oracle_rescue: float = V433_ORACLE_RESCUE_THRESHOLD,
) -> dict:
    """Classify the final v4.3 candidate-evidence falsification gate."""

    if not (0.0 < minimum_oracle_rescue <= 1.0):
        raise ValueError("minimum_oracle_rescue must be in (0, 1]")
    control_oracle = float(control["oracle_candidate_recall"])
    intervention_oracle = float(intervention["oracle_candidate_recall"])
    delta = intervention_oracle - control_oracle
    rescued = delta + 1e-12 >= minimum_oracle_rescue
    if rescued:
        status = "false_edge_contamination_limiter"
        statement = (
            "Deleting accepted cross-note edges rescues enough exact candidates "
            "to identify false-edge contamination as the current candidate-evidence limiter."
        )
        v44_priority = "reduce_false_pair_and_candidate_workload"
    else:
        status = "gap_proposal_candidate_construction_wall"
        statement = (
            "Deleting every accepted cross-note edge does not rescue enough exact "
            "candidates; stop the false-pair route and prioritize gap proposal and "
            "candidate construction."
        )
        v44_priority = "improve_gap_proposal_and_candidate_construction"
    return {
        "status": status,
        "minimum_oracle_rescue": minimum_oracle_rescue,
        "control_oracle_candidate_recall": control_oracle,
        "intervention_oracle_candidate_recall": intervention_oracle,
        "oracle_candidate_recall_delta": delta,
        "falsification_gate_passed": rescued,
        "v44_priority": v44_priority,
        "statement": statement,
    }


def run_v433_oracle_false_edge_diagnostic(
    *,
    notes: int = 100,
    pieces_per_note: int = 24,
    seed: int = 7,
    width: int = 180,
    height: int = 90,
    route_fragment_fraction_threshold: float = TEARFIT_V43_FINE_FRACTION,
    tolerance: int = 2,
    min_overlap_pixels: int = 14,
    min_effectiveness: float = 1.0,
    automatic_effectiveness: float = 2.0,
    min_contiguous_pixels: int = 3,
    automatic_contiguous_pixels: int = 5,
    coverage_threshold: float = 0.93,
    core_raw_coverage_threshold: float | None = None,
    gap_fill_radius: int = 2,
    beam_width: int = 32,
    max_partial_core_candidates: int = 128,
    candidate_states_per_pair_score: float = V433_SEED7_NORMALIZED_RATES[
        "candidate_states_per_pair_score"
    ],
    gap_states_per_fragment: float = V433_SEED7_NORMALIZED_RATES[
        "gap_states_per_fragment"
    ],
    partial_gap_states_per_fragment: float = V433_SEED7_NORMALIZED_RATES[
        "partial_gap_states_per_fragment"
    ],
    cover_nodes_per_note: float = V433_SEED7_NORMALIZED_RATES[
        "cover_nodes_per_note"
    ],
    minimum_oracle_rescue: float = V433_ORACLE_RESCUE_THRESHOLD,
) -> dict:
    """Run the control and oracle false-edge deletion counterfactual.

    This command is diagnostic only: it consumes simulator ``note_id`` truth
    and cannot be used on real fragments or in the production pipeline.
    """

    if notes < 1 or pieces_per_note < 2:
        raise ValueError("notes and pieces_per_note must be positive")
    config = FractalTearConfig(
        notes=notes,
        pieces_per_note=pieces_per_note,
        width=width,
        height=height,
        seed=seed,
        serial_ocr_rate=0.0,
    )
    common = {
        "algorithm": "v43_routed",
        "route_fragment_fraction_threshold": route_fragment_fraction_threshold,
        "tolerance": tolerance,
        "min_overlap_pixels": min_overlap_pixels,
        "min_effectiveness": min_effectiveness,
        "automatic_effectiveness": automatic_effectiveness,
        "min_contiguous_pixels": min_contiguous_pixels,
        "automatic_contiguous_pixels": automatic_contiguous_pixels,
        "coverage_threshold": coverage_threshold,
        "core_raw_coverage_threshold": core_raw_coverage_threshold,
        "gap_fill_radius": gap_fill_radius,
        "beam_width": beam_width,
        "max_partial_core_candidates": max_partial_core_candidates,
        "use_labels": False,
        "candidate_time_limit_seconds": None,
        "candidate_state_limit": None,
        "candidate_states_per_pair_score": candidate_states_per_pair_score,
        "partial_gap_time_limit_seconds": None,
        "gap_state_limit": None,
        "gap_states_per_fragment": gap_states_per_fragment,
        "partial_gap_state_limit": None,
        "partial_gap_states_per_fragment": partial_gap_states_per_fragment,
        "cover_time_limit_seconds": None,
        "cover_node_limit": None,
        "cover_nodes_per_note": cover_nodes_per_note,
    }
    control_result = run_tearfit_trial(config, **common)
    intervention_result = run_tearfit_trial(
        config,
        **common,
        diagnostic_oracle_drop_false_accepted_edges=True,
    )
    control = _v433_trial_snapshot(control_result)
    intervention = _v433_trial_snapshot(intervention_result)
    assessment = assess_v433_oracle_false_edge_deletion(
        control,
        intervention,
        minimum_oracle_rescue=minimum_oracle_rescue,
    )
    return {
        "config": {
            "schema_version": "4.3.3",
            "diagnostic": "oracle_false_edge_deletion",
            "notes": notes,
            "pieces_per_note": pieces_per_note,
            "seed": seed,
            "width": width,
            "height": height,
            "algorithm": "v43_routed",
            "normalized_rates": {
                "candidate_states_per_pair_score": candidate_states_per_pair_score,
                "gap_states_per_fragment": gap_states_per_fragment,
                "partial_gap_states_per_fragment": (
                    partial_gap_states_per_fragment
                ),
                "cover_nodes_per_note": cover_nodes_per_note,
            },
            "minimum_oracle_rescue": minimum_oracle_rescue,
            "simulation_boundary": (
                "placed fragments with simulator note_id truth; no locator or OCR errors"
            ),
        },
        "control": control,
        "oracle_false_edge_deleted": intervention,
        "comparison": {
            "oracle_candidate_recall_delta": (
                intervention["oracle_candidate_recall"]
                - control["oracle_candidate_recall"]
            ),
            "exact_yield_delta": intervention["exact_yield"] - control["exact_yield"],
            "candidate_count_delta": intervention["candidates"] - control["candidates"],
            "gap_candidate_count_delta": (
                intervention["gap_candidates"] - control["gap_candidates"]
            ),
            "complete_gap_seconds_delta": (
                intervention["stage_timings"]["complete_gap_search"]
                - control["stage_timings"]["complete_gap_search"]
            ),
            "total_seconds_delta": (
                intervention["stage_timings"]["total"]
                - control["stage_timings"]["total"]
            ),
        },
        "assessment": assessment,
    }


def _v44_trial_snapshot(result: TearFitTrialResult) -> dict:
    snapshot = _v433_trial_snapshot(result)
    snapshot["gap_proposal_pool"] = result.config["gap_proposal_pool"]
    snapshot["gap_proposal_stats"] = {
        stage: {
            key: int(stats.get(key, 0))
            for key in (
                "weak_pair_edges",
                "proposal_edges",
                "proposal_evaluations",
                "accepted_proposals",
                "no_weak_edge_evaluations",
                "accepted_no_weak_edge_proposals",
            )
        }
        for stage, stats in result.search_stats.items()
        if stage in {"complete_gap", "partial_gap"}
    }
    if result.candidate_funnel:
        snapshot["candidate_funnel"] = result.candidate_funnel
    return snapshot


def assess_v44_boundary_contact_proposal(
    control: dict,
    intervention: dict,
    *,
    minimum_oracle_rescue: float = V44_PROPOSAL_RESCUE_THRESHOLD,
    minimum_yield_rescue: float = V44_PROPOSAL_RESCUE_THRESHOLD,
    maximum_precision_drop: float = V44_MAX_PRECISION_DROP,
) -> dict:
    """Classify the one-variable boundary-contact proposal intervention."""

    if not (0.0 < minimum_oracle_rescue <= 1.0):
        raise ValueError("minimum_oracle_rescue must be in (0, 1]")
    if not (0.0 < minimum_yield_rescue <= 1.0):
        raise ValueError("minimum_yield_rescue must be in (0, 1]")
    if not (0.0 <= maximum_precision_drop <= 1.0):
        raise ValueError("maximum_precision_drop must be in [0, 1]")

    oracle_delta = float(intervention["oracle_candidate_recall"]) - float(
        control["oracle_candidate_recall"]
    )
    yield_delta = float(intervention["exact_yield"]) - float(control["exact_yield"])
    precision_drop = float(control["exact_precision"]) - float(
        intervention["exact_precision"]
    )
    oracle_rescued = oracle_delta + 1e-12 >= minimum_oracle_rescue
    quality_rescued = (
        yield_delta + 1e-12 >= minimum_yield_rescue
        and precision_drop <= maximum_precision_drop + 1e-12
    )
    gap_saturated = any(
        bool(intervention.get(key, False))
        for key in ("complete_gap_saturated", "partial_gap_saturated")
    )

    if oracle_rescued and quality_rescued:
        status = "weak_pair_proposal_gate_limiter"
        statement = (
            "Admitting boundary-contact fragments to the unchanged whole-assembly "
            "scorer rescues exact candidates and selected notes without an excessive "
            "precision loss; the weak-pair proposal gate is a measured limiter."
        )
    elif oracle_rescued:
        status = "proposal_recall_rescued_without_quality_rescue"
        statement = (
            "Boundary-contact proposals restore enough exact candidates, but the "
            "selected solution does not pass the preregistered yield/precision gate."
        )
    elif gap_saturated:
        status = "inconclusive_gap_budget_saturation"
        statement = (
            "The broader proposal pool does not meet the rescue gate and exhausts a "
            "gap-search budget, so this run cannot distinguish proposal quality from "
            "search capacity."
        )
    else:
        status = "deeper_candidate_construction_wall"
        statement = (
            "Removing the weak-pair eligibility gate does not rescue enough exact "
            "candidates under unsaturated gap search; the remaining wall lies in "
            "partial-core selection, whole-assembly scoring, or beam construction."
        )
    return {
        "status": status,
        "minimum_oracle_rescue": minimum_oracle_rescue,
        "minimum_yield_rescue": minimum_yield_rescue,
        "maximum_precision_drop": maximum_precision_drop,
        "oracle_candidate_recall_delta": oracle_delta,
        "exact_yield_delta": yield_delta,
        "exact_precision_drop": precision_drop,
        "oracle_rescue_gate_passed": oracle_rescued,
        "quality_gate_passed": quality_rescued,
        "intervention_gap_saturated": gap_saturated,
        "statement": statement,
    }


def run_v44_boundary_contact_proposal_diagnostic(
    *,
    notes: int = 100,
    pieces_per_note: int = 24,
    seed: int = 7,
    width: int = 180,
    height: int = 90,
    route_fragment_fraction_threshold: float = TEARFIT_V43_FINE_FRACTION,
    tolerance: int = 2,
    min_overlap_pixels: int = 14,
    min_effectiveness: float = 1.0,
    automatic_effectiveness: float = 2.0,
    min_contiguous_pixels: int = 3,
    automatic_contiguous_pixels: int = 5,
    coverage_threshold: float = 0.93,
    core_raw_coverage_threshold: float | None = None,
    gap_fill_radius: int = 2,
    beam_width: int = 32,
    max_partial_core_candidates: int = 128,
    candidate_states_per_pair_score: float = V433_SEED7_NORMALIZED_RATES[
        "candidate_states_per_pair_score"
    ],
    gap_states_per_fragment: float = V433_SEED7_NORMALIZED_RATES[
        "gap_states_per_fragment"
    ],
    partial_gap_states_per_fragment: float = V433_SEED7_NORMALIZED_RATES[
        "partial_gap_states_per_fragment"
    ],
    cover_nodes_per_note: float = V433_SEED7_NORMALIZED_RATES[
        "cover_nodes_per_note"
    ],
    minimum_oracle_rescue: float = V44_PROPOSAL_RESCUE_THRESHOLD,
    minimum_yield_rescue: float = V44_PROPOSAL_RESCUE_THRESHOLD,
    maximum_precision_drop: float = V44_MAX_PRECISION_DROP,
) -> dict:
    """Test whether the v4.3 weak-pair gate hides useful group-gap proposals."""

    if notes < 1 or pieces_per_note < 2:
        raise ValueError("notes and pieces_per_note must be positive")
    config = FractalTearConfig(
        notes=notes,
        pieces_per_note=pieces_per_note,
        width=width,
        height=height,
        seed=seed,
        serial_ocr_rate=0.0,
    )
    common = {
        "algorithm": "v43_routed",
        "route_fragment_fraction_threshold": route_fragment_fraction_threshold,
        "tolerance": tolerance,
        "min_overlap_pixels": min_overlap_pixels,
        "min_effectiveness": min_effectiveness,
        "automatic_effectiveness": automatic_effectiveness,
        "min_contiguous_pixels": min_contiguous_pixels,
        "automatic_contiguous_pixels": automatic_contiguous_pixels,
        "coverage_threshold": coverage_threshold,
        "core_raw_coverage_threshold": core_raw_coverage_threshold,
        "gap_fill_radius": gap_fill_radius,
        "beam_width": beam_width,
        "max_partial_core_candidates": max_partial_core_candidates,
        "use_labels": False,
        "candidate_time_limit_seconds": None,
        "candidate_state_limit": None,
        "candidate_states_per_pair_score": candidate_states_per_pair_score,
        "partial_gap_time_limit_seconds": None,
        "gap_state_limit": None,
        "gap_states_per_fragment": gap_states_per_fragment,
        "partial_gap_state_limit": None,
        "partial_gap_states_per_fragment": partial_gap_states_per_fragment,
        "cover_time_limit_seconds": None,
        "cover_node_limit": None,
        "cover_nodes_per_note": cover_nodes_per_note,
    }
    control_result = run_tearfit_trial(
        config,
        **common,
        gap_proposal_pool="weak_pair",
    )
    intervention_result = run_tearfit_trial(
        config,
        **common,
        gap_proposal_pool="boundary_contact",
    )
    control = _v44_trial_snapshot(control_result)
    intervention = _v44_trial_snapshot(intervention_result)
    assessment = assess_v44_boundary_contact_proposal(
        control,
        intervention,
        minimum_oracle_rescue=minimum_oracle_rescue,
        minimum_yield_rescue=minimum_yield_rescue,
        maximum_precision_drop=maximum_precision_drop,
    )
    return {
        "config": {
            "schema_version": "4.4.0-diagnostic",
            "diagnostic": "boundary_contact_proposal_pool",
            "notes": notes,
            "pieces_per_note": pieces_per_note,
            "seed": seed,
            "width": width,
            "height": height,
            "algorithm": "v43_routed",
            "control_proposal_pool": "weak_pair",
            "intervention_proposal_pool": "boundary_contact",
            "unchanged_components": [
                "pair_scoring",
                "whole_assembly_scorer",
                "beam_width",
                "state_and_node_budgets",
                "exact_cover",
            ],
            "normalized_rates": {
                "candidate_states_per_pair_score": candidate_states_per_pair_score,
                "gap_states_per_fragment": gap_states_per_fragment,
                "partial_gap_states_per_fragment": (
                    partial_gap_states_per_fragment
                ),
                "cover_nodes_per_note": cover_nodes_per_note,
            },
            "minimum_oracle_rescue": minimum_oracle_rescue,
            "minimum_yield_rescue": minimum_yield_rescue,
            "maximum_precision_drop": maximum_precision_drop,
            "simulation_boundary": (
                "placed fragments; boundary contact is proposal-only; no locator "
                "or OCR errors"
            ),
        },
        "weak_pair_control": control,
        "boundary_contact_intervention": intervention,
        "assessment": assessment,
    }


def run_v44_candidate_funnel_diagnostic(
    *,
    notes: int = 100,
    pieces_per_note: int = 24,
    seed: int = 7,
    width: int = 180,
    height: int = 90,
    route_fragment_fraction_threshold: float = TEARFIT_V43_FINE_FRACTION,
    tolerance: int = 2,
    min_overlap_pixels: int = 14,
    min_effectiveness: float = 1.0,
    automatic_effectiveness: float = 2.0,
    min_contiguous_pixels: int = 3,
    automatic_contiguous_pixels: int = 5,
    coverage_threshold: float = 0.93,
    core_raw_coverage_threshold: float | None = None,
    gap_fill_radius: int = 2,
    beam_width: int = 32,
    max_complete_core_candidates: int = 512,
    max_partial_core_candidates: int = 128,
    candidate_states_per_pair_score: float = V433_SEED7_NORMALIZED_RATES[
        "candidate_states_per_pair_score"
    ],
    gap_states_per_fragment: float = V433_SEED7_NORMALIZED_RATES[
        "gap_states_per_fragment"
    ],
    partial_gap_states_per_fragment: float = V433_SEED7_NORMALIZED_RATES[
        "partial_gap_states_per_fragment"
    ],
    cover_nodes_per_note: float = V433_SEED7_NORMALIZED_RATES[
        "cover_nodes_per_note"
    ],
) -> dict:
    """Run one production-equivalent trial plus a truth-restricted funnel audit."""

    config = FractalTearConfig(
        notes=notes,
        pieces_per_note=pieces_per_note,
        width=width,
        height=height,
        seed=seed,
        serial_ocr_rate=0.0,
    )
    result = run_tearfit_trial(
        config,
        algorithm="v43_routed",
        route_fragment_fraction_threshold=route_fragment_fraction_threshold,
        tolerance=tolerance,
        min_overlap_pixels=min_overlap_pixels,
        min_effectiveness=min_effectiveness,
        automatic_effectiveness=automatic_effectiveness,
        min_contiguous_pixels=min_contiguous_pixels,
        automatic_contiguous_pixels=automatic_contiguous_pixels,
        coverage_threshold=coverage_threshold,
        core_raw_coverage_threshold=core_raw_coverage_threshold,
        gap_fill_radius=gap_fill_radius,
        beam_width=beam_width,
        max_complete_core_candidates=max_complete_core_candidates,
        max_partial_core_candidates=max_partial_core_candidates,
        gap_proposal_pool="weak_pair",
        use_labels=False,
        candidate_time_limit_seconds=None,
        candidate_state_limit=None,
        candidate_states_per_pair_score=candidate_states_per_pair_score,
        partial_gap_time_limit_seconds=None,
        gap_state_limit=None,
        gap_states_per_fragment=gap_states_per_fragment,
        partial_gap_state_limit=None,
        partial_gap_states_per_fragment=partial_gap_states_per_fragment,
        cover_time_limit_seconds=None,
        cover_node_limit=None,
        cover_nodes_per_note=cover_nodes_per_note,
        diagnostic_candidate_funnel=True,
    )
    snapshot = _v44_trial_snapshot(result)
    funnel = result.candidate_funnel
    categories = dict(funnel["category_counts"])
    unresolved = {
        key: value
        for key, value in categories.items()
        if key not in {"core_exact", "production_gap_exact"}
    }
    max_count = max(unresolved.values(), default=0)
    dominant = sorted(
        key for key, value in unresolved.items() if value == max_count and value > 0
    )
    return {
        "config": {
            "schema_version": "4.4.0-diagnostic",
            "diagnostic": "truth_restricted_candidate_funnel",
            "notes": notes,
            "pieces_per_note": pieces_per_note,
            "seed": seed,
            "algorithm": "v43_routed",
            "gap_proposal_pool": "weak_pair",
            "max_complete_core_candidates": max_complete_core_candidates,
            "max_partial_core_candidates": max_partial_core_candidates,
            "simulation_boundary": (
                "placed fragments; simulator note_id is used only after production "
                "candidate generation to replay missing notes"
            ),
        },
        "trial": snapshot,
        "assessment": {
            "category_counts": categories,
            "unresolved_category_counts": unresolved,
            "dominant_unresolved_categories": dominant,
            "statement": (
                "The category counts localize missing exact candidates among core "
                "base construction, base selection, whole-assembly scoring, and "
                "cross-note beam competition without feeding truth to production."
            ),
        },
    }


def run_v44_core_connectivity_diagnostic(
    *,
    notes: int = 100,
    pieces_per_note: int = 24,
    seed: int = 7,
    width: int = 180,
    height: int = 90,
    tolerance: int = 2,
    min_effectiveness: float = 1.0,
    automatic_effectiveness: float = 2.0,
    min_contiguous_pixels: int = 3,
    automatic_contiguous_pixels: int = 5,
    core_raw_coverage_threshold: float = 0.78,
) -> dict:
    """Audit whether automatic true-edge components can reach the core threshold."""

    started = monotonic()
    config = FractalTearConfig(
        notes=notes,
        pieces_per_note=pieces_per_note,
        width=width,
        height=height,
        seed=seed,
        serial_ocr_rate=0.0,
    )
    _template, fragments = make_fractal_tear_fragments(config)
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
    pair_scoring_seconds = monotonic() - scoring_started
    connectivity = diagnose_true_core_connectivity(
        fragments,
        scores,
        minimum_raw_coverage=core_raw_coverage_threshold,
    )
    return {
        "config": {
            "schema_version": "4.4.0-diagnostic",
            "diagnostic": "true_core_connectivity",
            "notes": notes,
            "pieces_per_note": pieces_per_note,
            "seed": seed,
            "width": width,
            "height": height,
            "tolerance": tolerance,
            "min_effectiveness": min_effectiveness,
            "automatic_effectiveness": automatic_effectiveness,
            "min_contiguous_pixels": min_contiguous_pixels,
            "automatic_contiguous_pixels": automatic_contiguous_pixels,
            "core_raw_coverage_threshold": core_raw_coverage_threshold,
            "simulation_boundary": (
                "placed fragments; note_id only filters true automatic edges for "
                "post-hoc connected-component analysis"
            ),
        },
        "pair_scores": len(scores),
        "connectivity": connectivity,
        "timings": {
            "pair_scoring": pair_scoring_seconds,
            "total": monotonic() - started,
        },
    }
