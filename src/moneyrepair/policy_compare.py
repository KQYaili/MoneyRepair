from __future__ import annotations

from typing import Iterable

import numpy as np

from moneyrepair.llm_control import LLMAgentConfig, MockStrategy, llm_guided_assembly_loop
from moneyrepair.tearfit import (
    AssemblyCandidate,
    FractalTearConfig,
    TearFitComparisonCase,
    TearFitDiagnostics,
    diagnose_confirmed_candidates,
    generate_assembly_candidates,
    make_fractal_tear_fragments,
    score_absolute_tear_pairs,
    select_exact_cover_candidates,
    tearfit_comparison_cases,
)
from moneyrepair.types import Fragment

POLICY_COMPARE_STRATEGIES = (
    "static_score",
    "static_count",
    "static_broaden",
    "llm_balanced",
    "llm_coverage_first",
    "llm_score_first",
    "llm_broaden_search",
)


def _chimera_rate(diagnostics: TearFitDiagnostics) -> float:
    return diagnostics.chimeras / diagnostics.confirmed if diagnostics.confirmed else 0.0


def _policy_score(rows: list[dict]) -> tuple[float, float, float, float, float]:
    if not rows:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    precisions = [float(row["exact_precision"]) for row in rows]
    yields = [float(row["exact_yield"]) for row in rows]
    chimera_rates = [float(row["chimera_rate"]) for row in rows]
    manual_remaining = [float(row["manual_notes_remaining"]) for row in rows]
    return (
        min(precisions),
        sum(precisions) / len(precisions),
        sum(yields) / len(yields),
        -sum(chimera_rates) / len(chimera_rates),
        -sum(manual_remaining) / len(manual_remaining),
    )


def _selected_to_row(
    *,
    case: TearFitComparisonCase,
    serial_ocr_rate: float,
    policy: str,
    fragments: list[Fragment],
    accepted_edges: int,
    false_edge_rate: float,
    pair_scores: int,
    candidates: list[AssemblyCandidate] | None,
    selected: list[AssemblyCandidate],
) -> dict:
    diagnostics = diagnose_confirmed_candidates(selected, fragments)
    return {
        "case": case.name,
        "notes": case.notes,
        "pieces_per_note": case.pieces_per_note,
        "fray_probability": case.fray_probability,
        "serial_ocr_rate": serial_ocr_rate,
        "policy": policy,
        "fragments": len(fragments),
        "pair_scores": pair_scores,
        "accepted_edges": accepted_edges,
        "false_edge_rate": false_edge_rate,
        "candidates": len(candidates) if candidates is not None else None,
        "confirmed": diagnostics.confirmed,
        "exact_confirmed": diagnostics.exact_confirmed,
        "chimeras": diagnostics.chimeras,
        "chimera_rate": _chimera_rate(diagnostics),
        "exact_precision": diagnostics.exact_precision,
        "pure_precision": diagnostics.pure_precision,
        "exact_yield": diagnostics.exact_yield,
        "manual_notes_remaining": diagnostics.manual_notes_remaining,
    }


def _static_policy_selection(
    policy: str,
    fragments: list[Fragment],
    edges,
    *,
    coverage_threshold: float,
    gap_fill_radius: int,
    max_pieces: int,
    beam_width: int,
    candidate_time_limit_seconds: float | None,
    cover_time_limit_seconds: float | None,
) -> tuple[list[AssemblyCandidate], list[AssemblyCandidate]]:
    if policy == "static_score":
        threshold = coverage_threshold
        objective = "score_then_count"
        width = beam_width
    elif policy == "static_count":
        threshold = coverage_threshold
        objective = "count_then_score"
        width = beam_width
    elif policy == "static_broaden":
        threshold = max(0.5, coverage_threshold - 0.05)
        objective = "count_then_score"
        width = beam_width * 2
    else:
        raise ValueError(f"unknown static policy: {policy}")

    candidates = generate_assembly_candidates(
        fragments,
        edges,
        coverage_threshold=threshold,
        gap_fill_radius=gap_fill_radius,
        max_pieces=max_pieces,
        beam_width=width,
        seed_strategy="anchor_priority",
        time_limit_seconds=candidate_time_limit_seconds,
    )
    selected = select_exact_cover_candidates(
        candidates,
        time_limit_seconds=cover_time_limit_seconds,
        objective=objective,
    )
    return candidates, selected


def _llm_policy_selection(
    policy: str,
    fragments: list[Fragment],
    edges,
    *,
    coverage_threshold: float,
    max_pieces: int,
    beam_width: int,
    candidate_time_limit_seconds: float | None,
    cover_time_limit_seconds: float | None,
) -> list[AssemblyCandidate]:
    mock_strategy = policy.removeprefix("llm_")
    if mock_strategy not in {"balanced", "coverage_first", "score_first", "broaden_search"}:
        raise ValueError(f"unknown LLM policy: {policy}")
    time_limit = max(1.0, (candidate_time_limit_seconds or 0.0) + (cover_time_limit_seconds or 0.0))
    return llm_guided_assembly_loop(
        fragments,
        edges,
        LLMAgentConfig(use_mock=True, mock_strategy=mock_strategy),  # type: ignore[arg-type]
        max_iterations=3,
        coverage_threshold=coverage_threshold,
        max_pieces=max_pieces,
        beam_width=beam_width,
        time_limit_seconds=time_limit,
    )


def run_policy_controller_comparison(
    *,
    profile: str = "smoke",
    policies: Iterable[str] = POLICY_COMPARE_STRATEGIES,
    serial_ocr_rates: Iterable[float] = (0.6,),
    width: int = 120,
    height: int = 64,
    seed: int = 7,
    min_overlap_pixels: int = 10,
    tolerance: int = 2,
    coverage_threshold: float = 0.93,
    gap_fill_radius: int = 2,
    beam_width: int = 48,
    ensure_serial_anchor: bool = False,
    candidate_time_limit_seconds: float | None = 10.0,
    cover_time_limit_seconds: float | None = 5.0,
) -> dict:
    """Compare static exact-cover and LLM-guided search policies head-to-head."""

    requested_policies = tuple(policy.strip() for policy in policies if policy.strip())
    unknown = sorted(set(requested_policies) - set(POLICY_COMPARE_STRATEGIES))
    if unknown:
        raise ValueError(f"unknown policies: {', '.join(unknown)}")

    cases = tearfit_comparison_cases(profile)
    rates = tuple(float(rate) for rate in serial_ocr_rates)
    rows: list[dict] = []

    for case_index, case in enumerate(cases):
        for rate_index, rate in enumerate(rates):
            config = FractalTearConfig(
                notes=case.notes,
                pieces_per_note=case.pieces_per_note,
                width=width,
                height=height,
                seed=seed + case_index * 1009 + rate_index * 131,
                roughness=case.roughness,
                fray_probability=case.fray_probability,
                ensure_serial_anchor=ensure_serial_anchor,
                serial_ocr_rate=rate,
            )
            _template, fragments = make_fractal_tear_fragments(config)
            all_scores, _raw_edges = score_absolute_tear_pairs(
                fragments,
                tolerance=tolerance,
                min_overlap_pixels=1,
                use_labels=False,
            )
            _label_filtered_scores, edges = score_absolute_tear_pairs(
                fragments,
                tolerance=tolerance,
                min_overlap_pixels=min_overlap_pixels,
                use_labels=True,
            )
            false_edges = [
                edge
                for edge in edges
                if fragments[edge.left].meta.get("note_id") != fragments[edge.right].meta.get("note_id")
            ]
            false_edge_rate = len(false_edges) / len(edges) if edges else 0.0
            max_pieces = case.pieces_per_note + 2

            for policy in requested_policies:
                if policy.startswith("static_"):
                    candidates, selected = _static_policy_selection(
                        policy,
                        fragments,
                        edges,
                        coverage_threshold=coverage_threshold,
                        gap_fill_radius=gap_fill_radius,
                        max_pieces=max_pieces,
                        beam_width=beam_width,
                        candidate_time_limit_seconds=candidate_time_limit_seconds,
                        cover_time_limit_seconds=cover_time_limit_seconds,
                    )
                else:
                    candidates = None
                    selected = _llm_policy_selection(
                        policy,
                        fragments,
                        edges,
                        coverage_threshold=coverage_threshold,
                        max_pieces=max_pieces,
                        beam_width=beam_width,
                        candidate_time_limit_seconds=candidate_time_limit_seconds,
                        cover_time_limit_seconds=cover_time_limit_seconds,
                    )

                rows.append(
                    _selected_to_row(
                        case=case,
                        serial_ocr_rate=rate,
                        policy=policy,
                        fragments=fragments,
                        accepted_edges=len(edges),
                        false_edge_rate=false_edge_rate,
                        pair_scores=len(all_scores),
                        candidates=candidates,
                        selected=selected,
                    )
                )

    by_policy: dict[str, list[dict]] = {policy: [] for policy in requested_policies}
    for row in rows:
        by_policy[row["policy"]].append(row)

    summary = []
    for policy, policy_rows in by_policy.items():
        score = _policy_score(policy_rows)
        precisions = [float(row["exact_precision"]) for row in policy_rows]
        yields = [float(row["exact_yield"]) for row in policy_rows]
        chimera_rates = [float(row["chimera_rate"]) for row in policy_rows]
        summary.append(
            {
                "policy": policy,
                "min_exact_precision": min(precisions) if precisions else 0.0,
                "mean_exact_precision": float(np.mean(precisions)) if precisions else 0.0,
                "mean_exact_yield": float(np.mean(yields)) if yields else 0.0,
                "mean_chimera_rate": float(np.mean(chimera_rates)) if chimera_rates else 0.0,
                "score_tuple": score,
            }
        )
    summary.sort(key=lambda item: item["score_tuple"], reverse=True)
    best_policy = summary[0]["policy"] if summary else None
    for item in summary:
        item.pop("score_tuple", None)

    return {
        "config": {
            "profile": profile,
            "policies": requested_policies,
            "serial_ocr_rates": rates,
            "width": width,
            "height": height,
            "seed": seed,
            "min_overlap_pixels": min_overlap_pixels,
            "tolerance": tolerance,
            "coverage_threshold": coverage_threshold,
            "gap_fill_radius": gap_fill_radius,
            "beam_width": beam_width,
            "ensure_serial_anchor": ensure_serial_anchor,
            "candidate_time_limit_seconds": candidate_time_limit_seconds,
            "cover_time_limit_seconds": cover_time_limit_seconds,
        },
        "rows": rows,
        "summary": summary,
        "best_policy": best_policy,
    }
