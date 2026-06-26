from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter
from typing import Any

import numpy as np

from moneyrepair.compat import compute_compatibility_clustered, compute_compatibility_fast
from moneyrepair.diagnostics import diagnose_groups, diagnose_solutions
from moneyrepair.fingerprint import cluster_fragments_by_appearance
from moneyrepair.simulate import make_multi_note_fragments
from moneyrepair.solver import solve_covering_sets


def _diagnosis_summary(prefix: str, diagnosis: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "solutions",
        "chimeras",
        "pure",
        "chimera_rate",
        "pure_notes_found_count",
        "pure_notes_found_rate",
        "exactly_recovered_count",
        "exactly_recovered_rate",
        "uniquely_exact_recovered_count",
        "uniquely_exact_recovered_rate",
    )
    return {f"{prefix}_{key}": diagnosis[key] for key in keys if key in diagnosis}


def run_pressure_case(
    *,
    notes: int,
    appearance_spread: float,
    seed: int,
    pieces_per_note: int = 8,
    width: int = 160,
    height: int = 90,
    coverage: float = 0.95,
    max_solutions: int = 20,
    time_limit: float | None = 30.0,
    order_strategy: str = "area_degree",
    discriminate_tolerance: float = 0.05,
    wear_model: str = "global_gain",
    noise_sigma: float = 4.0,
    local_wear_strength: float = 0.0,
    gamma_spread: float = 0.0,
    stain_count: int = 0,
    stain_strength: float = 0.0,
) -> dict[str, Any]:
    """Run one chimera pressure case and return flat JSON-ready metrics."""

    started = perf_counter()
    template, fragments = make_multi_note_fragments(
        notes=notes,
        pieces_per_note=pieces_per_note,
        width=width,
        height=height,
        seed=seed,
        appearance_spread=appearance_spread,
        noise_sigma=noise_sigma,
        wear_model=wear_model,
        local_wear_strength=local_wear_strength,
        gamma_spread=gamma_spread,
        stain_count=stain_count,
        stain_strength=stain_strength,
    )
    generated_seconds = perf_counter() - started

    started = perf_counter()
    groups = cluster_fragments_by_appearance(fragments, template, tolerance=discriminate_tolerance)
    group_diag = diagnose_groups(fragments, groups)
    clustered_seconds = perf_counter() - started

    started = perf_counter()
    overlap_matrix = compute_compatibility_fast(fragments)
    overlap_build_seconds = perf_counter() - started

    started = perf_counter()
    overlap_solutions = solve_covering_sets(
        fragments,
        overlap_matrix,
        target_coverage=coverage,
        max_solutions=max_solutions,
        time_limit_seconds=time_limit,
        order_strategy=order_strategy,
    )
    overlap_solve_seconds = perf_counter() - started
    overlap_diag = diagnose_solutions(overlap_solutions, fragments)

    started = perf_counter()
    discriminative_matrix = compute_compatibility_clustered(fragments, groups)
    disc_build_seconds = perf_counter() - started

    started = perf_counter()
    disc_solutions = solve_covering_sets(
        fragments,
        discriminative_matrix,
        target_coverage=coverage,
        max_solutions=max_solutions,
        time_limit_seconds=time_limit,
        order_strategy=order_strategy,
    )
    disc_solve_seconds = perf_counter() - started
    disc_diag = diagnose_solutions(disc_solutions, fragments)

    row: dict[str, Any] = {
        "notes": notes,
        "pieces_per_note": pieces_per_note,
        "fragments": len(fragments),
        "appearance_spread": appearance_spread,
        "seed": seed,
        "wear_model": wear_model,
        "discriminate_tolerance": discriminate_tolerance,
        "coverage": coverage,
        "max_solutions": max_solutions,
        "cluster_count": group_diag["groups"],
        "cluster_deficit": group_diag["cluster_deficit"],
        "mixed_group_count": group_diag["mixed_group_count"],
        "mixed_note_count": group_diag["mixed_note_count"],
        "split_note_count": group_diag["split_note_count"],
        "cluster_exact_recoverable_count": group_diag["exact_recoverable_count"],
        "cluster_exact_recoverable_rate": group_diag["exact_recoverable_rate"],
        "generated_seconds": generated_seconds,
        "clustered_seconds": clustered_seconds,
        "overlap_build_seconds": overlap_build_seconds,
        "overlap_solve_seconds": overlap_solve_seconds,
        "disc_build_seconds": disc_build_seconds,
        "disc_solve_seconds": disc_solve_seconds,
    }
    row.update(_diagnosis_summary("overlap", overlap_diag))
    row.update(_diagnosis_summary("disc", disc_diag))
    return row


def aggregate_pressure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average seed-level rows by notes/spread/wear-model."""

    groups: dict[tuple[int, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["notes"]), float(row["appearance_spread"]), str(row["wear_model"]))
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (notes, spread, wear_model), items in groups.items():
        out: dict[str, Any] = {
            "notes": notes,
            "appearance_spread": spread,
            "wear_model": wear_model,
            "seeds": len(items),
        }
        fixed_keys = {"seed", "notes", "appearance_spread", "wear_model"}
        numeric_keys = sorted(
            {
                key
                for item in items
                for key, value in item.items()
                if key not in fixed_keys and isinstance(value, (int, float, np.integer, np.floating))
            }
        )
        for key in numeric_keys:
            out[key] = float(np.mean([float(item[key]) for item in items if key in item]))
        summary.append(out)
    return summary


def run_pressure_sweep(
    *,
    notes_values: Iterable[int],
    spread_values: Iterable[float],
    seeds: Iterable[int],
    pieces_per_note: int = 8,
    width: int = 160,
    height: int = 90,
    coverage: float = 0.95,
    max_solutions: int = 20,
    time_limit: float | None = 30.0,
    order_strategy: str = "area_degree",
    discriminate_tolerance: float = 0.05,
    wear_model: str = "global_gain",
    noise_sigma: float = 4.0,
    local_wear_strength: float = 0.0,
    gamma_spread: float = 0.0,
    stain_count: int = 0,
    stain_strength: float = 0.0,
) -> dict[str, Any]:
    rows = [
        run_pressure_case(
            notes=int(notes),
            appearance_spread=float(spread),
            seed=int(seed),
            pieces_per_note=pieces_per_note,
            width=width,
            height=height,
            coverage=coverage,
            max_solutions=max_solutions,
            time_limit=time_limit,
            order_strategy=order_strategy,
            discriminate_tolerance=discriminate_tolerance,
            wear_model=wear_model,
            noise_sigma=noise_sigma,
            local_wear_strength=local_wear_strength,
            gamma_spread=gamma_spread,
            stain_count=stain_count,
            stain_strength=stain_strength,
        )
        for notes in notes_values
        for spread in spread_values
        for seed in seeds
    ]
    return {
        "rows": rows,
        "summary": aggregate_pressure_rows(rows),
    }
