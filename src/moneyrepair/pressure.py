from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter
from typing import Any

import numpy as np

from moneyrepair.compat import compute_compatibility_clustered, compute_compatibility_fast
from moneyrepair.diagnostics import diagnose_groups, diagnose_solutions
from moneyrepair.baselines.fingerprint import cluster_fragments_by_appearance
from moneyrepair.baselines.interlock import apply_interlock_constraints_with_stats, compute_interlock_compatibility_with_stats
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


def _matrix_pair_summary(prefix: str, row: dict[str, Any], matrix, total_pairs: int) -> None:
    compatible = matrix.compatible_pair_count()
    row[f"{prefix}_compatible_pairs"] = compatible
    row[f"{prefix}_incompatible_pairs"] = total_pairs - compatible


def _interlock_stats_summary(prefix: str, row: dict[str, Any], stats) -> None:
    row[f"{prefix}_bbox_candidate_pairs"] = stats.bbox_candidate_pairs
    row[f"{prefix}_scored_contact_pairs"] = stats.scored_contact_pairs
    row[f"{prefix}_rejected_pairs"] = stats.rejected_pairs


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
    partition_model: str = "shared",
    include_interlock: bool = False,
    include_disc_interlock: bool = False,
    cell: int | None = None,
    min_interlock_contact: int = 8,
    min_interlock_ratio: float = 0.03,
    touch_priority: bool = True,
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
        partition_model=partition_model,
    )
    generated_seconds = perf_counter() - started

    started = perf_counter()
    groups = cluster_fragments_by_appearance(fragments, template, tolerance=discriminate_tolerance)
    group_diag = diagnose_groups(fragments, groups)
    clustered_seconds = perf_counter() - started

    started = perf_counter()
    overlap_matrix = compute_compatibility_fast(fragments, cell=cell)
    overlap_build_seconds = perf_counter() - started

    started = perf_counter()
    overlap_solutions = solve_covering_sets(
        fragments,
        overlap_matrix,
        target_coverage=coverage,
        max_solutions=max_solutions,
        time_limit_seconds=time_limit,
        order_strategy=order_strategy,
        touch_priority=touch_priority,
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
        touch_priority=touch_priority,
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
        "partition_model": partition_model,
        "discriminate_tolerance": discriminate_tolerance,
        "noise_sigma": noise_sigma,
        "local_wear_strength": local_wear_strength,
        "gamma_spread": gamma_spread,
        "stain_count": stain_count,
        "stain_strength": stain_strength,
        "coverage": coverage,
        "max_solutions": max_solutions,
        "touch_priority": touch_priority,
        "include_interlock": include_interlock,
        "include_disc_interlock": include_disc_interlock,
        "cell": cell,
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

    if include_interlock:
        started = perf_counter()
        interlock_matrix, interlock_stats = compute_interlock_compatibility_with_stats(
            fragments,
            cell=cell,
            min_contact_edges=min_interlock_contact,
            min_contact_ratio=min_interlock_ratio,
        )
        row["interlock_build_seconds"] = perf_counter() - started
        total_pairs = len(fragments) * (len(fragments) - 1) // 2
        _matrix_pair_summary("interlock", row, interlock_matrix, total_pairs)
        _interlock_stats_summary("interlock", row, interlock_stats)
        started = perf_counter()
        interlock_solutions = solve_covering_sets(
            fragments,
            interlock_matrix,
            target_coverage=coverage,
            max_solutions=max_solutions,
            time_limit_seconds=time_limit,
            order_strategy=order_strategy,
            touch_priority=touch_priority,
        )
        row["interlock_solve_seconds"] = perf_counter() - started
        row["min_interlock_contact"] = min_interlock_contact
        row["min_interlock_ratio"] = min_interlock_ratio
        row.update(_diagnosis_summary("interlock", diagnose_solutions(interlock_solutions, fragments)))

    if include_disc_interlock:
        started = perf_counter()
        disc_interlock_matrix, disc_interlock_stats = apply_interlock_constraints_with_stats(
            discriminative_matrix,
            fragments,
            cell=cell,
            min_contact_edges=min_interlock_contact,
            min_contact_ratio=min_interlock_ratio,
        )
        row["disc_interlock_build_seconds"] = perf_counter() - started
        row["disc_interlock_total_build_seconds"] = (
            row["disc_build_seconds"] + row["disc_interlock_build_seconds"]
        )
        total_pairs = len(fragments) * (len(fragments) - 1) // 2
        _matrix_pair_summary("disc_interlock", row, disc_interlock_matrix, total_pairs)
        _interlock_stats_summary("disc_interlock", row, disc_interlock_stats)
        started = perf_counter()
        disc_interlock_solutions = solve_covering_sets(
            fragments,
            disc_interlock_matrix,
            target_coverage=coverage,
            max_solutions=max_solutions,
            time_limit_seconds=time_limit,
            order_strategy=order_strategy,
            touch_priority=touch_priority,
        )
        row["disc_interlock_solve_seconds"] = perf_counter() - started
        row["min_interlock_contact"] = min_interlock_contact
        row["min_interlock_ratio"] = min_interlock_ratio
        row.update(_diagnosis_summary("disc_interlock", diagnose_solutions(disc_interlock_solutions, fragments)))
    return row


def aggregate_pressure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average seed-level rows by notes/spread/wear-model."""

    key_fields = (
        "notes",
        "appearance_spread",
        "wear_model",
        "partition_model",
        "discriminate_tolerance",
        "noise_sigma",
        "local_wear_strength",
        "gamma_spread",
        "stain_count",
        "stain_strength",
        "min_interlock_contact",
        "min_interlock_ratio",
        "touch_priority",
        "include_interlock",
        "include_disc_interlock",
        "cell",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key, items in groups.items():
        out: dict[str, Any] = {field: value for field, value in zip(key_fields, key) if value is not None}
        out["seeds"] = len(items)
        fixed_keys = {"seed", *key_fields}
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
    partition_model: str = "shared",
    include_interlock: bool = False,
    include_disc_interlock: bool = False,
    cell: int | None = None,
    min_interlock_contact: int = 8,
    min_interlock_ratio: float = 0.03,
    touch_priority: bool = True,
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
            partition_model=partition_model,
            include_interlock=include_interlock,
            include_disc_interlock=include_disc_interlock,
            cell=cell,
            min_interlock_contact=min_interlock_contact,
            min_interlock_ratio=min_interlock_ratio,
            touch_priority=touch_priority,
        )
        for notes in notes_values
        for spread in spread_values
        for seed in seeds
    ]
    return {
        "rows": rows,
        "summary": aggregate_pressure_rows(rows),
    }
