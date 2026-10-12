"""Distance between the bounded exact-cover search and the optimum of its pool.

Measurement only. Rebuilds one simulated v4.4.1 reconstruction, keeps its
final candidate pool and the depth-first selection, and re-solves the same
weighted set-packing problem as a MILP (HiGHS through SciPy): every fragment
id and every serial label is used at most once, and the objective is
lexicographic like ``score_then_count`` -- total candidate score first, then
the number of selected candidates. When the depth-first search stops at its
node budget its selection is only an incumbent; this audit reports how far that
incumbent is from the proven optimum (or the best bound if the MILP stops at
its time limit). Truth labels are read only afterwards, to show what the two
selections would mean for exact-note yield. Requires SciPy.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import monotonic

import numpy as np

from moneyrepair.tearfit import (
    FractalTearConfig,
    TearFitCoreConfig,
    diagnose_confirmed_candidates,
    make_fractal_tear_fragments,
    reconstruct_placed_fragments,
    select_exact_cover_candidates,
)


def packing_matrix(candidates):
    """Rows: fragment ids then serial labels; columns: candidates."""

    from scipy.sparse import csr_matrix

    rows: dict[str, int] = {}
    entries_row: list[int] = []
    entries_col: list[int] = []
    for column, candidate in enumerate(candidates):
        for key in [f"id:{fid}" for fid in set(candidate.fragment_ids)] + [
            f"label:{label}" for label in set(candidate.labels)
        ]:
            entries_row.append(rows.setdefault(key, len(rows)))
            entries_col.append(column)
    data = np.ones(len(entries_row), dtype=np.float64)
    return csr_matrix((data, (entries_row, entries_col)), shape=(len(rows), len(candidates)))


def solve_packing(candidates, *, time_limit: float) -> dict:
    """Lexicographic weighted set packing: max score, then max count."""

    from scipy.optimize import Bounds, LinearConstraint, milp

    if not candidates:
        return {"status": "empty", "score": 0.0, "count": 0, "selected": []}
    scores = np.array([candidate.score for candidate in candidates], dtype=np.float64)
    matrix = packing_matrix(candidates)
    packing = LinearConstraint(matrix, -np.inf, 1.0)
    bounds = Bounds(0.0, 1.0)
    integral = np.ones(len(candidates))
    started = monotonic()
    relaxation = milp(-scores, constraints=[packing], bounds=bounds, integrality=np.zeros(len(candidates)))
    lp_seconds = monotonic() - started
    started = monotonic()
    first = milp(
        -scores,
        constraints=[packing],
        bounds=bounds,
        integrality=integral,
        options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False},
    )
    first_seconds = monotonic() - started
    report = {
        "candidates": len(candidates),
        "rows": matrix.shape[0],
        "lp_relaxation_bound": float(-relaxation.fun) if relaxation.success else None,
        "lp_seconds": lp_seconds,
        "score_status": first.message,
        "score_optimal": bool(first.status == 0),
        "score_seconds": first_seconds,
        "score_nodes": getattr(first, "mip_node_count", None),
        "score_dual_bound": (
            float(-first.mip_dual_bound) if getattr(first, "mip_dual_bound", None) is not None else None
        ),
    }
    if first.x is None:
        report.update(score=None, count=None, selected=[])
        return report
    chosen = np.flatnonzero(first.x > 0.5)
    best_score = float(scores[chosen].sum())
    report["score"] = best_score
    if report["score_optimal"]:
        # Among maximum-score packings, maximise the count (tie-break).
        tolerance = 1e-9 * max(1.0, abs(best_score))
        started = monotonic()
        second = milp(
            -np.ones(len(candidates)),
            constraints=[packing, LinearConstraint(scores.reshape(1, -1), best_score - tolerance, np.inf)],
            bounds=bounds,
            integrality=integral,
            options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False},
        )
        report["count_seconds"] = monotonic() - started
        if second.x is not None and second.status == 0:
            chosen = np.flatnonzero(second.x > 0.5)
            report["score"] = float(scores[chosen].sum())
    report["count"] = int(len(chosen))
    report["selected"] = [int(index) for index in chosen]
    return report


def audit(notes: int, pieces: int, seed: int, strategy: str, *, multipliers, time_limit: float) -> dict:
    simulation = FractalTearConfig(notes=notes, pieces_per_note=pieces, seed=seed, serial_ocr_rate=0.0)
    _template, fragments = make_fractal_tear_fragments(simulation)
    config = replace(TearFitCoreConfig.v4_4_1(), base_selection_strategy=strategy)
    reconstruction = reconstruct_placed_fragments(
        fragments, config, max_pieces=pieces + 2, expected_notes=notes
    )
    pool = reconstruction.candidates
    budget = reconstruction.resolved_cover_node_limit
    position = {candidate.fragment_ids: index for index, candidate in enumerate(pool)}

    def describe(selected) -> dict:
        diagnostics = diagnose_confirmed_candidates(selected, fragments)
        return {
            "score": float(sum(candidate.score for candidate in selected)),
            "count": len(selected),
            "exact_yield": diagnostics.exact_yield,
            "exact_precision": diagnostics.exact_precision,
        }

    depth_first = []
    for multiplier in multipliers:
        stats: dict = {}
        started = monotonic()
        selected = select_exact_cover_candidates(
            pool,
            time_limit_seconds=None,
            objective=config.cover_objective,
            max_search_nodes=None if budget is None else int(budget * multiplier),
            search_stats=stats,
        )
        depth_first.append({
            "node_budget_multiplier": multiplier,
            "node_budget": None if budget is None else int(budget * multiplier),
            "seconds": monotonic() - started,
            **stats,
            **describe(selected),
            "selected": sorted(position[candidate.fragment_ids] for candidate in selected),
        })
    milp_report = solve_packing(pool, time_limit=time_limit)
    optimum = [pool[index] for index in milp_report.get("selected", [])]
    if optimum:
        milp_report.update({key: value for key, value in describe(optimum).items() if key not in ("score", "count")})
    incumbent = depth_first[0]
    reference = milp_report.get("score")
    bound = milp_report.get("score_dual_bound") or milp_report.get("lp_relaxation_bound")
    gap = None
    if reference:
        gap = (reference - incumbent["score"]) / reference
    shared = set(incumbent["selected"]) & set(milp_report.get("selected", []))
    return {
        "case": {"notes": notes, "pieces": pieces, "seed": seed, "strategy": strategy},
        "pool": {"candidates": len(pool), "cover_node_budget": budget},
        "depth_first": depth_first,
        "milp": milp_report,
        "incumbent_relative_gap_to_optimum": gap,
        "incumbent_relative_gap_to_bound": (
            None if not bound else (bound - incumbent["score"]) / bound
        ),
        "selection_overlap": {
            "shared": len(shared),
            "only_depth_first": len(set(incumbent["selected"]) - shared),
            "only_milp": len(set(milp_report.get("selected", [])) - shared),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notes", type=int, default=100)
    parser.add_argument("--pieces", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strategy", default="disjoint_round_robin")
    parser.add_argument("--node-multipliers", default="1,10,100")
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    multipliers = [float(value) for value in args.node_multipliers.split(",")]
    report = audit(
        args.notes, args.pieces, args.seed, args.strategy,
        multipliers=multipliers, time_limit=args.time_limit,
    )
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    summary = {key: report[key] for key in ("case", "pool", "incumbent_relative_gap_to_optimum",
                                            "incumbent_relative_gap_to_bound", "selection_overlap")}
    summary["milp"] = {key: value for key, value in report["milp"].items() if key != "selected"}
    summary["depth_first"] = [{key: value for key, value in row.items() if key != "selected"}
                              for row in report["depth_first"]]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
