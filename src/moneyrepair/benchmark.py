from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from moneyrepair.compat import compute_compatibility
from moneyrepair.simulate import make_synthetic_fragments
from moneyrepair.solver import solve_covering_sets


@dataclass(frozen=True)
class MatrixFootprint:
    fragments: int
    dense_bool_bytes: int
    packed_bytes: int

    @property
    def dense_bool_mb(self) -> float:
        return self.dense_bool_bytes / (1024 * 1024)

    @property
    def packed_mb(self) -> float:
        return self.packed_bytes / (1024 * 1024)

    def to_dict(self) -> dict:
        return {
            "fragments": self.fragments,
            "dense_bool_bytes": self.dense_bool_bytes,
            "dense_bool_mb": self.dense_bool_mb,
            "packed_bytes": self.packed_bytes,
            "packed_mb": self.packed_mb,
        }


def estimate_matrix_footprint(fragments: int) -> MatrixFootprint:
    if fragments < 0:
        raise ValueError("fragments must be non-negative")
    dense = fragments * fragments
    packed = fragments * ((fragments + 7) // 8)
    return MatrixFootprint(fragments=fragments, dense_bool_bytes=dense, packed_bytes=packed)


def write_matrix_footprint(path: str | Path, fragments: int) -> MatrixFootprint:
    footprint = estimate_matrix_footprint(fragments)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(footprint.to_dict(), indent=2), encoding="utf-8")
    return footprint


@dataclass(frozen=True)
class SyntheticBenchmark:
    pieces_requested: int
    pieces_generated: int
    width: int
    height: int
    target_coverage: float
    max_solutions: int
    timings_seconds: dict[str, float]
    matrix_footprint: MatrixFootprint
    solutions_found: int
    best_coverage: float | None
    order_strategy: str = "area"

    def to_dict(self) -> dict:
        return {
            "pieces_requested": self.pieces_requested,
            "pieces_generated": self.pieces_generated,
            "width": self.width,
            "height": self.height,
            "target_coverage": self.target_coverage,
            "max_solutions": self.max_solutions,
            "timings_seconds": self.timings_seconds,
            "matrix_footprint": self.matrix_footprint.to_dict(),
            "solutions_found": self.solutions_found,
            "best_coverage": self.best_coverage,
            "order_strategy": self.order_strategy,
        }


def run_synthetic_benchmark(
    pieces: int = 80,
    width: int = 480,
    height: int = 210,
    seed: int = 7,
    target_coverage: float = 0.98,
    max_solutions: int = 5,
    time_limit_seconds: float | None = 30.0,
    order_strategy: str = "area",
) -> SyntheticBenchmark:
    """Run a deterministic synthetic pipeline benchmark."""

    timings: dict[str, float] = {}

    started = perf_counter()
    _, fragments = make_synthetic_fragments(pieces=pieces, width=width, height=height, seed=seed)
    timings["simulate"] = perf_counter() - started

    started = perf_counter()
    matrix = compute_compatibility(fragments)
    timings["build_matrix"] = perf_counter() - started

    started = perf_counter()
    solutions = solve_covering_sets(
        fragments,
        matrix,
        target_coverage=target_coverage,
        max_solutions=max_solutions,
        time_limit_seconds=time_limit_seconds,
        order_strategy=order_strategy,
    )
    timings["solve"] = perf_counter() - started
    timings["total"] = sum(timings.values())

    return SyntheticBenchmark(
        pieces_requested=pieces,
        pieces_generated=len(fragments),
        width=width,
        height=height,
        target_coverage=target_coverage,
        max_solutions=max_solutions,
        timings_seconds=timings,
        matrix_footprint=estimate_matrix_footprint(len(fragments)),
        solutions_found=len(solutions),
        best_coverage=solutions[0].coverage if solutions else None,
        order_strategy=order_strategy,
    )


def write_synthetic_benchmark(path: str | Path, **kwargs) -> SyntheticBenchmark:
    result = run_synthetic_benchmark(**kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def compare_solver_strategies(
    pieces: int = 80,
    width: int = 480,
    height: int = 210,
    seed: int = 7,
    target_coverage: float = 0.98,
    max_solutions: int = 5,
    time_limit_seconds: float | None = 30.0,
    strategies: tuple[str, ...] = ("area", "degree", "area_degree"),
) -> list[SyntheticBenchmark]:
    return [
        run_synthetic_benchmark(
            pieces=pieces,
            width=width,
            height=height,
            seed=seed,
            target_coverage=target_coverage,
            max_solutions=max_solutions,
            time_limit_seconds=time_limit_seconds,
            order_strategy=strategy,
        )
        for strategy in strategies
    ]
