from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np

from moneyrepair.compat import CompatibilityMatrix, PackedCompatibilityMatrix
from moneyrepair.types import Fragment


@dataclass(frozen=True)
class CoverageSolution:
    fragment_ids: tuple[str, ...]
    coverage: float
    area: int


def solve_covering_sets(
    fragments: list[Fragment],
    compatibility: CompatibilityMatrix | PackedCompatibilityMatrix,
    target_coverage: float = 0.99,
    max_solutions: int = 20,
    start_id: str | None = None,
    time_limit_seconds: float | None = None,
    allowed_ids: set[str] | None = None,
) -> list[CoverageSolution]:
    """Depth-first search for compatible fragment sets covering the note."""

    if not fragments:
        return []
    if tuple(fragment.id for fragment in fragments) != compatibility.ids:
        raise ValueError("fragment order must match compatibility ids")
    if not (0.0 < target_coverage <= 1.0):
        raise ValueError("target_coverage must be in (0, 1]")

    allowed_indices = (
        set(range(len(fragments)))
        if allowed_ids is None
        else {index for index, fragment in enumerate(fragments) if fragment.id in allowed_ids}
    )
    if start_id is not None and compatibility.index(start_id) not in allowed_indices:
        return []
    if not allowed_indices:
        return []

    total_area = fragments[0].mask.size
    target_area = int(np.ceil(total_area * target_coverage))
    areas = np.array([fragment.area for fragment in fragments], dtype=np.int64)
    order = tuple(index for index in np.argsort(-areas).tolist() if index in allowed_indices)
    order_rank = {index: rank for rank, index in enumerate(order)}
    deadline = None if time_limit_seconds is None else monotonic() + time_limit_seconds
    solutions: list[CoverageSolution] = []
    seen: set[tuple[int, ...]] = set()

    def timed_out() -> bool:
        return deadline is not None and monotonic() >= deadline

    def make_solution(selected: tuple[int, ...], union_mask: np.ndarray) -> None:
        key = tuple(sorted(selected))
        if key in seen:
            return
        seen.add(key)
        area = int(union_mask.sum())
        solutions.append(
            CoverageSolution(
                fragment_ids=tuple(fragments[index].id for index in key),
                coverage=area / float(total_area),
                area=area,
            )
        )
        solutions.sort(key=lambda item: (-item.coverage, len(item.fragment_ids), item.fragment_ids))
        del solutions[max_solutions:]

    def upper_bound_area(union_mask: np.ndarray, candidates: tuple[int, ...]) -> int:
        if len(candidates) == 0:
            return int(union_mask.sum())
        candidate_union = union_mask.copy()
        for index in candidates:
            candidate_union |= fragments[index].mask
        return int(candidate_union.sum())

    def dfs(selected: tuple[int, ...], candidates: tuple[int, ...], union_mask: np.ndarray) -> None:
        if timed_out() or len(solutions) >= max_solutions:
            return

        current_area = int(union_mask.sum())
        if current_area >= target_area:
            make_solution(selected, union_mask)
            return
        if upper_bound_area(union_mask, candidates) < target_area:
            return

        for offset, index in enumerate(candidates):
            if timed_out() or len(solutions) >= max_solutions:
                return
            next_union = union_mask | fragments[index].mask
            remaining = candidates[offset + 1 :]
            next_candidates = compatibility.compatible_indices(index, remaining)
            dfs(selected + (index,), next_candidates, next_union)

    starts = (compatibility.index(start_id),) if start_id else order
    for start in starts:
        if timed_out() or len(solutions) >= max_solutions:
            break
        if start_id:
            candidates = compatibility.compatible_indices(start, tuple(index for index in order if index != start))
        else:
            start_rank = order_rank[start]
            candidates = compatibility.compatible_indices(start, order[start_rank + 1 :])
        dfs((start,), candidates, fragments[start].mask.copy())

    return solutions
