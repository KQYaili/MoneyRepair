from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np
import numba

from moneyrepair.compat import CompatibilityMatrix, PackedCompatibilityMatrix
from moneyrepair.types import Fragment


@numba.njit(fastmath=True, cache=True)
def sum_candidate_areas(areas: np.ndarray, candidates: np.ndarray) -> int:
    total = 0
    for idx in range(len(candidates)):
        total += areas[candidates[idx]]
    return total


def compute_touches_matrix(fragments: list[Fragment]) -> np.ndarray:
    """Precompute which fragments touch (are adjacent to) each other."""
    n = len(fragments)
    touches = np.zeros((n, n), dtype=bool)
    bboxes = [f.bbox for f in fragments]

    for i in range(n):
        x0_i, y0_i, x1_i, y1_i = bboxes[i]
        if x1_i <= x0_i:  # empty mask
            continue
        # Expand bbox by 1
        x0_i_exp, y0_i_exp, x1_i_exp, y1_i_exp = x0_i - 1, y0_i - 1, x1_i + 1, y1_i + 1

        mask_i = fragments[i].mask
        dilated_i = mask_i.copy()
        dilated_i[:-1] |= mask_i[1:]
        dilated_i[1:] |= mask_i[:-1]
        dilated_i[:, :-1] |= mask_i[:, 1:]
        dilated_i[:, 1:] |= mask_i[:, :-1]

        for j in range(i + 1, n):
            x0_j, y0_j, x1_j, y1_j = bboxes[j]
            if x1_j <= x0_j:
                continue
            # Check expanded bbox intersection
            if not (x0_i_exp < x1_j and x1_i_exp > x0_j and y0_i_exp < y1_j and y1_i_exp > y0_j):
                continue
            # Check pixel level overlap
            if np.any(dilated_i & fragments[j].mask):
                touches[i, j] = True
                touches[j, i] = True
    return touches


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
    order_strategy: str = "area",
    precise_bound_threshold: int = 24,
    touch_priority: bool = True,
) -> list[CoverageSolution]:
    """Depth-first search for compatible fragment sets covering the note."""

    if not fragments:
        return []
    if tuple(fragment.id for fragment in fragments) != compatibility.ids:
        raise ValueError("fragment order must match compatibility ids")
    if not (0.0 < target_coverage <= 1.0):
        raise ValueError("target_coverage must be in (0, 1]")

    if isinstance(compatibility, PackedCompatibilityMatrix):
        compatibility = compatibility.to_dense()

    allowed_indices = (
        set(range(len(fragments)))
        if allowed_ids is None
        else {index for index, fragment in enumerate(fragments) if fragment.id in allowed_ids}
    )
    if start_id is not None and compatibility.index(start_id) not in allowed_indices:
        return []
    if not allowed_indices:
        return []

    if order_strategy not in {"area", "degree", "area_degree", "max_degree", "max_variance"}:
        raise ValueError("order_strategy must be one of: area, degree, area_degree, max_degree, max_variance")

    total_area = fragments[0].mask.size
    target_area = int(np.ceil(total_area * target_coverage))
    areas = np.array([fragment.area for fragment in fragments], dtype=np.int64)
    allowed_tuple = tuple(sorted(allowed_indices))
    degrees = {
        index: len(compatibility.compatible_indices(index, tuple(candidate for candidate in allowed_tuple if candidate != index)))
        for index in allowed_tuple
    }
    if order_strategy == "area":
        order = tuple(index for index in np.argsort(-areas).tolist() if index in allowed_indices)
    elif order_strategy == "degree":
        order = tuple(sorted(allowed_tuple, key=lambda index: (degrees[index], -areas[index], index)))
    elif order_strategy == "max_degree":
        order = tuple(sorted(allowed_tuple, key=lambda index: (-degrees[index], -areas[index], index)))
    elif order_strategy == "max_variance":
        variances = []
        for f in fragments:
            if f.image is not None and int(f.mask.sum()) > 0:
                variances.append(float(np.var(f.image[f.mask])))
            else:
                variances.append(0.0)
        variances_arr = np.array(variances)
        order = tuple(index for index in np.argsort(-variances_arr).tolist() if index in allowed_indices)
    else:
        order = tuple(sorted(allowed_tuple, key=lambda index: (-areas[index], degrees[index], index)))
    order_rank = {index: rank for rank, index in enumerate(order)}
    deadline = None if time_limit_seconds is None else monotonic() + time_limit_seconds
    solutions: list[CoverageSolution] = []
    seen: set[tuple[int, ...]] = set()
    touches = compute_touches_matrix(fragments) if touch_priority else None

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

    def dfs(selected: tuple[int, ...], candidates: np.ndarray, union_mask: np.ndarray) -> None:
        if timed_out() or len(solutions) >= max_solutions:
            return

        current_area = int(union_mask.sum())
        if current_area >= target_area:
            make_solution(selected, union_mask)
            return

        # ====== Tier 1: Fast scalar upper bound area check ======
        if len(candidates) == 0:
            return

        scalar_upper_bound = current_area + sum_candidate_areas(areas, candidates)
        if scalar_upper_bound < target_area:
            return

        # ====== Tier 2: Precise geometry check for candidate sets ======
        # Trigger precise check when candidate count is below the threshold
        # or when the scalar bound is close to the target area (within 5% margin)
        if len(candidates) < precise_bound_threshold or (scalar_upper_bound - target_area < target_area * 0.05):
            combined_candidates_mask = np.logical_or.reduce([fragments[int(idx)].mask for idx in candidates])
            if int((union_mask | combined_candidates_mask).sum()) < target_area:
                return

        if touches is None:
            ordered_candidates = candidates
        else:
            # Prioritize candidates that physically touch any of the currently selected fragments
            touching_mask = np.zeros(len(fragments), dtype=bool)
            for sel_idx in selected:
                touching_mask |= touches[sel_idx]

            is_touching = touching_mask[candidates]
            offsets = np.arange(len(candidates))
            touching_offsets = offsets[is_touching]
            non_touching_offsets = offsets[~is_touching]
            reordered_offsets = np.concatenate((touching_offsets, non_touching_offsets))
            ordered_candidates = candidates[reordered_offsets]

        for pos in range(len(ordered_candidates)):
            if timed_out() or len(solutions) >= max_solutions:
                return
            index = int(ordered_candidates[pos])
            next_union = union_mask | fragments[index].mask
            remaining = ordered_candidates[pos + 1 :]
            # Vectorized bitset intersect using boolean indexing over the pre-unpacked matrix
            next_candidates = remaining[compatibility.compatible[index, remaining]]
            dfs(selected + (index,), next_candidates, next_union)

    starts = [compatibility.index(start_id)] if start_id else order
    for start in starts:
        if timed_out() or len(solutions) >= max_solutions:
            break
        if start_id:
            remaining = np.array([idx for idx in order if idx != start], dtype=np.int64)
            candidates = remaining[compatibility.compatible[start, remaining]]
        else:
            start_rank = order_rank[start]
            remaining = np.array(order[start_rank + 1 :], dtype=np.int64)
            candidates = remaining[compatibility.compatible[start, remaining]]
        dfs((start,), candidates, fragments[start].mask.copy())

    return solutions
