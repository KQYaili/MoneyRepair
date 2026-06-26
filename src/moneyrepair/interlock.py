from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from moneyrepair.compat import PackedCompatibilityMatrix, compute_compatibility_fast
from moneyrepair.types import Fragment


@dataclass(frozen=True)
class TearInterlockScore:
    """Raster tear-fit evidence for two already placed fragments."""

    contact_edges: int
    contact_ratio: float


@dataclass(frozen=True)
class InterlockCompatibilityStats:
    """Sparse interlock pruning counters."""

    bbox_candidate_pairs: int
    scored_contact_pairs: int
    rejected_pairs: int


def boundary_edge_count(mask: np.ndarray) -> int:
    """Count foreground-to-background 4-neighbour boundary edges."""

    mask = mask.astype(bool)
    if mask.size == 0:
        return 0
    vertical = int(np.count_nonzero(mask[:-1, :] != mask[1:, :]))
    horizontal = int(np.count_nonzero(mask[:, :-1] != mask[:, 1:]))
    border = int(mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum())
    return vertical + horizontal + border


def contact_edge_count(left: np.ndarray, right: np.ndarray) -> int:
    """Count 4-neighbour edges where two masks bite into each other."""

    left = left.astype(bool)
    right = right.astype(bool)
    if left.shape != right.shape:
        raise ValueError("masks must share shape")
    return int(
        np.count_nonzero(left[:-1, :] & right[1:, :])
        + np.count_nonzero(left[1:, :] & right[:-1, :])
        + np.count_nonzero(left[:, :-1] & right[:, 1:])
        + np.count_nonzero(left[:, 1:] & right[:, :-1])
    )


def tear_interlock_score(left: Fragment, right: Fragment) -> TearInterlockScore:
    """Score complementary tear contact for two placed fragments.

    This is deliberately geometric, not photometric. A long shared jagged
    contact is strong evidence that two pieces came from one physical tear; no
    contact means the pair is simply non-adjacent and should not be penalised by
    this local test.
    """

    contact = contact_edge_count(left.mask, right.mask)
    if contact == 0:
        return TearInterlockScore(contact_edges=0, contact_ratio=0.0)
    denom = max(1, min(boundary_edge_count(left.mask), boundary_edge_count(right.mask)))
    return TearInterlockScore(contact_edges=contact, contact_ratio=contact / float(denom))


def _auto_contact_cell_size(bboxes: list[tuple[int, int, int, int]]) -> int:
    dims = sorted(max(x1 - x0, y1 - y0) for x0, y0, x1, y1 in bboxes if x1 > x0 and y1 > y0)
    if not dims:
        return 1
    return max(1, int(dims[len(dims) // 2]))


def _expanded_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return x0 - 1, y0 - 1, x1 + 1, y1 + 1


def _bbox_intersects(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return left[0] < right[2] and left[2] > right[0] and left[1] < right[3] and left[3] > right[1]


def _bbox_cells(bbox: tuple[int, int, int, int], cell: int) -> Iterable[tuple[int, int]]:
    x0, y0, x1, y1 = bbox
    for cy in range(y0 // cell, (y1 - 1) // cell + 1):
        for cx in range(x0 // cell, (x1 - 1) // cell + 1):
            yield cx, cy


def iter_contact_candidate_pairs(fragments: list[Fragment], cell: int | None = None) -> Iterable[tuple[int, int]]:
    """Yield pairs whose expanded bboxes could share a tear boundary."""

    bboxes = [fragment.bbox for fragment in fragments]
    expanded = [_expanded_bbox(bbox) for bbox in bboxes]
    cell_size = max(1, int(cell)) if cell is not None else _auto_contact_cell_size(bboxes)
    buckets: dict[tuple[int, int], list[int]] = {}
    seen: set[tuple[int, int]] = set()
    for right, bbox in enumerate(expanded):
        candidates: set[int] = set()
        for cell_id in _bbox_cells(bbox, cell_size):
            candidates.update(buckets.get(cell_id, ()))
        for left in candidates:
            pair = (left, right) if left < right else (right, left)
            if pair in seen:
                continue
            seen.add(pair)
            if _bbox_intersects(expanded[pair[0]], expanded[pair[1]]):
                yield pair
        for cell_id in _bbox_cells(bbox, cell_size):
            buckets.setdefault(cell_id, []).append(right)


def compute_interlock_compatibility_with_stats(
    fragments: list[Fragment],
    *,
    max_overlap_pixels: int = 0,
    max_overlap_ratio: float = 0.0,
    cell: int | None = None,
    min_contact_edges: int = 8,
    min_contact_ratio: float = 0.03,
) -> tuple[PackedCompatibilityMatrix, InterlockCompatibilityStats]:
    """Compatibility from non-overlap plus local tear-contact evidence.

    Fragments must already be placed in the same note coordinate frame. This is
    not a raw-crop edge-searcher over arbitrary rotation/translation; it is a
    local validation pass for candidate poses.

    Non-adjacent pairs stay compatible after the overlap test: they do not need
    to prove a tear fit because they are not touching. Adjacent pairs with
    enough contact to be meaningful must have a minimum shared-boundary ratio;
    otherwise they are treated as a false join.
    """

    packed = compute_compatibility_fast(
        fragments,
        max_overlap_pixels=max_overlap_pixels,
        max_overlap_ratio=max_overlap_ratio,
        cell=cell,
    )
    bbox_candidate_pairs = 0
    scored_contact_pairs = 0
    rejected_pairs = 0
    for left_index, right_index in iter_contact_candidate_pairs(fragments, cell=cell):
        bbox_candidate_pairs += 1
        if not packed.is_compatible(left_index, right_index):
            continue
        score = tear_interlock_score(fragments[left_index], fragments[right_index])
        if score.contact_edges < min_contact_edges:
            continue
        scored_contact_pairs += 1
        if score.contact_ratio < min_contact_ratio:
            packed.set_pair_compatible(left_index, right_index, False)
            rejected_pairs += 1
    return packed, InterlockCompatibilityStats(
        bbox_candidate_pairs=bbox_candidate_pairs,
        scored_contact_pairs=scored_contact_pairs,
        rejected_pairs=rejected_pairs,
    )


def compute_interlock_compatibility(
    fragments: list[Fragment],
    *,
    max_overlap_pixels: int = 0,
    max_overlap_ratio: float = 0.0,
    cell: int | None = None,
    min_contact_edges: int = 8,
    min_contact_ratio: float = 0.03,
) -> PackedCompatibilityMatrix:
    matrix, _ = compute_interlock_compatibility_with_stats(
        fragments,
        max_overlap_pixels=max_overlap_pixels,
        max_overlap_ratio=max_overlap_ratio,
        cell=cell,
        min_contact_edges=min_contact_edges,
        min_contact_ratio=min_contact_ratio,
    )
    return matrix
