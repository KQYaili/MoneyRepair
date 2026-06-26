from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moneyrepair.compat import PackedCompatibilityMatrix, compute_compatibility_fast
from moneyrepair.types import Fragment


@dataclass(frozen=True)
class TearInterlockScore:
    """Raster tear-fit evidence for two already placed fragments."""

    contact_edges: int
    contact_ratio: float


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


def compute_interlock_compatibility(
    fragments: list[Fragment],
    *,
    max_overlap_pixels: int = 0,
    max_overlap_ratio: float = 0.0,
    cell: int | None = None,
    min_contact_edges: int = 8,
    min_contact_ratio: float = 0.03,
) -> PackedCompatibilityMatrix:
    """Compatibility from non-overlap plus local tear-contact evidence.

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
    matrix = packed.to_dense()
    for left_index in range(len(fragments)):
        for right_index in range(left_index + 1, len(fragments)):
            if not matrix.compatible[left_index, right_index]:
                continue
            score = tear_interlock_score(fragments[left_index], fragments[right_index])
            if score.contact_edges < min_contact_edges:
                continue
            if score.contact_ratio < min_contact_ratio:
                matrix.compatible[left_index, right_index] = False
                matrix.compatible[right_index, left_index] = False
    return PackedCompatibilityMatrix.from_dense(matrix)
