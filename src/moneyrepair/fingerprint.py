from __future__ import annotations

import numpy as np

from moneyrepair.compat import PackedCompatibilityMatrix, compute_compatibility_clustered
from moneyrepair.types import Fragment

# Appearance fingerprinting: the discrimination signal the overlap-only matrix
# is missing. Two fragments of the SAME physical note share one global tone
# transform (wear / yellowing / ink density); fragments of DIFFERENT notes do
# not. Measuring that transform relative to the standard template cancels the
# per-region content, so the fingerprint depends on the note, not on which part
# of the note a fragment happens to cover.


def fragment_appearance(fragment: Fragment, template: np.ndarray) -> np.ndarray:
    """Per-channel least-squares gain fitting ``observed ~= gain * template``.

    Returned over the fragment's masked pixels, so it recovers the note's tone
    transform independent of the region the fragment covers. Falls back to ones
    where there is no signal.
    """

    if fragment.image is None:
        return np.ones(3, dtype=np.float64)
    mask = fragment.mask
    if int(mask.sum()) == 0:
        return np.ones(3, dtype=np.float64)
    observed = fragment.image.astype(np.float64)[mask]
    reference = template.astype(np.float64)[mask]
    gain = np.ones(3, dtype=np.float64)
    for channel in range(3):
        ref = reference[:, channel]
        denom = float(np.dot(ref, ref))
        if denom > 1e-6:
            gain[channel] = float(np.dot(observed[:, channel], ref) / denom)
    return gain


def fragment_appearances(fragments: list[Fragment], template: np.ndarray) -> dict[str, np.ndarray]:
    return {fragment.id: fragment_appearance(fragment, template) for fragment in fragments}


def cluster_fragments_by_appearance(
    fragments: list[Fragment],
    template: np.ndarray,
    tolerance: float = 0.05,
) -> dict[str, int]:
    """Greedy clustering of fragments by appearance gain.

    Fragments within ``tolerance`` (Euclidean distance on the 3-vector gain) of
    an existing cluster centroid join it; otherwise they seed a new cluster.
    Returns a mapping from fragment id to integer cluster id. With well-separated
    per-note tones this recovers one cluster per note.
    """

    appearances = fragment_appearances(fragments, template)
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    groups: dict[str, int] = {}
    for fragment in fragments:
        vector = appearances[fragment.id]
        best_index = -1
        best_distance = tolerance
        for index, centroid in enumerate(centroids):
            distance = float(np.linalg.norm(vector - centroid))
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        if best_index < 0:
            centroids.append(vector.copy())
            counts.append(1)
            groups[fragment.id] = len(centroids) - 1
        else:
            count = counts[best_index] + 1
            centroids[best_index] = (centroids[best_index] * counts[best_index] + vector) / count
            counts[best_index] = count
            groups[fragment.id] = best_index
    return groups


def discriminative_groups(
    fragments: list[Fragment],
    template: np.ndarray,
    mode: str = "appearance",
    tolerance: float = 0.05,
) -> dict[str, int]:
    if mode == "appearance":
        return cluster_fragments_by_appearance(fragments, template, tolerance=tolerance)
    if mode == "serial":
        return groups_from_labels(fragments)
    raise ValueError("mode must be 'appearance' or 'serial'")


def discriminative_compatibility(
    fragments: list[Fragment],
    template: np.ndarray,
    mode: str = "appearance",
    tolerance: float = 0.05,
    max_overlap_pixels: int = 0,
    max_overlap_ratio: float = 0.0,
) -> PackedCompatibilityMatrix:
    """Build a compatibility matrix that uses both overlap and discrimination."""

    groups = discriminative_groups(fragments, template, mode=mode, tolerance=tolerance)
    return compute_compatibility_clustered(
        fragments,
        groups,
        max_overlap_pixels=max_overlap_pixels,
        max_overlap_ratio=max_overlap_ratio,
    )


def groups_from_labels(fragments: list[Fragment]) -> dict[str, int]:
    """Group fragments by their serial label, when present.

    Fragments without a label each get a unique singleton group, so unlabelled
    pieces are never linked by serial alone — that is the realistic case the
    appearance fingerprint has to cover.
    """

    groups: dict[str, int] = {}
    label_to_group: dict[str, int] = {}
    next_group = 0
    for fragment in fragments:
        label = fragment.label
        if label:
            if label not in label_to_group:
                label_to_group[label] = next_group
                next_group += 1
            groups[fragment.id] = label_to_group[label]
        else:
            groups[fragment.id] = next_group
            next_group += 1
    return groups
