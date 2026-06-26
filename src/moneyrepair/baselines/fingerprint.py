"""SUPERSEDED baseline. Kept for comparison only. Appearance wear gain fails under spatially non-uniform wear. Not used by the supported pipeline."""

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
        denom = float(np.dot(ref, ref)) + 1e-9
        gain[channel] = float(np.dot(observed[:, channel], ref) / denom)
    return gain


def fragment_appearances(fragments: list[Fragment], template: np.ndarray) -> dict[str, np.ndarray]:
    return {fragment.id: fragment_appearance(fragment, template) for fragment in fragments}


def cluster_fragments_by_appearance(
    fragments: list[Fragment],
    template: np.ndarray,
    tolerance: float = 0.05,
    min_samples: int = 2,
) -> dict[str, int]:
    """Cluster fragments by appearance gain using a pure NumPy DBSCAN algorithm.

    This replaces the sequential greedy clustering to ensure order-independence
    and prevent centroid drift ("rolling snowball" effect) under continuous wear.
    Noise fragments (DBSCAN label -1) are assigned to unique singleton groups.
    """
    if not fragments:
        return {}

    appearances = fragment_appearances(fragments, template)
    ids = [frag.id for frag in fragments]
    X = np.array([appearances[fid] for fid in ids], dtype=np.float64)

    n = len(fragments)
    labels = np.full(n, -1, dtype=np.int32)

    # Compute pairwise Euclidean distance
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))

    # Neighborhoods (indices within tolerance)
    neighbors = [np.flatnonzero(dist[i] <= tolerance) for i in range(n)]

    cluster_id = 0
    for i in range(n):
        if labels[i] != -1:
            continue

        if len(neighbors[i]) >= min_samples:
            labels[i] = cluster_id
            queue = list(neighbors[i])
            head = 0
            while head < len(queue):
                curr = queue[head]
                head += 1

                if labels[curr] == -1:
                    labels[curr] = cluster_id
                    if len(neighbors[curr]) >= min_samples:
                        for neighbor in neighbors[curr]:
                            if labels[neighbor] == -1:
                                queue.append(neighbor)
            cluster_id += 1

    # Map groups: cluster labels map to groups, noise map to unique singletons
    groups: dict[str, int] = {}
    next_unique_id = cluster_id
    for idx, fid in enumerate(ids):
        lbl = labels[idx]
        if lbl == -1:
            groups[fid] = next_unique_id
            next_unique_id += 1
        else:
            groups[fid] = int(lbl)

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


def compute_anchor_score(fragment: Fragment) -> float:
    """Compute an anchor score for a fragment to prioritize search start.

    Fragments containing serial number labels, specific key features, or large areas
    are given higher anchor scores to prune the DFS tree early.
    """
    score = 0.0
    if fragment.label:
        score += 100.0
    score += float(fragment.area) * 0.05
    for tag in fragment.tags:
        if "serial" in tag or "ocr" in tag:
            score += 50.0
        elif "edge" in tag:
            score += 10.0
        elif "corner" in tag:
            score += 20.0
    return score


def prioritize_fragments_by_anchor(fragments: list[Fragment]) -> list[Fragment]:
    """Sort a list of fragments in descending order of their anchor score."""
    return sorted(fragments, key=compute_anchor_score, reverse=True)
