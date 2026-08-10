from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Iterable

import numpy as np

from moneyrepair.simulate import synthetic_banknote
from moneyrepair.types import Fragment


def binary_dilation_3x3(mask: np.ndarray) -> np.ndarray:
    """Dilate a binary 2D array with a 3x3 structuring element (all ones) using numpy shifts."""
    if mask.size == 0 or mask.ndim != 2:
        return mask
    mask = mask.astype(bool)
    dilated = mask.copy()
    dilated[1:, :] |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:] |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    dilated[1:, 1:] |= mask[:-1, :-1]
    dilated[:-1, :-1] |= mask[1:, 1:]
    dilated[1:, :-1] |= mask[:-1, 1:]
    dilated[:-1, 1:] |= mask[1:, :-1]
    return dilated


TEARFIT_SEED_STRATEGIES = ("anchor_only", "anchor_priority", "all")
TEARFIT_COVER_OBJECTIVES = ("count_then_score", "score_then_count")
TEARFIT_EDGE_SCORING = ("overlap", "effectiveness")
TEARFIT_ALGORITHMS = ("baseline", "effectiveness", "effectiveness_gap", "v43_routed")
TEARFIT_EVIDENCE_LEVELS = ("automatic", "review", "insufficient-evidence")
TEARFIT_V43_FINE_FRACTION = 0.05


@dataclass(frozen=True)
class FractalTearConfig:
    """Parameters for the research tear-fit sandbox.

    The generated fragments are already placed in the common banknote coordinate
    frame. This module deliberately tests the geometry kernel after pose
    estimation, not the raw-crop locator.
    """

    notes: int = 20
    pieces_per_note: int = 8
    width: int = 180
    height: int = 90
    seed: int = 7
    roughness: float = 4.0
    fray_layers: int = 2
    fray_probability: float = 0.18
    ensure_serial_anchor: bool = False
    serial_ocr_rate: float = 0.6


@dataclass(frozen=True)
class TearBoundaryEvidence:
    boundary: np.ndarray
    dilated_boundary: np.ndarray
    bbox: tuple[int, int, int, int]
    boundary_mask: np.ndarray
    dilated_boundary_mask: np.ndarray
    normal_y: np.ndarray
    normal_x: np.ndarray


@dataclass(frozen=True)
class TearFitEdge:
    left: int
    right: int
    overlap_pixels: int
    left_hits: int
    right_hits: int
    overlap_ratio: float
    contiguous_pixels: int = 0
    continuity_ratio: float = 0.0
    bidirectional_balance: float = 0.0
    normal_opposition: float = 0.0
    curvature_entropy: float = 0.0
    expected_accidental_hits: float = 0.0
    mask_overlap_pixels: int = 0
    pose_uncertainty: float = 0.0
    effectiveness: float = 0.0
    evidence_level: str = "insufficient-evidence"


@dataclass(frozen=True)
class GroupGapEvidence:
    fragment: int
    seam_count: int
    matched_pixels: int
    gap_fill_pixels: int
    gap_fill_ratio: float
    overlap_pixels: int
    unmatched_perimeter_ratio: float
    pose_uncertainty: float
    score: float
    evidence_level: str


@dataclass(frozen=True)
class AssemblyCandidate:
    fragment_ids: tuple[str, ...]
    coverage: float
    raw_coverage: float
    score: float
    support_pixels: int
    labels: tuple[str, ...] = field(default_factory=tuple)
    base_score: float = 0.0
    constraint_bonus: float = 0.0
    evidence_score: float = 1.0
    evidence_level: str = "automatic"
    gap_steps: int = 0


@dataclass(frozen=True)
class TearFitDiagnostics:
    confirmed: int
    exact_confirmed: int
    pure_confirmed: int
    chimeras: int
    true_notes: int
    exact_yield: float
    exact_precision: float
    pure_precision: float
    manual_notes_remaining: int
    confirmed_candidates: tuple[AssemblyCandidate, ...]
    automatic_candidates: int = 0
    review_candidates: int = 0
    automatic_exact_confirmed: int = 0
    automatic_exact_yield: float = 0.0
    automatic_exact_precision: float = 0.0


@dataclass(frozen=True)
class TearFitTrialResult:
    config: dict
    fragments: int
    pair_scores: int
    accepted_edges: int
    false_edge_rate: float
    true_edge_median: float
    false_edge_median: float
    candidates: int
    diagnostics: TearFitDiagnostics
    edge_decisions: dict[str, int] = field(default_factory=dict)
    candidate_decisions: dict[str, int] = field(default_factory=dict)
    search_stats: dict[str, dict[str, int | bool]] = field(default_factory=dict)
    core_candidates: int = 0
    partial_core_candidates: int = 0
    gap_candidates: int = 0
    partial_gap_candidates: int = 0
    selected_partial_gap_candidates: int = 0

    def to_jsonable(self) -> dict:
        return {
            "config": self.config,
            "fragments": self.fragments,
            "pair_scores": self.pair_scores,
            "accepted_edges": self.accepted_edges,
            "false_edge_rate": self.false_edge_rate,
            "true_edge_median": self.true_edge_median,
            "false_edge_median": self.false_edge_median,
            "candidates": self.candidates,
            "core_candidates": self.core_candidates,
            "partial_core_candidates": self.partial_core_candidates,
            "gap_candidates": self.gap_candidates,
            "partial_gap_candidates": self.partial_gap_candidates,
            "selected_partial_gap_candidates": self.selected_partial_gap_candidates,
            "edge_decisions": dict(self.edge_decisions),
            "candidate_decisions": dict(self.candidate_decisions),
            "search_stats": self.search_stats,
            "diagnostics": {
                "confirmed": self.diagnostics.confirmed,
                "exact_confirmed": self.diagnostics.exact_confirmed,
                "pure_confirmed": self.diagnostics.pure_confirmed,
                "chimeras": self.diagnostics.chimeras,
                "true_notes": self.diagnostics.true_notes,
                "exact_yield": self.diagnostics.exact_yield,
                "exact_precision": self.diagnostics.exact_precision,
                "pure_precision": self.diagnostics.pure_precision,
                "manual_notes_remaining": self.diagnostics.manual_notes_remaining,
                "automatic_candidates": self.diagnostics.automatic_candidates,
                "review_candidates": self.diagnostics.review_candidates,
                "automatic_exact_confirmed": self.diagnostics.automatic_exact_confirmed,
                "automatic_exact_yield": self.diagnostics.automatic_exact_yield,
                "automatic_exact_precision": self.diagnostics.automatic_exact_precision,
                "confirmed_candidates": [
                    {
                        "fragment_ids": item.fragment_ids,
                        "coverage": item.coverage,
                        "raw_coverage": item.raw_coverage,
                        "score": item.score,
                        "support_pixels": item.support_pixels,
                        "labels": item.labels,
                        "evidence_score": item.evidence_score,
                        "evidence_level": item.evidence_level,
                        "gap_steps": item.gap_steps,
                    }
                    for item in self.diagnostics.confirmed_candidates
                ],
            },
        }


@dataclass(frozen=True)
class TearFitComparisonCase:
    name: str
    notes: int
    pieces_per_note: int
    fray_probability: float = 0.18
    roughness: float = 4.0


def _serial_roi(height: int, width: int) -> tuple[int, int, int, int]:
    return int(width * 0.06), int(height * 0.62), int(width * 0.42), int(height * 0.92)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _smooth_noise(
    length: int, rng: np.random.Generator, roughness: float
) -> np.ndarray:
    if length <= 1:
        return np.zeros(max(1, length), dtype=np.float32)
    walk = np.cumsum(rng.normal(0.0, roughness, size=length)).astype(np.float32)
    walk -= float(walk.mean())
    for window in (9, 5, 3):
        if length >= window:
            kernel = np.ones(window, dtype=np.float32) / float(window)
            walk = np.convolve(walk, kernel, mode="same").astype(np.float32)
    limit = max(2.0, roughness * 3.0)
    return np.clip(walk, -limit, limit)


def _split_mask_once(
    mask: np.ndarray,
    rng: np.random.Generator,
    roughness: float,
    min_area: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    x0, y0, x1, y1 = _bbox(mask)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    vertical = (x1 - x0) >= (y1 - y0)
    if vertical:
        span = max(2, x1 - x0)
        base = int(rng.integers(x0 + span // 3, x1 - span // 3 + 1))
        line = base + _smooth_noise(y1 - y0, rng, roughness)
        threshold = np.empty(mask.shape[0], dtype=np.float32)
        threshold[:] = base
        threshold[y0:y1] = line
        left = mask & (xx <= threshold[:, None])
        right = mask & ~left
    else:
        span = max(2, y1 - y0)
        base = int(rng.integers(y0 + span // 3, y1 - span // 3 + 1))
        line = base + _smooth_noise(x1 - x0, rng, roughness)
        threshold = np.empty(mask.shape[1], dtype=np.float32)
        threshold[:] = base
        threshold[x0:x1] = line
        left = mask & (yy <= threshold[None, :])
        right = mask & ~left
    if int(left.sum()) < min_area or int(right.sum()) < min_area:
        return None
    return left, right


def fractal_tear_partition(
    height: int,
    width: int,
    pieces: int,
    seed: int,
    *,
    roughness: float = 4.0,
) -> list[np.ndarray]:
    """Partition a note rectangle by recursive jagged tears."""

    if pieces < 1:
        raise ValueError("pieces must be >= 1")
    rng = np.random.default_rng(seed)
    masks = [np.ones((height, width), dtype=bool)]
    min_area = max(16, height * width // max(pieces * 24, 1))
    attempts = 0
    while len(masks) < pieces and attempts < pieces * 80:
        attempts += 1
        index = int(np.argmax([item.sum() for item in masks]))
        split = _split_mask_once(
            masks[index], rng, roughness=roughness, min_area=min_area
        )
        if split is None:
            continue
        masks.pop(index)
        masks.extend(split)
    return masks


def tear_boundary(mask: np.ndarray, *, outer_margin: int = 1) -> np.ndarray:
    """Return placed tear-boundary pixels, excluding clean note-frame edges."""

    mask = mask.astype(bool)
    if mask.size == 0:
        return mask.copy()
    padded = np.pad(mask, 1, constant_values=False)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    boundary = mask & ~(up & down & left & right)
    if outer_margin >= 0:
        clean = np.zeros_like(boundary)
        clean[: outer_margin + 1, :] = True
        clean[-(outer_margin + 1) :, :] = True
        clean[:, : outer_margin + 1] = True
        clean[:, -(outer_margin + 1) :] = True
        boundary &= ~clean
    return boundary


def _apply_fray(
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    layers: int,
    probability: float,
) -> np.ndarray:
    frayed = mask.copy()
    for layer in range(max(0, layers)):
        boundary = tear_boundary(frayed, outer_margin=1)
        if not np.any(boundary):
            break
        drop = boundary & (rng.random(frayed.shape) < probability / float(layer + 1))
        candidate = frayed & ~drop
        if candidate.sum() >= max(8, mask.sum() * 0.75):
            frayed = candidate
    return frayed


def make_fractal_tear_fragments(
    config: FractalTearConfig,
) -> tuple[np.ndarray, list[Fragment]]:
    """Generate placed fragments with per-note jagged tears and edge fray."""

    if config.notes < 1:
        raise ValueError("notes must be >= 1")
    if not (0.0 <= config.serial_ocr_rate <= 1.0):
        raise ValueError("serial_ocr_rate must be in [0, 1]")
    template = synthetic_banknote(config.width, config.height, seed=config.seed)
    x0, y0, x1, y1 = _serial_roi(config.height, config.width)
    fragments: list[Fragment] = []
    for note_index in range(config.notes):
        rng = np.random.default_rng(config.seed + 10_003 * (note_index + 1))
        masks = fractal_tear_partition(
            config.height,
            config.width,
            config.pieces_per_note,
            seed=config.seed + 101 * (note_index + 1),
            roughness=config.roughness,
        )
        note_fragments: list[Fragment] = []
        serial = f"SN{note_index:08d}"
        note_id = f"note-{note_index:03d}"
        serial_overlaps: list[int] = []
        for piece_index, raw_mask in enumerate(masks):
            mask = _apply_fray(
                raw_mask,
                rng,
                layers=config.fray_layers,
                probability=config.fray_probability,
            )
            overlap = int(raw_mask[y0:y1, x0:x1].sum())
            serial_overlaps.append(overlap)
            label = serial if overlap >= max(10, (y1 - y0) * (x1 - x0) // 20) else None
            image = np.where(mask[..., None], template, 0)
            note_fragments.append(
                Fragment(
                    id=f"n{note_index:03d}f{piece_index:03d}",
                    label=label,
                    mask=mask,
                    image=image,
                    meta={
                        "note_id": note_id,
                        "serial": serial,
                        "partition_model": "fractal",
                        "fray_layers": config.fray_layers,
                        "fray_probability": config.fray_probability,
                    },
                )
            )
        if config.ensure_serial_anchor and not any(
            fragment.label for fragment in note_fragments
        ):
            anchor = int(np.argmax(serial_overlaps))
            old = note_fragments[anchor]
            note_fragments[anchor] = Fragment(
                id=old.id,
                label=serial,
                side=old.side,
                mask=old.mask,
                image=old.image,
                tags=old.tags,
                meta=old.meta,
            )
        if rng.random() > config.serial_ocr_rate:
            note_fragments = [
                Fragment(
                    id=fragment.id,
                    label=None,
                    side=fragment.side,
                    mask=fragment.mask,
                    image=fragment.image,
                    tags=fragment.tags,
                    meta=fragment.meta,
                )
                for fragment in note_fragments
            ]
        fragments.extend(note_fragments)
    return template, fragments


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(0, int(radius))):
        out = binary_dilation_3x3(out)
    return out


def _boundary_normals(
    mask: np.ndarray, boundary: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate outward raster normals on boundary pixels."""

    padded = np.pad(mask.astype(np.float32), 1, constant_values=0.0)
    normal_x = padded[1:-1, :-2] - padded[1:-1, 2:]
    normal_y = padded[:-2, 1:-1] - padded[2:, 1:-1]
    magnitude = np.hypot(normal_y, normal_x)
    valid = boundary & (magnitude > 0.0)
    out_y = np.zeros(mask.shape, dtype=np.float32)
    out_x = np.zeros(mask.shape, dtype=np.float32)
    out_y[valid] = normal_y[valid] / magnitude[valid]
    out_x[valid] = normal_x[valid] / magnitude[valid]
    return out_y, out_x


def _largest_connected_support(mask: np.ndarray) -> int:
    """Return the largest 8-connected component in a sparse support mask."""

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    active = mask[y0:y1, x0:x1]
    visited = np.zeros(active.shape, dtype=bool)
    best = 0
    for seed_y, seed_x in np.argwhere(active):
        sy = int(seed_y)
        sx = int(seed_x)
        if visited[sy, sx]:
            continue
        visited[sy, sx] = True
        stack = [(sy, sx)]
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny = y + dy
                    nx = x + dx
                    if (
                        ny < 0
                        or nx < 0
                        or ny >= active.shape[0]
                        or nx >= active.shape[1]
                    ):
                        continue
                    if active[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        best = max(best, size)
    return best


def _matched_normal_opposition(
    left: TearBoundaryEvidence,
    right: TearBoundaryEvidence,
    left_match: np.ndarray,
    *,
    tolerance: int,
    sample_limit: int = 128,
) -> float:
    """Measure whether nearby outward normals point in opposite directions."""

    points = np.argwhere(left_match)
    if len(points) == 0:
        return 0.0
    if len(points) > sample_limit:
        points = points[np.linspace(0, len(points) - 1, sample_limit, dtype=np.int64)]
    radius = max(0, int(tolerance))
    height, width = left_match.shape
    scores: list[float] = []
    for raw_y, raw_x in points:
        y = int(raw_y)
        x = int(raw_x)
        left_ny = float(left.normal_y[y, x])
        left_nx = float(left.normal_x[y, x])
        if left_ny == 0.0 and left_nx == 0.0:
            continue
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        local = right.boundary_mask[y0:y1, x0:x1]
        candidates = np.argwhere(local)
        if len(candidates) == 0:
            continue
        candidate_scores: list[float] = []
        for local_y, local_x in candidates:
            ry = y0 + int(local_y)
            rx = x0 + int(local_x)
            right_ny = float(right.normal_y[ry, rx])
            right_nx = float(right.normal_x[ry, rx])
            if right_ny == 0.0 and right_nx == 0.0:
                continue
            candidate_scores.append(-(left_ny * right_ny + left_nx * right_nx))
        if candidate_scores:
            scores.append(float(np.clip(max(candidate_scores), 0.0, 1.0)))
    return float(np.mean(scores)) if scores else 0.0


def _curvature_entropy(
    left: TearBoundaryEvidence,
    right: TearBoundaryEvidence,
    left_match: np.ndarray,
    right_match: np.ndarray,
    *,
    bins: int = 8,
) -> float:
    """Raster curvature proxy from the entropy of matched boundary normals."""

    angles: list[np.ndarray] = []
    for evidence, matched in ((left, left_match), (right, right_match)):
        valid = matched & ((evidence.normal_y != 0.0) | (evidence.normal_x != 0.0))
        if np.any(valid):
            angles.append(
                np.mod(
                    np.arctan2(evidence.normal_y[valid], evidence.normal_x[valid]),
                    2.0 * np.pi,
                )
            )
    if not angles:
        return 0.0
    values = np.concatenate(angles)
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 2.0 * np.pi))
    probabilities = counts[counts > 0].astype(np.float64)
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / float(np.log(bins)) if bins > 1 else 0.0


def tear_match_effectiveness(
    *,
    contiguous_pixels: int,
    overlap_pixels: int,
    left_hits: int,
    right_hits: int,
    overlap_ratio: float = 1.0,
    normal_opposition: float,
    curvature_entropy: float,
    expected_accidental_hits: float,
    mask_overlap_pixels: int = 0,
) -> float:
    """Benefit-to-damage score for one placed tear-edge match.

    The numerator rewards one continuous, bidirectional seam with opposing
    normals. The denominator charges scattered hits, one-sided support, actual
    mask overlap, and the random coincidence expected from boundary density.
    This is inspired by tolerance-derived restoration effectiveness, but it is
    a MoneyRepair tear-matching metric rather than the paper's physical formula.
    """

    if (
        min(
            contiguous_pixels,
            overlap_pixels,
            left_hits,
            right_hits,
            mask_overlap_pixels,
        )
        < 0
    ):
        raise ValueError("tear-match counts must be non-negative")
    if expected_accidental_hits < 0.0:
        raise ValueError("expected_accidental_hits must be non-negative")
    if overlap_ratio < 0.0:
        raise ValueError("overlap_ratio must be non-negative")
    if overlap_pixels == 0:
        return 0.0
    balance = overlap_pixels / float(max(1, max(left_hits, right_hits)))
    scattered = max(0, overlap_pixels - contiguous_pixels)
    one_sided = abs(left_hits - right_hits)
    geometry_factor = 0.5 + 0.5 * float(np.clip(normal_opposition, 0.0, 1.0))
    specificity_factor = 0.75 + 0.5 * float(np.clip(curvature_entropy, 0.0, 1.0))
    explained_perimeter_factor = float(np.clip(overlap_ratio / 0.30, 0.25, 1.5))
    benefit = (
        contiguous_pixels
        * balance
        * geometry_factor
        * specificity_factor
        * explained_perimeter_factor
    )
    damage = (
        1.0
        + scattered
        + one_sided
        + 2.0 * mask_overlap_pixels
        + expected_accidental_hits
    )
    return float(benefit / damage)


def tear_boundary_evidence(
    fragments: list[Fragment], *, tolerance: int = 2
) -> list[TearBoundaryEvidence]:
    evidence: list[TearBoundaryEvidence] = []
    for fragment in fragments:
        boundary = tear_boundary(fragment.mask)
        dilated = _dilate(boundary, tolerance)
        normal_y, normal_x = _boundary_normals(fragment.mask, boundary)
        evidence.append(
            TearBoundaryEvidence(
                boundary=np.flatnonzero(boundary),
                dilated_boundary=np.flatnonzero(dilated),
                bbox=_bbox(dilated),
                boundary_mask=boundary,
                dilated_boundary_mask=dilated,
                normal_y=normal_y,
                normal_x=normal_x,
            )
        )
    return evidence


def _labels_compatible(left: str | None, right: str | None) -> bool:
    return not left or not right or left == right


def _bbox_intersects(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> bool:
    return (
        left[0] < right[2]
        and left[2] > right[0]
        and left[1] < right[3]
        and left[3] > right[1]
    )


def _mask_overlap(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left & right))


def score_absolute_tear_pairs(
    fragments: list[Fragment],
    *,
    tolerance: int = 2,
    min_overlap_pixels: int = 1,
    max_pair_overlap_pixels: int = 0,
    use_labels: bool = True,
    scoring: str = "overlap",
    min_effectiveness: float = 1.0,
    automatic_effectiveness: float = 2.0,
    min_contiguous_pixels: int = 3,
    automatic_contiguous_pixels: int = 5,
) -> tuple[list[TearFitEdge], list[TearFitEdge]]:
    """Score torn-edge coincidence in the placed coordinate frame.

    Returns ``(all_scored_pairs, accepted_edges)``. ``overlap`` preserves the
    historical fixed-pixel baseline. ``effectiveness`` accepts short seams when
    their continuous, bidirectional geometry outweighs scattered and accidental
    support; accepted edges are labelled ``automatic`` or ``review``.
    """

    if scoring not in TEARFIT_EDGE_SCORING:
        raise ValueError(f"scoring must be one of: {', '.join(TEARFIT_EDGE_SCORING)}")
    if min_effectiveness < 0.0 or automatic_effectiveness < min_effectiveness:
        raise ValueError("effectiveness thresholds must be ordered and non-negative")
    if min_contiguous_pixels < 1 or automatic_contiguous_pixels < min_contiguous_pixels:
        raise ValueError("contiguous-pixel thresholds must be positive and ordered")
    evidence = tear_boundary_evidence(fragments, tolerance=tolerance)
    all_scores: list[TearFitEdge] = []
    accepted: list[TearFitEdge] = []
    for i in range(len(fragments)):
        left_ev = evidence[i]
        for j in range(i + 1, len(fragments)):
            if use_labels and not _labels_compatible(
                fragments[i].label, fragments[j].label
            ):
                continue
            mask_overlap = _mask_overlap(fragments[i].mask, fragments[j].mask)
            if max_pair_overlap_pixels >= 0 and mask_overlap > max_pair_overlap_pixels:
                continue
            right_ev = evidence[j]
            if not _bbox_intersects(left_ev.bbox, right_ev.bbox):
                continue
            left_hits = int(
                np.intersect1d(
                    left_ev.boundary, right_ev.dilated_boundary, assume_unique=True
                ).size
            )
            right_hits = int(
                np.intersect1d(
                    right_ev.boundary, left_ev.dilated_boundary, assume_unique=True
                ).size
            )
            overlap = min(left_hits, right_hits)
            denom = max(1, min(len(left_ev.boundary), len(right_ev.boundary)))
            contiguous = 0
            continuity_ratio = 0.0
            balance = overlap / float(max(1, max(left_hits, right_hits)))
            normal_opposition = 0.0
            curvature_entropy = 0.0
            expected_accidental_hits = 0.0
            pose_uncertainty = 0.0
            effectiveness = 0.0
            evidence_level = "insufficient-evidence"
            if scoring == "effectiveness" and overlap > 0:
                left_match = left_ev.boundary_mask & right_ev.dilated_boundary_mask
                right_match = right_ev.boundary_mask & left_ev.dilated_boundary_mask
                contiguous = max(
                    _largest_connected_support(left_match),
                    _largest_connected_support(right_match),
                )
                continuity_ratio = contiguous / float(max(1, overlap))
                normal_opposition = _matched_normal_opposition(
                    left_ev,
                    right_ev,
                    left_match,
                    tolerance=tolerance,
                )
                curvature_entropy = _curvature_entropy(
                    left_ev, right_ev, left_match, right_match
                )
                total_pixels = float(fragments[i].mask.size)
                expected_left = (
                    len(left_ev.boundary)
                    * len(right_ev.dilated_boundary)
                    / total_pixels
                )
                expected_right = (
                    len(right_ev.boundary)
                    * len(left_ev.dilated_boundary)
                    / total_pixels
                )
                expected_accidental_hits = 0.5 * (expected_left + expected_right)
                effectiveness = tear_match_effectiveness(
                    contiguous_pixels=contiguous,
                    overlap_pixels=overlap,
                    left_hits=left_hits,
                    right_hits=right_hits,
                    overlap_ratio=overlap / float(denom),
                    normal_opposition=normal_opposition,
                    curvature_entropy=curvature_entropy,
                    expected_accidental_hits=expected_accidental_hits,
                    mask_overlap_pixels=mask_overlap,
                )
                pose_uncertainty = max(
                    _normalised_pose_uncertainty(fragments[i], tolerance),
                    _normalised_pose_uncertainty(fragments[j], tolerance),
                )
                effectiveness *= 1.0 - 0.5 * pose_uncertainty
                if (
                    effectiveness >= automatic_effectiveness
                    and contiguous >= automatic_contiguous_pixels
                ):
                    evidence_level = "automatic"
                elif (
                    effectiveness >= min_effectiveness
                    and contiguous >= min_contiguous_pixels
                ):
                    evidence_level = "review"
            elif overlap >= min_overlap_pixels:
                evidence_level = "automatic"
            edge = TearFitEdge(
                left=i,
                right=j,
                overlap_pixels=overlap,
                left_hits=left_hits,
                right_hits=right_hits,
                overlap_ratio=overlap / float(denom),
                contiguous_pixels=contiguous,
                continuity_ratio=continuity_ratio,
                bidirectional_balance=balance,
                normal_opposition=normal_opposition,
                curvature_entropy=curvature_entropy,
                expected_accidental_hits=expected_accidental_hits,
                mask_overlap_pixels=mask_overlap,
                pose_uncertainty=pose_uncertainty,
                effectiveness=effectiveness,
                evidence_level=evidence_level,
            )
            all_scores.append(edge)
            if evidence_level != "insufficient-evidence":
                accepted.append(edge)
    return all_scores, accepted


def _edge_graph(
    edges: Iterable[TearFitEdge], count: int
) -> tuple[list[dict[int, TearFitEdge]], dict[tuple[int, int], TearFitEdge]]:
    graph: list[dict[int, TearFitEdge]] = [dict() for _ in range(count)]
    lookup: dict[tuple[int, int], TearFitEdge] = {}
    for edge in edges:
        graph[edge.left][edge.right] = edge
        graph[edge.right][edge.left] = edge
        lookup[(min(edge.left, edge.right), max(edge.left, edge.right))] = edge
    return graph, lookup


def _group_labels(fragments: list[Fragment], indices: Iterable[int]) -> tuple[str, ...]:
    return tuple(
        sorted({fragments[index].label for index in indices if fragments[index].label})
    )


def _labels_ok(fragments: list[Fragment], indices: Iterable[int]) -> bool:
    return len(_group_labels(fragments, indices)) <= 1


def _group_masks_ok(
    fragments: list[Fragment], indices: Iterable[int], max_overlap_pixels: int
) -> tuple[bool, np.ndarray, int]:
    selected = list(indices)
    if not selected:
        raise ValueError("indices must not be empty")
    union = np.zeros_like(fragments[selected[0]].mask)
    area_sum = 0
    for index in selected:
        area_sum += fragments[index].area
        union |= fragments[index].mask
    overlap = area_sum - int(union.sum())
    return overlap <= max_overlap_pixels, union, overlap


def _support_for_state(
    state: frozenset[int], edge_lookup: dict[tuple[int, int], TearFitEdge]
) -> int:
    members = sorted(state)
    support = 0
    for pos, left in enumerate(members):
        for right in members[pos + 1 :]:
            edge = edge_lookup.get((left, right))
            if edge is not None:
                support += edge.overlap_pixels
    return support


def _candidate_key(fragment_ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(fragment_ids))


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _normalise_candidate_keys(
    values: Iterable[Iterable[str]] | None,
) -> set[tuple[str, ...]]:
    if values is None:
        return set()
    return {_candidate_key(value) for value in values}


def _normalise_index_pairs(
    values: Iterable[tuple[str, str]] | None,
    fragment_id_to_index: dict[str, int],
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    if values is None:
        return pairs
    for left_id, right_id in values:
        if left_id == right_id:
            continue
        left = fragment_id_to_index.get(left_id)
        right = fragment_id_to_index.get(right_id)
        if left is None or right is None:
            continue
        pairs.add((left, right) if left <= right else (right, left))
    return pairs


def _state_has_pair(state: frozenset[int], pair: tuple[int, int]) -> bool:
    return pair[0] in state and pair[1] in state


def _state_has_forbidden_pair(
    state: frozenset[int], pairs: set[tuple[int, int]]
) -> bool:
    return any(_state_has_pair(state, pair) for pair in pairs)


def _forced_pairs_satisfied(state: frozenset[int], pairs: set[tuple[int, int]]) -> bool:
    for left, right in pairs:
        if (left in state) != (right in state):
            return False
    return True


def _pair_preference_bonus(
    state: frozenset[int], pairs: set[tuple[int, int]], bonus: float
) -> float:
    if not pairs:
        return 0.0
    return bonus * sum(1 for pair in pairs if _state_has_pair(state, pair))


def _resolve_seed_strategy(seed_strategy: str, require_anchor: bool | None) -> str:
    if require_anchor is not None:
        return "anchor_priority"
    if seed_strategy not in TEARFIT_SEED_STRATEGIES:
        raise ValueError(
            f"seed_strategy must be one of: {', '.join(TEARFIT_SEED_STRATEGIES)}"
        )
    return seed_strategy


def _seed_order(fragments: list[Fragment], seed_strategy: str) -> list[int]:
    labelled = [index for index, fragment in enumerate(fragments) if fragment.label]
    unlabelled = [
        index for index, fragment in enumerate(fragments) if not fragment.label
    ]
    if seed_strategy == "anchor_only":
        return labelled
    if seed_strategy == "anchor_priority":
        return labelled + unlabelled
    if seed_strategy == "all":
        return list(range(len(fragments)))
    raise ValueError(
        f"seed_strategy must be one of: {', '.join(TEARFIT_SEED_STRATEGIES)}"
    )


def _normalised_pose_uncertainty(fragment: Fragment, tolerance: int) -> float:
    """Read optional locator uncertainty from fragment metadata on a 0..1 scale."""

    sigma_x = float(
        fragment.meta.get("sigma_x", fragment.meta.get("pose_sigma_x", 0.0)) or 0.0
    )
    sigma_y = float(
        fragment.meta.get("sigma_y", fragment.meta.get("pose_sigma_y", 0.0)) or 0.0
    )
    sigma_theta = abs(
        float(
            fragment.meta.get("sigma_theta", fragment.meta.get("pose_sigma_theta", 0.0))
            or 0.0
        )
    )
    sigma_scale = abs(
        float(
            fragment.meta.get("sigma_scale", fragment.meta.get("pose_sigma_scale", 0.0))
            or 0.0
        )
    )
    translation = np.hypot(sigma_x, sigma_y) / float(max(1, 2 * tolerance))
    rotation = sigma_theta / 5.0
    scale = sigma_scale / 0.02
    return float(np.clip((translation + rotation + scale) / 3.0, 0.0, 1.0))


def score_fragment_against_assembly(
    fragments: list[Fragment],
    state: frozenset[int],
    union_mask: np.ndarray,
    candidate_index: int,
    edge_lookup: dict[tuple[int, int], TearFitEdge],
    boundary_evidence: list[TearBoundaryEvidence],
    *,
    tolerance: int = 2,
    min_score: float = 0.35,
    automatic_score: float = 0.55,
) -> GroupGapEvidence:
    """Score one fragment against the full boundary of a partial assembly."""

    fragment = fragments[candidate_index]
    overlap = int(np.count_nonzero(union_mask & fragment.mask))
    gap_fill_pixels = max(0, fragment.area - overlap)
    gap_fill_ratio = gap_fill_pixels / float(max(1, fragment.area))

    assembly_boundary = tear_boundary(union_mask)
    candidate_boundary = boundary_evidence[candidate_index].boundary_mask
    matched_mask = candidate_boundary & _dilate(assembly_boundary, tolerance)
    matched_pixels = int(np.count_nonzero(matched_mask))
    boundary_pixels = int(np.count_nonzero(candidate_boundary))
    matched_ratio = matched_pixels / float(max(1, boundary_pixels))
    unmatched_ratio = 1.0 - matched_ratio

    supporting: list[TearFitEdge] = []
    for member in state:
        edge = edge_lookup.get(
            (min(member, candidate_index), max(member, candidate_index))
        )
        if (
            edge is not None
            and edge.contiguous_pixels >= 2
            and edge.effectiveness > 0.0
        ):
            supporting.append(edge)
    seam_count = len(supporting)
    edge_strength = (
        float(
            np.mean(
                [edge.effectiveness / (1.0 + edge.effectiveness) for edge in supporting]
            )
        )
        if supporting
        else 0.0
    )
    multi_seam_support = min(1.0, seam_count / 2.0)
    overlap_ratio = overlap / float(max(1, fragment.area))
    pose_uncertainty = _normalised_pose_uncertainty(fragment, tolerance)
    score = (
        0.15 * gap_fill_ratio
        + 0.25 * multi_seam_support
        + 0.25 * matched_ratio
        + 0.25 * edge_strength
        - 0.35 * overlap_ratio
        - 0.10 * unmatched_ratio
        - 0.15 * pose_uncertainty
    )
    if score >= automatic_score and seam_count >= 2:
        evidence_level = "automatic"
    elif score >= min_score and matched_pixels >= 2:
        evidence_level = "review"
    else:
        evidence_level = "insufficient-evidence"
    return GroupGapEvidence(
        fragment=candidate_index,
        seam_count=seam_count,
        matched_pixels=matched_pixels,
        gap_fill_pixels=gap_fill_pixels,
        gap_fill_ratio=gap_fill_ratio,
        overlap_pixels=overlap,
        unmatched_perimeter_ratio=unmatched_ratio,
        pose_uncertainty=pose_uncertainty,
        score=float(score),
        evidence_level=evidence_level,
    )


class StateInfo:
    __slots__ = (
        "state",
        "union",
        "area_sum",
        "labels",
        "support",
        "score",
        "raw_coverage",
        "coverage",
        "evidence_score",
        "evidence_level",
        "gap_steps",
    )

    def __init__(
        self,
        state: frozenset[int],
        union: np.ndarray,
        area_sum: int,
        labels: set[str],
        support: int,
        score: float,
        raw_coverage: float,
        coverage: float,
        evidence_score: float = 1.0,
        evidence_level: str = "automatic",
        gap_steps: int = 0,
    ) -> None:
        self.state = state
        self.union = union
        self.area_sum = area_sum
        self.labels = labels
        self.support = support
        self.score = score
        self.raw_coverage = raw_coverage
        self.coverage = coverage
        self.evidence_score = evidence_score
        self.evidence_level = evidence_level
        self.gap_steps = gap_steps


def generate_assembly_candidates(
    fragments: list[Fragment],
    edges: list[TearFitEdge],
    *,
    gap_edges: list[TearFitEdge] | None = None,
    enable_group_gap: bool = False,
    group_tolerance: int = 2,
    core_min_pieces: int = 2,
    min_group_gap_score: float = 0.35,
    automatic_group_gap_score: float = 0.55,
    coverage_threshold: float = 0.93,
    minimum_candidate_raw_coverage: float | None = None,
    gap_fill_radius: int = 2,
    max_pieces: int = 12,
    beam_width: int = 64,
    seed_strategy: str = "anchor_priority",
    require_anchor: bool | None = None,
    seed_whitelist: Iterable[str] | None = None,
    seed_labels: Iterable[str] | None = None,
    forced_pairs: Iterable[tuple[str, str]] | None = None,
    preferred_pairs: Iterable[tuple[str, str]] | None = None,
    forbidden_pairs: Iterable[tuple[str, str]] | None = None,
    forbidden_candidates: Iterable[Iterable[str]] | None = None,
    preferred_pair_bonus: float = 500.0,
    max_group_overlap_pixels: int = 0,
    time_limit_seconds: float | None = 20.0,
    max_expanded_states: int | None = None,
    search_stats: dict[str, int | bool] | None = None,
) -> list[AssemblyCandidate]:
    """Generate full-note candidates by connected tear graph search.

    The search is label-aware and overlap-constrained. Serial labels act as
    constraints and priority seeds, not as the only legal entry point unless
    ``seed_strategy="anchor_only"`` is explicitly requested for comparison.
    With ``enable_group_gap``, high-confidence ``edges`` first build a core.
    Weaker ``gap_edges`` can then add a fragment only when its placement is
    supported by the boundary of the whole partial assembly. It produces
    candidate note groups; :func:`select_exact_cover_candidates` then performs
    the global set-packing pass over those candidates.
    """

    if not fragments:
        if search_stats is not None:
            search_stats.update(
                expanded_states=0, state_limit_reached=False, time_limit_reached=False
            )
        return []
    if core_min_pieces < 1:
        raise ValueError("core_min_pieces must be positive")
    if min_group_gap_score < 0.0 or automatic_group_gap_score < min_group_gap_score:
        raise ValueError("group-gap thresholds must be ordered and non-negative")
    if minimum_candidate_raw_coverage is not None and not (
        0.0 < minimum_candidate_raw_coverage <= 1.0
    ):
        raise ValueError("minimum_candidate_raw_coverage must be in (0, 1]")
    if max_expanded_states is not None and max_expanded_states < 1:
        raise ValueError("max_expanded_states must be positive")
    seed_strategy = _resolve_seed_strategy(seed_strategy, require_anchor)
    fragment_id_to_index = {
        fragment.id: index for index, fragment in enumerate(fragments)
    }
    seed_whitelist_set = set(seed_whitelist or ())
    seed_label_set = set(seed_labels or ())
    forced_pair_indices = _normalise_index_pairs(forced_pairs, fragment_id_to_index)
    preferred_pair_indices = (
        _normalise_index_pairs(preferred_pairs, fragment_id_to_index)
        | forced_pair_indices
    )
    forbidden_pair_indices = _normalise_index_pairs(
        forbidden_pairs, fragment_id_to_index
    )
    forbidden_candidate_keys = _normalise_candidate_keys(forbidden_candidates)
    graph, edge_lookup = _edge_graph(edges, len(fragments))
    eligible_gap_edges = [
        edge
        for edge in (gap_edges or ())
        if edge.contiguous_pixels >= 2 and edge.effectiveness > 0.0
    ]
    gap_graph, gap_edge_lookup = _edge_graph(eligible_gap_edges, len(fragments))
    group_boundary_evidence = (
        tear_boundary_evidence(fragments, tolerance=group_tolerance)
        if enable_group_gap
        else []
    )
    starts = _seed_order(fragments, seed_strategy)
    if seed_whitelist_set or seed_label_set:
        starts = [
            index
            for index in starts
            if fragments[index].id in seed_whitelist_set
            or (fragments[index].label and fragments[index].label in seed_label_set)
        ]
    if not starts:
        if search_stats is not None:
            search_stats.update(
                expanded_states=0, state_limit_reached=False, time_limit_reached=False
            )
        return []
    total_area = fragments[0].mask.size
    record_raw_threshold = (
        coverage_threshold
        if minimum_candidate_raw_coverage is None
        else minimum_candidate_raw_coverage
    )
    deadline = None if time_limit_seconds is None else monotonic() + time_limit_seconds
    expanded_states = 0
    state_limit_reached = False
    time_limit_reached = False

    def budget_exhausted() -> bool:
        nonlocal state_limit_reached, time_limit_reached
        if max_expanded_states is not None and expanded_states >= max_expanded_states:
            state_limit_reached = True
            return True
        if deadline is not None and monotonic() >= deadline:
            time_limit_reached = True
            return True
        return False

    seen_states: set[frozenset[int]] = set()
    candidates: dict[tuple[str, ...], AssemblyCandidate] = {}

    for start in starts:
        if budget_exhausted():
            break
        start_fragment = fragments[start]
        start_union = start_fragment.mask.copy()
        start_area_sum = start_fragment.area
        start_labels = {start_fragment.label} if start_fragment.label else set()
        start_support = 0
        start_raw_coverage = start_area_sum / float(total_area)
        if start_raw_coverage >= coverage_threshold - 0.15:
            start_coverage = int(_dilate(start_union, gap_fill_radius).sum()) / float(
                total_area
            )
        else:
            start_coverage = start_raw_coverage
        start_score = (
            start_coverage * 10_000.0 + start_support + 100.0 * len(start_labels)
        )

        initial_state = StateInfo(
            state=frozenset({start}),
            union=start_union,
            area_sum=start_area_sum,
            labels=start_labels,
            support=start_support,
            score=start_score,
            raw_coverage=start_raw_coverage,
            coverage=start_coverage,
        )

        frontier: list[StateInfo] = [initial_state]
        for _depth in range(max(1, max_pieces - 1)):
            if budget_exhausted():
                break
            next_frontier: dict[frozenset[int], StateInfo] = {}
            for state_info in frontier:
                if budget_exhausted():
                    break
                expanded_states += 1
                neighbours: set[int] = set()
                for member in state_info.state:
                    neighbours.update(graph[member])
                neighbours.difference_update(state_info.state)
                gap_options: dict[int, GroupGapEvidence] = {}
                if enable_group_gap and len(state_info.state) >= core_min_pieces:
                    gap_pool: set[int] = set()
                    for member in state_info.state:
                        gap_pool.update(gap_graph[member])
                    gap_pool.difference_update(state_info.state)
                    gap_pool.difference_update(neighbours)
                    for gap_index in gap_pool:
                        gap_evidence = score_fragment_against_assembly(
                            fragments,
                            state_info.state,
                            state_info.union,
                            gap_index,
                            gap_edge_lookup,
                            group_boundary_evidence,
                            tolerance=group_tolerance,
                            min_score=min_group_gap_score,
                            automatic_score=automatic_group_gap_score,
                        )
                        if gap_evidence.evidence_level != "insufficient-evidence":
                            gap_options[gap_index] = gap_evidence
                    neighbours.update(gap_options)
                for neighbour in neighbours:
                    new_state_set = frozenset((*state_info.state, neighbour))
                    if new_state_set in seen_states or len(new_state_set) > max_pieces:
                        continue
                    if _state_has_forbidden_pair(new_state_set, forbidden_pair_indices):
                        continue

                    # Check labels incrementally
                    neighbour_label = fragments[neighbour].label
                    if neighbour_label and neighbour_label not in state_info.labels:
                        if len(state_info.labels) > 0:
                            continue
                        new_labels = state_info.labels | {neighbour_label}
                    else:
                        new_labels = state_info.labels

                    # Check overlap and mask incrementally
                    new_area_sum = state_info.area_sum + fragments[neighbour].area
                    new_union = state_info.union | fragments[neighbour].mask
                    overlap = new_area_sum - int(new_union.sum())
                    if overlap > max_group_overlap_pixels:
                        continue

                    # Calculate pairwise or whole-assembly support incrementally.
                    added_support = 0
                    incident_edges: list[TearFitEdge] = []
                    for member in state_info.state:
                        key = (min(member, neighbour), max(member, neighbour))
                        edge = edge_lookup.get(key)
                        if edge is not None:
                            if edge.effectiveness > 0.0:
                                added_support += int(
                                    round(
                                        edge.contiguous_pixels
                                        * min(edge.effectiveness, 8.0)
                                    )
                                )
                            else:
                                added_support += edge.overlap_pixels
                            incident_edges.append(edge)
                    gap_evidence = gap_options.get(neighbour)
                    if gap_evidence is not None:
                        added_support = max(added_support, gap_evidence.matched_pixels)
                        step_score = gap_evidence.score
                        step_level = gap_evidence.evidence_level
                        new_gap_steps = state_info.gap_steps + 1
                    else:
                        adaptive_scores = [
                            edge.effectiveness / (1.0 + edge.effectiveness)
                            for edge in incident_edges
                            if edge.effectiveness > 0.0
                        ]
                        step_score = max(adaptive_scores, default=1.0)
                        step_level = (
                            "automatic"
                            if not incident_edges
                            or any(
                                edge.evidence_level == "automatic"
                                for edge in incident_edges
                            )
                            else "review"
                        )
                        new_gap_steps = state_info.gap_steps
                    new_support = state_info.support + added_support
                    new_evidence_score = min(state_info.evidence_score, step_score)
                    new_evidence_level = (
                        "automatic"
                        if state_info.evidence_level == "automatic"
                        and step_level == "automatic"
                        else "review"
                    )

                    raw_coverage = int(new_union.sum()) / float(total_area)
                    if raw_coverage >= coverage_threshold - 0.15:
                        coverage = int(
                            _dilate(new_union, gap_fill_radius).sum()
                        ) / float(total_area)
                    else:
                        coverage = raw_coverage

                    base_score = (
                        coverage * 10_000.0
                        + new_support
                        + 100.0 * len(new_labels)
                        + 250.0 * new_evidence_score
                        - 25.0 * new_gap_steps
                    )
                    preference_bonus = _pair_preference_bonus(
                        new_state_set, preferred_pair_indices, preferred_pair_bonus
                    )
                    score = base_score + preference_bonus

                    new_info = StateInfo(
                        state=new_state_set,
                        union=new_union,
                        area_sum=new_area_sum,
                        labels=new_labels,
                        support=new_support,
                        score=score,
                        raw_coverage=raw_coverage,
                        coverage=coverage,
                        evidence_score=new_evidence_score,
                        evidence_level=new_evidence_level,
                        gap_steps=new_gap_steps,
                    )

                    existing = next_frontier.get(new_state_set)
                    if existing is None or new_info.score > existing.score:
                        next_frontier[new_state_set] = new_info

                    if (
                        coverage >= coverage_threshold
                        or raw_coverage >= record_raw_threshold
                    ):
                        ids = tuple(
                            sorted(fragments[index].id for index in new_state_set)
                        )
                        if ids in forbidden_candidate_keys:
                            continue
                        if not _forced_pairs_satisfied(
                            new_state_set, forced_pair_indices
                        ):
                            continue
                        existing_candidate = candidates.get(ids)
                        candidate = AssemblyCandidate(
                            fragment_ids=ids,
                            coverage=coverage,
                            raw_coverage=raw_coverage,
                            score=score,
                            support_pixels=new_support,
                            labels=tuple(sorted(new_labels)),
                            base_score=base_score,
                            constraint_bonus=preference_bonus,
                            evidence_score=new_evidence_score,
                            evidence_level=new_evidence_level,
                            gap_steps=new_gap_steps,
                        )
                        if (
                            existing_candidate is None
                            or candidate.score > existing_candidate.score
                        ):
                            candidates[ids] = candidate
                seen_states.add(state_info.state)
            if not next_frontier:
                break
            ordered = sorted(
                next_frontier.values(),
                key=lambda item: (
                    -item.score,
                    len(item.state),
                    tuple(sorted(item.state)),
                ),
            )
            frontier = ordered[:beam_width]

    if search_stats is not None:
        search_stats.update(
            expanded_states=expanded_states,
            state_limit_reached=state_limit_reached,
            time_limit_reached=time_limit_reached,
        )
    return sorted(
        candidates.values(), key=lambda item: (-item.score, item.fragment_ids)
    )


def augment_candidates_with_group_gap(
    fragments: list[Fragment],
    candidates: list[AssemblyCandidate],
    gap_edges: list[TearFitEdge],
    *,
    tolerance: int = 2,
    coverage_threshold: float = 0.93,
    gap_fill_radius: int = 2,
    max_pieces: int = 12,
    core_min_pieces: int = 2,
    beam_width: int = 8,
    max_base_candidates: int = 512,
    min_group_gap_score: float = 0.35,
    automatic_group_gap_score: float = 0.55,
    max_group_overlap_pixels: int = 0,
    time_limit_seconds: float | None = 20.0,
    max_expanded_states: int | None = None,
    search_stats: dict[str, int | bool] | None = None,
) -> list[AssemblyCandidate]:
    """Add weak pairwise fragments only after a high-confidence core exists.

    Every input candidate that already reaches ``coverage_threshold`` remains
    in the output. Lower-coverage cores are expansion seeds only. The second
    stage searches for fragments whose joint fit against the whole core is
    stronger than any individual weak seam, so enabling it cannot erase a
    complete core-only solution.
    """

    if not candidates:
        if search_stats is not None:
            search_stats.update(
                expanded_states=0, state_limit_reached=False, time_limit_reached=False
            )
        return []
    if max_expanded_states is not None and max_expanded_states < 1:
        raise ValueError("max_expanded_states must be positive")
    id_to_index = {fragment.id: index for index, fragment in enumerate(fragments)}
    eligible_gap_edges = [
        edge
        for edge in gap_edges
        if edge.contiguous_pixels >= 2 and edge.effectiveness > 0.0
    ]
    gap_graph, gap_lookup = _edge_graph(eligible_gap_edges, len(fragments))
    boundary_evidence = tear_boundary_evidence(fragments, tolerance=tolerance)
    total_area = fragments[0].mask.size
    deadline = None if time_limit_seconds is None else monotonic() + time_limit_seconds
    expanded_states = 0
    state_limit_reached = False
    time_limit_reached = False

    def budget_exhausted() -> bool:
        nonlocal state_limit_reached, time_limit_reached
        if max_expanded_states is not None and expanded_states >= max_expanded_states:
            state_limit_reached = True
            return True
        if deadline is not None and monotonic() >= deadline:
            time_limit_reached = True
            return True
        return False

    output: dict[tuple[str, ...], AssemblyCandidate] = {
        _candidate_key(candidate.fragment_ids): candidate
        for candidate in candidates
        if candidate.coverage >= coverage_threshold
    }

    base_order = sorted(
        candidates,
        key=lambda item: (
            -len(item.fragment_ids),
            -item.raw_coverage,
            -item.evidence_score,
            -item.score,
            item.fragment_ids,
        ),
    )
    for base in base_order[: max(1, max_base_candidates)]:
        if budget_exhausted():
            break
        try:
            start_state = frozenset(
                id_to_index[fragment_id] for fragment_id in base.fragment_ids
            )
        except KeyError:
            continue
        if len(start_state) < core_min_pieces or len(start_state) >= max_pieces:
            continue
        start_union = np.logical_or.reduce(
            [fragments[index].mask for index in start_state]
        )
        start_area_sum = sum(fragments[index].area for index in start_state)
        start_labels = set(base.labels)
        frontier = [
            StateInfo(
                state=start_state,
                union=start_union,
                area_sum=start_area_sum,
                labels=start_labels,
                support=base.support_pixels,
                score=base.score,
                raw_coverage=base.raw_coverage,
                coverage=base.coverage,
                evidence_score=base.evidence_score,
                evidence_level=base.evidence_level,
                gap_steps=base.gap_steps,
            )
        ]
        seen = {start_state}
        for _depth in range(max(0, max_pieces - len(start_state))):
            if budget_exhausted():
                break
            next_frontier: dict[frozenset[int], StateInfo] = {}
            for state_info in frontier:
                if budget_exhausted():
                    break
                expanded_states += 1
                pool: set[int] = set()
                for member in state_info.state:
                    pool.update(gap_graph[member])
                pool.difference_update(state_info.state)
                for neighbour in pool:
                    new_state = frozenset((*state_info.state, neighbour))
                    if new_state in seen:
                        continue
                    neighbour_label = fragments[neighbour].label
                    if (
                        neighbour_label
                        and state_info.labels
                        and neighbour_label not in state_info.labels
                    ):
                        continue
                    evidence = score_fragment_against_assembly(
                        fragments,
                        state_info.state,
                        state_info.union,
                        neighbour,
                        gap_lookup,
                        boundary_evidence,
                        tolerance=tolerance,
                        min_score=min_group_gap_score,
                        automatic_score=automatic_group_gap_score,
                    )
                    if evidence.evidence_level == "insufficient-evidence":
                        continue
                    new_area_sum = state_info.area_sum + fragments[neighbour].area
                    new_union = state_info.union | fragments[neighbour].mask
                    overlap = new_area_sum - int(new_union.sum())
                    if overlap > max_group_overlap_pixels:
                        continue
                    new_labels = set(state_info.labels)
                    if neighbour_label:
                        new_labels.add(neighbour_label)
                    raw_coverage = int(new_union.sum()) / float(total_area)
                    coverage = (
                        int(_dilate(new_union, gap_fill_radius).sum())
                        / float(total_area)
                        if raw_coverage >= coverage_threshold - 0.15
                        else raw_coverage
                    )
                    evidence_score = min(state_info.evidence_score, evidence.score)
                    evidence_level = (
                        "automatic"
                        if state_info.evidence_level == "automatic"
                        and evidence.evidence_level == "automatic"
                        else "review"
                    )
                    gap_steps = state_info.gap_steps + 1
                    support = state_info.support + evidence.matched_pixels
                    base_score = (
                        coverage * 10_000.0
                        + support
                        + 100.0 * len(new_labels)
                        + 250.0 * evidence_score
                        - 25.0 * gap_steps
                    )
                    info = StateInfo(
                        state=new_state,
                        union=new_union,
                        area_sum=new_area_sum,
                        labels=new_labels,
                        support=support,
                        score=base_score,
                        raw_coverage=raw_coverage,
                        coverage=coverage,
                        evidence_score=evidence_score,
                        evidence_level=evidence_level,
                        gap_steps=gap_steps,
                    )
                    existing = next_frontier.get(new_state)
                    if existing is None or info.score > existing.score:
                        next_frontier[new_state] = info
                    if coverage >= coverage_threshold:
                        ids = tuple(sorted(fragments[index].id for index in new_state))
                        candidate = AssemblyCandidate(
                            fragment_ids=ids,
                            coverage=coverage,
                            raw_coverage=raw_coverage,
                            score=base_score,
                            support_pixels=support,
                            labels=tuple(sorted(new_labels)),
                            base_score=base_score,
                            evidence_score=evidence_score,
                            evidence_level=evidence_level,
                            gap_steps=gap_steps,
                        )
                        previous = output.get(ids)
                        if previous is None or candidate.score > previous.score:
                            output[ids] = candidate
                seen.add(state_info.state)
            if not next_frontier:
                break
            frontier = sorted(
                next_frontier.values(),
                key=lambda item: (
                    -item.score,
                    len(item.state),
                    tuple(sorted(item.state)),
                ),
            )[:beam_width]
    if search_stats is not None:
        search_stats.update(
            expanded_states=expanded_states,
            state_limit_reached=state_limit_reached,
            time_limit_reached=time_limit_reached,
        )
    return sorted(output.values(), key=lambda item: (-item.score, item.fragment_ids))


def select_exact_cover_candidates(
    candidates: list[AssemblyCandidate],
    *,
    time_limit_seconds: float | None = 10.0,
    objective: str = "score_then_count",
    forbidden_candidates: Iterable[Iterable[str]] | None = None,
    locked_candidates: Iterable[Iterable[str]] | None = None,
    preferred_candidates: Iterable[Iterable[str]] | None = None,
    preferred_candidate_bonus: float = 50_000.0,
    max_search_nodes: int | None = None,
    search_stats: dict[str, int | bool] | None = None,
) -> list[AssemblyCandidate]:
    """Globally choose disjoint confirmed candidates.

    This is a maximum set-packing solver over generated full-note candidates:
    no fragment may appear twice and no serial label may be confirmed twice.
    ``score_then_count`` is the weighted set-packing variant: it maximises
    total score first and uses count as the tie-breaker. ``count_then_score``
    maximises confirmed note count first and uses total candidate score as a
    tie-breaker.
    """

    if objective not in TEARFIT_COVER_OBJECTIVES:
        raise ValueError(
            f"objective must be one of: {', '.join(TEARFIT_COVER_OBJECTIVES)}"
        )
    if max_search_nodes is not None and max_search_nodes < 1:
        raise ValueError("max_search_nodes must be positive")
    forbidden_keys = _normalise_candidate_keys(forbidden_candidates)
    locked_keys = _normalise_candidate_keys(locked_candidates)
    preferred_keys = _normalise_candidate_keys(preferred_candidates)

    filtered = [
        candidate
        for candidate in candidates
        if _candidate_key(candidate.fragment_ids) not in forbidden_keys
    ]

    def adjusted_score(item: AssemblyCandidate) -> float:
        score = item.score
        if _candidate_key(item.fragment_ids) in preferred_keys:
            score += preferred_candidate_bonus
        return score

    ordered = sorted(
        filtered,
        key=lambda item: (
            -adjusted_score(item),
            -len(item.fragment_ids),
            item.fragment_ids,
        ),
    )
    locked: list[AssemblyCandidate] = []
    remaining: list[AssemblyCandidate] = []
    used_locked_ids: set[str] = set()
    used_locked_labels: set[str] = set()
    locked_score = 0.0
    available_locked_keys = {
        _candidate_key(candidate.fragment_ids) for candidate in ordered
    }
    missing_locked = locked_keys - available_locked_keys
    if missing_locked:
        raise ValueError(f"locked candidates are unavailable: {sorted(missing_locked)}")
    for candidate in ordered:
        key = _candidate_key(candidate.fragment_ids)
        if key in locked_keys:
            item_ids = set(candidate.fragment_ids)
            item_labels = set(candidate.labels)
            if item_ids & used_locked_ids or item_labels & used_locked_labels:
                raise ValueError(
                    "locked candidates conflict by fragment id or serial label"
                )
            locked.append(candidate)
            used_locked_ids.update(item_ids)
            used_locked_labels.update(item_labels)
            locked_score += adjusted_score(candidate)
        else:
            remaining.append(candidate)

    # Pre-compute optimistic suffix scores for score_then_count DFS pruning
    suffix_score = [0.0] * (len(remaining) + 1)
    for i in range(len(remaining) - 1, -1, -1):
        suffix_score[i] = suffix_score[i + 1] + max(0.0, adjusted_score(remaining[i]))

    deadline = None if time_limit_seconds is None else monotonic() + time_limit_seconds
    locked_count = len(locked)
    best_count = locked_count
    best_score = locked_score
    best_choice: tuple[int, ...] = ()

    def better(count: int, score: float) -> bool:
        if objective == "count_then_score":
            return count > best_count or (count == best_count and score > best_score)
        return score > best_score or (score == best_score and count > best_count)

    # Iterative branch-and-bound. The recursive form recursed to depth equal to
    # the candidate count, so large pools (N=200+) overflowed Python's recursion
    # limit and crashed mid-search instead of returning the best packing found so
    # far. An explicit LIFO stack keeps the identical include-first DFS order and
    # both pruning bounds, with no depth limit. Per-candidate frozensets/scores
    # are precomputed once instead of rebuilt at every visited node.
    remaining_count = len(remaining)
    item_ids_list = [frozenset(item.fragment_ids) for item in remaining]
    item_labels_list = [frozenset(item.labels) for item in remaining]
    item_scores = [adjusted_score(item) for item in remaining]

    stack: list[tuple[int, frozenset[str], frozenset[str], tuple[int, ...], float]] = [
        (0, frozenset(used_locked_ids), frozenset(used_locked_labels), (), locked_score)
    ]
    search_nodes = 0
    node_limit_reached = False
    time_limit_reached = False
    while stack:
        if max_search_nodes is not None and search_nodes >= max_search_nodes:
            node_limit_reached = True
            break
        if deadline is not None and monotonic() >= deadline:
            time_limit_reached = True
            break
        pos, used_ids, used_labels, chosen, score = stack.pop()
        search_nodes += 1
        if (
            objective == "count_then_score"
            and locked_count + len(chosen) + (remaining_count - pos) < best_count
        ):
            continue
        if objective == "score_then_count" and score + suffix_score[pos] < best_score:
            continue
        if pos >= remaining_count:
            count = locked_count + len(chosen)
            if better(count, score):
                best_count = count
                best_score = score
                best_choice = chosen
            continue

        # Push skip-branch first so the include-branch is popped first, matching
        # the recursive preorder (a good packing found early tightens the bound).
        stack.append((pos + 1, used_ids, used_labels, chosen, score))
        item_ids = item_ids_list[pos]
        item_labels = item_labels_list[pos]
        if not (item_ids & used_ids) and not (item_labels & used_labels):
            stack.append(
                (
                    pos + 1,
                    used_ids | item_ids,
                    used_labels | item_labels,
                    chosen + (pos,),
                    score + item_scores[pos],
                )
            )
    selected_out = locked + [remaining[index] for index in best_choice]
    final_out = []
    for candidate in selected_out:
        key = _candidate_key(candidate.fragment_ids)
        if key in preferred_keys:
            candidate = AssemblyCandidate(
                fragment_ids=candidate.fragment_ids,
                coverage=candidate.coverage,
                raw_coverage=candidate.raw_coverage,
                score=candidate.score + preferred_candidate_bonus,
                support_pixels=candidate.support_pixels,
                labels=candidate.labels,
                base_score=candidate.base_score,
                constraint_bonus=candidate.constraint_bonus + preferred_candidate_bonus,
                evidence_score=candidate.evidence_score,
                evidence_level=candidate.evidence_level,
                gap_steps=candidate.gap_steps,
            )
        final_out.append(candidate)
    if search_stats is not None:
        search_stats.update(
            search_nodes=search_nodes,
            node_limit_reached=node_limit_reached,
            time_limit_reached=time_limit_reached,
        )
    return final_out


def diagnose_confirmed_candidates(
    candidates: list[AssemblyCandidate], fragments: list[Fragment]
) -> TearFitDiagnostics:
    lookup = {fragment.id: fragment for fragment in fragments}
    true_notes: dict[str, set[str]] = {}
    for fragment in fragments:
        note_id = fragment.meta.get("note_id")
        if note_id:
            true_notes.setdefault(note_id, set()).add(fragment.id)

    exact = 0
    pure = 0
    chimeras = 0
    automatic_exact = 0
    automatic_exact_notes: set[str] = set()
    for candidate in candidates:
        ids = set(candidate.fragment_ids)
        notes = {
            lookup[fid].meta.get("note_id")
            for fid in ids
            if fid in lookup and lookup[fid].meta.get("note_id")
        }
        if len(notes) > 1:
            chimeras += 1
            continue
        if len(notes) == 1:
            pure += 1
            note_id = next(iter(notes))
            if ids == true_notes.get(note_id, set()):
                exact += 1
                if candidate.evidence_level == "automatic":
                    automatic_exact += 1
                    automatic_exact_notes.add(note_id)
    confirmed = len(candidates)
    true_count = len(true_notes)
    automatic_candidates = sum(
        candidate.evidence_level == "automatic" for candidate in candidates
    )
    review_candidates = sum(
        candidate.evidence_level == "review" for candidate in candidates
    )
    return TearFitDiagnostics(
        confirmed=confirmed,
        exact_confirmed=exact,
        pure_confirmed=pure,
        chimeras=chimeras,
        true_notes=true_count,
        exact_yield=exact / true_count if true_count else 0.0,
        exact_precision=exact / confirmed if confirmed else 0.0,
        pure_precision=pure / confirmed if confirmed else 0.0,
        manual_notes_remaining=true_count - len(automatic_exact_notes),
        confirmed_candidates=tuple(candidates),
        automatic_candidates=automatic_candidates,
        review_candidates=review_candidates,
        automatic_exact_confirmed=automatic_exact,
        automatic_exact_yield=automatic_exact / true_count if true_count else 0.0,
        automatic_exact_precision=(
            automatic_exact / automatic_candidates if automatic_candidates else 0.0
        ),
    )


def run_tearfit_trial(
    config: FractalTearConfig,
    *,
    algorithm: str = "baseline",
    route_fragment_fraction_threshold: float = TEARFIT_V43_FINE_FRACTION,
    tolerance: int = 2,
    min_overlap_pixels: int = 14,
    min_effectiveness: float = 1.0,
    automatic_effectiveness: float = 2.0,
    min_contiguous_pixels: int = 3,
    automatic_contiguous_pixels: int = 5,
    core_min_pieces: int = 2,
    min_group_gap_score: float = 0.35,
    automatic_group_gap_score: float = 0.55,
    coverage_threshold: float = 0.93,
    core_raw_coverage_threshold: float | None = None,
    gap_fill_radius: int = 2,
    max_pieces: int | None = None,
    beam_width: int = 64,
    max_partial_core_candidates: int = 128,
    use_labels: bool = True,
    seed_strategy: str = "anchor_priority",
    require_anchor: bool | None = None,
    serial_ocr_rate: float | None = None,
    candidate_time_limit_seconds: float | None = 20.0,
    candidate_state_limit: int | None = None,
    partial_gap_time_limit_seconds: float | None = 5.0,
    gap_state_limit: int | None = None,
    partial_gap_state_limit: int | None = None,
    cover_time_limit_seconds: float | None = 10.0,
    cover_node_limit: int | None = None,
    cover_objective: str = "score_then_count",
) -> TearFitTrialResult:
    """Run one labelled exact-cover tear-fit trial."""

    if algorithm not in TEARFIT_ALGORITHMS:
        raise ValueError(f"algorithm must be one of: {', '.join(TEARFIT_ALGORITHMS)}")
    if not (0.0 < route_fragment_fraction_threshold < 1.0):
        raise ValueError("route_fragment_fraction_threshold must be in (0, 1)")
    if core_raw_coverage_threshold is None:
        core_raw_coverage_threshold = max(0.05, coverage_threshold - 0.15)
    if not (0.0 < core_raw_coverage_threshold < coverage_threshold):
        raise ValueError(
            "core_raw_coverage_threshold must be in (0, coverage_threshold)"
        )
    if max_partial_core_candidates < 1:
        raise ValueError("max_partial_core_candidates must be positive")
    seed_strategy = _resolve_seed_strategy(seed_strategy, require_anchor)
    if serial_ocr_rate is not None:
        config = FractalTearConfig(
            **{**config.__dict__, "serial_ocr_rate": serial_ocr_rate}
        )
    _template, fragments = make_fractal_tear_fragments(config)
    median_fragment_fraction = float(
        np.median([fragment.area for fragment in fragments])
    ) / float(fragments[0].mask.size)
    resolved_algorithm = algorithm
    if algorithm == "v43_routed":
        resolved_algorithm = (
            "baseline"
            if median_fragment_fraction >= route_fragment_fraction_threshold
            else "effectiveness_gap"
        )
    scoring = "overlap" if resolved_algorithm == "baseline" else "effectiveness"
    all_scores, raw_edges = score_absolute_tear_pairs(
        fragments,
        tolerance=tolerance,
        min_overlap_pixels=min_overlap_pixels,
        use_labels=False,
        scoring=scoring,
        min_effectiveness=min_effectiveness,
        automatic_effectiveness=automatic_effectiveness,
        min_contiguous_pixels=min_contiguous_pixels,
        automatic_contiguous_pixels=automatic_contiguous_pixels,
    )
    if use_labels:
        label_filtered_scores, edges = score_absolute_tear_pairs(
            fragments,
            tolerance=tolerance,
            min_overlap_pixels=min_overlap_pixels,
            use_labels=True,
            scoring=scoring,
            min_effectiveness=min_effectiveness,
            automatic_effectiveness=automatic_effectiveness,
            min_contiguous_pixels=min_contiguous_pixels,
            automatic_contiguous_pixels=automatic_contiguous_pixels,
        )
    else:
        label_filtered_scores = all_scores
        edges = raw_edges
    core_edges = (
        edges
        if resolved_algorithm == "baseline"
        else [edge for edge in edges if edge.evidence_level == "automatic"]
    )
    core_search_stats: dict[str, int | bool] = {}
    generated_candidates = generate_assembly_candidates(
        fragments,
        core_edges,
        coverage_threshold=coverage_threshold,
        minimum_candidate_raw_coverage=(
            core_raw_coverage_threshold
            if resolved_algorithm == "effectiveness_gap"
            else None
        ),
        gap_fill_radius=gap_fill_radius,
        max_pieces=max_pieces or config.pieces_per_note + 2,
        beam_width=beam_width,
        seed_strategy=seed_strategy,
        time_limit_seconds=candidate_time_limit_seconds,
        max_expanded_states=candidate_state_limit,
        search_stats=core_search_stats,
    )
    complete_core_candidates = [
        candidate
        for candidate in generated_candidates
        if candidate.coverage >= coverage_threshold
    ]
    partial_core_candidates = [
        candidate
        for candidate in generated_candidates
        if candidate.coverage < coverage_threshold
    ]
    candidates = complete_core_candidates
    gap_candidate_count = 0
    partial_gap_keys: set[tuple[str, ...]] = set()
    complete_gap_search_stats: dict[str, int | bool] = {
        "expanded_states": 0,
        "state_limit_reached": False,
        "time_limit_reached": False,
    }
    partial_gap_search_stats = dict(complete_gap_search_stats)
    if resolved_algorithm == "effectiveness_gap":
        from_complete = augment_candidates_with_group_gap(
            fragments,
            complete_core_candidates,
            label_filtered_scores,
            tolerance=tolerance,
            coverage_threshold=coverage_threshold,
            gap_fill_radius=gap_fill_radius,
            max_pieces=max_pieces or config.pieces_per_note + 2,
            core_min_pieces=core_min_pieces,
            min_group_gap_score=min_group_gap_score,
            automatic_group_gap_score=automatic_group_gap_score,
            time_limit_seconds=candidate_time_limit_seconds,
            max_expanded_states=gap_state_limit,
            search_stats=complete_gap_search_stats,
        )
        from_partial = augment_candidates_with_group_gap(
            fragments,
            partial_core_candidates,
            label_filtered_scores,
            tolerance=tolerance,
            coverage_threshold=coverage_threshold,
            gap_fill_radius=gap_fill_radius,
            max_pieces=max_pieces or config.pieces_per_note + 2,
            core_min_pieces=core_min_pieces,
            max_base_candidates=max_partial_core_candidates,
            min_group_gap_score=min_group_gap_score,
            automatic_group_gap_score=automatic_group_gap_score,
            time_limit_seconds=partial_gap_time_limit_seconds,
            max_expanded_states=partial_gap_state_limit,
            search_stats=partial_gap_search_stats,
        )
        merged: dict[tuple[str, ...], AssemblyCandidate] = {}
        for candidate in (*from_complete, *from_partial):
            key = _candidate_key(candidate.fragment_ids)
            previous = merged.get(key)
            if previous is None or candidate.score > previous.score:
                merged[key] = candidate
        candidates = sorted(
            merged.values(), key=lambda item: (-item.score, item.fragment_ids)
        )
        complete_keys = {
            _candidate_key(candidate.fragment_ids)
            for candidate in complete_core_candidates
        }
        from_complete_keys = {
            _candidate_key(candidate.fragment_ids) for candidate in from_complete
        }
        from_partial_keys = {
            _candidate_key(candidate.fragment_ids) for candidate in from_partial
        }
        gap_candidate_count = sum(key not in complete_keys for key in merged)
        partial_gap_keys = from_partial_keys - from_complete_keys - complete_keys
    cover_search_stats: dict[str, int | bool] = {}
    selected = select_exact_cover_candidates(
        candidates,
        time_limit_seconds=cover_time_limit_seconds,
        max_search_nodes=cover_node_limit,
        search_stats=cover_search_stats,
        objective=cover_objective,
    )
    diagnostics = diagnose_confirmed_candidates(selected, fragments)

    true_scores = [
        edge.overlap_pixels if resolved_algorithm == "baseline" else edge.effectiveness
        for edge in all_scores
        if fragments[edge.left].meta.get("note_id")
        == fragments[edge.right].meta.get("note_id")
    ]
    false_scores = [
        edge.overlap_pixels if resolved_algorithm == "baseline" else edge.effectiveness
        for edge in all_scores
        if fragments[edge.left].meta.get("note_id")
        != fragments[edge.right].meta.get("note_id")
    ]
    false_edges = [
        edge
        for edge in core_edges
        if fragments[edge.left].meta.get("note_id")
        != fragments[edge.right].meta.get("note_id")
    ]
    edge_decisions = {
        level: sum(edge.evidence_level == level for edge in label_filtered_scores)
        for level in TEARFIT_EVIDENCE_LEVELS
    }
    candidate_decisions = {
        level: sum(candidate.evidence_level == level for candidate in candidates)
        for level in ("automatic", "review")
    }
    return TearFitTrialResult(
        config={
            **config.__dict__,
            "algorithm": algorithm,
            "resolved_algorithm": resolved_algorithm,
            "median_fragment_fraction": median_fragment_fraction,
            "route_fragment_fraction_threshold": route_fragment_fraction_threshold,
            "tolerance": tolerance,
            "min_overlap_pixels": min_overlap_pixels,
            "min_effectiveness": min_effectiveness,
            "automatic_effectiveness": automatic_effectiveness,
            "min_contiguous_pixels": min_contiguous_pixels,
            "automatic_contiguous_pixels": automatic_contiguous_pixels,
            "core_min_pieces": core_min_pieces,
            "min_group_gap_score": min_group_gap_score,
            "automatic_group_gap_score": automatic_group_gap_score,
            "coverage_threshold": coverage_threshold,
            "core_raw_coverage_threshold": core_raw_coverage_threshold,
            "gap_fill_radius": gap_fill_radius,
            "beam_width": beam_width,
            "max_partial_core_candidates": max_partial_core_candidates,
            "use_labels": use_labels,
            "seed_strategy": seed_strategy,
            "candidate_time_limit_seconds": candidate_time_limit_seconds,
            "candidate_state_limit": candidate_state_limit,
            "partial_gap_time_limit_seconds": partial_gap_time_limit_seconds,
            "gap_state_limit": gap_state_limit,
            "partial_gap_state_limit": partial_gap_state_limit,
            "cover_time_limit_seconds": cover_time_limit_seconds,
            "cover_node_limit": cover_node_limit,
            "cover_objective": cover_objective,
        },
        fragments=len(fragments),
        pair_scores=len(all_scores),
        accepted_edges=len(core_edges),
        false_edge_rate=len(false_edges) / len(core_edges) if core_edges else 0.0,
        true_edge_median=float(np.median(true_scores)) if true_scores else 0.0,
        false_edge_median=float(np.median(false_scores)) if false_scores else 0.0,
        candidates=len(candidates),
        diagnostics=diagnostics,
        edge_decisions=edge_decisions,
        candidate_decisions=candidate_decisions,
        search_stats={
            "core": core_search_stats,
            "complete_gap": complete_gap_search_stats,
            "partial_gap": partial_gap_search_stats,
            "exact_cover": cover_search_stats,
        },
        core_candidates=len(generated_candidates),
        partial_core_candidates=len(partial_core_candidates),
        gap_candidates=gap_candidate_count,
        partial_gap_candidates=len(partial_gap_keys),
        selected_partial_gap_candidates=sum(
            _candidate_key(candidate.fragment_ids) in partial_gap_keys
            for candidate in selected
        ),
    )


def run_tearfit_sweep(
    notes_list: Iterable[int],
    *,
    pieces_per_note: int = 8,
    width: int = 180,
    height: int = 90,
    seed: int = 7,
    min_overlap_pixels: int = 14,
    tolerance: int = 2,
    coverage_threshold: float = 0.93,
    gap_fill_radius: int = 2,
    beam_width: int = 64,
    serial_ocr_rate: float = 0.6,
    seed_strategy: str = "anchor_priority",
    require_anchor: bool | None = None,
    ensure_serial_anchor: bool = False,
    candidate_time_limit_seconds: float | None = 20.0,
    cover_time_limit_seconds: float | None = 10.0,
    cover_objective: str = "score_then_count",
) -> list[dict]:
    seed_strategy = _resolve_seed_strategy(seed_strategy, require_anchor)
    rows = []
    for offset, notes in enumerate(notes_list):
        result = run_tearfit_trial(
            FractalTearConfig(
                notes=int(notes),
                pieces_per_note=pieces_per_note,
                width=width,
                height=height,
                seed=seed + offset * 997,
                serial_ocr_rate=serial_ocr_rate,
                ensure_serial_anchor=ensure_serial_anchor,
            ),
            min_overlap_pixels=min_overlap_pixels,
            tolerance=tolerance,
            coverage_threshold=coverage_threshold,
            gap_fill_radius=gap_fill_radius,
            beam_width=beam_width,
            seed_strategy=seed_strategy,
            candidate_time_limit_seconds=candidate_time_limit_seconds,
            cover_time_limit_seconds=cover_time_limit_seconds,
            cover_objective=cover_objective,
        )
        rows.append(result.to_jsonable())
    return rows


def tearfit_comparison_cases(
    profile: str = "smoke",
) -> tuple[TearFitComparisonCase, ...]:
    if profile == "smoke":
        return (
            TearFitComparisonCase("small_n8_p5", notes=8, pieces_per_note=5),
            TearFitComparisonCase("base_n20_p8", notes=20, pieces_per_note=8),
        )
    if profile == "pressure":
        return (
            TearFitComparisonCase("base_n20_p8", notes=20, pieces_per_note=8),
            TearFitComparisonCase("scale_n50_p8", notes=50, pieces_per_note=8),
            TearFitComparisonCase("scale_n100_p8", notes=100, pieces_per_note=8),
            TearFitComparisonCase("fine_n50_p16", notes=50, pieces_per_note=16),
            TearFitComparisonCase(
                "fray_n50_p8", notes=50, pieces_per_note=8, fray_probability=0.40
            ),
        )
    raise ValueError("profile must be 'smoke' or 'pressure'")


def _strategy_score(rows: list[dict]) -> tuple[float, float, float, float]:
    if not rows:
        return (0.0, 0.0, 0.0, 0.0)
    precisions = [float(row["exact_precision"]) for row in rows]
    yields = [float(row["exact_yield"]) for row in rows]
    chimera_rates = [
        float(row["chimeras"]) / float(row["confirmed"]) if row["confirmed"] else 0.0
        for row in rows
    ]
    return (
        min(precisions),
        sum(precisions) / len(precisions),
        sum(yields) / len(yields),
        -sum(chimera_rates) / len(chimera_rates),
    )


def run_tearfit_strategy_comparison(
    *,
    profile: str = "smoke",
    seed_strategies: Iterable[str] = ("anchor_only", "anchor_priority", "all"),
    cover_objectives: Iterable[str] = ("count_then_score",),
    serial_ocr_rates: Iterable[float] = (0.0, 0.6, 1.0),
    width: int = 120,
    height: int = 64,
    seed: int = 7,
    min_overlap_pixels: int = 10,
    tolerance: int = 2,
    coverage_threshold: float = 0.93,
    gap_fill_radius: int = 2,
    beam_width: int = 48,
    ensure_serial_anchor: bool = False,
    candidate_time_limit_seconds: float | None = 10.0,
    cover_time_limit_seconds: float | None = 5.0,
) -> dict:
    rows: list[dict] = []
    cases = tearfit_comparison_cases(profile)
    strategies = tuple(seed_strategies)
    objectives = tuple(cover_objectives)
    rates = tuple(float(rate) for rate in serial_ocr_rates)
    for strategy in strategies:
        _resolve_seed_strategy(strategy, None)
    for objective in objectives:
        if objective not in TEARFIT_COVER_OBJECTIVES:
            raise ValueError(
                f"cover_objective must be one of: {', '.join(TEARFIT_COVER_OBJECTIVES)}"
            )
    for case_index, case in enumerate(cases):
        for rate_index, rate in enumerate(rates):
            for strategy in strategies:
                for objective in objectives:
                    result = run_tearfit_trial(
                        FractalTearConfig(
                            notes=case.notes,
                            pieces_per_note=case.pieces_per_note,
                            width=width,
                            height=height,
                            seed=seed + case_index * 1009 + rate_index * 131,
                            roughness=case.roughness,
                            fray_probability=case.fray_probability,
                            ensure_serial_anchor=ensure_serial_anchor,
                            serial_ocr_rate=rate,
                        ),
                        tolerance=tolerance,
                        min_overlap_pixels=min_overlap_pixels,
                        coverage_threshold=coverage_threshold,
                        gap_fill_radius=gap_fill_radius,
                        beam_width=beam_width,
                        seed_strategy=strategy,
                        candidate_time_limit_seconds=candidate_time_limit_seconds,
                        cover_time_limit_seconds=cover_time_limit_seconds,
                        cover_objective=objective,
                    )
                    diag = result.diagnostics
                    rows.append(
                        {
                            "case": case.name,
                            "notes": case.notes,
                            "pieces_per_note": case.pieces_per_note,
                            "fray_probability": case.fray_probability,
                            "serial_ocr_rate": rate,
                            "seed_strategy": strategy,
                            "cover_objective": objective,
                            "fragments": result.fragments,
                            "accepted_edges": result.accepted_edges,
                            "false_edge_rate": result.false_edge_rate,
                            "candidates": result.candidates,
                            "confirmed": diag.confirmed,
                            "exact_confirmed": diag.exact_confirmed,
                            "chimeras": diag.chimeras,
                            "exact_precision": diag.exact_precision,
                            "pure_precision": diag.pure_precision,
                            "exact_yield": diag.exact_yield,
                            "manual_notes_remaining": diag.manual_notes_remaining,
                        }
                    )

    by_strategy: dict[tuple[str, str], list[dict]] = {
        (strategy, objective): [] for strategy in strategies for objective in objectives
    }
    for row in rows:
        by_strategy[(row["seed_strategy"], row["cover_objective"])].append(row)
    summary = []
    for (strategy, objective), strategy_rows in by_strategy.items():
        score = _strategy_score(strategy_rows)
        summary.append(
            {
                "seed_strategy": strategy,
                "cover_objective": objective,
                "min_exact_precision": score[0],
                "mean_exact_precision": score[1],
                "mean_exact_yield": score[2],
                "mean_negative_chimera_rate": score[3],
                "score_tuple": score,
            }
        )
    summary.sort(key=lambda item: item["score_tuple"], reverse=True)
    best = (
        {
            "seed_strategy": summary[0]["seed_strategy"],
            "cover_objective": summary[0]["cover_objective"],
        }
        if summary
        else None
    )
    for item in summary:
        item.pop("score_tuple", None)
    return {
        "config": {
            "profile": profile,
            "seed_strategies": strategies,
            "cover_objectives": objectives,
            "serial_ocr_rates": rates,
            "width": width,
            "height": height,
            "seed": seed,
            "min_overlap_pixels": min_overlap_pixels,
            "tolerance": tolerance,
            "coverage_threshold": coverage_threshold,
            "gap_fill_radius": gap_fill_radius,
            "beam_width": beam_width,
            "ensure_serial_anchor": ensure_serial_anchor,
            "candidate_time_limit_seconds": candidate_time_limit_seconds,
            "cover_time_limit_seconds": cover_time_limit_seconds,
        },
        "rows": rows,
        "summary": summary,
        "best_strategy": best,
        "best_seed_strategy": best,
    }


def run_tearfit_v43_ablation(
    *,
    notes: int = 10,
    pieces_list: Iterable[int] = (8, 16, 24),
    seeds: Iterable[int] = (7,),
    algorithms: Iterable[str] = TEARFIT_ALGORITHMS,
    route_fragment_fraction_threshold: float = TEARFIT_V43_FINE_FRACTION,
    width: int = 180,
    height: int = 90,
    tolerance: int = 2,
    min_overlap_pixels: int = 14,
    min_effectiveness: float = 1.0,
    automatic_effectiveness: float = 2.0,
    min_contiguous_pixels: int = 3,
    automatic_contiguous_pixels: int = 5,
    coverage_threshold: float = 0.93,
    core_raw_coverage_threshold: float | None = None,
    gap_fill_radius: int = 2,
    beam_width: int = 32,
    max_partial_core_candidates: int = 128,
    candidate_time_limit_seconds: float | None = 15.0,
    candidate_state_limit: int | None = 100_000,
    partial_gap_time_limit_seconds: float | None = 5.0,
    gap_state_limit: int | None = 20_000,
    partial_gap_state_limit: int | None = 5_000,
    cover_time_limit_seconds: float | None = 5.0,
    cover_node_limit: int | None = 250_000,
) -> dict:
    """Run same-seed baseline -> Etear -> group-gap head-to-head trials."""

    algorithms = tuple(algorithms)
    pieces_values = tuple(int(value) for value in pieces_list)
    seed_values = tuple(int(value) for value in seeds)
    for algorithm in algorithms:
        if algorithm not in TEARFIT_ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of: {', '.join(TEARFIT_ALGORITHMS)}"
            )
    rows: list[dict] = []
    for pieces in pieces_values:
        for seed in seed_values:
            cached_trials: dict[str, tuple[TearFitTrialResult, float, str]] = {}
            median_fragment_fraction: float | None = None
            for algorithm in algorithms:
                reused_from_algorithm: str | None = None
                cached: tuple[TearFitTrialResult, float, str] | None = None
                if algorithm == "v43_routed" and median_fragment_fraction is not None:
                    route = (
                        "baseline"
                        if median_fragment_fraction >= route_fragment_fraction_threshold
                        else "effectiveness_gap"
                    )
                    cached = cached_trials.get(route)
                if cached is not None:
                    result, elapsed_seconds, reused_from_algorithm = cached
                else:
                    started = monotonic()
                    result = run_tearfit_trial(
                        FractalTearConfig(
                            notes=notes,
                            pieces_per_note=pieces,
                            width=width,
                            height=height,
                            seed=seed,
                            serial_ocr_rate=0.0,
                        ),
                        algorithm=algorithm,
                        route_fragment_fraction_threshold=route_fragment_fraction_threshold,
                        tolerance=tolerance,
                        min_overlap_pixels=min_overlap_pixels,
                        min_effectiveness=min_effectiveness,
                        automatic_effectiveness=automatic_effectiveness,
                        min_contiguous_pixels=min_contiguous_pixels,
                        automatic_contiguous_pixels=automatic_contiguous_pixels,
                        coverage_threshold=coverage_threshold,
                        core_raw_coverage_threshold=core_raw_coverage_threshold,
                        gap_fill_radius=gap_fill_radius,
                        beam_width=beam_width,
                        max_partial_core_candidates=max_partial_core_candidates,
                        use_labels=False,
                        candidate_time_limit_seconds=candidate_time_limit_seconds,
                        candidate_state_limit=candidate_state_limit,
                        partial_gap_time_limit_seconds=partial_gap_time_limit_seconds,
                        gap_state_limit=gap_state_limit,
                        partial_gap_state_limit=partial_gap_state_limit,
                        cover_time_limit_seconds=cover_time_limit_seconds,
                        cover_node_limit=cover_node_limit,
                    )
                    elapsed_seconds = monotonic() - started
                    resolved = str(result.config["resolved_algorithm"])
                    cached_trials.setdefault(
                        resolved, (result, elapsed_seconds, algorithm)
                    )
                    median_fragment_fraction = float(
                        result.config["median_fragment_fraction"]
                    )
                diagnostics = result.diagnostics
                rows.append(
                    {
                        "algorithm": algorithm,
                        "resolved_algorithm": result.config["resolved_algorithm"],
                        "notes": notes,
                        "pieces_per_note": pieces,
                        "seed": seed,
                        "elapsed_seconds": elapsed_seconds,
                        "reused_from_algorithm": reused_from_algorithm,
                        "fragments": result.fragments,
                        "accepted_edges": result.accepted_edges,
                        "false_edge_rate": result.false_edge_rate,
                        "edge_decisions": result.edge_decisions,
                        "candidates": result.candidates,
                        "core_candidates": result.core_candidates,
                        "partial_core_candidates": result.partial_core_candidates,
                        "gap_candidates": result.gap_candidates,
                        "partial_gap_candidates": result.partial_gap_candidates,
                        "selected_partial_gap_candidates": result.selected_partial_gap_candidates,
                        "candidate_decisions": result.candidate_decisions,
                        "search_stats": result.search_stats,
                        "confirmed": diagnostics.confirmed,
                        "exact_confirmed": diagnostics.exact_confirmed,
                        "exact_yield": diagnostics.exact_yield,
                        "exact_precision": diagnostics.exact_precision,
                        "automatic_exact_yield": diagnostics.automatic_exact_yield,
                        "automatic_exact_precision": diagnostics.automatic_exact_precision,
                        "pure_precision": diagnostics.pure_precision,
                        "manual_notes_remaining": diagnostics.manual_notes_remaining,
                        "automatic_candidates": diagnostics.automatic_candidates,
                        "review_candidates": diagnostics.review_candidates,
                    }
                )

    summary: list[dict] = []
    for pieces in pieces_values:
        for algorithm in algorithms:
            selected = [
                row
                for row in rows
                if row["pieces_per_note"] == pieces and row["algorithm"] == algorithm
            ]
            if not selected:
                continue
            summary.append(
                {
                    "algorithm": algorithm,
                    "pieces_per_note": pieces,
                    "runs": len(selected),
                    "mean_exact_yield": float(
                        np.mean([row["exact_yield"] for row in selected])
                    ),
                    "mean_exact_precision": float(
                        np.mean([row["exact_precision"] for row in selected])
                    ),
                    "min_exact_precision": float(
                        np.min([row["exact_precision"] for row in selected])
                    ),
                    "mean_automatic_exact_yield": float(
                        np.mean([row["automatic_exact_yield"] for row in selected])
                    ),
                    "mean_automatic_exact_precision": float(
                        np.mean([row["automatic_exact_precision"] for row in selected])
                    ),
                    "mean_pure_precision": float(
                        np.mean([row["pure_precision"] for row in selected])
                    ),
                    "mean_false_edge_rate": float(
                        np.mean([row["false_edge_rate"] for row in selected])
                    ),
                    "mean_core_candidates": float(
                        np.mean([row["core_candidates"] for row in selected])
                    ),
                    "mean_partial_core_candidates": float(
                        np.mean([row["partial_core_candidates"] for row in selected])
                    ),
                    "mean_gap_candidates": float(
                        np.mean([row["gap_candidates"] for row in selected])
                    ),
                    "mean_partial_gap_candidates": float(
                        np.mean([row["partial_gap_candidates"] for row in selected])
                    ),
                    "mean_selected_partial_gap_candidates": float(
                        np.mean(
                            [row["selected_partial_gap_candidates"] for row in selected]
                        )
                    ),
                    "mean_manual_notes_remaining": float(
                        np.mean([row["manual_notes_remaining"] for row in selected])
                    ),
                    "mean_elapsed_seconds": float(
                        np.mean([row["elapsed_seconds"] for row in selected])
                    ),
                }
            )

    baseline_by_pieces = {
        row["pieces_per_note"]: row for row in summary if row["algorithm"] == "baseline"
    }
    for row in summary:
        baseline = baseline_by_pieces.get(row["pieces_per_note"])
        if baseline is None:
            continue
        row["delta_exact_yield_vs_baseline"] = (
            row["mean_exact_yield"] - baseline["mean_exact_yield"]
        )
        row["delta_exact_precision_vs_baseline"] = (
            row["mean_exact_precision"] - baseline["mean_exact_precision"]
        )
        row["delta_automatic_exact_yield_vs_baseline"] = (
            row["mean_automatic_exact_yield"] - baseline["mean_automatic_exact_yield"]
        )
        row["delta_automatic_exact_precision_vs_baseline"] = (
            row["mean_automatic_exact_precision"]
            - baseline["mean_automatic_exact_precision"]
        )
        row["delta_false_edge_rate_vs_baseline"] = (
            row["mean_false_edge_rate"] - baseline["mean_false_edge_rate"]
        )
        row["delta_manual_notes_vs_baseline"] = (
            row["mean_manual_notes_remaining"] - baseline["mean_manual_notes_remaining"]
        )
    return {
        "config": {
            "notes": notes,
            "pieces_list": pieces_values,
            "seeds": seed_values,
            "algorithms": algorithms,
            "route_fragment_fraction_threshold": route_fragment_fraction_threshold,
            "width": width,
            "height": height,
            "tolerance": tolerance,
            "min_overlap_pixels": min_overlap_pixels,
            "min_effectiveness": min_effectiveness,
            "automatic_effectiveness": automatic_effectiveness,
            "min_contiguous_pixels": min_contiguous_pixels,
            "automatic_contiguous_pixels": automatic_contiguous_pixels,
            "coverage_threshold": coverage_threshold,
            "core_raw_coverage_threshold": (
                max(0.05, coverage_threshold - 0.15)
                if core_raw_coverage_threshold is None
                else core_raw_coverage_threshold
            ),
            "gap_fill_radius": gap_fill_radius,
            "beam_width": beam_width,
            "max_partial_core_candidates": max_partial_core_candidates,
            "candidate_time_limit_seconds": candidate_time_limit_seconds,
            "candidate_state_limit": candidate_state_limit,
            "partial_gap_time_limit_seconds": partial_gap_time_limit_seconds,
            "gap_state_limit": gap_state_limit,
            "partial_gap_state_limit": partial_gap_state_limit,
            "cover_time_limit_seconds": cover_time_limit_seconds,
            "cover_node_limit": cover_node_limit,
            "serial_ocr_rate": 0.0,
            "use_labels": False,
        },
        "rows": rows,
        "summary": summary,
    }


def _parse_notes_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the placed tear-fit research sandbox"
    )
    parser.add_argument("--notes-list", default="20,50,100")
    parser.add_argument("--pieces-per-note", type=int, default=8)
    parser.add_argument("--width", type=int, default=180)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-overlap-pixels", type=int, default=14)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--coverage-threshold", type=float, default=0.93)
    parser.add_argument("--gap-fill-radius", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--serial-ocr-rate", type=float, default=0.6)
    parser.add_argument(
        "--seed-strategy", choices=TEARFIT_SEED_STRATEGIES, default="anchor_priority"
    )
    parser.add_argument("--ensure-serial-anchor", action="store_true")
    parser.add_argument("--ideal-serial-upper-bound", action="store_true")
    parser.add_argument("--candidate-time-limit", type=float, default=20.0)
    parser.add_argument("--cover-time-limit", type=float, default=10.0)
    parser.add_argument(
        "--cover-objective",
        choices=TEARFIT_COVER_OBJECTIVES,
        default="score_then_count",
    )
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    rows = run_tearfit_sweep(
        _parse_notes_list(args.notes_list),
        pieces_per_note=args.pieces_per_note,
        width=args.width,
        height=args.height,
        seed=args.seed,
        min_overlap_pixels=args.min_overlap_pixels,
        tolerance=args.tolerance,
        coverage_threshold=args.coverage_threshold,
        gap_fill_radius=args.gap_fill_radius,
        beam_width=args.beam_width,
        serial_ocr_rate=1.0 if args.ideal_serial_upper_bound else args.serial_ocr_rate,
        seed_strategy=args.seed_strategy,
        ensure_serial_anchor=True
        if args.ideal_serial_upper_bound
        else args.ensure_serial_anchor,
        candidate_time_limit_seconds=args.candidate_time_limit,
        cover_time_limit_seconds=args.cover_time_limit,
        cover_objective=args.cover_objective,
    )
    payload = {"rows": rows}
    text = json.dumps(payload, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
