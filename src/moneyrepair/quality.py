from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from moneyrepair.types import Fragment


@dataclass(frozen=True)
class QualityThresholds:
    """Acquisition contract thresholds for a production capture batch.

    A frame is accepted only when every metric stays on the good side of its
    threshold. Defaults are tuned for clear flatbed scans / steady phone photos
    of banknote fragments on a plain background.
    """

    min_focus: float = 60.0
    max_glare: float = 0.06
    min_segmentation: float = 0.55
    max_color_drift: float = 26.0

    def to_dict(self) -> dict:
        return {
            "min_focus": self.min_focus,
            "max_glare": self.max_glare,
            "min_segmentation": self.min_segmentation,
            "max_color_drift": self.max_color_drift,
        }


@dataclass(frozen=True)
class FrameQuality:
    """Per-frame acquisition quality metrics and accept/reject reasons."""

    source: str
    focus: float
    glare: float
    segmentation_confidence: float
    color_drift: float
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "focus": self.focus,
            "glare": self.glare,
            "segmentation_confidence": self.segmentation_confidence,
            "color_drift": self.color_drift,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def _to_gray(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _interior_mask(mask: np.ndarray) -> np.ndarray:
    """Mask pixels whose full 3x3 neighbourhood is also foreground.

    Used to keep boundary pixels (fragment edge against background) out of the
    focus estimate so a sharp cut edge is not mistaken for sharp content.
    """

    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return np.zeros_like(mask, dtype=bool)
    interior = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(mask)
            ys = slice(max(0, dy), mask.shape[0] + min(0, dy))
            xs = slice(max(0, dx), mask.shape[1] + min(0, dx))
            sy = slice(max(0, -dy), mask.shape[0] + min(0, -dy))
            sx = slice(max(0, -dx), mask.shape[1] + min(0, -dx))
            shifted[ys, xs] = mask[sy, sx]
            interior &= shifted
    return interior


def focus_score(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance of the Laplacian over interior content. Higher is sharper."""

    gray = _to_gray(image)
    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    if mask is None:
        region = np.ones(gray.shape, dtype=bool)
    else:
        region = _interior_mask(mask.astype(bool))
    if region.sum() < 16:
        return 0.0
    return float(np.var(lap[region]))


def glare_fraction(image: np.ndarray, mask: np.ndarray | None = None, clip: int = 250) -> float:
    """Fraction of foreground pixels with clipped (blown-out) luminance."""

    gray = _to_gray(image)
    region = np.ones(gray.shape, dtype=bool) if mask is None else mask.astype(bool)
    total = int(region.sum())
    if total == 0:
        return 0.0
    return float(np.count_nonzero(gray[region] >= clip) / total)


def segmentation_confidence(mask: np.ndarray) -> float:
    """Solidity of a mask: foreground area over its row/column convex fill.

    Ragged, holey, or speckled segmentations score low; a clean blob scores
    near 1.0. Fully numpy, no SciPy dependency.
    """

    mask = mask.astype(bool)
    area = int(mask.sum())
    if area == 0:
        return 0.0
    left = np.maximum.accumulate(mask, axis=1)
    right = np.maximum.accumulate(mask[:, ::-1], axis=1)[:, ::-1]
    row_span = left & right
    top = np.maximum.accumulate(mask, axis=0)
    bottom = np.maximum.accumulate(mask[::-1], axis=0)[::-1]
    col_span = top & bottom
    filled = mask | (row_span & col_span)
    return float(area / int(filled.sum()))


def color_drift(image: np.ndarray, mask: np.ndarray | None = None, reference: np.ndarray | None = None) -> float:
    """Per-channel colour drift.

    With a reference note image, drift is the largest per-channel mean
    difference over the shared foreground pixels. Without a reference, drift is
    the white-balance cast estimated from the brightest foreground pixels.
    """

    rgb = image.astype(np.float32)
    region = np.ones(rgb.shape[:2], dtype=bool) if mask is None else mask.astype(bool)
    if int(region.sum()) == 0:
        return 0.0

    if reference is not None:
        ref = reference.astype(np.float32)
        if ref.shape[:2] != rgb.shape[:2]:
            raise ValueError("reference must share the fragment image height/width")
        diff = np.abs(rgb[region].mean(axis=0) - ref[region].mean(axis=0))
        return float(diff.max())

    gray = _to_gray(image)[region]
    pixels = rgb[region]
    threshold = np.quantile(gray, 0.85)
    bright = pixels[gray >= threshold]
    if bright.shape[0] < 8:
        bright = pixels
    channel_means = bright.mean(axis=0)
    return float(np.abs(channel_means - channel_means.mean()).max())


def assess_frame(
    image: np.ndarray | None,
    mask: np.ndarray | None = None,
    *,
    source: str = "",
    thresholds: QualityThresholds | None = None,
    reference: np.ndarray | None = None,
) -> FrameQuality:
    """Score one captured frame/fragment against the acquisition contract."""

    thresholds = thresholds or QualityThresholds()
    if image is None:
        focus = 0.0
        glare = 0.0
        drift = 0.0
    else:
        focus = focus_score(image, mask)
        glare = glare_fraction(image, mask)
        drift = color_drift(image, mask, reference)
    seg = segmentation_confidence(mask) if mask is not None else 1.0

    reasons: list[str] = []
    if image is None:
        reasons.append("missing_image")
    else:
        if focus < thresholds.min_focus:
            reasons.append("blurry")
        if glare > thresholds.max_glare:
            reasons.append("glare")
        if drift > thresholds.max_color_drift:
            reasons.append("color_drift")
    if seg < thresholds.min_segmentation:
        reasons.append("weak_segmentation")

    return FrameQuality(
        source=source,
        focus=focus,
        glare=glare,
        segmentation_confidence=seg,
        color_drift=drift,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def assess_fragments(
    fragments: Iterable[Fragment],
    *,
    thresholds: QualityThresholds | None = None,
    reference: np.ndarray | None = None,
) -> list[FrameQuality]:
    thresholds = thresholds or QualityThresholds()
    reports: list[FrameQuality] = []
    for fragment in fragments:
        reports.append(
            assess_frame(
                fragment.image,
                fragment.mask,
                source=fragment.id,
                thresholds=thresholds,
                reference=reference,
            )
        )
    return reports


def summarize_quality(reports: list[FrameQuality], thresholds: QualityThresholds | None = None) -> dict:
    """Aggregate per-frame reports into a batch acceptance summary."""

    thresholds = thresholds or QualityThresholds()
    total = len(reports)
    accepted = [report for report in reports if report.passed]
    rejected = [report for report in reports if not report.passed]
    reason_counts: dict[str, int] = {}
    for report in rejected:
        for reason in report.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "thresholds": thresholds.to_dict(),
        "frames": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": (len(accepted) / total) if total else 0.0,
        "reason_counts": reason_counts,
        "rejected_sources": [report.source for report in rejected],
        "mean_metrics": {
            "focus": _mean([report.focus for report in reports]),
            "glare": _mean([report.glare for report in reports]),
            "segmentation_confidence": _mean([report.segmentation_confidence for report in reports]),
            "color_drift": _mean([report.color_drift for report in reports]),
        },
    }
