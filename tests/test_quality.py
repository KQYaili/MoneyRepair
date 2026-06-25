import numpy as np

from moneyrepair.quality import (
    QualityThresholds,
    assess_fragments,
    focus_score,
    segmentation_confidence,
    summarize_quality,
)
from moneyrepair.types import Fragment


def _solid_rectangle_mask(size: int = 20, margin: int = 2) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[margin : size - margin, margin : size - margin] = True
    return mask


def _checker_image(mask: np.ndarray, low: int = 50, high: int = 200) -> np.ndarray:
    height, width = mask.shape
    yy, xx = np.mgrid[0:height, 0:width]
    checker = np.where((xx + yy) % 2 == 0, low, high).astype(np.uint8)
    image = np.repeat(checker[..., None], 3, axis=2)
    return np.where(mask[..., None], image, 0)


def _sharp_fragment() -> Fragment:
    mask = _solid_rectangle_mask()
    return Fragment(id="sharp", mask=mask, image=_checker_image(mask))


def _blurry_fragment() -> Fragment:
    mask = _solid_rectangle_mask()
    image = np.where(mask[..., None], np.full((*mask.shape, 3), 120, dtype=np.uint8), 0)
    return Fragment(id="blurry", mask=mask, image=image)


def test_focus_score_separates_sharp_from_flat():
    assert focus_score(_sharp_fragment().image, _sharp_fragment().mask) > 100.0
    assert focus_score(_blurry_fragment().image, _blurry_fragment().mask) < 1.0


def test_segmentation_confidence_rewards_solid_masks():
    solid = _solid_rectangle_mask()
    assert segmentation_confidence(solid) == 1.0

    speckled = solid.copy()
    speckled[5, 5] = False
    speckled[10, 12] = False
    assert segmentation_confidence(speckled) < 1.0


def test_assess_fragments_accepts_sharp_rejects_blurry():
    thresholds = QualityThresholds()
    reports = assess_fragments([_sharp_fragment(), _blurry_fragment()], thresholds=thresholds)
    by_id = {report.source: report for report in reports}

    assert by_id["sharp"].passed
    assert not by_id["blurry"].passed
    assert "blurry" in by_id["blurry"].reasons

    summary = summarize_quality(reports, thresholds)
    assert summary["frames"] == 2
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["reason_counts"]["blurry"] == 1
