import numpy as np

from moneyrepair.vision import (
    apply_affine,
    contour_points,
    curve_distance,
    direction_histogram,
    estimate_affine,
    tag_contour,
)


def test_contour_tags_edge_corner_and_center():
    corner = np.zeros((20, 30), dtype=bool)
    corner[:5, :5] = True
    tags = tag_contour(contour_points(corner), corner.shape, margin=1)
    assert "corner" in tags
    assert "edge" in tags

    center = np.zeros((20, 30), dtype=bool)
    center[8:12, 12:16] = True
    assert tag_contour(contour_points(center), center.shape) == ("center",)


def test_curve_distance_handles_rotation():
    line = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
    rotated = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.float32)
    zigzag = np.array([[0, 0], [1, 1], [2, 0], [3, 1]], dtype=np.float32)

    assert curve_distance(line, rotated, samples=16) < curve_distance(line, zigzag, samples=16)


def test_affine_estimation_roundtrip():
    src = np.array([[0, 0], [1, 0], [0, 1], [2, 3]], dtype=np.float32)
    matrix = np.array([[2, 0.5, 3], [-0.25, 1.5, 4]], dtype=np.float32)
    dst = apply_affine(src, matrix)

    estimated = estimate_affine(src, dst)
    np.testing.assert_allclose(apply_affine(src, estimated), dst, atol=1e-5)


def test_direction_histogram_is_normalized():
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
    hist = direction_histogram(square, bins=4, samples=16)
    np.testing.assert_allclose(hist.sum(), 1.0)
