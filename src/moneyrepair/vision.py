from __future__ import annotations

import numpy as np


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    eroded = center.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        eroded &= padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
    return mask & ~eroded


def contour_points(mask: np.ndarray) -> np.ndarray:
    """Return boundary points ordered by angle around the fragment centroid."""

    ys, xs = np.nonzero(boundary_mask(mask))
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)
    points = np.column_stack([xs, ys]).astype(np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def resample_curve(points: np.ndarray, samples: int = 64) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points.astype(np.float32), samples, axis=0)

    points = points.astype(np.float32)
    closed = np.vstack([points, points[0]])
    steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(steps)])
    if distance[-1] == 0:
        return np.repeat(points[:1], samples, axis=0)
    targets = np.linspace(0, distance[-1], samples, endpoint=False)
    xs = np.interp(targets, distance, closed[:, 0])
    ys = np.interp(targets, distance, closed[:, 1])
    return np.column_stack([xs, ys]).astype(np.float32)


def tag_contour(points: np.ndarray, canvas_shape: tuple[int, int], margin: int = 3) -> tuple[str, ...]:
    """Assign coarse tags useful for filtering curve comparisons."""

    if len(points) == 0:
        return ("center",)
    height, width = canvas_shape
    x = points[:, 0]
    y = points[:, 1]
    near_left = np.mean(x <= margin) > 0.08
    near_right = np.mean(x >= width - 1 - margin) > 0.08
    near_top = np.mean(y <= margin) > 0.08
    near_bottom = np.mean(y >= height - 1 - margin) > 0.08
    edge_count = sum((near_left, near_right, near_top, near_bottom))

    tags: list[str] = []
    if edge_count >= 2:
        tags.append("corner")
    if edge_count >= 1:
        tags.append("edge")
    if not tags:
        tags.append("center")
    if near_left:
        tags.append("left")
    if near_right:
        tags.append("right")
    if near_top:
        tags.append("top")
    if near_bottom:
        tags.append("bottom")
    return tuple(tags)


def direction_histogram(points: np.ndarray, bins: int = 8, samples: int = 64) -> np.ndarray:
    """Histogram of local contour directions."""

    curve = resample_curve(points, samples=samples)
    if len(curve) < 2:
        return np.zeros(bins, dtype=np.float32)
    closed = np.vstack([curve, curve[0]])
    deltas = np.diff(closed, axis=0)
    angles = (np.arctan2(deltas[:, 1], deltas[:, 0]) + 2 * np.pi) % (2 * np.pi)
    hist, _ = np.histogram(angles, bins=bins, range=(0, 2 * np.pi))
    total = hist.sum()
    if total == 0:
        return hist.astype(np.float32)
    return (hist / total).astype(np.float32)


def apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to Nx2 points."""

    points = np.asarray(points, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (2, 3):
        raise ValueError("affine matrix must have shape (2, 3)")
    hom = np.column_stack([points, np.ones(len(points), dtype=np.float32)])
    return (hom @ matrix.T).astype(np.float32)


def estimate_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares affine transform from corresponding 2D points."""

    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) < 3:
        raise ValueError("src and dst must be matching Nx2 arrays with at least 3 points")
    design = np.column_stack([src[:, 0], src[:, 1], np.ones(len(src), dtype=np.float32)])
    x_params, *_ = np.linalg.lstsq(design, dst[:, 0], rcond=None)
    y_params, *_ = np.linalg.lstsq(design, dst[:, 1], rcond=None)
    return np.vstack([x_params, y_params]).astype(np.float32)


def normalized_curve(points: np.ndarray, samples: int = 64) -> np.ndarray:
    curve = resample_curve(points, samples=samples)
    if len(curve) == 0:
        return curve
    curve = curve - curve.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(curve**2, axis=1))))
    if scale > 0:
        curve = curve / scale
    return curve.astype(np.float32)


def curve_distance(left: np.ndarray, right: np.ndarray, samples: int = 64) -> float:
    """Rotation/reversal tolerant distance between two contours."""

    a = normalized_curve(left, samples=samples)
    b = normalized_curve(right, samples=samples)
    if len(a) == 0 or len(b) == 0:
        return float("inf")

    za = a[:, 0] + 1j * a[:, 1]
    base = [b, b[::-1]]
    best = float("inf")
    for candidate in base:
        zb0 = candidate[:, 0] + 1j * candidate[:, 1]
        for shift in range(samples):
            zb = np.roll(zb0, shift)
            denom = np.vdot(zb, zb)
            rotation = 1.0 if denom == 0 else np.vdot(zb, za) / denom
            aligned = zb * rotation
            best = min(best, float(np.sqrt(np.mean(np.abs(za - aligned) ** 2))))
    return best
