from __future__ import annotations

from dataclasses import dataclass

from moneyrepair.types import Fragment
from moneyrepair.vision import contour_points, curve_distance, direction_histogram, tag_contour


@dataclass(frozen=True)
class ContourRecord:
    fragment_id: str
    tags: tuple[str, ...]
    boundary_points: int
    bbox: tuple[int, int, int, int]
    direction_histogram: tuple[float, ...]


def describe_contours(fragments: list[Fragment], direction_bins: int = 8) -> list[ContourRecord]:
    records: list[ContourRecord] = []
    for fragment in fragments:
        points = contour_points(fragment.mask)
        tags = tag_contour(points, fragment.mask.shape)
        histogram = direction_histogram(points, bins=direction_bins)
        records.append(
            ContourRecord(
                fragment_id=fragment.id,
                tags=tags,
                boundary_points=len(points),
                bbox=fragment.bbox,
                direction_histogram=tuple(float(value) for value in histogram),
            )
        )
    return records


def match_similar_contours(
    fragments: list[Fragment],
    max_distance: float = 0.25,
    limit: int = 100,
) -> list[dict]:
    """Pair fragments by coarse tags first, then by contour similarity."""

    points = [contour_points(fragment.mask) for fragment in fragments]
    tags = [set(tag_contour(item, fragments[index].mask.shape)) for index, item in enumerate(points)]
    primary_tags = [{"edge", "corner", "center"} & item for item in tags]
    matches: list[dict] = []

    for left in range(len(fragments)):
        for right in range(left + 1, len(fragments)):
            shared = primary_tags[left] & primary_tags[right]
            if not shared:
                continue
            distance = curve_distance(points[left], points[right])
            if distance <= max_distance:
                matches.append(
                    {
                        "left": fragments[left].id,
                        "right": fragments[right].id,
                        "distance": distance,
                        "shared_tags": sorted(shared),
                    }
                )

    matches.sort(key=lambda item: (item["distance"], item["left"], item["right"]))
    return matches[:limit]


def match_raw_crop_contours(
    fragments: list[Fragment],
    segment_length: int = 16,
    max_distance: float = 0.15,
    limit: int = 100,
) -> list[dict]:
    """Match raw-crop fragments by finding similar sub-segments on their boundaries.

    Translation- and rotation-invariant matching of local torn edges.
    """
    import numpy as np
    from moneyrepair.vision import contour_points, resample_curve

    curves = []
    for f in fragments:
        pts = contour_points(f.mask)
        if len(pts) < segment_length:
            curves.append(np.empty((0, 2), dtype=np.float32))
        else:
            curves.append(resample_curve(pts, samples=64))

    matches = []
    n = len(fragments)

    for i in range(n):
        c_i = curves[i]
        if len(c_i) == 0:
            continue
        c_i_complex = c_i[:, 0] + 1j * c_i[:, 1]
        c_i_wrapped = np.concatenate([c_i_complex, c_i_complex[:segment_length-1]])

        windows_i = []
        for start in range(len(c_i)):
            w = c_i_wrapped[start : start + segment_length]
            w = w - w.mean()
            norm = np.linalg.norm(w)
            w = w / norm if norm > 1e-9 else w
            windows_i.append(w)

        W_i = np.vstack(windows_i)

        for j in range(i + 1, n):
            c_j = curves[j]
            if len(c_j) == 0:
                continue

            c_j_complex = c_j[:, 0] + 1j * c_j[:, 1]
            best_dist = float("inf")
            best_match_info = {}

            for dir_idx, direction in enumerate((c_j_complex, c_j_complex[::-1])):
                is_reversed = (dir_idx == 1)
                windows_j = []
                c_j_wrapped = np.concatenate([direction, direction[:segment_length-1]])
                for start in range(len(c_j)):
                    w_j = c_j_wrapped[start : start + segment_length]
                    w_j = w_j - w_j.mean()
                    norm_j = np.linalg.norm(w_j)
                    w_j = w_j / norm_j if norm_j > 1e-9 else w_j
                    windows_j.append(w_j)

                W_j = np.vstack(windows_j)

                # Vectorized dot products: shape (len(c_i), len(c_j))
                dots = W_i @ W_j.conj().T
                abs_dots = np.abs(dots)
                max_idx = np.argmax(abs_dots)
                idx_i, idx_j = np.unravel_index(max_idx, abs_dots.shape)
                max_abs_dot = abs_dots[idx_i, idx_j]
                dist = np.sqrt(max(0.0, 2.0 - 2.0 * max_abs_dot))

                if dist < best_dist:
                    best_dist = dist
                    dot_val = dots[idx_i, idx_j]
                    theta = np.angle(dot_val)
                    
                    sub_A = c_i_wrapped[idx_i : idx_i + segment_length]
                    sub_B = c_j_wrapped[idx_j : idx_j + segment_length]
                    mean_A = sub_A.mean()
                    mean_B = sub_B.mean()
                    t = mean_A - mean_B * np.exp(1j * theta)

                    left_start = int(idx_i)
                    if is_reversed:
                        right_start = int((len(c_j) - 1 - idx_j) % len(c_j))
                    else:
                        right_start = int(idx_j)

                    best_match_info = {
                        "left_start": left_start,
                        "right_start": right_start,
                        "reversed": is_reversed,
                        "estimated_rotation": float(theta),
                        "estimated_translation": [float(t.real), float(t.imag)],
                    }

            if best_dist <= max_distance:
                matches.append(
                    {
                        "left": fragments[i].id,
                        "right": fragments[j].id,
                        "left_start": best_match_info["left_start"],
                        "right_start": best_match_info["right_start"],
                        "reversed": best_match_info["reversed"],
                        "estimated_rotation": best_match_info["estimated_rotation"],
                        "estimated_translation": best_match_info["estimated_translation"],
                        "distance": float(best_dist),
                    }
                )

    matches.sort(key=lambda item: (item["distance"], item["left"], item["right"]))
    return matches[:limit]
