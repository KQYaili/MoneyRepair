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
