from __future__ import annotations

from collections import Counter
from math import pi, sqrt
from typing import Iterable

import numpy as np

from moneyrepair.solver import CoverageSolution
from moneyrepair.types import Fragment


def diagnose_pair_retrieval(
    fragments: list[Fragment],
    pair_scores: Iterable[tuple[int, int, float]],
    *,
    truth_key: str = "note_id",
    top_ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Measure pair-score retrieval quality without feeding truth to a solver.

    The metric is intentionally post-hoc: ``truth_key`` is read only while
    evaluating a scored candidate graph. Missing positive candidates contribute
    zero to MRR and recall@K, so a spatial prefilter cannot hide retrieval loss.
    """

    if not truth_key:
        raise ValueError("truth_key must not be empty")
    requested_ks = tuple(sorted(set(top_ks)))
    if not requested_ks or any(value < 1 for value in requested_ks):
        raise ValueError("top_ks must contain positive integers")

    labels = [fragment.meta.get(truth_key) for fragment in fragments]
    label_counts = Counter(label for label in labels if label is not None)
    eligible = {
        index
        for index, label in enumerate(labels)
        if label is not None and label_counts[label] > 1
    }
    neighbours: list[dict[int, float]] = [dict() for _ in fragments]
    unique_pairs: dict[tuple[int, int], float] = {}
    for left, right, raw_score in pair_scores:
        if left == right:
            raise ValueError("pair scores must connect distinct fragments")
        if not (0 <= left < len(fragments) and 0 <= right < len(fragments)):
            raise IndexError("pair score references an unknown fragment")
        score = float(raw_score)
        if not np.isfinite(score):
            raise ValueError("pair scores must be finite")
        key = (min(left, right), max(left, right))
        previous = unique_pairs.get(key)
        if previous is None or score > previous:
            unique_pairs[key] = score

    for (left, right), score in unique_pairs.items():
        neighbours[left][right] = score
        neighbours[right][left] = score

    ranks: dict[int, int | None] = {}
    ordered_candidates: dict[int, list[int]] = {}
    queries_with_candidates = 0
    queries_with_positive_candidates = 0
    for index in sorted(eligible):
        ordered = sorted(
            neighbours[index],
            key=lambda other: (
                -neighbours[index][other],
                fragments[other].id,
                other,
            ),
        )
        ordered_candidates[index] = ordered
        if ordered:
            queries_with_candidates += 1
        first_positive = next(
            (
                rank
                for rank, other in enumerate(ordered, start=1)
                if labels[other] == labels[index]
            ),
            None,
        )
        ranks[index] = first_positive
        if first_positive is not None:
            queries_with_positive_candidates += 1

    eligible_queries = len(eligible)
    reciprocal_pairs: set[tuple[int, int]] = set()
    best = {
        index: ordered[0]
        for index, ordered in ordered_candidates.items()
        if ordered
    }
    for left, right in best.items():
        if best.get(right) == left:
            reciprocal_pairs.add((min(left, right), max(left, right)))
    true_reciprocal_pairs = {
        pair for pair in reciprocal_pairs if labels[pair[0]] == labels[pair[1]]
    }
    true_scored_pairs = {
        pair
        for pair in unique_pairs
        if labels[pair[0]] is not None and labels[pair[0]] == labels[pair[1]]
    }

    reciprocal_ranks = [1.0 / rank if rank is not None else 0.0 for rank in ranks.values()]
    found_reciprocal_ranks = [1.0 / rank for rank in ranks.values() if rank is not None]
    rank_histogram = {
        "1": sum(rank == 1 for rank in ranks.values()),
        "2": sum(rank == 2 for rank in ranks.values()),
        "3-5": sum(rank is not None and 3 <= rank <= 5 for rank in ranks.values()),
        "6-10": sum(rank is not None and 6 <= rank <= 10 for rank in ranks.values()),
        ">10": sum(rank is not None and rank > 10 for rank in ranks.values()),
        "missing": sum(rank is None for rank in ranks.values()),
    }
    return {
        "evaluation_truth_only": True,
        "truth_key": truth_key,
        "fragments": len(fragments),
        "eligible_queries": eligible_queries,
        "scored_pairs": len(unique_pairs),
        "same_truth_scored_pairs": len(true_scored_pairs),
        "queries_with_candidates": queries_with_candidates,
        "queries_with_positive_candidates": queries_with_positive_candidates,
        "candidate_graph_recall": (
            queries_with_positive_candidates / eligible_queries
            if eligible_queries
            else 0.0
        ),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "mrr_given_positive_candidate": (
            float(np.mean(found_reciprocal_ranks))
            if found_reciprocal_ranks
            else 0.0
        ),
        "recall_at_k": {
            str(k): (
                sum(rank is not None and rank <= k for rank in ranks.values())
                / eligible_queries
                if eligible_queries
                else 0.0
            )
            for k in requested_ks
        },
        "rank_histogram": rank_histogram,
        "reciprocal_best_buddy_pairs": len(reciprocal_pairs),
        "true_reciprocal_best_buddy_pairs": len(true_reciprocal_pairs),
        "reciprocal_best_buddy_precision": (
            len(true_reciprocal_pairs) / len(reciprocal_pairs)
            if reciprocal_pairs
            else 0.0
        ),
        "reciprocal_best_buddy_recall_over_scored_truth_pairs": (
            len(true_reciprocal_pairs) / len(true_scored_pairs)
            if true_scored_pairs
            else 0.0
        ),
    }


def _convex_hull(points: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(
        origin: tuple[int, int],
        left: tuple[int, int],
        right: tuple[int, int],
    ) -> int:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower: list[tuple[int, int]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[int, int]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _polygon_area(points: list[tuple[int, int]]) -> float:
    if len(points) < 3:
        return 0.0
    return 0.5 * abs(
        sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
        )
    )


def mask_geometry_features(mask: np.ndarray) -> dict[str, float]:
    """Return deterministic pixel-geometry descriptors for one fragment mask."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("fragment masks must be two-dimensional")
    area = int(np.count_nonzero(binary))
    if area == 0:
        raise ValueError("fragment masks must contain foreground pixels")

    ys, xs = np.nonzero(binary)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    padded = np.pad(binary, 1, constant_values=False)
    centre = padded[1:-1, 1:-1]
    perimeter = int(
        np.count_nonzero(centre & ~padded[:-2, 1:-1])
        + np.count_nonzero(centre & ~padded[2:, 1:-1])
        + np.count_nonzero(centre & ~padded[1:-1, :-2])
        + np.count_nonzero(centre & ~padded[1:-1, 2:])
    )
    cell_corners = (
        (int(x + dx), int(y + dy))
        for y, x in zip(ys, xs)
        for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))
    )
    hull = _convex_hull(cell_corners)
    hull_area = _polygon_area(hull)
    solidity = area / hull_area if hull_area > 0.0 else 1.0
    circularity = 4.0 * pi * area / float(perimeter * perimeter)
    compactness = 1.0 / circularity if circularity > 0.0 else 0.0
    return {
        "area_pixels": float(area),
        "area_fraction": area / float(binary.size),
        "perimeter_pixels": float(perimeter),
        "perimeter_over_sqrt_area": perimeter / sqrt(float(area)),
        "bbox_width_pixels": float(width),
        "bbox_height_pixels": float(height),
        "bbox_width_fraction": width / float(binary.shape[1]),
        "bbox_height_fraction": height / float(binary.shape[0]),
        "bbox_aspect_ratio": width / float(height),
        "bbox_fill_ratio": area / float(width * height),
        "circularity": circularity,
        "compactness": compactness,
        "convex_hull_area_pixels": hull_area,
        "convex_solidity": solidity,
        "concavity_ratio": max(0.0, 1.0 - solidity),
        "convex_hull_vertices": float(len(hull)),
    }


def summarize_fragment_geometry(fragments: Iterable[Fragment]) -> dict:
    """Summarize mask-shape distributions for simulator/real-data audits."""

    items = list(fragments)
    if not items:
        return {
            "measurement_only": True,
            "fragments": 0,
            "mask_shapes": {},
            "features": {},
        }
    rows = [mask_geometry_features(fragment.mask) for fragment in items]
    shape_counts = Counter(f"{fragment.mask.shape[0]}x{fragment.mask.shape[1]}" for fragment in items)
    feature_names = tuple(rows[0])
    summaries: dict[str, dict[str, float]] = {}
    for name in feature_names:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        summaries[name] = {
            "min": float(values.min()),
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return {
        "measurement_only": True,
        "coordinate_unit": "pixels; compare raw pixel features only at a common canonical scale",
        "fragments": len(items),
        "mask_shapes": dict(sorted(shape_counts.items())),
        "features": summaries,
    }


def solution_purity(solution: CoverageSolution, lookup: dict[str, Fragment]) -> dict:
    """Score one solution against the secret true ``note_id`` of its fragments."""

    note_ids = [lookup[fid].meta.get("note_id") for fid in solution.fragment_ids if fid in lookup]
    known = [note_id for note_id in note_ids if note_id]
    distinct = sorted(set(known))
    dominant: str | None = None
    dominant_fraction = 0.0
    if known:
        dominant, count = Counter(known).most_common(1)[0]
        dominant_fraction = count / len(known)
    return {
        "fragments": len(solution.fragment_ids),
        "distinct_notes": len(distinct),
        "is_chimera": len(distinct) > 1,
        "dominant_note": dominant,
        "dominant_fraction": dominant_fraction,
        "coverage": solution.coverage,
    }


def diagnose_solutions(solutions: list[CoverageSolution], fragments: list[Fragment]) -> dict:
    """Aggregate chimera and recovery statistics over a solution set.

    A chimera is a solution mixing fragments from more than one true note. A note
    is "exactly recovered" when some solution's fragment set equals that note's
    full set of fragments.
    """

    lookup = {fragment.id: fragment for fragment in fragments}
    per_solution = [solution_purity(solution, lookup) for solution in solutions]
    chimeras = sum(1 for item in per_solution if item["is_chimera"])
    pure = sum(1 for item in per_solution if not item["is_chimera"] and item["distinct_notes"] == 1)

    true_notes: dict[str, set[str]] = {}
    for fragment in fragments:
        note_id = fragment.meta.get("note_id")
        if note_id:
            true_notes.setdefault(note_id, set()).add(fragment.id)

    solution_sets = [frozenset(solution.fragment_ids) for solution in solutions]
    solution_set_counts = Counter(solution_sets)
    exactly_recovered = sorted(
        note_id for note_id, frag_set in true_notes.items() if frozenset(frag_set) in solution_sets
    )
    uniquely_exact_recovered = sorted(
        note_id
        for note_id, frag_set in true_notes.items()
        if solution_set_counts[frozenset(frag_set)] == 1
    )
    pure_notes_found = sorted(
        {item["dominant_note"] for item in per_solution if not item["is_chimera"] and item["dominant_note"]}
    )
    true_note_count = len(true_notes)

    return {
        "solutions": len(solutions),
        "chimeras": chimeras,
        "pure": pure,
        "chimera_rate": chimeras / len(solutions) if solutions else 0.0,
        "true_notes": true_note_count,
        "pure_notes_found": pure_notes_found,
        "pure_notes_found_count": len(pure_notes_found),
        "pure_notes_found_rate": len(pure_notes_found) / true_note_count if true_note_count else 0.0,
        "exactly_recovered_notes": exactly_recovered,
        "exactly_recovered_count": len(exactly_recovered),
        "exactly_recovered_rate": len(exactly_recovered) / true_note_count if true_note_count else 0.0,
        "uniquely_exact_recovered_notes": uniquely_exact_recovered,
        "uniquely_exact_recovered_count": len(uniquely_exact_recovered),
        "uniquely_exact_recovered_rate": len(uniquely_exact_recovered) / true_note_count if true_note_count else 0.0,
        "per_solution": per_solution,
    }


def diagnose_groups(fragments: list[Fragment], groups: dict[str, int]) -> dict:
    """Diagnose whether discrimination groups uniquely identify true notes.

    Unlike :func:`diagnose_solutions`, this does not depend on the DFS top-k
    output. It is the pressure-test metric for large pools where
    ``max_solutions`` can hide merged identities behind the first pure-looking
    candidates.
    """

    true_notes: dict[str, set[str]] = {}
    group_members: dict[int, set[str]] = {}
    group_notes: dict[int, set[str]] = {}
    note_groups: dict[str, set[int]] = {}
    for fragment in fragments:
        note_id = fragment.meta.get("note_id")
        group_id = groups.get(fragment.id)
        if note_id is not None:
            true_notes.setdefault(note_id, set()).add(fragment.id)
        if group_id is None:
            continue
        group_members.setdefault(group_id, set()).add(fragment.id)
        if note_id is not None:
            group_notes.setdefault(group_id, set()).add(note_id)
            note_groups.setdefault(note_id, set()).add(group_id)

    note_set_to_id = {frozenset(fragment_ids): note_id for note_id, fragment_ids in true_notes.items()}
    exact_recoverable_notes = sorted(
        note_set_to_id[frozenset(member_set)]
        for member_set in group_members.values()
        if frozenset(member_set) in note_set_to_id
    )
    mixed_groups = sorted(group_id for group_id, note_ids in group_notes.items() if len(note_ids) > 1)
    mixed_notes = sorted({note_id for group_id in mixed_groups for note_id in group_notes[group_id]})
    split_notes = sorted(note_id for note_id, group_ids in note_groups.items() if len(group_ids) > 1)
    true_note_count = len(true_notes)

    return {
        "groups": len(group_members),
        "true_notes": true_note_count,
        "cluster_deficit": true_note_count - len(group_members),
        "mixed_groups": mixed_groups,
        "mixed_group_count": len(mixed_groups),
        "mixed_notes": mixed_notes,
        "mixed_note_count": len(mixed_notes),
        "split_notes": split_notes,
        "split_note_count": len(split_notes),
        "exact_recoverable_notes": exact_recoverable_notes,
        "exact_recoverable_count": len(exact_recoverable_notes),
        "exact_recoverable_rate": len(exact_recoverable_notes) / true_note_count if true_note_count else 0.0,
    }
