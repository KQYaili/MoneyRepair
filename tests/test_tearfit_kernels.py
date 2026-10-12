"""The tear-fit core's fast paths must reproduce the plain computations exactly.

Each test pins one optimisation to a direct reference implementation, so a
later change cannot silently move the frozen core's scores or selections.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pytest

from moneyrepair.tearfit import (
    AssemblyCandidate,
    FractalTearConfig,
    StateInfo,
    _AssemblyGeometry,
    _bbox,
    _beam,
    _clip,
    _crop_overlap_count,
    _cropped_boundary_evidence,
    _dilate,
    _edge_graph,
    _largest_connected_support,
    _matched_normal_opposition,
    _NormalTable,
    _pack_crops,
    _packed_crop_counts,
    _packed_crops_meet,
    _packed_cross_counts,
    _packed_overlap_counts,
    _packed_point_counts,
    _supporting_edges,
    _uniform_histogram,
    make_fractal_tear_fragments,
    score_absolute_tear_pairs,
    select_exact_cover_candidates,
    tear_boundary_evidence,
)


def _fragments(notes: int = 4, pieces: int = 12, seed: int = 3):
    _template, fragments = make_fractal_tear_fragments(
        FractalTearConfig(notes=notes, pieces_per_note=pieces, seed=seed, serial_ocr_rate=0.0)
    )
    return fragments


def _same_float(left: float, right: float) -> bool:
    return left == right and math.copysign(1.0, left) == math.copysign(1.0, right)


def test_uniform_histogram_matches_numpy_including_bin_edges():
    rng = np.random.default_rng(0)
    edges = np.linspace(0.0, 2.0 * np.pi, 9).astype(np.float32)
    for dtype in (np.float32, np.float64):
        for _ in range(200):
            values = rng.uniform(-0.5, 2.0 * np.pi + 0.5, size=rng.integers(0, 60))
            values = np.concatenate([values, edges, np.nextafter(edges, 0.0)])
            values = values.astype(dtype)
            expected, _ = np.histogram(values, bins=8, range=(0.0, 2.0 * np.pi))
            np.testing.assert_array_equal(
                _uniform_histogram(values, 8, 0.0, 2.0 * np.pi), expected
            )


def _reference_largest_component(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    best = 0
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        queue = deque([(y, x)])
        size = 0
        while queue:
            cy, cx = queue.popleft()
            size += 1
            for ny in range(cy - 1, cy + 2):
                for nx in range(cx - 1, cx + 2):
                    inside = 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                    if inside and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        best = max(best, size)
    return best


def test_largest_connected_support_matches_flood_fill():
    rng = np.random.default_rng(1)
    for _ in range(100):
        mask = rng.random((rng.integers(1, 30), rng.integers(1, 30))) < rng.uniform(0.05, 0.6)
        assert _largest_connected_support(mask) == _reference_largest_component(mask)
    assert _largest_connected_support(np.zeros((4, 5), dtype=bool)) == 0


def _packed(masks):
    boxes = [_bbox(mask) for mask in masks]
    crops = [(y0, x0, mask[y0:y1, x0:x1]) for mask, (x0, y0, x1, y1) in zip(masks, boxes)]
    return boxes, _pack_crops(crops)


def test_packed_kernels_match_numpy():
    rng = np.random.default_rng(2)
    shape = (20, 30)
    for _ in range(20):
        masks = [rng.random(shape) < rng.uniform(0.05, 0.5) for _ in range(6)]
        corner = np.zeros(shape, dtype=bool)
        corner[: rng.integers(1, 5), : rng.integers(1, 5)] = True
        masks += [np.zeros(shape, dtype=bool), corner]
        boxes, (offsets, shapes, origins, buffer) = _packed(masks)
        everyone = np.arange(len(masks), dtype=np.int64)
        for index, mask in enumerate(masks):
            expected = [int(np.count_nonzero(mask & other)) for other in masks]
            counts = np.empty(len(masks), dtype=np.int64)
            _packed_overlap_counts(index, everyone, offsets, shapes, origins, buffer, counts)
            assert counts.tolist() == expected
            meets = np.empty(len(masks), dtype=np.bool_)
            _packed_crops_meet(index, everyone, offsets, shapes, origins, buffer, meets)
            assert meets.tolist() == [value > 0 for value in expected]
        canvas = rng.random(shape) < 0.5
        expected = [int(np.count_nonzero(canvas & mask)) for mask in masks]
        counts = np.empty(len(masks), dtype=np.int64)
        _packed_crop_counts(canvas, everyone, offsets, shapes, origins, buffer, counts)
        assert counts.tolist() == expected
        for mask, (x0, y0, x1, y1), count in zip(masks, boxes, expected):
            crop = np.ascontiguousarray(mask[y0:y1, x0:x1])
            assert _crop_overlap_count(canvas, crop, y0, x0) == count
        points = [np.flatnonzero(mask) for mask in masks]
        lengths = np.array([len(item) for item in points], dtype=np.int64)
        point_offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        flat = np.concatenate(points).astype(np.int64)
        counts = np.empty(len(masks), dtype=np.int64)
        _packed_point_counts(canvas.reshape(-1), everyone, point_offsets, lengths, flat, counts)
        assert counts.tolist() == expected


def test_packed_cross_counts_match_numpy():
    rng = np.random.default_rng(3)
    shape = (24, 24)
    for _ in range(20):
        first = [rng.random(shape) < 0.15 for _ in range(5)]
        second = [_dilate(mask, 1) for mask in first]
        boxes = [_bbox(mask) for mask in second]
        offsets, shapes, origins, first_buffer = _pack_crops(
            [(y0, x0, mask[y0:y1, x0:x1]) for mask, (x0, y0, x1, y1) in zip(first, boxes)]
        )
        second_buffer = _pack_crops(
            [(y0, x0, mask[y0:y1, x0:x1]) for mask, (x0, y0, x1, y1) in zip(second, boxes)]
        )[3]
        everyone = np.arange(len(first), dtype=np.int64)
        for index in range(len(first)):
            forward = np.empty(len(first), dtype=np.int64)
            backward = np.empty(len(first), dtype=np.int64)
            _packed_cross_counts(
                index, everyone, offsets, shapes, origins, first_buffer, second_buffer, forward, backward
            )
            assert forward.tolist() == [
                int(np.count_nonzero(first[index] & other)) for other in second
            ]
            assert backward.tolist() == [
                int(np.count_nonzero(other & second[index])) for other in first
            ]


def test_scalar_clip_matches_numpy_clip():
    values = [0.0, -0.0, 0.3, -0.2, 1.2, 1.5, 0.25, float("inf"), float("-inf"), float("nan")]
    for low, high in ((0.0, 1.0), (0.25, 1.5)):
        for value in values:
            expected = float(np.clip(value, low, high))
            got = _clip(value, low, high)
            if math.isnan(expected):
                assert math.isnan(got)
            else:
                assert _same_float(got, expected)


def _reference_normal_opposition(left, right, left_match, *, tolerance, sample_limit=128):
    """The original per-pixel scan over full-canvas evidence."""

    points = np.argwhere(left_match)
    if len(points) == 0:
        return 0.0
    if len(points) > sample_limit:
        points = points[np.linspace(0, len(points) - 1, sample_limit, dtype=np.int64)]
    height, width = left_match.shape
    scores = []
    for raw_y, raw_x in points:
        y, x = int(raw_y), int(raw_x)
        left_ny = float(left.normal_y[y, x])
        left_nx = float(left.normal_x[y, x])
        if left_ny == 0.0 and left_nx == 0.0:
            continue
        y0, y1 = max(0, y - tolerance), min(height, y + tolerance + 1)
        x0, x1 = max(0, x - tolerance), min(width, x + tolerance + 1)
        candidate_scores = []
        for local_y, local_x in np.argwhere(right.boundary_mask[y0:y1, x0:x1]):
            right_ny = float(right.normal_y[y0 + local_y, x0 + local_x])
            right_nx = float(right.normal_x[y0 + local_y, x0 + local_x])
            if right_ny == 0.0 and right_nx == 0.0:
                continue
            candidate_scores.append(-(left_ny * right_ny + left_nx * right_nx))
        if candidate_scores:
            scores.append(float(np.clip(max(candidate_scores), 0.0, 1.0)))
    return float(np.mean(scores)) if scores else 0.0


def test_normal_opposition_table_matches_pixel_scan():
    fragments = _fragments()
    tolerance = 2
    full = tear_boundary_evidence(fragments, tolerance=tolerance)
    cropped = _cropped_boundary_evidence(fragments, tolerance=tolerance)
    table = _NormalTable(cropped)
    compared = 0
    for i in range(len(fragments)):
        for j in range(len(fragments)):
            if i == j:
                continue
            left_match = full[i].boundary_mask & full[j].dilated_boundary_mask
            if not left_match.any():
                continue
            ys, xs = np.nonzero(left_match)
            window = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            x0, y0, x1, y1 = window
            fast = _matched_normal_opposition(
                table,
                i,
                j,
                cropped,
                left_match[y0:y1, x0:x1],
                (y0, x0),
                tolerance=tolerance,
            )
            slow = _reference_normal_opposition(
                full[i], full[j], left_match, tolerance=tolerance
            )
            assert _same_float(fast, slow), (i, j, fast, slow)
            compared += 1
    assert compared > 20


def test_beam_matches_full_sort_with_ties():
    rng = np.random.default_rng(4)
    for width in (1, 3, 8, 50):
        states = []
        seen: set[frozenset[int]] = set()
        while len(states) < 40:
            members = frozenset(rng.choice(30, size=rng.integers(1, 5), replace=False).tolist())
            if members in seen:
                continue
            seen.add(members)
            score = float(rng.choice([0.0, -0.0, 1.5, 2.5, float(rng.random())]))
            states.append(
                StateInfo(
                    state=members,
                    union=np.zeros((2, 2), dtype=bool),
                    area_sum=0,
                    labels=set(),
                    support=0,
                    score=score,
                    raw_coverage=0.0,
                    coverage=0.0,
                )
            )
        expected = sorted(
            states,
            key=lambda item: (-item.score, len(item.state), tuple(sorted(item.state))),
        )[:width]
        assert [item.state for item in _beam(states, width)] == [
            item.state for item in expected
        ]


def test_incremental_dilated_union_matches_direct_dilation():
    fragments = _fragments()
    geometry = _AssemblyGeometry(fragments, tolerance=2, gap_fill_radius=2)
    root = StateInfo(
        state=frozenset({0}),
        union=fragments[0].mask.copy(),
        area_sum=geometry.areas[0],
        labels=set(),
        support=0,
        score=0.0,
        raw_coverage=0.0,
        coverage=0.0,
    )
    state = root
    for index in (1, 2, 5, 7):
        assert geometry.dilated_union_pixels(state, index) == int(
            np.count_nonzero(_dilate(state.union | fragments[index].mask, 2))
        )
        state = StateInfo(
            state=state.state | {index},
            union=None,
            area_sum=0,
            labels=set(),
            support=0,
            score=0.0,
            raw_coverage=0.0,
            coverage=0.0,
            union_count=int(np.count_nonzero(state.union | fragments[index].mask)),
            parent=state,
            added_index=index,
            added_mask=fragments[index].mask,
        )
        dilated, count = geometry.dilated_union(state)
        np.testing.assert_array_equal(dilated, _dilate(state.union, 2))
        assert count == int(np.count_nonzero(dilated))
        others = [index for index in range(len(fragments)) if index not in state.state]
        assert geometry.dilated_union_counts(state, others) == [
            int(np.count_nonzero(_dilate(state.union | fragments[index].mask, 2)))
            for index in others
        ]


def test_overlap_bits_match_pairwise_masks():
    fragments = _fragments(notes=3)
    geometry = _AssemblyGeometry(fragments, tolerance=2, gap_fill_radius=2)
    for index, fragment in enumerate(fragments):
        expected = sum(
            1 << other
            for other, candidate in enumerate(fragments)
            if other != index and np.any(fragment.mask & candidate.mask)
        )
        assert geometry.overlap_bits(index) == expected


def _reference_cover(candidates, *, objective, max_search_nodes):
    """The original frozenset branch-and-bound, unlimited by time."""

    ordered = sorted(candidates, key=lambda item: (-item.score, -len(item.fragment_ids), item.fragment_ids))
    suffix = [0.0] * (len(ordered) + 1)
    for index in range(len(ordered) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + max(0.0, ordered[index].score)
    best_count, best_score, best_choice = 0, 0.0, ()
    stack = [(0, frozenset(), frozenset(), (), 0.0)]
    nodes = 0
    limited = False
    while stack:
        if max_search_nodes is not None and nodes >= max_search_nodes:
            limited = True
            break
        pos, used_ids, used_labels, chosen, score = stack.pop()
        nodes += 1
        if objective == "count_then_score" and len(chosen) + (len(ordered) - pos) < best_count:
            continue
        if objective == "score_then_count" and score + suffix[pos] < best_score:
            continue
        if pos >= len(ordered):
            count = len(chosen)
            if objective == "count_then_score":
                better = count > best_count or (count == best_count and score > best_score)
            else:
                better = score > best_score or (score == best_score and count > best_count)
            if better:
                best_count, best_score, best_choice = count, score, chosen
            continue
        stack.append((pos + 1, used_ids, used_labels, chosen, score))
        ids = frozenset(ordered[pos].fragment_ids)
        labels = frozenset(ordered[pos].labels)
        if not (ids & used_ids) and not (labels & used_labels):
            stack.append(
                (pos + 1, used_ids | ids, used_labels | labels, chosen + (pos,), score + ordered[pos].score)
            )
    return [ordered[index] for index in best_choice], nodes, limited


@pytest.mark.parametrize("objective", ["score_then_count", "count_then_score"])
def test_bitset_exact_cover_matches_set_search(objective):
    rng = np.random.default_rng(5)
    for trial in range(25):
        pool = [f"f{index}" for index in range(18)]
        candidates = []
        for _ in range(rng.integers(5, 22)):
            ids = tuple(sorted(rng.choice(pool, size=rng.integers(1, 5), replace=False).tolist()))
            labels = tuple(sorted({f"S{rng.integers(0, 4)}"})) if rng.random() < 0.3 else ()
            candidates.append(
                AssemblyCandidate(
                    fragment_ids=ids,
                    coverage=1.0,
                    raw_coverage=1.0,
                    score=float(rng.choice([1.0, 2.0, float(rng.random() * 3)])),
                    support_pixels=0,
                    labels=labels,
                )
            )
        for limit in (None, 7, 40):
            stats: dict = {}
            selected = select_exact_cover_candidates(
                candidates,
                objective=objective,
                time_limit_seconds=None,
                max_search_nodes=limit,
                search_stats=stats,
            )
            expected, nodes, limited = _reference_cover(
                candidates, objective=objective, max_search_nodes=limit
            )
            assert [item.fragment_ids for item in selected] == [
                item.fragment_ids for item in expected
            ], (trial, limit)
            assert stats["search_nodes"] == nodes
            assert stats["node_limit_reached"] == limited


def test_pair_overlap_rows_match_pairwise_masks():
    fragments = _fragments(notes=3)
    geometry = _AssemblyGeometry(fragments, tolerance=2, gap_fill_radius=2)
    offsets, partners, pixels = geometry._pair_overlap_rows()
    for index, fragment in enumerate(fragments):
        row = dict(zip(partners[offsets[index] : offsets[index + 1]].tolist(),
                       pixels[offsets[index] : offsets[index + 1]].tolist()))
        expected = {
            other: int(np.count_nonzero(fragment.mask & candidate.mask))
            for other, candidate in enumerate(fragments)
            if other != index and np.any(fragment.mask & candidate.mask)
        }
        assert row == expected


def _state(fragments, members):
    union = np.zeros_like(fragments[0].mask)
    for index in members:
        union |= fragments[index].mask
    return StateInfo(
        state=frozenset(members),
        union=union,
        area_sum=sum(fragments[index].area for index in members),
        labels=set(),
        support=0,
        score=0.0,
        raw_coverage=0.0,
        coverage=0.0,
    )


@pytest.mark.parametrize("thresholds", [(0.35, 0.55), (0.0, 0.2)])
def test_batched_proposals_match_scalar_scoring(thresholds):
    min_score, automatic_score = thresholds
    fragments = _fragments(notes=4, pieces=12, seed=5)
    geometry = _AssemblyGeometry(fragments, tolerance=2, gap_fill_radius=2)
    scores, _accepted = score_absolute_tear_pairs(fragments, scoring="effectiveness")
    eligible = [edge for edge in scores if edge.contiguous_pixels >= 2 and edge.effectiveness > 0.0]
    lookup = _edge_graph(eligible, len(fragments))[1]
    by_note: dict = {}
    for index, fragment in enumerate(fragments):
        by_note.setdefault(fragment.meta["note_id"], []).append(index)
    states = [members[: len(members) // 2] for members in by_note.values()]
    overlapping = [
        (left, right)
        for left in range(len(fragments))
        for right in range(left + 1, len(fragments))
        if np.any(fragments[left].mask & fragments[right].mask)
    ]
    states.append(list(overlapping[0]))
    accepted_any = False
    for members in states:
        info = _state(fragments, members)
        disjoint = info.area_sum == info.union_count
        assert disjoint == (members != states[-1])
        proposals = [index for index in range(len(fragments)) if index not in info.state]
        seams = {}
        strongest = {}
        for index in proposals:
            supporting = _supporting_edges(info.state, index, lookup)
            if supporting:
                seams[index] = supporting
                strongest[index] = max(edge.effectiveness / (1.0 + edge.effectiveness) for edge in supporting)
        expected = []
        for index in proposals:
            evidence = geometry.score_against(
                info,
                index,
                seams.get(index, []),
                min_score=min_score,
                automatic_score=automatic_score,
            )
            if evidence is not None:
                expected.append((index, evidence))
        got = geometry.score_proposals(
            info, proposals, seams, strongest, min_score=min_score, automatic_score=automatic_score
        )
        assert got == expected
        accepted_any = accepted_any or bool(got)
    assert accepted_any or min_score > 0.0
