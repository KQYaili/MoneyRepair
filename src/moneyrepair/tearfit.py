from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Iterable

import numpy as np

from moneyrepair.interlock import binary_dilation_3x3
from moneyrepair.simulate import synthetic_banknote
from moneyrepair.types import Fragment

TEARFIT_SEED_STRATEGIES = ("anchor_only", "anchor_priority", "all")
TEARFIT_COVER_OBJECTIVES = ("count_then_score", "score_then_count")


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


@dataclass(frozen=True)
class TearFitEdge:
    left: int
    right: int
    overlap_pixels: int
    left_hits: int
    right_hits: int
    overlap_ratio: float


@dataclass(frozen=True)
class AssemblyCandidate:
    fragment_ids: tuple[str, ...]
    coverage: float
    raw_coverage: float
    score: float
    support_pixels: int
    labels: tuple[str, ...] = field(default_factory=tuple)


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
                "confirmed_candidates": [
                    {
                        "fragment_ids": item.fragment_ids,
                        "coverage": item.coverage,
                        "raw_coverage": item.raw_coverage,
                        "score": item.score,
                        "support_pixels": item.support_pixels,
                        "labels": item.labels,
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


def _smooth_noise(length: int, rng: np.random.Generator, roughness: float) -> np.ndarray:
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
        split = _split_mask_once(masks[index], rng, roughness=roughness, min_area=min_area)
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


def make_fractal_tear_fragments(config: FractalTearConfig) -> tuple[np.ndarray, list[Fragment]]:
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
        if config.ensure_serial_anchor and not any(fragment.label for fragment in note_fragments):
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


def tear_boundary_evidence(fragments: list[Fragment], *, tolerance: int = 2) -> list[TearBoundaryEvidence]:
    evidence: list[TearBoundaryEvidence] = []
    for fragment in fragments:
        boundary = tear_boundary(fragment.mask)
        dilated = _dilate(boundary, tolerance)
        evidence.append(
            TearBoundaryEvidence(
                boundary=np.flatnonzero(boundary),
                dilated_boundary=np.flatnonzero(dilated),
                bbox=_bbox(dilated),
            )
        )
    return evidence


def _labels_compatible(left: str | None, right: str | None) -> bool:
    return not left or not right or left == right


def _bbox_intersects(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return left[0] < right[2] and left[2] > right[0] and left[1] < right[3] and left[3] > right[1]


def _mask_overlap(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left & right))


def score_absolute_tear_pairs(
    fragments: list[Fragment],
    *,
    tolerance: int = 2,
    min_overlap_pixels: int = 1,
    max_pair_overlap_pixels: int = 0,
    use_labels: bool = True,
) -> tuple[list[TearFitEdge], list[TearFitEdge]]:
    """Score torn-edge coincidence in the placed coordinate frame.

    Returns ``(all_scored_pairs, accepted_edges)``. A pair is accepted when the
    two tear boundaries occupy the same absolute coordinates within
    ``tolerance`` and the masks do not overlap past ``max_pair_overlap_pixels``.
    """

    evidence = tear_boundary_evidence(fragments, tolerance=tolerance)
    all_scores: list[TearFitEdge] = []
    accepted: list[TearFitEdge] = []
    for i in range(len(fragments)):
        left_ev = evidence[i]
        for j in range(i + 1, len(fragments)):
            if use_labels and not _labels_compatible(fragments[i].label, fragments[j].label):
                continue
            if max_pair_overlap_pixels >= 0 and _mask_overlap(fragments[i].mask, fragments[j].mask) > max_pair_overlap_pixels:
                continue
            right_ev = evidence[j]
            if not _bbox_intersects(left_ev.bbox, right_ev.bbox):
                continue
            left_hits = int(np.intersect1d(left_ev.boundary, right_ev.dilated_boundary, assume_unique=True).size)
            right_hits = int(np.intersect1d(right_ev.boundary, left_ev.dilated_boundary, assume_unique=True).size)
            overlap = min(left_hits, right_hits)
            denom = max(1, min(len(left_ev.boundary), len(right_ev.boundary)))
            edge = TearFitEdge(
                left=i,
                right=j,
                overlap_pixels=overlap,
                left_hits=left_hits,
                right_hits=right_hits,
                overlap_ratio=overlap / float(denom),
            )
            all_scores.append(edge)
            if overlap >= min_overlap_pixels:
                accepted.append(edge)
    return all_scores, accepted


def _edge_graph(edges: Iterable[TearFitEdge], count: int) -> tuple[list[dict[int, TearFitEdge]], dict[tuple[int, int], TearFitEdge]]:
    graph: list[dict[int, TearFitEdge]] = [dict() for _ in range(count)]
    lookup: dict[tuple[int, int], TearFitEdge] = {}
    for edge in edges:
        graph[edge.left][edge.right] = edge
        graph[edge.right][edge.left] = edge
        lookup[(min(edge.left, edge.right), max(edge.left, edge.right))] = edge
    return graph, lookup


def _group_labels(fragments: list[Fragment], indices: Iterable[int]) -> tuple[str, ...]:
    return tuple(sorted({fragments[index].label for index in indices if fragments[index].label}))


def _labels_ok(fragments: list[Fragment], indices: Iterable[int]) -> bool:
    return len(_group_labels(fragments, indices)) <= 1


def _group_masks_ok(fragments: list[Fragment], indices: Iterable[int], max_overlap_pixels: int) -> tuple[bool, np.ndarray, int]:
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


def _support_for_state(state: frozenset[int], edge_lookup: dict[tuple[int, int], TearFitEdge]) -> int:
    members = sorted(state)
    support = 0
    for pos, left in enumerate(members):
        for right in members[pos + 1 :]:
            edge = edge_lookup.get((left, right))
            if edge is not None:
                support += edge.overlap_pixels
    return support


def _resolve_seed_strategy(seed_strategy: str, require_anchor: bool | None) -> str:
    if require_anchor is not None:
        return "anchor_priority"
    if seed_strategy not in TEARFIT_SEED_STRATEGIES:
        raise ValueError(f"seed_strategy must be one of: {', '.join(TEARFIT_SEED_STRATEGIES)}")
    return seed_strategy


def _seed_order(fragments: list[Fragment], seed_strategy: str) -> list[int]:
    labelled = [index for index, fragment in enumerate(fragments) if fragment.label]
    unlabelled = [index for index, fragment in enumerate(fragments) if not fragment.label]
    if seed_strategy == "anchor_only":
        return labelled
    if seed_strategy == "anchor_priority":
        return labelled + unlabelled
    if seed_strategy == "all":
        return list(range(len(fragments)))
    raise ValueError(f"seed_strategy must be one of: {', '.join(TEARFIT_SEED_STRATEGIES)}")


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
    ) -> None:
        self.state = state
        self.union = union
        self.area_sum = area_sum
        self.labels = labels
        self.support = support
        self.score = score
        self.raw_coverage = raw_coverage
        self.coverage = coverage


def generate_assembly_candidates(
    fragments: list[Fragment],
    edges: list[TearFitEdge],
    *,
    coverage_threshold: float = 0.93,
    gap_fill_radius: int = 2,
    max_pieces: int = 12,
    beam_width: int = 64,
    seed_strategy: str = "anchor_priority",
    require_anchor: bool | None = None,
    max_group_overlap_pixels: int = 0,
    time_limit_seconds: float | None = 20.0,
) -> list[AssemblyCandidate]:
    """Generate full-note candidates by connected tear graph search.

    The search is label-aware and overlap-constrained. Serial labels act as
    constraints and priority seeds, not as the only legal entry point unless
    ``seed_strategy="anchor_only"`` is explicitly requested for comparison.
    It produces candidate note groups; :func:`select_exact_cover_candidates`
    then performs the global set-packing pass over those candidates.
    """

    if not fragments:
        return []
    seed_strategy = _resolve_seed_strategy(seed_strategy, require_anchor)
    graph, edge_lookup = _edge_graph(edges, len(fragments))
    starts = _seed_order(fragments, seed_strategy)
    if not starts:
        return []
    total_area = fragments[0].mask.size
    deadline = None if time_limit_seconds is None else monotonic() + time_limit_seconds
    seen_states: set[frozenset[int]] = set()
    candidates: dict[tuple[str, ...], AssemblyCandidate] = {}

    for start in starts:
        if deadline is not None and monotonic() >= deadline:
            break
        start_fragment = fragments[start]
        start_union = start_fragment.mask.copy()
        start_area_sum = start_fragment.area
        start_labels = {start_fragment.label} if start_fragment.label else set()
        start_support = 0
        start_raw_coverage = start_area_sum / float(total_area)
        if start_raw_coverage >= coverage_threshold - 0.15:
            start_coverage = int(_dilate(start_union, gap_fill_radius).sum()) / float(total_area)
        else:
            start_coverage = start_raw_coverage
        start_score = start_coverage * 10_000.0 + start_support + 100.0 * len(start_labels)

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
            next_frontier: dict[frozenset[int], StateInfo] = {}
            for state_info in frontier:
                if deadline is not None and monotonic() >= deadline:
                    break
                neighbours: set[int] = set()
                for member in state_info.state:
                    neighbours.update(graph[member])
                neighbours.difference_update(state_info.state)
                for neighbour in neighbours:
                    new_state_set = frozenset((*state_info.state, neighbour))
                    if new_state_set in seen_states or len(new_state_set) > max_pieces:
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

                    # Calculate support incrementally
                    added_support = 0
                    for member in state_info.state:
                        key = (min(member, neighbour), max(member, neighbour))
                        edge = edge_lookup.get(key)
                        if edge is not None:
                            added_support += edge.overlap_pixels
                    new_support = state_info.support + added_support

                    raw_coverage = int(new_union.sum()) / float(total_area)
                    if raw_coverage >= coverage_threshold - 0.15:
                        coverage = int(_dilate(new_union, gap_fill_radius).sum()) / float(total_area)
                    else:
                        coverage = raw_coverage

                    score = coverage * 10_000.0 + new_support + 100.0 * len(new_labels)

                    new_info = StateInfo(
                        state=new_state_set,
                        union=new_union,
                        area_sum=new_area_sum,
                        labels=new_labels,
                        support=new_support,
                        score=score,
                        raw_coverage=raw_coverage,
                        coverage=coverage,
                    )

                    existing = next_frontier.get(new_state_set)
                    if existing is None or new_info.score > existing.score:
                        next_frontier[new_state_set] = new_info

                    if coverage >= coverage_threshold:
                        ids = tuple(sorted(fragments[index].id for index in new_state_set))
                        existing_candidate = candidates.get(ids)
                        candidate = AssemblyCandidate(
                            fragment_ids=ids,
                            coverage=coverage,
                            raw_coverage=raw_coverage,
                            score=score,
                            support_pixels=new_support,
                            labels=tuple(sorted(new_labels)),
                        )
                        if existing_candidate is None or candidate.score > existing_candidate.score:
                            candidates[ids] = candidate
                seen_states.add(state_info.state)
            if not next_frontier:
                break
            ordered = sorted(next_frontier.values(), key=lambda item: (-item.score, len(item.state), tuple(sorted(item.state))))
            frontier = ordered[:beam_width]

    return sorted(candidates.values(), key=lambda item: (-item.score, item.fragment_ids))


def select_exact_cover_candidates(
    candidates: list[AssemblyCandidate],
    *,
    time_limit_seconds: float | None = 10.0,
    objective: str = "score_then_count",
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
        raise ValueError(f"objective must be one of: {', '.join(TEARFIT_COVER_OBJECTIVES)}")
    ordered = sorted(candidates, key=lambda item: (-item.score, -len(item.fragment_ids), item.fragment_ids))
    
    # Pre-compute optimistic suffix scores for score_then_count DFS pruning
    suffix_score = [0.0] * (len(ordered) + 1)
    for i in range(len(ordered) - 1, -1, -1):
        suffix_score[i] = suffix_score[i + 1] + max(0.0, ordered[i].score)

    deadline = None if time_limit_seconds is None else monotonic() + time_limit_seconds
    best_count = 0
    best_score = float("-inf")
    best_choice: tuple[int, ...] = ()

    def better(count: int, score: float) -> bool:
        if objective == "count_then_score":
            return count > best_count or (count == best_count and score > best_score)
        return score > best_score or (score == best_score and count > best_count)

    def dfs(pos: int, used_ids: frozenset[str], used_labels: frozenset[str], chosen: tuple[int, ...], score: float) -> None:
        nonlocal best_count, best_score, best_choice
        if deadline is not None and monotonic() >= deadline:
            return
        if objective == "count_then_score" and len(chosen) + (len(ordered) - pos) < best_count:
            return
        if objective == "score_then_count" and score + suffix_score[pos] < best_score:
            return
        if pos >= len(ordered):
            count = len(chosen)
            if better(count, score):
                best_count = count
                best_score = score
                best_choice = chosen
            return

        item = ordered[pos]
        item_ids = frozenset(item.fragment_ids)
        item_labels = frozenset(item.labels)
        if not (item_ids & used_ids) and not (item_labels & used_labels):
            dfs(pos + 1, used_ids | item_ids, used_labels | item_labels, chosen + (pos,), score + item.score)
        dfs(pos + 1, used_ids, used_labels, chosen, score)

    dfs(0, frozenset(), frozenset(), (), 0.0)
    return [ordered[index] for index in best_choice]


def diagnose_confirmed_candidates(candidates: list[AssemblyCandidate], fragments: list[Fragment]) -> TearFitDiagnostics:
    lookup = {fragment.id: fragment for fragment in fragments}
    true_notes: dict[str, set[str]] = {}
    for fragment in fragments:
        note_id = fragment.meta.get("note_id")
        if note_id:
            true_notes.setdefault(note_id, set()).add(fragment.id)

    exact = 0
    pure = 0
    chimeras = 0
    exact_notes: set[str] = set()
    for candidate in candidates:
        ids = set(candidate.fragment_ids)
        notes = {lookup[fid].meta.get("note_id") for fid in ids if fid in lookup and lookup[fid].meta.get("note_id")}
        if len(notes) > 1:
            chimeras += 1
            continue
        if len(notes) == 1:
            pure += 1
            note_id = next(iter(notes))
            if ids == true_notes.get(note_id, set()):
                exact += 1
                exact_notes.add(note_id)
    confirmed = len(candidates)
    true_count = len(true_notes)
    return TearFitDiagnostics(
        confirmed=confirmed,
        exact_confirmed=exact,
        pure_confirmed=pure,
        chimeras=chimeras,
        true_notes=true_count,
        exact_yield=exact / true_count if true_count else 0.0,
        exact_precision=exact / confirmed if confirmed else 0.0,
        pure_precision=pure / confirmed if confirmed else 0.0,
        manual_notes_remaining=true_count - len(exact_notes),
        confirmed_candidates=tuple(candidates),
    )


def run_tearfit_trial(
    config: FractalTearConfig,
    *,
    tolerance: int = 2,
    min_overlap_pixels: int = 14,
    coverage_threshold: float = 0.93,
    gap_fill_radius: int = 2,
    max_pieces: int | None = None,
    beam_width: int = 64,
    use_labels: bool = True,
    seed_strategy: str = "anchor_priority",
    require_anchor: bool | None = None,
    serial_ocr_rate: float | None = None,
    candidate_time_limit_seconds: float | None = 20.0,
    cover_time_limit_seconds: float | None = 10.0,
    cover_objective: str = "score_then_count",
) -> TearFitTrialResult:
    """Run one labelled exact-cover tear-fit trial."""

    seed_strategy = _resolve_seed_strategy(seed_strategy, require_anchor)
    if serial_ocr_rate is not None:
        config = FractalTearConfig(**{**config.__dict__, "serial_ocr_rate": serial_ocr_rate})
    _template, fragments = make_fractal_tear_fragments(config)
    all_scores, _raw_edges = score_absolute_tear_pairs(
        fragments,
        tolerance=tolerance,
        min_overlap_pixels=1,
        use_labels=False,
    )
    _label_filtered_scores, edges = score_absolute_tear_pairs(
        fragments,
        tolerance=tolerance,
        min_overlap_pixels=min_overlap_pixels,
        use_labels=use_labels,
    )
    candidates = generate_assembly_candidates(
        fragments,
        edges,
        coverage_threshold=coverage_threshold,
        gap_fill_radius=gap_fill_radius,
        max_pieces=max_pieces or config.pieces_per_note + 2,
        beam_width=beam_width,
        seed_strategy=seed_strategy,
        time_limit_seconds=candidate_time_limit_seconds,
    )
    selected = select_exact_cover_candidates(
        candidates,
        time_limit_seconds=cover_time_limit_seconds,
        objective=cover_objective,
    )
    diagnostics = diagnose_confirmed_candidates(selected, fragments)

    true_scores = [
        edge.overlap_pixels
        for edge in all_scores
        if fragments[edge.left].meta.get("note_id") == fragments[edge.right].meta.get("note_id")
    ]
    false_scores = [
        edge.overlap_pixels
        for edge in all_scores
        if fragments[edge.left].meta.get("note_id") != fragments[edge.right].meta.get("note_id")
    ]
    false_edges = [
        edge
        for edge in edges
        if fragments[edge.left].meta.get("note_id") != fragments[edge.right].meta.get("note_id")
    ]
    return TearFitTrialResult(
        config={
            **config.__dict__,
            "tolerance": tolerance,
            "min_overlap_pixels": min_overlap_pixels,
            "coverage_threshold": coverage_threshold,
            "gap_fill_radius": gap_fill_radius,
            "beam_width": beam_width,
            "use_labels": use_labels,
            "seed_strategy": seed_strategy,
            "candidate_time_limit_seconds": candidate_time_limit_seconds,
            "cover_time_limit_seconds": cover_time_limit_seconds,
            "cover_objective": cover_objective,
        },
        fragments=len(fragments),
        pair_scores=len(all_scores),
        accepted_edges=len(edges),
        false_edge_rate=len(false_edges) / len(edges) if edges else 0.0,
        true_edge_median=float(np.median(true_scores)) if true_scores else 0.0,
        false_edge_median=float(np.median(false_scores)) if false_scores else 0.0,
        candidates=len(candidates),
        diagnostics=diagnostics,
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
    serial_ocr_rate: float = 1.0,
    seed_strategy: str = "anchor_priority",
    require_anchor: bool | None = None,
    ensure_serial_anchor: bool = True,
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


def tearfit_comparison_cases(profile: str = "smoke") -> tuple[TearFitComparisonCase, ...]:
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
            TearFitComparisonCase("fray_n50_p8", notes=50, pieces_per_note=8, fray_probability=0.40),
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
            raise ValueError(f"cover_objective must be one of: {', '.join(TEARFIT_COVER_OBJECTIVES)}")
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


def _parse_notes_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the placed tear-fit research sandbox")
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
    parser.add_argument("--serial-ocr-rate", type=float, default=1.0)
    parser.add_argument("--seed-strategy", choices=TEARFIT_SEED_STRATEGIES, default="anchor_priority")
    parser.add_argument("--ensure-serial-anchor", action="store_true")
    parser.add_argument("--candidate-time-limit", type=float, default=20.0)
    parser.add_argument("--cover-time-limit", type=float, default=10.0)
    parser.add_argument("--cover-objective", choices=TEARFIT_COVER_OBJECTIVES, default="score_then_count")
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
        serial_ocr_rate=args.serial_ocr_rate,
        seed_strategy=args.seed_strategy,
        ensure_serial_anchor=args.ensure_serial_anchor,
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
