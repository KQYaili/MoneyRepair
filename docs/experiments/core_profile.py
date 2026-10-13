"""Where the simulated v4.4.1 core spends its work, and why proposals fail.

Measurement only. Wrappers around the core's private helpers count outcomes
and recompute classifications on the side; every wrapped call returns exactly
what the core would have returned, so the reconstruction is unchanged.

For one simulated case it reports:

* the pair-scoring funnel: all pairs, pairs whose dilated tear boundaries
  share a bounding box, pairs dropped for mask overlap, scored pairs, pairs
  with touching boundaries, and accepted review/automatic edges;
* core-search size: expanded states and core candidates;
* the memory the reconstruction keeps: deep sizes of the fragments, the
  scored-pair records, and the core and final candidates (objects shared
  between them are counted once, in the first);
* whole-assembly gap proposals by fate: no reachable evidence level, rejected
  because of the overlap penalty (the same placement without overlap would
  reach a threshold), rejected on weak seam/boundary evidence, accepted but
  pruned for union overlap, and kept;
* with ``--memory``: traced Python memory per stage and the allocation sites
  that hold the most memory near the core-search peak. ``tracemalloc`` slows
  allocation-heavy code many times over, so use it on small pools (N <= 50).
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import tracemalloc
from time import monotonic

import numpy as np

from moneyrepair import tearfit
from moneyrepair.tearfit import (
    FractalTearConfig,
    TearFitCoreConfig,
    tear_boundary_evidence,
)

ROOT = Path(__file__).resolve().parents[2]


def deep_size(value, seen: set[int]) -> int:
    """Bytes held by ``value`` and everything it references, each object once."""

    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size + (deep_size(value.base, seen) if value.base is not None else 0)
    if isinstance(value, dict):
        return size + sum(deep_size(key, seen) + deep_size(item, seen) for key, item in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(deep_size(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return size + deep_size(vars(value), seen)
    return size


def pair_funnel(fragments, all_scores, *, tolerance: int, max_pair_overlap_pixels: int) -> dict:
    """Count pairs surviving each filter of ``score_absolute_tear_pairs``."""

    evidence = tear_boundary_evidence(fragments, tolerance=tolerance)
    boxes = np.array([ev.bbox for ev in evidence], dtype=np.int64).reshape(len(fragments), 4)
    masks = [fragment.mask for fragment in fragments]
    bbox_pairs = 0
    overlap_dropped = 0
    for i in range(len(fragments)):
        rest = boxes[i + 1 :]
        partners = np.flatnonzero(
            (boxes[i, 0] < rest[:, 2])
            & (boxes[i, 2] > rest[:, 0])
            & (boxes[i, 1] < rest[:, 3])
            & (boxes[i, 3] > rest[:, 1])
        ) + (i + 1)
        bbox_pairs += len(partners)
        for j in partners.tolist():
            if max_pair_overlap_pixels >= 0 and int(
                np.count_nonzero(masks[i] & masks[j])
            ) > max_pair_overlap_pixels:
                overlap_dropped += 1
    count = len(fragments)
    levels = Counter(edge.evidence_level for edge in all_scores)
    return {
        "fragments": count,
        "all_pairs": count * (count - 1) // 2,
        "bbox_candidate_pairs": bbox_pairs,
        "mask_overlap_dropped": overlap_dropped,
        "scored_pairs": len(all_scores),
        "scored_pairs_consistent": bbox_pairs - overlap_dropped == len(all_scores),
        "touching_pairs": sum(edge.overlap_pixels > 0 for edge in all_scores),
        "review_edges": levels.get("review", 0),
        "automatic_edges": levels.get("automatic", 0),
    }


class _Probe:
    """Counters filled by wrappers around the core's private helpers."""

    def __init__(self) -> None:
        self.stage = "core"
        self.gap: dict[str, Counter] = {}
        self.memory: dict[str, dict] = {}
        self._restore: list[tuple[object, str, object]] = []

    def patch(self, owner: object, name: str, value: object) -> None:
        self._restore.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def uninstall(self) -> None:
        while self._restore:
            owner, name, value = self._restore.pop()
            setattr(owner, name, value)

    def install(self) -> None:
        geometry = tearfit._AssemblyGeometry
        original_score = geometry.score_against
        original_batch = getattr(geometry, "score_proposals", None)
        probe = self

        def score_against(self, info, candidate_index, supporting, **kwargs):
            result = original_score(self, info, candidate_index, supporting, **kwargs)
            probe.gap.setdefault(probe.stage, Counter())[
                probe.classify(self, info, candidate_index, supporting, result, **kwargs)
            ] += 1
            return result

        def score_proposals(self, info, proposals, seams, strongest, **kwargs):
            accepted = original_batch(self, info, proposals, seams, strongest, **kwargs)
            evidence = dict(accepted)
            counts = probe.gap.setdefault(probe.stage, Counter())
            for index in proposals:
                counts[
                    probe.classify(self, info, index, seams.get(index, []), evidence.get(index), **kwargs)
                ] += 1
            return accepted

        self.patch(geometry, "score_against", score_against)
        if original_batch is not None:
            self.patch(geometry, "score_proposals", score_proposals)

        original_augment = tearfit.augment_candidates_with_group_gap
        calls = {"count": 0}

        def augment(*args, **kwargs):
            calls["count"] += 1
            probe.stage = "complete_gap" if calls["count"] == 1 else "partial_gap"
            try:
                return original_augment(*args, **kwargs)
            finally:
                probe.stage = "after_gap"

        self.patch(tearfit, "augment_candidates_with_group_gap", augment)

    @staticmethod
    def classify(geometry, info, index, supporting, result, *, min_score, automatic_score, strongest=None):
        boundary = geometry.boundary(index)
        assembly = info.assembly_boundary(geometry.tolerance)
        matched = int(tearfit._crop_overlap_count(assembly, boundary.boundary, boundary.y0, boundary.x0))
        seam_count = len(supporting)
        if seam_count < 2 and matched < 2:
            return "no_reachable_level"
        overlap = geometry.union_overlap(info.union, index)
        if result is not None:
            internal = info.area_sum - info.union_count
            if internal + overlap > 0:
                return "accepted_then_pruned_for_union_overlap"
            return "accepted_and_kept"
        threshold = math.inf
        if seam_count >= 2:
            threshold = automatic_score
        if matched >= 2:
            threshold = min(threshold, min_score)
        overlap_free = tearfit._group_gap_evidence(
            index,
            overlap=0,
            area=geometry.areas[index],
            matched_pixels=matched,
            boundary_pixels=boundary.boundary_pixels,
            supporting=supporting,
            pose_uncertainty=geometry.pose_uncertainty(index),
            min_score=min_score,
            automatic_score=automatic_score,
        )
        if overlap > 0 and overlap_free.score >= threshold:
            return "rejected_by_overlap_penalty"
        return "rejected_on_weak_evidence"


def run(notes: int, pieces: int, seed: int, strategy: str, memory: bool) -> dict:
    probe = _Probe()
    try:
        return _run(probe, notes, pieces, seed, strategy, memory)
    finally:
        probe.uninstall()
        if tracemalloc.is_tracing():
            tracemalloc.stop()


def _run(probe: _Probe, notes: int, pieces: int, seed: int, strategy: str, memory: bool) -> dict:
    probe.install()
    config = TearFitCoreConfig.v4_4_1()
    trial_kwargs = asdict(config)
    del trial_kwargs["fray_layers"]
    trial_kwargs["base_selection_strategy"] = strategy
    simulation = FractalTearConfig(
        notes=notes, pieces_per_note=pieces, seed=seed, serial_ocr_rate=0.0
    )
    captured = {}
    original_reconstruct = tearfit.reconstruct_placed_fragments

    def reconstruct(*args, **kwargs):
        captured["fragments"] = args[0]
        captured["reconstruction"] = original_reconstruct(*args, **kwargs)
        return captured["reconstruction"]

    probe.patch(tearfit, "reconstruct_placed_fragments", reconstruct)
    if memory:
        _track_memory(probe)
    started = monotonic()
    result = tearfit.run_tearfit_trial(simulation, **trial_kwargs)
    wall = monotonic() - started
    reconstruction = captured["reconstruction"]
    funnel = pair_funnel(
        captured["fragments"],
        reconstruction.all_scores,
        tolerance=config.tolerance,
        max_pair_overlap_pixels=0,
    )
    seen: set[int] = set()
    retained = {
        name: round(deep_size(value, seen) / 2**20, 1)
        for name, value in (
            ("fragments_mb", captured["fragments"]),
            ("scored_pairs_mb", reconstruction.all_scores),
            ("core_candidates_mb", reconstruction.generated_candidates),
            ("pool_candidates_mb", reconstruction.candidates),
        )
    }
    return {
        "case": {"notes": notes, "pieces": pieces, "seed": seed, "strategy": strategy},
        "resolved_algorithm": reconstruction.resolved_algorithm,
        "wall_seconds_instrumented": wall,
        "stage_timings_instrumented": dict(result.stage_timings),
        "pair_funnel": funnel,
        "retained_memory": retained,
        "core_search": {
            "expanded_states": reconstruction.search_stats["core"]["expanded_states"],
            "core_candidates": len(reconstruction.generated_candidates),
        },
        "gap_proposals": {stage: dict(counts) for stage, counts in probe.gap.items()},
        "gap_search_stats": {
            stage: reconstruction.search_stats[stage]
            for stage in ("complete_gap", "partial_gap")
        },
        "memory": probe.memory,
    }


def _track_memory(probe: _Probe, top: int = 12) -> None:
    """Record traced memory per stage and the largest sites near the core peak."""

    tracemalloc.start(1)
    stage_functions = {
        "pair_scoring": "score_absolute_tear_pairs",
        "core_search": "generate_assembly_candidates",
        "gap_search": "augment_candidates_with_group_gap",
        "exact_cover": "select_exact_cover_candidates",
    }
    for label, name in stage_functions.items():
        original = getattr(tearfit, name)

        def wrapped(*args, __original=original, __label=label, **kwargs):
            before = tracemalloc.get_traced_memory()[0]
            tracemalloc.reset_peak()
            value = __original(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            record = probe.memory.setdefault(__label, {"calls": 0, "peak_mb": 0.0})
            record["calls"] += 1
            record["before_mb"] = before / 1e6
            record["peak_mb"] = max(record["peak_mb"], peak / 1e6)
            record["after_mb"] = current / 1e6
            return value

        probe.patch(tearfit, name, wrapped)

    original_beam = tearfit._beam_records
    state = {"best": 0}

    def beam_records(records, width):
        current = tracemalloc.get_traced_memory()[0]
        # Snapshots are slow; take one only after traced memory grew by 50 MB.
        if current > state["best"] + 50_000_000:
            state["best"] = current
            snapshot = tracemalloc.take_snapshot().filter_traces(
                [tracemalloc.Filter(False, tracemalloc.__file__)]
            )
            probe.memory["core_peak_sites"] = {
                "traced_mb": current / 1e6,
                "sites": [
                    {"site": f"{Path(stat.traceback[0].filename).name}:{stat.traceback[0].lineno}",
                     "mb": stat.size / 1e6, "blocks": stat.count}
                    for stat in snapshot.statistics("lineno")[:top]
                ],
            }
        return original_beam(records, width)

    probe.patch(tearfit, "_beam_records", beam_records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notes", type=int, default=100)
    parser.add_argument("--pieces", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strategy", default="disjoint_round_robin")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run(args.notes, args.pieces, args.seed, args.strategy, args.memory)
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
