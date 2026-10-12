from __future__ import annotations

import argparse
from bisect import bisect_left
import heapq
from hashlib import sha256
import json
from math import ceil
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from pathlib import Path
from time import monotonic
from typing import AbstractSet, Any, Callable, Collection, Iterable, cast

import numba
import numpy as np

from moneyrepair.simulate import load_dataset, synthetic_banknote
from moneyrepair.types import Fragment


def binary_dilation_3x3(mask: np.ndarray) -> np.ndarray:
    """Dilate a binary 2D array with a 3x3 structuring element (all ones) using numpy shifts."""
    if mask.size == 0 or mask.ndim != 2:
        return mask
    mask = mask.astype(bool)
    dilated = mask.copy()
    dilated[1:, :] |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:] |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    dilated[1:, 1:] |= mask[:-1, :-1]
    dilated[:-1, :-1] |= mask[1:, 1:]
    dilated[1:, :-1] |= mask[:-1, 1:]
    dilated[:-1, 1:] |= mask[1:, :-1]
    return dilated


TEARFIT_SEED_STRATEGIES = ("anchor_only", "anchor_priority", "all")
TEARFIT_COVER_OBJECTIVES = ("count_then_score", "score_then_count")
TEARFIT_EDGE_SCORING = ("overlap", "effectiveness")
TEARFIT_ALGORITHMS = (
    "baseline",
    "effectiveness",
    "effectiveness_gap",
    "v43_routed",
    "v44_gap_first",
)
TEARFIT_GAP_ROUTING = ("complexity", "uniform")
TEARFIT_EVIDENCE_LEVELS = ("automatic", "review", "insufficient-evidence")
TEARFIT_GAP_PROPOSAL_POOLS = ("weak_pair", "boundary_contact")
TEARFIT_BASE_SELECTION_STRATEGIES = ("global", "disjoint_round_robin")
TEARFIT_V43_FINE_FRACTION = 0.05
TEARFIT_OBSERVED_META_KEYS = (
    "sigma_x",
    "pose_sigma_x",
    "sigma_y",
    "pose_sigma_y",
    "sigma_theta",
    "pose_sigma_theta",
    "sigma_scale",
    "pose_sigma_scale",
)


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
    boundary_mask: np.ndarray
    dilated_boundary_mask: np.ndarray
    normal_y: np.ndarray
    normal_x: np.ndarray


@dataclass(frozen=True)
class _CroppedBoundary:
    """Tear-boundary evidence cropped to its dilated-boundary bounding box.

    It holds exactly the pixels of :class:`TearBoundaryEvidence`, so scores
    computed from it are identical, while memory and per-pair work scale with
    the fragment rather than the canvas. ``x0, y0, x1, y1`` is the crop in
    canvas coordinates.
    """

    x0: int
    y0: int
    x1: int
    y1: int
    boundary: np.ndarray
    dilated: np.ndarray
    normal_y: np.ndarray
    normal_x: np.ndarray
    boundary_pixels: int
    dilated_pixels: int

    def crop(
        self, array: np.ndarray, window: tuple[int, int, int, int]
    ) -> np.ndarray:
        """View ``array`` (one of this crop's fields) over a canvas window."""

        x0, y0, x1, y1 = window
        return array[y0 - self.y0 : y1 - self.y0, x0 - self.x0 : x1 - self.x0]


@dataclass(frozen=True)
class TearFitEdge:
    left: int
    right: int
    overlap_pixels: int
    left_hits: int
    right_hits: int
    overlap_ratio: float
    contiguous_pixels: int = 0
    continuity_ratio: float = 0.0
    bidirectional_balance: float = 0.0
    normal_opposition: float = 0.0
    curvature_entropy: float = 0.0
    expected_accidental_hits: float = 0.0
    mask_overlap_pixels: int = 0
    pose_uncertainty: float = 0.0
    effectiveness: float = 0.0
    evidence_level: str = "insufficient-evidence"


@dataclass(frozen=True)
class GroupGapEvidence:
    fragment: int
    seam_count: int
    matched_pixels: int
    gap_fill_pixels: int
    gap_fill_ratio: float
    overlap_pixels: int
    unmatched_perimeter_ratio: float
    pose_uncertainty: float
    score: float
    evidence_level: str


@dataclass(frozen=True)
class ResidualGapRegion:
    """An explicit uncovered gap left by a high-confidence assembly core.

    The mask is the connected component of the note-frame complement of a core
    union. Geometry here is never synthesised: the region only records where
    real fragment evidence is still missing, which fragments border it, and how
    hard it is expected to be to close.
    """

    mask: np.ndarray
    area: int
    perimeter_pixels: int
    component_id: int
    adjacent_fragment_indices: tuple[int, ...]
    complexity: float = 0.0
    routing_class: str = "simple"


@dataclass(frozen=True)
class GapProposalEvaluation:
    """Proposal-effectiveness / tolerance verdict for one fragment-set proposal.

    ``effectiveness`` is the ``E_proposal`` ratio: the closed exposed-perimeter
    plus independent seam support divided by the physically penalised terms
    (slivers, tolerance instability, residual gap, newly exposed perimeter).
    """

    accepted: bool
    reason: str
    effectiveness: float
    closed_perimeter_delta: float
    independent_seam_support: float
    sliver_area: float
    tolerance_instability: float
    residual_gap_delta: float
    newly_exposed_perimeter: float
    net_new_area: int
    overlap_pixels: int
    coverage: float
    raw_coverage: float
    support_pixels: int
    evidence_score: float
    evidence_level: str


@dataclass(frozen=True)
class AssemblyCandidate:
    fragment_ids: tuple[str, ...]
    coverage: float
    raw_coverage: float
    score: float
    support_pixels: int
    labels: tuple[str, ...] = field(default_factory=tuple)
    base_score: float = 0.0
    constraint_bonus: float = 0.0
    evidence_score: float = 1.0
    evidence_level: str = "automatic"
    gap_steps: int = 0


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
    automatic_candidates: int = 0
    review_candidates: int = 0
    automatic_exact_confirmed: int = 0
    automatic_exact_yield: float = 0.0
    automatic_exact_precision: float = 0.0


@dataclass(frozen=True)
class TearFitTrialResult:
    config: dict
    fragments: int
    pair_scores: int
    accepted_edges: int
    true_possible_edges: int
    true_accepted_edges: int
    false_accepted_edges: int
    false_edge_rate: float
    true_edge_recall: float
    true_edge_median: float
    false_edge_median: float
    candidates: int
    diagnostics: TearFitDiagnostics
    edge_decisions: dict[str, int] = field(default_factory=dict)
    candidate_decisions: dict[str, int] = field(default_factory=dict)
    search_stats: dict[str, dict[str, int | bool]] = field(default_factory=dict)
    core_candidates: int = 0
    partial_core_candidates: int = 0
    gap_candidates: int = 0
    partial_gap_candidates: int = 0
    selected_core_candidates: int = 0
    selected_gap_candidates: int = 0
    selected_complete_gap_candidates: int = 0
    selected_partial_gap_candidates: int = 0
    true_gap_candidates: int = 0
    false_gap_candidates: int = 0
    selected_true_gap_candidates: int = 0
    selected_false_gap_candidates: int = 0
    oracle_candidate_notes: int = 0
    oracle_candidate_recall: float = 0.0
    selected_score_total: float = 0.0
    selected_solution_fingerprint: str = ""
    candidate_provenance_fingerprint: str = ""
    stage_timings: dict[str, float] = field(default_factory=dict)
    diagnostic_removed_false_accepted_edges: int = 0
    diagnostic_removed_false_core_edges: int = 0
    candidate_funnel: dict = field(default_factory=dict)
    proposal_efficiency: float = 0.0

    def to_jsonable(self) -> dict:
        return {
            "config": self.config,
            "fragments": self.fragments,
            "pair_scores": self.pair_scores,
            "accepted_edges": self.accepted_edges,
            "true_possible_edges": self.true_possible_edges,
            "true_accepted_edges": self.true_accepted_edges,
            "false_accepted_edges": self.false_accepted_edges,
            "false_edge_rate": self.false_edge_rate,
            "true_edge_recall": self.true_edge_recall,
            "true_edge_median": self.true_edge_median,
            "false_edge_median": self.false_edge_median,
            "candidates": self.candidates,
            "core_candidates": self.core_candidates,
            "partial_core_candidates": self.partial_core_candidates,
            "gap_candidates": self.gap_candidates,
            "partial_gap_candidates": self.partial_gap_candidates,
            "selected_core_candidates": self.selected_core_candidates,
            "selected_gap_candidates": self.selected_gap_candidates,
            "selected_complete_gap_candidates": self.selected_complete_gap_candidates,
            "selected_partial_gap_candidates": self.selected_partial_gap_candidates,
            "true_gap_candidates": self.true_gap_candidates,
            "false_gap_candidates": self.false_gap_candidates,
            "selected_true_gap_candidates": self.selected_true_gap_candidates,
            "selected_false_gap_candidates": self.selected_false_gap_candidates,
            "oracle_candidate_notes": self.oracle_candidate_notes,
            "oracle_candidate_recall": self.oracle_candidate_recall,
            "selected_score_total": self.selected_score_total,
            "selected_solution_fingerprint": self.selected_solution_fingerprint,
            "candidate_provenance_fingerprint": self.candidate_provenance_fingerprint,
            "stage_timings": dict(self.stage_timings),
            "diagnostic_removed_false_accepted_edges": (
                self.diagnostic_removed_false_accepted_edges
            ),
            "diagnostic_removed_false_core_edges": (
                self.diagnostic_removed_false_core_edges
            ),
            "candidate_funnel": self.candidate_funnel,
            "proposal_efficiency": self.proposal_efficiency,
            "edge_decisions": dict(self.edge_decisions),
            "candidate_decisions": dict(self.candidate_decisions),
            "search_stats": self.search_stats,
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
                "automatic_candidates": self.diagnostics.automatic_candidates,
                "review_candidates": self.diagnostics.review_candidates,
                "automatic_exact_confirmed": self.diagnostics.automatic_exact_confirmed,
                "automatic_exact_yield": self.diagnostics.automatic_exact_yield,
                "automatic_exact_precision": self.diagnostics.automatic_exact_precision,
                "confirmed_candidates": [
                    {
                        "fragment_ids": item.fragment_ids,
                        "coverage": item.coverage,
                        "raw_coverage": item.raw_coverage,
                        "score": item.score,
                        "support_pixels": item.support_pixels,
                        "labels": item.labels,
                        "evidence_score": item.evidence_score,
                        "evidence_level": item.evidence_level,
                        "gap_steps": item.gap_steps,
                    }
                    for item in self.diagnostics.confirmed_candidates
                ],
            },
        }


@dataclass(frozen=True)
class TearFitCoreConfig:
    """Decision parameters of the tear-fit core, independent of how fragments arose.

    ``fray_layers`` is the expected boundary fray depth in pixels; only the
    v4.4 residual-gap-first proposal reads it.
    """

    algorithm: str = "baseline"
    route_fragment_fraction_threshold: float = TEARFIT_V43_FINE_FRACTION
    tolerance: int = 2
    min_overlap_pixels: int = 14
    min_effectiveness: float = 1.0
    automatic_effectiveness: float = 2.0
    min_contiguous_pixels: int = 3
    automatic_contiguous_pixels: int = 5
    core_min_pieces: int = 2
    min_group_gap_score: float = 0.35
    automatic_group_gap_score: float = 0.55
    coverage_threshold: float = 0.93
    core_raw_coverage_threshold: float | None = None
    gap_fill_radius: int = 2
    beam_width: int = 64
    max_complete_core_candidates: int = 512
    max_partial_core_candidates: int = 128
    base_selection_strategy: str = "global"
    gap_proposal_pool: str = "weak_pair"
    use_labels: bool = True
    seed_strategy: str = "anchor_priority"
    candidate_time_limit_seconds: float | None = 20.0
    candidate_state_limit: int | None = None
    candidate_states_per_pair_score: float | None = None
    partial_gap_time_limit_seconds: float | None = 5.0
    gap_state_limit: int | None = None
    gap_states_per_fragment: float | None = None
    partial_gap_state_limit: int | None = None
    partial_gap_states_per_fragment: float | None = None
    cover_time_limit_seconds: float | None = 10.0
    cover_node_limit: int | None = None
    cover_nodes_per_note: float | None = None
    cover_objective: str = "score_then_count"
    enable_residual_gap_proposals: bool = True
    gap_routing: str = "complexity"
    gap_region_max_core_candidates: int = 8
    gap_proposal_alpha: float = 1.0
    gap_proposal_gamma: float = 1.0
    gap_proposal_delta: float = 1.0
    gap_proposal_eps: float = 1.0
    gap_proposal_zeta: float = 1.0
    gap_proposal_min_effectiveness: float = 0.0
    fray_layers: int = 2

    def __post_init__(self) -> None:
        if self.algorithm not in TEARFIT_ALGORITHMS:
            raise ValueError(f"algorithm must be one of: {', '.join(TEARFIT_ALGORITHMS)}")
        if not (0.0 < self.route_fragment_fraction_threshold < 1.0):
            raise ValueError("route_fragment_fraction_threshold must be in (0, 1)")
        if not (0.0 < self.resolved_core_raw_coverage_threshold < self.coverage_threshold):
            raise ValueError(
                "core_raw_coverage_threshold must be in (0, coverage_threshold)"
            )
        if self.max_complete_core_candidates < 1 or self.max_partial_core_candidates < 1:
            raise ValueError("core candidate limits must be positive")
        if self.base_selection_strategy not in TEARFIT_BASE_SELECTION_STRATEGIES:
            raise ValueError(
                "base_selection_strategy must be one of: "
                + ", ".join(TEARFIT_BASE_SELECTION_STRATEGIES)
            )
        if self.gap_proposal_pool not in TEARFIT_GAP_PROPOSAL_POOLS:
            raise ValueError(
                "gap_proposal_pool must be one of: "
                + ", ".join(TEARFIT_GAP_PROPOSAL_POOLS)
            )
        if self.gap_routing not in TEARFIT_GAP_ROUTING:
            raise ValueError(
                "gap_routing must be one of: " + ", ".join(TEARFIT_GAP_ROUTING)
            )
        if self.seed_strategy not in TEARFIT_SEED_STRATEGIES:
            raise ValueError(
                f"seed_strategy must be one of: {', '.join(TEARFIT_SEED_STRATEGIES)}"
            )

    @property
    def resolved_core_raw_coverage_threshold(self) -> float:
        if self.core_raw_coverage_threshold is None:
            return max(0.05, self.coverage_threshold - 0.15)
        return self.core_raw_coverage_threshold

    @classmethod
    def v4_4_1(cls) -> TearFitCoreConfig:
        """The frozen core: the disjoint-round arm of the v4.4.1 base-selection run."""

        return cls(
            algorithm="v43_routed",
            beam_width=32,
            base_selection_strategy="disjoint_round_robin",
            use_labels=False,
            candidate_time_limit_seconds=None,
            candidate_states_per_pair_score=12.691965985531159,
            partial_gap_time_limit_seconds=None,
            gap_states_per_fragment=83.33333333333333,
            partial_gap_states_per_fragment=20.833333333333332,
            cover_time_limit_seconds=None,
            cover_nodes_per_note=25_000.0,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["resolved_core_raw_coverage_threshold"] = (
            self.resolved_core_raw_coverage_threshold
        )
        return payload


@dataclass(frozen=True)
class TearFitReconstruction:
    """Truth-blind output of :func:`reconstruct_placed_fragments`."""

    config: TearFitCoreConfig
    resolved_algorithm: str
    median_fragment_fraction: float
    max_pieces: int
    expected_notes: int
    resolved_candidate_state_limit: int | None
    resolved_gap_state_limit: int | None
    resolved_partial_gap_state_limit: int | None
    resolved_cover_node_limit: int | None
    all_scores: list[TearFitEdge]
    label_filtered_scores: list[TearFitEdge]
    core_edges: list[TearFitEdge]
    generated_candidates: list[AssemblyCandidate]
    partial_core_candidates: list[AssemblyCandidate]
    candidates: list[AssemblyCandidate]
    selected: list[AssemblyCandidate]
    complete_core_keys: frozenset[tuple[str, ...]]
    gap_candidate_keys: frozenset[tuple[str, ...]]
    complete_gap_keys: frozenset[tuple[str, ...]]
    partial_gap_keys: frozenset[tuple[str, ...]]
    selected_keys: frozenset[tuple[str, ...]]
    search_stats: dict[str, dict[str, int | bool]]
    stage_timings: dict[str, float]
    edge_decisions: dict[str, int]
    candidate_decisions: dict[str, int]
    proposal_efficiency: float
    selected_solution_fingerprint: str
    candidate_provenance_fingerprint: str
    edge_intervention_applied: bool = False
    intervention_removed_edges: int = 0
    intervention_removed_core_edges: int = 0

    def candidate_source(self, fragment_ids: Iterable[str]) -> str:
        return _candidate_source(
            _candidate_key(fragment_ids),
            complete_core_keys=self.complete_core_keys,
            complete_gap_keys=self.complete_gap_keys,
            partial_gap_keys=self.partial_gap_keys,
        )

    def to_jsonable(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "resolved_algorithm": self.resolved_algorithm,
            "median_fragment_fraction": self.median_fragment_fraction,
            "max_pieces": self.max_pieces,
            "expected_notes": self.expected_notes,
            "resolved_budgets": {
                "candidate_state_limit": self.resolved_candidate_state_limit,
                "gap_state_limit": self.resolved_gap_state_limit,
                "partial_gap_state_limit": self.resolved_partial_gap_state_limit,
                "cover_node_limit": self.resolved_cover_node_limit,
            },
            "edge_intervention_applied": self.edge_intervention_applied,
            "pair_scores": len(self.all_scores),
            "core_edges": len(self.core_edges),
            "edge_decisions": dict(self.edge_decisions),
            "core_candidates": len(self.generated_candidates),
            "candidates": len(self.candidates),
            "candidate_decisions": dict(self.candidate_decisions),
            "selected": [
                {
                    "fragment_ids": list(candidate.fragment_ids),
                    "source": self.candidate_source(candidate.fragment_ids),
                    "evidence_level": candidate.evidence_level,
                    "coverage": candidate.coverage,
                    "raw_coverage": candidate.raw_coverage,
                    "score": candidate.score,
                    "evidence_score": candidate.evidence_score,
                    "gap_steps": candidate.gap_steps,
                    "labels": list(candidate.labels),
                }
                for candidate in self.selected
            ],
            "search_stats": self.search_stats,
            "stage_timings": dict(self.stage_timings),
            "selected_solution_fingerprint": self.selected_solution_fingerprint,
            "candidate_provenance_fingerprint": self.candidate_provenance_fingerprint,
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


def _smooth_noise(
    length: int, rng: np.random.Generator, roughness: float
) -> np.ndarray:
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
        split = _split_mask_once(
            masks[index], rng, roughness=roughness, min_area=min_area
        )
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


def make_fractal_tear_fragments(
    config: FractalTearConfig,
) -> tuple[np.ndarray, list[Fragment]]:
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
        if config.ensure_serial_anchor and not any(
            fragment.label for fragment in note_fragments
        ):
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


def _boundary_normals(
    mask: np.ndarray, boundary: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate outward raster normals on boundary pixels."""

    padded = np.pad(mask.astype(np.float32), 1, constant_values=0.0)
    normal_x = padded[1:-1, :-2] - padded[1:-1, 2:]
    normal_y = padded[:-2, 1:-1] - padded[2:, 1:-1]
    magnitude = np.hypot(normal_y, normal_x)
    valid = boundary & (magnitude > 0.0)
    out_y = np.zeros(mask.shape, dtype=np.float32)
    out_x = np.zeros(mask.shape, dtype=np.float32)
    out_y[valid] = normal_y[valid] / magnitude[valid]
    out_x[valid] = normal_x[valid] / magnitude[valid]
    return out_y, out_x


def _largest_connected_support(mask: np.ndarray) -> int:
    """Return the largest 8-connected component in a sparse support mask."""

    return int(_largest_component_size(np.ascontiguousarray(mask, dtype=np.bool_)))


@numba.njit(cache=True)
def _largest_component_size(mask: np.ndarray) -> int:  # pragma: no cover - compiled
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=np.bool_)
    stack = np.empty((height * width, 2), dtype=np.int64)
    best = 0
    for seed_y in range(height):
        for seed_x in range(width):
            if not mask[seed_y, seed_x] or visited[seed_y, seed_x]:
                continue
            visited[seed_y, seed_x] = True
            stack[0, 0] = seed_y
            stack[0, 1] = seed_x
            top = 1
            size = 0
            while top > 0:
                top -= 1
                y = stack[top, 0]
                x = stack[top, 1]
                size += 1
                for dy in range(-1, 2):
                    ny = y + dy
                    if ny < 0 or ny >= height:
                        continue
                    for dx in range(-1, 2):
                        nx = x + dx
                        if (dy == 0 and dx == 0) or nx < 0 or nx >= width:
                            continue
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack[top, 0] = ny
                            stack[top, 1] = nx
                            top += 1
            if size > best:
                best = size
    return best


class _NormalTable:
    """Distinct raster normals of a fragment set and their pairwise opposition.

    Placed-mask normals take only a handful of values, so each boundary pixel
    stores the index of its normal and ``opposition[i, j]`` holds
    ``-(ny_i * ny_j + nx_i * nx_j)`` evaluated in float64 exactly as it would
    be for one pixel pair.
    """

    def __init__(self, evidence: list[_CroppedBoundary]) -> None:
        keys = [self._keys(ev) for ev in evidence]
        flat = [key.ravel() for key in keys if key.size]
        self.keys = np.unique(np.concatenate(flat)) if flat else np.zeros(1, np.uint64)
        normal_y = (self.keys >> np.uint64(32)).astype(np.uint32).view(np.float32)
        normal_x = (self.keys & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.float32)
        vectors = [(float(y), float(x)) for y, x in zip(normal_y, normal_x)]
        self.is_zero = np.array([y == 0.0 and x == 0.0 for y, x in vectors], dtype=np.bool_)
        self.opposition = np.array(
            [[-(ly * ry + lx * rx) for ry, rx in vectors] for ly, lx in vectors],
            dtype=np.float64,
        ).reshape(len(vectors), len(vectors))
        self.codes = [
            np.searchsorted(self.keys, key).astype(np.int64) for key in keys
        ]

    @staticmethod
    def _keys(evidence: _CroppedBoundary) -> np.ndarray:
        return (evidence.normal_y.view(np.uint32).astype(np.uint64) << np.uint64(32)) | (
            evidence.normal_x.view(np.uint32).astype(np.uint64)
        )


@numba.njit(cache=True)
def _best_opposition(
    ys: np.ndarray,
    xs: np.ndarray,
    left_codes: np.ndarray,
    left_y0: int,
    left_x0: int,
    right_codes: np.ndarray,
    right_boundary: np.ndarray,
    right_y0: int,
    right_x0: int,
    radius: int,
    opposition: np.ndarray,
    is_zero: np.ndarray,
    best: np.ndarray,
    usable: np.ndarray,
) -> None:  # pragma: no cover - compiled
    right_y1 = right_y0 + right_boundary.shape[0]
    right_x1 = right_x0 + right_boundary.shape[1]
    for point in range(ys.shape[0]):
        y = ys[point]
        x = xs[point]
        usable[point] = False
        left_code = left_codes[y - left_y0, x - left_x0]
        if is_zero[left_code]:
            continue
        found = False
        current = 0.0
        for near_y in range(y - radius, y + radius + 1):
            if near_y < right_y0 or near_y >= right_y1:
                continue
            for near_x in range(x - radius, x + radius + 1):
                if near_x < right_x0 or near_x >= right_x1:
                    continue
                if not right_boundary[near_y - right_y0, near_x - right_x0]:
                    continue
                right_code = right_codes[near_y - right_y0, near_x - right_x0]
                if is_zero[right_code]:
                    continue
                value = opposition[left_code, right_code]
                # Keep the first of equal values, as max() over a list does.
                if not found or value > current:
                    current = value
                    found = True
        if found:
            best[point] = current
            usable[point] = True


def _matched_normal_opposition(
    table: _NormalTable,
    left: int,
    right: int,
    evidence: list[_CroppedBoundary],
    left_match: np.ndarray,
    origin: tuple[int, int],
    *,
    tolerance: int,
    sample_limit: int = 128,
) -> float:
    """Measure whether nearby outward normals point in opposite directions.

    ``left_match`` is a canvas window whose top-left pixel is ``origin``
    ``(y, x)``. Each sampled matched pixel takes its best-opposed right
    boundary normal within ``tolerance``, scanning the window row by row;
    the result is the mean over pixels that found one.
    """

    points = np.argwhere(left_match)
    if len(points) == 0:
        return 0.0
    if len(points) > sample_limit:
        points = points[np.linspace(0, len(points) - 1, sample_limit, dtype=np.int64)]
    ys = np.ascontiguousarray(points[:, 0] + origin[0], dtype=np.int64)
    xs = np.ascontiguousarray(points[:, 1] + origin[1], dtype=np.int64)
    left_ev = evidence[left]
    right_ev = evidence[right]
    best = np.empty(len(points), dtype=np.float64)
    usable = np.empty(len(points), dtype=np.bool_)
    _best_opposition(
        ys,
        xs,
        table.codes[left],
        left_ev.y0,
        left_ev.x0,
        table.codes[right],
        right_ev.boundary,
        right_ev.y0,
        right_ev.x0,
        max(0, int(tolerance)),
        table.opposition,
        table.is_zero,
        best,
        usable,
    )
    if not np.any(usable):
        return 0.0
    return float(np.mean(np.clip(best[usable], 0.0, 1.0)))


def _curvature_entropy(
    windows: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    bins: int = 8,
) -> float:
    """Raster curvature proxy from the entropy of matched boundary normals.

    ``windows`` holds ``(normal_y, normal_x, matched)`` arrays over one canvas
    window per side, left side first.
    """

    angles: list[np.ndarray] = []
    for normal_y, normal_x, matched in windows:
        valid = matched & ((normal_y != 0.0) | (normal_x != 0.0))
        if np.any(valid):
            angles.append(
                np.mod(
                    np.arctan2(normal_y[valid], normal_x[valid]),
                    2.0 * np.pi,
                )
            )
    if not angles:
        return 0.0
    values = np.concatenate(angles)
    counts = _uniform_histogram(values, bins, 0.0, 2.0 * np.pi)
    probabilities = counts[counts > 0].astype(np.float64)
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / float(np.log(bins)) if bins > 1 else 0.0


_HISTOGRAM_EDGES: dict[tuple[str, int, float, float], np.ndarray] = {}


def _uniform_histogram(
    values: np.ndarray, bins: int, first: float, last: float
) -> np.ndarray:
    """``np.histogram(values, bins, range=(first, last))[0]`` for 1-D input.

    np.histogram places each value in the bin whose computed edges bracket it
    (the last bin is closed); binary search over the same edges gives the same
    counts without its per-call overhead.
    """

    key = (values.dtype.str, bins, first, last)
    edges = _HISTOGRAM_EDGES.get(key)
    if edges is None:
        edges = np.histogram_bin_edges(values[:0], bins=bins, range=(first, last))
        _HISTOGRAM_EDGES[key] = edges
    kept = values[(values >= first) & (values <= last)]
    index = np.searchsorted(edges, kept, side="right") - 1
    index[index == bins] = bins - 1
    return np.bincount(index, minlength=bins)


def _clip(value: float, low: float, high: float) -> float:
    """``float(np.clip(value, low, high))`` for one number, without NumPy overhead.

    Like ``np.clip`` it keeps NaN and the sign of a zero that equals ``low``.
    """

    return float(min(max(value, low), high))


def tear_match_effectiveness(
    *,
    contiguous_pixels: int,
    overlap_pixels: int,
    left_hits: int,
    right_hits: int,
    overlap_ratio: float = 1.0,
    normal_opposition: float,
    curvature_entropy: float,
    expected_accidental_hits: float,
    mask_overlap_pixels: int = 0,
) -> float:
    """Benefit-to-damage score for one placed tear-edge match.

    The numerator rewards one continuous, bidirectional seam with opposing
    normals. The denominator charges scattered hits, one-sided support, actual
    mask overlap, and the random coincidence expected from boundary density.
    This is inspired by tolerance-derived restoration effectiveness, but it is
    a MoneyRepair tear-matching metric rather than the paper's physical formula.
    """

    if (
        min(
            contiguous_pixels,
            overlap_pixels,
            left_hits,
            right_hits,
            mask_overlap_pixels,
        )
        < 0
    ):
        raise ValueError("tear-match counts must be non-negative")
    if expected_accidental_hits < 0.0:
        raise ValueError("expected_accidental_hits must be non-negative")
    if overlap_ratio < 0.0:
        raise ValueError("overlap_ratio must be non-negative")
    if overlap_pixels == 0:
        return 0.0
    balance = overlap_pixels / float(max(1, max(left_hits, right_hits)))
    scattered = max(0, overlap_pixels - contiguous_pixels)
    one_sided = abs(left_hits - right_hits)
    geometry_factor = 0.5 + 0.5 * _clip(normal_opposition, 0.0, 1.0)
    specificity_factor = 0.75 + 0.5 * _clip(curvature_entropy, 0.0, 1.0)
    explained_perimeter_factor = _clip(overlap_ratio / 0.30, 0.25, 1.5)
    benefit = (
        contiguous_pixels
        * balance
        * geometry_factor
        * specificity_factor
        * explained_perimeter_factor
    )
    damage = (
        1.0
        + scattered
        + one_sided
        + 2.0 * mask_overlap_pixels
        + expected_accidental_hits
    )
    return float(benefit / damage)


def tear_boundary_evidence(
    fragments: list[Fragment], *, tolerance: int = 2
) -> list[TearBoundaryEvidence]:
    evidence: list[TearBoundaryEvidence] = []
    for fragment in fragments:
        boundary = tear_boundary(fragment.mask)
        dilated = _dilate(boundary, tolerance)
        normal_y, normal_x = _boundary_normals(fragment.mask, boundary)
        evidence.append(
            TearBoundaryEvidence(
                boundary=np.flatnonzero(boundary),
                dilated_boundary=np.flatnonzero(dilated),
                bbox=_bbox(dilated),
                boundary_mask=boundary,
                dilated_boundary_mask=dilated,
                normal_y=normal_y,
                normal_x=normal_x,
            )
        )
    return evidence


def _cropped_boundary_evidence(
    fragments: list[Fragment], *, tolerance: int = 2
) -> list[_CroppedBoundary]:
    """Build :func:`tear_boundary_evidence` per fragment and keep only its crop."""

    evidence: list[_CroppedBoundary] = []
    for fragment in fragments:
        boundary = tear_boundary(fragment.mask)
        dilated = _dilate(boundary, tolerance)
        normal_y, normal_x = _boundary_normals(fragment.mask, boundary)
        x0, y0, x1, y1 = _bbox(dilated)
        evidence.append(
            _CroppedBoundary(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                boundary=boundary[y0:y1, x0:x1].copy(),
                dilated=dilated[y0:y1, x0:x1].copy(),
                normal_y=normal_y[y0:y1, x0:x1].copy(),
                normal_x=normal_x[y0:y1, x0:x1].copy(),
                boundary_pixels=int(np.count_nonzero(boundary)),
                dilated_pixels=int(np.count_nonzero(dilated)),
            )
        )
    return evidence


def _pack_crops(
    crops: list[tuple[int, int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack ``(y0, x0, crop)`` boolean crops into one flat buffer.

    Returns ``(offsets, shapes, origins, buffer)`` with one ``(height,
    width)`` and one ``(y0, x0)`` row per crop.
    """

    sizes = np.array([crop.size for _y0, _x0, crop in crops], dtype=np.int64)
    offsets = np.zeros(len(crops), dtype=np.int64)
    if len(crops) > 1:
        offsets[1:] = np.cumsum(sizes)[:-1]
    shapes = np.array([crop.shape for _y0, _x0, crop in crops], dtype=np.int64).reshape(-1, 2)
    origins = np.array([(y0, x0) for y0, x0, _crop in crops], dtype=np.int64).reshape(-1, 2)
    buffer = (
        np.concatenate([crop.ravel() for _y0, _x0, crop in crops]).astype(np.bool_)
        if crops
        else np.zeros(0, dtype=np.bool_)
    )
    return offsets, shapes, origins, buffer


def _box_intersection(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """Intersect two ``(x0, y0, x1, y1)`` boxes; ``None`` when they are disjoint."""

    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


@numba.njit(cache=True)
def _crop_overlap_count(
    canvas: np.ndarray, crop: np.ndarray, y0: int, x0: int
) -> int:  # pragma: no cover - compiled
    """Count pixels set in ``crop`` and in the window of ``canvas`` at ``(y0, x0)``."""

    total = 0
    for y in range(crop.shape[0]):
        for x in range(crop.shape[1]):
            if crop[y, x] and canvas[y0 + y, x0 + x]:
                total += 1
    return total


@numba.njit(cache=True)
def _packed_crop_counts(
    canvas: np.ndarray,
    indices: np.ndarray,
    offsets: np.ndarray,
    shapes: np.ndarray,
    origins: np.ndarray,
    buffer: np.ndarray,
    out: np.ndarray,
) -> None:  # pragma: no cover - compiled
    """For each fragment in ``indices``, count its packed crop pixels set on ``canvas``."""

    for slot in range(indices.shape[0]):
        index = indices[slot]
        height = shapes[index, 0]
        width = shapes[index, 1]
        y0 = origins[index, 0]
        x0 = origins[index, 1]
        base = offsets[index]
        total = 0
        for y in range(height):
            row = base + y * width
            for x in range(width):
                if buffer[row + x] and canvas[y0 + y, x0 + x]:
                    total += 1
        out[slot] = total


@numba.njit(cache=True)
def _packed_point_counts(
    canvas: np.ndarray,
    indices: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    points: np.ndarray,
    out: np.ndarray,
) -> None:  # pragma: no cover - compiled
    """For each fragment in ``indices``, count its packed flat pixel positions set on ``canvas``."""

    for slot in range(indices.shape[0]):
        index = indices[slot]
        base = offsets[index]
        total = 0
        for position in range(base, base + lengths[index]):
            if canvas[points[position]]:
                total += 1
        out[slot] = total


@numba.njit(cache=True)
def _packed_overlap_counts(
    index: int,
    others: np.ndarray,
    offsets: np.ndarray,
    shapes: np.ndarray,
    origins: np.ndarray,
    buffer: np.ndarray,
    out: np.ndarray,
) -> None:  # pragma: no cover - compiled
    """Count pixels set in both fragment ``index``'s packed crop and each other crop."""

    y0 = origins[index, 0]
    x0 = origins[index, 1]
    y1 = y0 + shapes[index, 0]
    x1 = x0 + shapes[index, 1]
    for slot in range(others.shape[0]):
        other = others[slot]
        oy0 = origins[other, 0]
        ox0 = origins[other, 1]
        total = 0
        for y in range(max(y0, oy0), min(y1, oy0 + shapes[other, 0])):
            row = offsets[index] + (y - y0) * shapes[index, 1] - x0
            other_row = offsets[other] + (y - oy0) * shapes[other, 1] - ox0
            for x in range(max(x0, ox0), min(x1, ox0 + shapes[other, 1])):
                if buffer[row + x] and buffer[other_row + x]:
                    total += 1
        out[slot] = total


@numba.njit(cache=True)
def _packed_cross_counts(
    index: int,
    others: np.ndarray,
    offsets: np.ndarray,
    shapes: np.ndarray,
    origins: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    forward: np.ndarray,
    backward: np.ndarray,
) -> None:  # pragma: no cover - compiled
    """``|first_i & second_j|`` and ``|first_j & second_i|`` for each other crop ``j``.

    ``first`` and ``second`` share one packing (the same boxes).
    """

    y0 = origins[index, 0]
    x0 = origins[index, 1]
    y1 = y0 + shapes[index, 0]
    x1 = x0 + shapes[index, 1]
    for slot in range(others.shape[0]):
        other = others[slot]
        oy0 = origins[other, 0]
        ox0 = origins[other, 1]
        there = 0
        back = 0
        for y in range(max(y0, oy0), min(y1, oy0 + shapes[other, 0])):
            row = offsets[index] + (y - y0) * shapes[index, 1] - x0
            other_row = offsets[other] + (y - oy0) * shapes[other, 1] - ox0
            for x in range(max(x0, ox0), min(x1, ox0 + shapes[other, 1])):
                if first[row + x] and second[other_row + x]:
                    there += 1
                if first[other_row + x] and second[row + x]:
                    back += 1
        forward[slot] = there
        backward[slot] = back


@numba.njit(cache=True)
def _packed_crops_meet(
    index: int,
    others: np.ndarray,
    offsets: np.ndarray,
    shapes: np.ndarray,
    origins: np.ndarray,
    buffer: np.ndarray,
    out: np.ndarray,
) -> None:  # pragma: no cover - compiled
    """Whether fragment ``index``'s packed crop shares a set pixel with each other crop."""

    y0 = origins[index, 0]
    x0 = origins[index, 1]
    y1 = y0 + shapes[index, 0]
    x1 = x0 + shapes[index, 1]
    for slot in range(others.shape[0]):
        other = others[slot]
        oy0 = origins[other, 0]
        ox0 = origins[other, 1]
        top_y = max(y0, oy0)
        bottom_y = min(y1, oy0 + shapes[other, 0])
        left_x = max(x0, ox0)
        right_x = min(x1, ox0 + shapes[other, 1])
        found = False
        for y in range(top_y, bottom_y):
            row = offsets[index] + (y - y0) * shapes[index, 1]
            other_row = offsets[other] + (y - oy0) * shapes[other, 1]
            for x in range(left_x, right_x):
                if buffer[row + x - x0] and buffer[other_row + x - ox0]:
                    found = True
                    break
            if found:
                break
        out[slot] = found


@numba.njit(cache=True)
def _member_overlap_totals(
    members: np.ndarray,
    offsets: np.ndarray,
    partners: np.ndarray,
    pixels: np.ndarray,
    out: np.ndarray,
) -> None:  # pragma: no cover - compiled
    """Add every member's pairwise mask-overlap row to ``out`` (zeroed by the caller)."""

    for slot in range(members.shape[0]):
        member = members[slot]
        for position in range(offsets[member], offsets[member + 1]):
            out[partners[position]] += pixels[position]


def _labels_compatible(left: str | None, right: str | None) -> bool:
    return not left or not right or left == right


def score_absolute_tear_pairs(
    fragments: list[Fragment],
    *,
    tolerance: int = 2,
    min_overlap_pixels: int = 1,
    max_pair_overlap_pixels: int = 0,
    use_labels: bool = True,
    scoring: str = "overlap",
    min_effectiveness: float = 1.0,
    automatic_effectiveness: float = 2.0,
    min_contiguous_pixels: int = 3,
    automatic_contiguous_pixels: int = 5,
) -> tuple[list[TearFitEdge], list[TearFitEdge]]:
    """Score torn-edge coincidence in the placed coordinate frame.

    Returns ``(all_scored_pairs, accepted_edges)``. ``overlap`` preserves the
    historical fixed-pixel baseline. ``effectiveness`` accepts short seams when
    their continuous, bidirectional geometry outweighs scattered and accidental
    support; accepted edges are labelled ``automatic`` or ``review``.
    """

    if scoring not in TEARFIT_EDGE_SCORING:
        raise ValueError(f"scoring must be one of: {', '.join(TEARFIT_EDGE_SCORING)}")
    if min_effectiveness < 0.0 or automatic_effectiveness < min_effectiveness:
        raise ValueError("effectiveness thresholds must be ordered and non-negative")
    if min_contiguous_pixels < 1 or automatic_contiguous_pixels < min_contiguous_pixels:
        raise ValueError("contiguous-pixel thresholds must be positive and ordered")
    if any(fragment.mask.shape != fragments[0].mask.shape for fragment in fragments):
        raise ValueError("all fragment masks must share one coordinate frame")
    evidence = _cropped_boundary_evidence(fragments, tolerance=tolerance)
    normals = _NormalTable(evidence) if scoring == "effectiveness" else None
    boxes = np.array(
        [(ev.x0, ev.y0, ev.x1, ev.y1) for ev in evidence], dtype=np.int64
    ).reshape(len(fragments), 4)
    mask_boxes = [_bbox(fragment.mask) for fragment in fragments]
    mask_offsets, mask_shapes, mask_origins, mask_buffer = _pack_crops(
        [
            (y0, x0, fragment.mask[y0:y1, x0:x1])
            for fragment, (x0, y0, x1, y1) in zip(fragments, mask_boxes)
        ]
    )
    edge_offsets, edge_shapes, edge_origins, boundary_buffer = _pack_crops(
        [(ev.y0, ev.x0, ev.boundary) for ev in evidence]
    )
    dilated_buffer = _pack_crops([(ev.y0, ev.x0, ev.dilated) for ev in evidence])[3]
    pose_uncertainties: dict[int, float] = {}

    def pose_uncertainty_of(index: int) -> float:
        value = pose_uncertainties.get(index)
        if value is None:
            value = _normalised_pose_uncertainty(fragments[index], tolerance)
            pose_uncertainties[index] = value
        return value

    all_scores: list[TearFitEdge] = []
    accepted: list[TearFitEdge] = []
    for i in range(len(fragments)):
        left_ev = evidence[i]
        # Only pairs whose dilated tear boundaries share a bounding box can
        # touch; this prefilter gives the same pairs, in the same order, as
        # testing every j > i.
        rest = boxes[i + 1 :]
        partners = np.flatnonzero(
            (boxes[i, 0] < rest[:, 2])
            & (boxes[i, 2] > rest[:, 0])
            & (boxes[i, 1] < rest[:, 3])
            & (boxes[i, 3] > rest[:, 1])
        ) + (i + 1)
        if use_labels:
            partners = partners[
                np.array(
                    [
                        _labels_compatible(fragments[i].label, fragments[j].label)
                        for j in partners.tolist()
                    ],
                    dtype=bool,
                ).reshape(-1)
            ]
        # The label and mask-overlap filters have no side effects, so both
        # counts for every partner come from one compiled pass each.
        mask_overlaps = np.empty(len(partners), dtype=np.int64)
        _packed_overlap_counts(
            i, partners, mask_offsets, mask_shapes, mask_origins, mask_buffer, mask_overlaps
        )
        if max_pair_overlap_pixels >= 0:
            kept = mask_overlaps <= max_pair_overlap_pixels
            partners = partners[kept]
            mask_overlaps = mask_overlaps[kept]
        hits_forward = np.empty(len(partners), dtype=np.int64)
        hits_backward = np.empty(len(partners), dtype=np.int64)
        _packed_cross_counts(
            i,
            partners,
            edge_offsets,
            edge_shapes,
            edge_origins,
            boundary_buffer,
            dilated_buffer,
            hits_forward,
            hits_backward,
        )
        for slot, j in enumerate(partners.tolist()):
            mask_overlap = int(mask_overlaps[slot])
            right_ev = evidence[j]
            left_hits = int(hits_forward[slot])
            right_hits = int(hits_backward[slot])
            overlap = min(left_hits, right_hits)
            denom = max(1, min(left_ev.boundary_pixels, right_ev.boundary_pixels))
            contiguous = 0
            continuity_ratio = 0.0
            balance = overlap / float(max(1, max(left_hits, right_hits)))
            normal_opposition = 0.0
            curvature_entropy = 0.0
            expected_accidental_hits = 0.0
            pose_uncertainty = 0.0
            effectiveness = 0.0
            evidence_level = "insufficient-evidence"
            if normals is not None and overlap > 0:
                window = _box_intersection(
                    (left_ev.x0, left_ev.y0, left_ev.x1, left_ev.y1),
                    (right_ev.x0, right_ev.y0, right_ev.x1, right_ev.y1),
                )
                assert window is not None
                left_match = left_ev.crop(left_ev.boundary, window) & right_ev.crop(
                    right_ev.dilated, window
                )
                right_match = right_ev.crop(right_ev.boundary, window) & left_ev.crop(
                    left_ev.dilated, window
                )
                contiguous = max(
                    _largest_connected_support(left_match),
                    _largest_connected_support(right_match),
                )
                continuity_ratio = contiguous / float(max(1, overlap))
                normal_opposition = _matched_normal_opposition(
                    normals,
                    i,
                    j,
                    evidence,
                    left_match,
                    (window[1], window[0]),
                    tolerance=tolerance,
                )
                curvature_entropy = _curvature_entropy(
                    (
                        (
                            left_ev.crop(left_ev.normal_y, window),
                            left_ev.crop(left_ev.normal_x, window),
                            left_match,
                        ),
                        (
                            right_ev.crop(right_ev.normal_y, window),
                            right_ev.crop(right_ev.normal_x, window),
                            right_match,
                        ),
                    )
                )
                total_pixels = float(fragments[i].mask.size)
                expected_left = (
                    left_ev.boundary_pixels * right_ev.dilated_pixels / total_pixels
                )
                expected_right = (
                    right_ev.boundary_pixels * left_ev.dilated_pixels / total_pixels
                )
                expected_accidental_hits = 0.5 * (expected_left + expected_right)
                effectiveness = tear_match_effectiveness(
                    contiguous_pixels=contiguous,
                    overlap_pixels=overlap,
                    left_hits=left_hits,
                    right_hits=right_hits,
                    overlap_ratio=overlap / float(denom),
                    normal_opposition=normal_opposition,
                    curvature_entropy=curvature_entropy,
                    expected_accidental_hits=expected_accidental_hits,
                    mask_overlap_pixels=mask_overlap,
                )
                pose_uncertainty = max(pose_uncertainty_of(i), pose_uncertainty_of(j))
                effectiveness *= 1.0 - 0.5 * pose_uncertainty
                if (
                    effectiveness >= automatic_effectiveness
                    and contiguous >= automatic_contiguous_pixels
                ):
                    evidence_level = "automatic"
                elif (
                    effectiveness >= min_effectiveness
                    and contiguous >= min_contiguous_pixels
                ):
                    evidence_level = "review"
            elif overlap >= min_overlap_pixels:
                evidence_level = "automatic"
            edge = TearFitEdge(
                left=i,
                right=j,
                overlap_pixels=overlap,
                left_hits=left_hits,
                right_hits=right_hits,
                overlap_ratio=overlap / float(denom),
                contiguous_pixels=contiguous,
                continuity_ratio=continuity_ratio,
                bidirectional_balance=balance,
                normal_opposition=normal_opposition,
                curvature_entropy=curvature_entropy,
                expected_accidental_hits=expected_accidental_hits,
                mask_overlap_pixels=mask_overlap,
                pose_uncertainty=pose_uncertainty,
                effectiveness=effectiveness,
                evidence_level=evidence_level,
            )
            all_scores.append(edge)
            if evidence_level != "insufficient-evidence":
                accepted.append(edge)
    return all_scores, accepted


def _edge_graph(
    edges: Iterable[TearFitEdge], count: int
) -> tuple[list[dict[int, TearFitEdge]], dict[tuple[int, int], TearFitEdge]]:
    graph: list[dict[int, TearFitEdge]] = [dict() for _ in range(count)]
    lookup: dict[tuple[int, int], TearFitEdge] = {}
    for edge in edges:
        graph[edge.left][edge.right] = edge
        graph[edge.right][edge.left] = edge
        lookup[(min(edge.left, edge.right), max(edge.left, edge.right))] = edge
    return graph, lookup


def diagnose_true_core_connectivity(
    fragments: list[Fragment],
    scored_edges: list[TearFitEdge],
    *,
    minimum_raw_coverage: float,
) -> dict:
    """Measure simulator-truth connectivity of the automatic core-edge graph."""

    if not (0.0 < minimum_raw_coverage <= 1.0):
        raise ValueError("minimum_raw_coverage must be in (0, 1]")
    if any("note_id" not in fragment.meta for fragment in fragments):
        raise ValueError("core connectivity requires simulated note_id ground truth")

    note_indices: dict[str, list[int]] = {}
    for index, fragment in enumerate(fragments):
        note_indices.setdefault(str(fragment.meta["note_id"]), []).append(index)
    adjacency: list[set[int]] = [set() for _ in fragments]
    automatic_true_edges = 0
    for edge in scored_edges:
        if edge.evidence_level != "automatic":
            continue
        left_note = str(fragments[edge.left].meta["note_id"])
        right_note = str(fragments[edge.right].meta["note_id"])
        if left_note != right_note:
            continue
        adjacency[edge.left].add(edge.right)
        adjacency[edge.right].add(edge.left)
        automatic_true_edges += 1

    total_area = fragments[0].mask.size if fragments else 0
    rows: list[dict] = []
    for note_id, indices in sorted(note_indices.items()):
        remaining = set(indices)
        components: list[list[int]] = []
        while remaining:
            start = min(remaining)
            stack = [start]
            remaining.remove(start)
            component: list[int] = []
            while stack:
                current = stack.pop()
                component.append(current)
                neighbours = adjacency[current] & remaining
                remaining.difference_update(neighbours)
                stack.extend(sorted(neighbours, reverse=True))
            components.append(component)
        component_rows = []
        for component in components:
            union = np.any(
                np.stack([fragments[index].mask for index in component]), axis=0
            )
            component_rows.append(
                {
                    "pieces": len(component),
                    "raw_coverage": int(union.sum()) / float(max(1, total_area)),
                }
            )
        component_rows.sort(
            key=lambda item: (-item["raw_coverage"], -item["pieces"])
        )
        best = component_rows[0]
        rows.append(
            {
                "note_id": note_id,
                "components": len(component_rows),
                "largest_component_pieces": best["pieces"],
                "largest_component_raw_coverage": best["raw_coverage"],
                "recordable_at_threshold": (
                    best["raw_coverage"] >= minimum_raw_coverage
                ),
            }
        )
    return {
        "simulation_truth_only": True,
        "minimum_raw_coverage": minimum_raw_coverage,
        "automatic_true_edges": automatic_true_edges,
        "recordable_notes": sum(row["recordable_at_threshold"] for row in rows),
        "unrecordable_notes": sum(
            not row["recordable_at_threshold"] for row in rows
        ),
        "notes": rows,
    }


def _group_labels(fragments: list[Fragment], indices: Iterable[int]) -> tuple[str, ...]:
    return tuple(
        sorted({label for index in indices if (label := fragments[index].label)})
    )


def _labels_ok(fragments: list[Fragment], indices: Iterable[int]) -> bool:
    return len(_group_labels(fragments, indices)) <= 1


def _group_masks_ok(
    fragments: list[Fragment], indices: Iterable[int], max_overlap_pixels: int
) -> tuple[bool, np.ndarray, int]:
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


def _support_for_state(
    state: frozenset[int], edge_lookup: dict[tuple[int, int], TearFitEdge]
) -> int:
    members = sorted(state)
    support = 0
    for pos, left in enumerate(members):
        for right in members[pos + 1 :]:
            edge = edge_lookup.get((left, right))
            if edge is not None:
                support += edge.overlap_pixels
    return support


def _candidate_key(fragment_ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(fragment_ids))


def _fingerprint_records(records: Iterable[str]) -> str:
    payload = "\n".join(sorted(records)).encode("utf-8")
    return sha256(payload).hexdigest()


def _true_note_candidate_keys(
    fragments: list[Fragment],
) -> dict[tuple[str, ...], str]:
    ids_by_note: dict[str, list[str]] = {}
    for fragment in fragments:
        note_id = fragment.meta.get("note_id")
        if note_id:
            ids_by_note.setdefault(str(note_id), []).append(fragment.id)
    return {
        _candidate_key(fragment_ids): note_id
        for note_id, fragment_ids in ids_by_note.items()
    }


def _resolve_workload_budget(
    absolute_limit: int | None,
    units: int,
    per_unit: float | None,
    *,
    name: str,
) -> int | None:
    if per_unit is None:
        return absolute_limit
    if absolute_limit is not None:
        raise ValueError(f"{name} absolute and per-unit limits are mutually exclusive")
    if per_unit <= 0.0:
        raise ValueError(f"{name} per-unit limit must be positive")
    return max(1, ceil(per_unit * units))


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _normalise_candidate_keys(
    values: Iterable[Iterable[str]] | None,
) -> set[tuple[str, ...]]:
    if values is None:
        return set()
    return {_candidate_key(value) for value in values}


def _normalise_index_pairs(
    values: Iterable[tuple[str, str]] | None,
    fragment_id_to_index: dict[str, int],
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    if values is None:
        return pairs
    for left_id, right_id in values:
        if left_id == right_id:
            continue
        left = fragment_id_to_index.get(left_id)
        right = fragment_id_to_index.get(right_id)
        if left is None or right is None:
            continue
        pairs.add((left, right) if left <= right else (right, left))
    return pairs


def _state_has_pair(state: frozenset[int], pair: tuple[int, int]) -> bool:
    return pair[0] in state and pair[1] in state


def _state_has_forbidden_pair(
    state: frozenset[int], pairs: set[tuple[int, int]]
) -> bool:
    return any(_state_has_pair(state, pair) for pair in pairs)


def _forced_pairs_satisfied(state: frozenset[int], pairs: set[tuple[int, int]]) -> bool:
    for left, right in pairs:
        if (left in state) != (right in state):
            return False
    return True


def _pair_preference_bonus(
    state: frozenset[int], pairs: set[tuple[int, int]], bonus: float
) -> float:
    if not pairs:
        return 0.0
    return bonus * sum(1 for pair in pairs if _state_has_pair(state, pair))


def _resolve_seed_strategy(seed_strategy: str, require_anchor: bool | None) -> str:
    if require_anchor is not None:
        return "anchor_priority"
    if seed_strategy not in TEARFIT_SEED_STRATEGIES:
        raise ValueError(
            f"seed_strategy must be one of: {', '.join(TEARFIT_SEED_STRATEGIES)}"
        )
    return seed_strategy


def _seed_order(fragments: list[Fragment], seed_strategy: str) -> list[int]:
    labelled = [index for index, fragment in enumerate(fragments) if fragment.label]
    unlabelled = [
        index for index, fragment in enumerate(fragments) if not fragment.label
    ]
    if seed_strategy == "anchor_only":
        return labelled
    if seed_strategy == "anchor_priority":
        return labelled + unlabelled
    if seed_strategy == "all":
        return list(range(len(fragments)))
    raise ValueError(
        f"seed_strategy must be one of: {', '.join(TEARFIT_SEED_STRATEGIES)}"
    )


def _normalised_pose_uncertainty(fragment: Fragment, tolerance: int) -> float:
    """Read optional locator uncertainty from fragment metadata on a 0..1 scale."""

    sigma_x = float(
        fragment.meta.get("sigma_x", fragment.meta.get("pose_sigma_x", 0.0)) or 0.0
    )
    sigma_y = float(
        fragment.meta.get("sigma_y", fragment.meta.get("pose_sigma_y", 0.0)) or 0.0
    )
    sigma_theta = abs(
        float(
            fragment.meta.get("sigma_theta", fragment.meta.get("pose_sigma_theta", 0.0))
            or 0.0
        )
    )
    sigma_scale = abs(
        float(
            fragment.meta.get("sigma_scale", fragment.meta.get("pose_sigma_scale", 0.0))
            or 0.0
        )
    )
    translation = np.hypot(sigma_x, sigma_y) / float(max(1, 2 * tolerance))
    rotation = sigma_theta / 5.0
    scale = sigma_scale / 0.02
    return float(np.clip((translation + rotation + scale) / 3.0, 0.0, 1.0))


def score_fragment_against_assembly(
    fragments: list[Fragment],
    state: frozenset[int],
    union_mask: np.ndarray,
    candidate_index: int,
    edge_lookup: dict[tuple[int, int], TearFitEdge],
    boundary_evidence: list[TearBoundaryEvidence],
    *,
    tolerance: int = 2,
    min_score: float = 0.35,
    automatic_score: float = 0.55,
) -> GroupGapEvidence:
    """Score one fragment against the full boundary of a partial assembly."""

    fragment = fragments[candidate_index]
    candidate_boundary = boundary_evidence[candidate_index].boundary_mask
    assembly_boundary = _dilate(tear_boundary(union_mask), tolerance)
    return _group_gap_evidence(
        candidate_index,
        overlap=int(np.count_nonzero(union_mask & fragment.mask)),
        area=fragment.area,
        matched_pixels=int(np.count_nonzero(candidate_boundary & assembly_boundary)),
        boundary_pixels=int(np.count_nonzero(candidate_boundary)),
        supporting=_supporting_edges(state, candidate_index, edge_lookup),
        pose_uncertainty=_normalised_pose_uncertainty(fragment, tolerance),
        min_score=min_score,
        automatic_score=automatic_score,
    )


def _supporting_edges(
    state: frozenset[int],
    candidate_index: int,
    edge_lookup: dict[tuple[int, int], TearFitEdge],
) -> list[TearFitEdge]:
    """Seams from state members to the candidate that can support a gap step."""

    supporting: list[TearFitEdge] = []
    for member in state:
        edge = edge_lookup.get(
            (min(member, candidate_index), max(member, candidate_index))
        )
        if (
            edge is not None
            and edge.contiguous_pixels >= 2
            and edge.effectiveness > 0.0
        ):
            supporting.append(edge)
    return supporting


def _supporting_from_adjacency(
    state: frozenset[int], adjacency: dict[int, TearFitEdge]
) -> list[TearFitEdge]:
    """:func:`_supporting_edges` from the candidate's row of an edge graph.

    The graph row holds the same edge objects as the lookup it was built
    with, and members are visited in the same state order.
    """

    supporting: list[TearFitEdge] = []
    for member in state:
        edge = adjacency.get(member)
        if (
            edge is not None
            and edge.contiguous_pixels >= 2
            and edge.effectiveness > 0.0
        ):
            supporting.append(edge)
    return supporting


def _group_gap_evidence(
    candidate_index: int,
    *,
    overlap: int,
    area: int,
    matched_pixels: int,
    boundary_pixels: int,
    supporting: list[TearFitEdge],
    pose_uncertainty: float,
    min_score: float,
    automatic_score: float,
) -> GroupGapEvidence:
    """Whole-assembly gap score from the pixel counts of one placement."""

    gap_fill_pixels = max(0, area - overlap)
    gap_fill_ratio = gap_fill_pixels / float(max(1, area))
    matched_ratio = matched_pixels / float(max(1, boundary_pixels))
    unmatched_ratio = 1.0 - matched_ratio
    seam_count = len(supporting)
    edge_strength = (
        float(
            np.mean(
                [edge.effectiveness / (1.0 + edge.effectiveness) for edge in supporting]
            )
        )
        if supporting
        else 0.0
    )
    multi_seam_support = min(1.0, seam_count / 2.0)
    overlap_ratio = overlap / float(max(1, area))
    score = (
        0.15 * gap_fill_ratio
        + 0.25 * multi_seam_support
        + 0.25 * matched_ratio
        + 0.25 * edge_strength
        - 0.35 * overlap_ratio
        - 0.10 * unmatched_ratio
        - 0.15 * pose_uncertainty
    )
    if score >= automatic_score and seam_count >= 2:
        evidence_level = "automatic"
    elif score >= min_score and matched_pixels >= 2:
        evidence_level = "review"
    else:
        evidence_level = "insufficient-evidence"
    return GroupGapEvidence(
        fragment=candidate_index,
        seam_count=seam_count,
        matched_pixels=matched_pixels,
        gap_fill_pixels=gap_fill_pixels,
        gap_fill_ratio=gap_fill_ratio,
        overlap_pixels=overlap,
        unmatched_perimeter_ratio=unmatched_ratio,
        pose_uncertainty=pose_uncertainty,
        score=float(score),
        evidence_level=evidence_level,
    )


class _AssemblyGeometry:
    """Per-fragment pixel data that the assembly searches reuse across states.

    Everything is derived from the fragment masks alone and computed at most
    once per fragment, so the searches see exactly the values they would
    recompute from full-canvas arrays at every step.
    """

    def __init__(
        self, fragments: list[Fragment], *, tolerance: int, gap_fill_radius: int
    ) -> None:
        self.fragments = fragments
        self.tolerance = tolerance
        self.gap_fill_radius = gap_fill_radius
        self.areas = [fragment.area for fragment in fragments]
        self._area_array = np.asarray(self.areas, dtype=np.int64)
        self.mask_boxes = [_bbox(fragment.mask) for fragment in fragments]
        self._box_array = np.array(self.mask_boxes, dtype=np.int64).reshape(-1, 4)
        self._overlap_bits: dict[int, int] = {}
        self._mask_crops: dict[int, np.ndarray] = {}
        self._boundaries: dict[int, _CroppedBoundary] = {}
        self._pose_uncertainty: dict[int, float] = {}
        self._dilated_masks: dict[
            int, tuple[tuple[int, int, int, int], np.ndarray, int]
        ] = {}
        self._masks_packed: tuple[np.ndarray, ...] | None = None
        self._boundary_points: tuple[np.ndarray, ...] | None = None
        self._pose_values: np.ndarray | None = None
        self._pair_overlaps: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._dilated_packed: tuple[np.ndarray, ...] | None = None

    def serves(
        self, fragments: list[Fragment], *, tolerance: int, gap_fill_radius: int
    ) -> bool:
        """Whether these caches were built for exactly these fragments and radii."""

        return (
            self.fragments is fragments
            and self.tolerance == tolerance
            and self.gap_fill_radius == gap_fill_radius
        )

    def _box_hits(self, index: int) -> np.ndarray:
        """Other fragments whose mask bounding boxes intersect fragment ``index``'s."""

        x0, y0, x1, y1 = self.mask_boxes[index]
        boxes = self._box_array
        hits = np.flatnonzero(
            (boxes[:, 0] < x1)
            & (boxes[:, 2] > x0)
            & (boxes[:, 1] < y1)
            & (boxes[:, 3] > y0)
        )
        return hits[hits != index]

    def _pose_array(self) -> np.ndarray:
        """:meth:`pose_uncertainty` of every fragment."""

        if self._pose_values is None:
            self._pose_values = np.array(
                [self.pose_uncertainty(index) for index in range(len(self.fragments))],
                dtype=np.float64,
            )
        return self._pose_values

    def _pair_overlap_rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nonzero pairwise mask overlaps in CSR form: ``(offsets, partners, pixels)``.

        Row ``i`` lists every other fragment whose mask shares pixels with
        fragment ``i``, and how many it shares.
        """

        if self._pair_overlaps is None:
            offsets, shapes, origins, buffer = self._mask_pack()
            partner_rows: list[np.ndarray] = []
            pixel_rows: list[np.ndarray] = []
            for index in range(len(self.fragments)):
                hits = self._box_hits(index)
                pixels = np.empty(len(hits), dtype=np.int64)
                _packed_overlap_counts(index, hits, offsets, shapes, origins, buffer, pixels)
                shared = pixels > 0
                partner_rows.append(hits[shared].astype(np.int32))
                pixel_rows.append(pixels[shared].astype(np.int32))
            row_offsets = np.zeros(len(partner_rows) + 1, dtype=np.int64)
            row_offsets[1:] = np.cumsum([len(row) for row in partner_rows])
            empty = np.zeros(0, dtype=np.int32)
            self._pair_overlaps = (
                row_offsets,
                np.concatenate(partner_rows) if partner_rows else empty,
                np.concatenate(pixel_rows) if pixel_rows else empty,
            )
        return self._pair_overlaps

    def _pack_boundary_points(self) -> tuple[np.ndarray, ...]:
        """Every fragment's tear-boundary pixels as flat canvas positions.

        Returns ``(offsets, lengths, points)``; counting a canvas at these
        positions equals counting it against the boundary crop.
        """

        if self._boundary_points is None:
            width = self.fragments[0].mask.shape[1]
            lists = []
            for index in range(len(self.fragments)):
                boundary = self.boundary(index)
                ys, xs = np.nonzero(boundary.boundary)
                lists.append((ys + boundary.y0) * width + (xs + boundary.x0))
            lengths = np.array([len(item) for item in lists], dtype=np.int64)
            offsets = np.zeros(len(lists), dtype=np.int64)
            if len(lists) > 1:
                offsets[1:] = np.cumsum(lengths)[:-1]
            points = (
                np.concatenate(lists).astype(np.int64) if lists else np.zeros(0, dtype=np.int64)
            )
            self._boundary_points = (offsets, lengths, points)
        return self._boundary_points

    def _mask_pack(self) -> tuple[np.ndarray, ...]:
        """Every fragment's mask crop, packed by :func:`_pack_crops`."""

        if self._masks_packed is None:
            crops = []
            for index, fragment in enumerate(self.fragments):
                x0, y0, x1, y1 = self.mask_boxes[index]
                crops.append((y0, x0, fragment.mask[y0:y1, x0:x1]))
            self._masks_packed = _pack_crops(crops)
        return self._masks_packed

    def union_overlap(self, union: np.ndarray, index: int) -> int:
        """``|union & mask|`` counted inside the fragment's bounding box."""

        crop = self._mask_crops.get(index)
        x0, y0, x1, y1 = self.mask_boxes[index]
        if crop is None:
            crop = np.ascontiguousarray(self.fragments[index].mask[y0:y1, x0:x1])
            self._mask_crops[index] = crop
        return int(_crop_overlap_count(union, crop, y0, x0))

    def overlap_bits(self, index: int) -> int:
        """Fragments whose masks share a pixel with fragment ``index``, as a bitset.

        Bit ``j`` is set when fragment ``j`` overlaps fragment ``index``. The
        relation is symmetric, so a union of member masks meets fragment ``n``
        exactly when bit ``n`` is set in the members' combined bitset.
        """

        bits = self._overlap_bits.get(index)
        if bits is None:
            hits = self._box_hits(index)
            offsets, shapes, origins, buffer = self._mask_pack()
            meets = np.empty(len(hits), dtype=np.bool_)
            _packed_crops_meet(index, hits, offsets, shapes, origins, buffer, meets)
            flags = np.zeros(len(self.fragments), dtype=np.bool_)
            flags[hits[meets]] = True
            bits = int.from_bytes(np.packbits(flags, bitorder="little").tobytes(), "little")
            self._overlap_bits[index] = bits
        return bits

    def boundary(self, index: int) -> _CroppedBoundary:
        cached = self._boundaries.get(index)
        if cached is None:
            cached = _cropped_boundary_evidence(
                [self.fragments[index]], tolerance=self.tolerance
            )[0]
            self._boundaries[index] = cached
        return cached

    def pose_uncertainty(self, index: int) -> float:
        cached = self._pose_uncertainty.get(index)
        if cached is None:
            cached = _normalised_pose_uncertainty(self.fragments[index], self.tolerance)
            self._pose_uncertainty[index] = cached
        return cached

    def dilated_mask(
        self, index: int
    ) -> tuple[tuple[int, int, int, int], np.ndarray, int]:
        """The fragment's mask dilated by ``gap_fill_radius``: box, crop, count."""

        cached = self._dilated_masks.get(index)
        if cached is None:
            dilated = _dilate(self.fragments[index].mask, self.gap_fill_radius)
            box = _bbox(dilated)
            x0, y0, x1, y1 = box
            cached = (box, dilated[y0:y1, x0:x1].copy(), int(np.count_nonzero(dilated)))
            self._dilated_masks[index] = cached
        return cached

    def dilated_union(self, info: StateInfo) -> tuple[np.ndarray, int]:
        """``dilate(union, gap_fill_radius)`` of a state and its pixel count.

        Binary dilation distributes over union, so a child's dilation is its
        parent's plus the added fragment's own; only search roots dilate the
        whole canvas.
        """

        cached = info.dilated_union_cache
        if cached is not None:
            return cached
        if info.parent is None or info.added_index is None:
            dilated = _dilate(info.union, self.gap_fill_radius)
            cached = (dilated, int(np.count_nonzero(dilated)))
        else:
            parent_dilated, parent_count = self.dilated_union(info.parent)
            (x0, y0, x1, y1), crop, count = self.dilated_mask(info.added_index)
            shared = int(_crop_overlap_count(parent_dilated, crop, y0, x0))
            dilated = parent_dilated.copy()
            dilated[y0:y1, x0:x1] |= crop
            cached = (dilated, parent_count + count - shared)
        info.dilated_union_cache = cached
        return cached

    def _dilated_pack(self) -> tuple[np.ndarray, ...]:
        """Every fragment's dilated mask, packed: offsets, shapes, origins, buffer, counts."""

        if self._dilated_packed is None:
            crops = []
            counts = np.empty(len(self.fragments), dtype=np.int64)
            for index in range(len(self.fragments)):
                (x0, y0, _x1, _y1), crop, count = self.dilated_mask(index)
                crops.append((y0, x0, crop))
                counts[index] = count
            self._dilated_packed = (*_pack_crops(crops), counts)
        return self._dilated_packed

    def dilated_union_counts(self, parent: StateInfo, indices: list[int]) -> list[int]:
        """:meth:`dilated_union_pixels` for several fragments added to one state."""

        dilated_union, dilated_count = self.dilated_union(parent)
        offsets, shapes, origins, buffer, counts = self._dilated_pack()
        index_array = np.fromiter(indices, dtype=np.int64, count=len(indices))
        shared = np.empty(len(indices), dtype=np.int64)
        _packed_crop_counts(dilated_union, index_array, offsets, shapes, origins, buffer, shared)
        return (dilated_count + counts[index_array] - shared).tolist()

    def dilated_union_pixels(self, parent: StateInfo, index: int) -> int:
        """``|dilate(parent union | mask)|`` without building the new union."""

        dilated_union, dilated_count = self.dilated_union(parent)
        (x0, y0, x1, y1), crop, count = self.dilated_mask(index)
        shared = int(_crop_overlap_count(dilated_union, crop, y0, x0))
        return dilated_count + count - shared

    def score_against(
        self,
        info: StateInfo,
        candidate_index: int,
        supporting: list[TearFitEdge],
        *,
        min_score: float,
        automatic_score: float,
    ) -> GroupGapEvidence | None:
        """:func:`score_fragment_against_assembly` with cached inputs.

        ``supporting`` is :func:`_supporting_edges` for this state and
        candidate. Returns ``None`` exactly when the evidence level would be
        ``insufficient-evidence``. Almost every proposal is rejected, mostly
        for overlapping the assembly, so the exact score is only built when an
        upper bound (the strongest seam in place of the mean) can still reach
        a threshold the placement is eligible for.
        """

        boundary = self.boundary(candidate_index)
        assembly = info.assembly_boundary(self.tolerance)
        matched = int(
            _crop_overlap_count(assembly, boundary.boundary, boundary.y0, boundary.x0)
        )
        seam_count = len(supporting)
        threshold = float("inf")
        if seam_count >= 2:
            threshold = automatic_score
        if matched >= 2:
            threshold = min(threshold, min_score)
        if threshold == float("inf"):
            return None
        area = self.areas[candidate_index]
        overlap = self.union_overlap(info.union, candidate_index)
        pose_uncertainty = self.pose_uncertainty(candidate_index)
        matched_ratio = matched / float(max(1, boundary.boundary_pixels))
        strongest = max(
            (edge.effectiveness / (1.0 + edge.effectiveness) for edge in supporting),
            default=0.0,
        )
        upper_bound = (
            0.15 * (max(0, area - overlap) / float(max(1, area)))
            + 0.25 * min(1.0, seam_count / 2.0)
            + 0.25 * matched_ratio
            + 0.25 * strongest
            - 0.35 * (overlap / float(max(1, area)))
            - 0.10 * (1.0 - matched_ratio)
            - 0.15 * pose_uncertainty
        )
        if upper_bound < threshold - 1e-9:
            return None
        evidence = _group_gap_evidence(
            candidate_index,
            overlap=overlap,
            area=area,
            matched_pixels=matched,
            boundary_pixels=boundary.boundary_pixels,
            supporting=supporting,
            pose_uncertainty=pose_uncertainty,
            min_score=min_score,
            automatic_score=automatic_score,
        )
        if evidence.evidence_level == "insufficient-evidence":
            return None
        return evidence

    def score_proposals(
        self,
        info: StateInfo,
        proposals: list[int],
        seams: dict[int, list[TearFitEdge]],
        strongest: dict[int, float],
        *,
        min_score: float,
        automatic_score: float,
    ) -> list[tuple[int, GroupGapEvidence]]:
        """:meth:`score_against` for every proposal of one state at once.

        Both pixel counts come from one compiled pass, and the upper bound is
        evaluated elementwise with the same operations in the same order as
        the scalar path; only proposals that survive it are scored exactly.
        Returns the accepted proposals with their evidence, in input order.
        """

        if not proposals:
            return []
        count = len(proposals)
        indices = np.fromiter(proposals, dtype=np.int64, count=count)
        assembly = info.assembly_boundary(self.tolerance)
        matched = np.empty(count, dtype=np.int64)
        point_offsets, point_lengths, points = self._pack_boundary_points()
        _packed_point_counts(
            np.ascontiguousarray(assembly).reshape(-1),
            indices,
            point_offsets,
            point_lengths,
            points,
            matched,
        )
        if info.area_sum == info.union_count:
            # No pixel lies in two members, so a proposal's overlap with the
            # union is the sum of its pairwise overlaps with the members.
            row_offsets, partners, pixels = self._pair_overlap_rows()
            totals = np.zeros(len(self.fragments), dtype=np.int64)
            members = np.fromiter(info.state, dtype=np.int64, count=len(info.state))
            _member_overlap_totals(members, row_offsets, partners, pixels, totals)
            overlap = totals[indices]
        else:
            overlap = np.empty(count, dtype=np.int64)
            offsets, shapes, origins, buffer = self._mask_pack()
            _packed_crop_counts(info.union, indices, offsets, shapes, origins, buffer, overlap)
        seam_count = np.fromiter(
            (len(seams.get(index, ())) for index in proposals), dtype=np.int64, count=count
        )
        strength = np.fromiter(
            (strongest.get(index, 0.0) for index in proposals), dtype=np.float64, count=count
        )
        pose = self._pose_array()[indices]
        area = self._area_array[indices]
        boundary_pixels = point_lengths[indices]
        threshold = np.where(seam_count >= 2, automatic_score, np.inf)
        threshold = np.where(matched >= 2, np.minimum(threshold, min_score), threshold)
        area_scale = np.maximum(1, area).astype(np.float64)
        matched_ratio = matched / np.maximum(1, boundary_pixels).astype(np.float64)
        upper_bound = (
            0.15 * (np.maximum(0, area - overlap) / area_scale)
            + 0.25 * np.minimum(1.0, seam_count / 2.0)
            + 0.25 * matched_ratio
            + 0.25 * strength
            - 0.35 * (overlap / area_scale)
            - 0.10 * (1.0 - matched_ratio)
            - 0.15 * pose
        )
        survivors = np.flatnonzero(
            np.isfinite(threshold) & ~(upper_bound < threshold - 1e-9)
        )
        accepted: list[tuple[int, GroupGapEvidence]] = []
        for slot in survivors.tolist():
            index = proposals[slot]
            evidence = _group_gap_evidence(
                index,
                overlap=int(overlap[slot]),
                area=self.areas[index],
                matched_pixels=int(matched[slot]),
                boundary_pixels=self.boundary(index).boundary_pixels,
                supporting=seams.get(index, []),
                pose_uncertainty=self.pose_uncertainty(index),
                min_score=min_score,
                automatic_score=automatic_score,
            )
            if evidence.evidence_level != "insufficient-evidence":
                accepted.append((index, evidence))
        return accepted


class StateInfo:
    """One partial assembly in a beam search.

    A state created by adding one fragment to ``parent`` builds its union only
    when it is expanded; most children are pruned by the beam before that.
    ``union_count`` is always exact.
    """

    __slots__ = (
        "state",
        "members",
        "blocked",
        "_union",
        "parent",
        "added_index",
        "_added_mask",
        "union_count",
        "area_sum",
        "labels",
        "support",
        "score",
        "raw_coverage",
        "coverage",
        "evidence_score",
        "evidence_level",
        "gap_steps",
        "dilated_union_cache",
        "_assembly_boundary",
    )

    def __init__(
        self,
        state: frozenset[int],
        union: np.ndarray | None,
        area_sum: int,
        labels: set[str],
        support: int,
        score: float,
        raw_coverage: float,
        coverage: float,
        evidence_score: float = 1.0,
        evidence_level: str = "automatic",
        gap_steps: int = 0,
        *,
        union_count: int | None = None,
        parent: StateInfo | None = None,
        added_index: int | None = None,
        added_mask: np.ndarray | None = None,
        members: tuple[int, ...] | None = None,
        blocked: int | None = None,
    ) -> None:
        if union is None:
            if parent is None or added_mask is None or union_count is None:
                raise ValueError(
                    "a lazy state needs its parent, added mask and union count"
                )
        elif union_count is None:
            union_count = int(np.count_nonzero(union))
        self.state = state
        self.members = members
        self.blocked = blocked
        self._union = union
        self.parent = parent
        self.added_index = added_index
        self._added_mask = added_mask
        self.union_count = int(union_count)
        self.area_sum = area_sum
        self.labels = labels
        self.support = support
        self.score = score
        self.raw_coverage = raw_coverage
        self.coverage = coverage
        self.evidence_score = evidence_score
        self.evidence_level = evidence_level
        self.gap_steps = gap_steps
        self.dilated_union_cache: tuple[np.ndarray, int] | None = None
        self._assembly_boundary: tuple[int, np.ndarray] | None = None

    @property
    def union(self) -> np.ndarray:
        if self._union is None:
            assert self.parent is not None and self._added_mask is not None
            self._union = self.parent.union | self._added_mask
            self._added_mask = None
        return self._union

    def assembly_boundary(self, tolerance: int) -> np.ndarray:
        """Dilated tear boundary of the whole assembly, built once per state."""

        cached = self._assembly_boundary
        if cached is None or cached[0] != tolerance:
            cached = (tolerance, _dilate(tear_boundary(self.union), tolerance))
            self._assembly_boundary = cached
        return cached[1]


class _BeamKey:
    """Sort key ``(-score, size, sorted members)`` that sorts members only on ties."""

    __slots__ = ("score", "state", "_members")

    def __init__(
        self,
        score: float,
        state: AbstractSet[int] | tuple[int, ...],
        members: tuple[int, ...] | None = None,
    ) -> None:
        self.score = score
        self.state = state
        self._members = members

    def sorted_members(self) -> tuple[int, ...]:
        if self._members is None:
            self._members = tuple(sorted(self.state))
        return self._members

    def __lt__(self, other: _BeamKey) -> bool:
        if self.score != other.score:
            return -self.score < -other.score
        if len(self.state) != len(other.state):
            return len(self.state) < len(other.state)
        return self.sorted_members() < other.sorted_members()


def _beam(states: Collection[StateInfo], width: int) -> list[StateInfo]:
    """``sorted(states, key=(-score, size, sorted members))[:width]``.

    States are distinct, so the key order is total and ``heapq.nsmallest``
    returns the same states in the same order without sorting every one.
    """

    def key(info: StateInfo) -> _BeamKey:
        return _BeamKey(info.score, info.state)

    if width < 1:
        return sorted(states, key=key)[:width]
    return heapq.nsmallest(width, states, key=key)


def _beam_records(
    records: dict[tuple[int, ...], tuple[Any, ...]], width: int
) -> list[tuple[tuple[int, ...], tuple[Any, ...]]]:
    """:func:`_beam` over ``sorted members -> (score, ...)`` records."""

    def key(item: tuple[tuple[int, ...], tuple[Any, ...]]) -> _BeamKey:
        return _BeamKey(item[1][0], item[0], item[0])

    if width < 1:
        return sorted(records.items(), key=key)[:width]
    return heapq.nsmallest(width, records.items(), key=key)


def generate_assembly_candidates(
    fragments: list[Fragment],
    edges: list[TearFitEdge],
    *,
    gap_edges: list[TearFitEdge] | None = None,
    enable_group_gap: bool = False,
    group_tolerance: int = 2,
    core_min_pieces: int = 2,
    min_group_gap_score: float = 0.35,
    automatic_group_gap_score: float = 0.55,
    coverage_threshold: float = 0.93,
    minimum_candidate_raw_coverage: float | None = None,
    gap_fill_radius: int = 2,
    max_pieces: int = 12,
    beam_width: int = 64,
    seed_strategy: str = "anchor_priority",
    require_anchor: bool | None = None,
    seed_whitelist: Iterable[str] | None = None,
    seed_labels: Iterable[str] | None = None,
    forced_pairs: Iterable[tuple[str, str]] | None = None,
    preferred_pairs: Iterable[tuple[str, str]] | None = None,
    forbidden_pairs: Iterable[tuple[str, str]] | None = None,
    forbidden_candidates: Iterable[Iterable[str]] | None = None,
    preferred_pair_bonus: float = 500.0,
    max_group_overlap_pixels: int = 0,
    time_limit_seconds: float | None = 20.0,
    max_expanded_states: int | None = None,
    search_stats: dict[str, int | bool] | None = None,
) -> list[AssemblyCandidate]:
    """Generate full-note candidates by connected tear graph search.

    The search is label-aware and overlap-constrained. Serial labels act as
    constraints and priority seeds, not as the only legal entry point unless
    ``seed_strategy="anchor_only"`` is explicitly requested for comparison.
    With ``enable_group_gap``, high-confidence ``edges`` first build a core.
    Weaker ``gap_edges`` can then add a fragment only when its placement is
    supported by the boundary of the whole partial assembly. It produces
    candidate note groups; :func:`select_exact_cover_candidates` then performs
    the global set-packing pass over those candidates.
    """

    if not fragments:
        if search_stats is not None:
            search_stats.update(
                expanded_states=0, state_limit_reached=False, time_limit_reached=False
            )
        return []
    if core_min_pieces < 1:
        raise ValueError("core_min_pieces must be positive")
    if min_group_gap_score < 0.0 or automatic_group_gap_score < min_group_gap_score:
        raise ValueError("group-gap thresholds must be ordered and non-negative")
    if minimum_candidate_raw_coverage is not None and not (
        0.0 < minimum_candidate_raw_coverage <= 1.0
    ):
        raise ValueError("minimum_candidate_raw_coverage must be in (0, 1]")
    if max_expanded_states is not None and max_expanded_states < 1:
        raise ValueError("max_expanded_states must be positive")
    seed_strategy = _resolve_seed_strategy(seed_strategy, require_anchor)
    fragment_id_to_index = {
        fragment.id: index for index, fragment in enumerate(fragments)
    }
    seed_whitelist_set = set(seed_whitelist or ())
    seed_label_set = set(seed_labels or ())
    forced_pair_indices = _normalise_index_pairs(forced_pairs, fragment_id_to_index)
    preferred_pair_indices = (
        _normalise_index_pairs(preferred_pairs, fragment_id_to_index)
        | forced_pair_indices
    )
    forbidden_pair_indices = _normalise_index_pairs(
        forbidden_pairs, fragment_id_to_index
    )
    forbidden_candidate_keys = _normalise_candidate_keys(forbidden_candidates)
    graph = _edge_graph(edges, len(fragments))[0]
    eligible_gap_edges = [
        edge
        for edge in (gap_edges or ())
        if edge.contiguous_pixels >= 2 and edge.effectiveness > 0.0
    ]
    gap_graph = _edge_graph(eligible_gap_edges, len(fragments))[0]
    geometry = _AssemblyGeometry(
        fragments, tolerance=group_tolerance, gap_fill_radius=gap_fill_radius
    )
    starts = _seed_order(fragments, seed_strategy)
    if seed_whitelist_set or seed_label_set:
        starts = [
            index
            for index in starts
            if fragments[index].id in seed_whitelist_set
            or (fragments[index].label and fragments[index].label in seed_label_set)
        ]
    if not starts:
        if search_stats is not None:
            search_stats.update(
                expanded_states=0, state_limit_reached=False, time_limit_reached=False
            )
        return []
    total_area = fragments[0].mask.size
    record_raw_threshold = (
        coverage_threshold
        if minimum_candidate_raw_coverage is None
        else minimum_candidate_raw_coverage
    )
    deadline = None if time_limit_seconds is None else monotonic() + time_limit_seconds
    expanded_states = 0
    state_limit_reached = False
    time_limit_reached = False

    def budget_exhausted() -> bool:
        nonlocal state_limit_reached, time_limit_reached
        if max_expanded_states is not None and expanded_states >= max_expanded_states:
            state_limit_reached = True
            return True
        if deadline is not None and monotonic() >= deadline:
            time_limit_reached = True
            return True
        return False

    # Expanded states are remembered by their sorted members: equal tuples are
    # exactly equal sets, and a tuple takes a fraction of a frozenset's memory.
    seen_states: set[tuple[int, ...]] = set()
    candidates: dict[tuple[str, ...], AssemblyCandidate] = {}

    for start in starts:
        if budget_exhausted():
            break
        start_fragment = fragments[start]
        start_union = start_fragment.mask.copy()
        start_area_sum = geometry.areas[start]
        start_labels = {start_fragment.label} if start_fragment.label else set()
        start_support = 0
        start_raw_coverage = start_area_sum / float(total_area)
        if start_raw_coverage >= coverage_threshold - 0.15:
            start_coverage = int(_dilate(start_union, gap_fill_radius).sum()) / float(
                total_area
            )
        else:
            start_coverage = start_raw_coverage
        start_score = (
            start_coverage * 10_000.0 + start_support + 100.0 * len(start_labels)
        )

        initial_state = StateInfo(
            state=frozenset({start}),
            union=start_union,
            area_sum=start_area_sum,
            labels=start_labels,
            support=start_support,
            score=start_score,
            raw_coverage=start_raw_coverage,
            coverage=start_coverage,
            union_count=start_area_sum,
            members=(start,),
            blocked=geometry.overlap_bits(start),
        )

        frontier: list[StateInfo] = [initial_state]
        for _depth in range(max(1, max_pieces - 1)):
            if budget_exhausted():
                break
            # Children are kept as plain records keyed by sorted members; only
            # the states that survive the beam become StateInfo objects.
            next_frontier: dict[tuple[int, ...], tuple[Any, ...]] = {}
            for state_info in frontier:
                if budget_exhausted():
                    break
                expanded_states += 1
                state = state_info.state
                members = state_info.members
                assert members is not None
                neighbours: set[int] = set()
                for member in state:
                    neighbours.update(graph[member])
                neighbours.difference_update(state)
                gap_options: dict[int, GroupGapEvidence] = {}
                if enable_group_gap and len(state) >= core_min_pieces:
                    gap_pool: set[int] = set()
                    for member in state:
                        gap_pool.update(gap_graph[member])
                    gap_pool.difference_update(state)
                    gap_pool.difference_update(neighbours)
                    for gap_index in gap_pool:
                        gap_evidence = geometry.score_against(
                            state_info,
                            gap_index,
                            _supporting_from_adjacency(state, gap_graph[gap_index]),
                            min_score=min_group_gap_score,
                            automatic_score=automatic_group_gap_score,
                        )
                        if gap_evidence is not None:
                            gap_options[gap_index] = gap_evidence
                    neighbours.update(gap_options)
                # Pass 1 applies the filters, all free of side effects and the
                # overlap test first, and gathers each passing child's terms.
                # Pass 2 scores and records the children in the same order,
                # after one batched count of their dilated coverage.
                blocked = state_info.blocked
                assert blocked is not None
                passed: list[tuple[Any, ...]] = []
                dilate: list[int] = []
                for neighbour in neighbours:
                    # The union meets the neighbour exactly when its bit is set
                    # in the members' overlap bitset; the overlap is then at
                    # least one pixel.
                    overlapping = (blocked >> neighbour) & 1
                    if overlapping and max_group_overlap_pixels < 1:
                        continue
                    position = bisect_left(members, neighbour)
                    child = members[:position] + (neighbour,) + members[position:]
                    if child in seen_states or len(child) > max_pieces:
                        continue
                    if forbidden_pair_indices and _state_has_forbidden_pair(
                        frozenset(child), forbidden_pair_indices
                    ):
                        continue

                    # Check labels incrementally
                    neighbour_label = fragments[neighbour].label
                    if neighbour_label and neighbour_label not in state_info.labels:
                        if len(state_info.labels) > 0:
                            continue
                        new_labels = state_info.labels | {neighbour_label}
                    else:
                        new_labels = state_info.labels

                    # Check overlap incrementally; |U | M| = |U| + |M| - |U & M|.
                    shared = (
                        geometry.union_overlap(state_info.union, neighbour)
                        if overlapping
                        else 0
                    )
                    neighbour_area = geometry.areas[neighbour]
                    new_area_sum = state_info.area_sum + neighbour_area
                    new_union_count = state_info.union_count + neighbour_area - shared
                    overlap = new_area_sum - new_union_count
                    if overlap > max_group_overlap_pixels:
                        continue

                    # Calculate pairwise or whole-assembly support incrementally.
                    # The neighbour's few graph edges are the same edges a
                    # lookup per state member finds; the sum, the maximum and
                    # the any() below are independent of their order.
                    added_support = 0
                    has_incident = False
                    any_automatic = False
                    best_adaptive: float | None = None
                    for member, edge in graph[neighbour].items():
                        if member in state:
                            has_incident = True
                            if edge.effectiveness > 0.0:
                                added_support += int(
                                    round(
                                        edge.contiguous_pixels
                                        * min(edge.effectiveness, 8.0)
                                    )
                                )
                                adaptive = edge.effectiveness / (1.0 + edge.effectiveness)
                                if best_adaptive is None or adaptive > best_adaptive:
                                    best_adaptive = adaptive
                            else:
                                added_support += edge.overlap_pixels
                            if edge.evidence_level == "automatic":
                                any_automatic = True
                    step_gap = gap_options.get(neighbour)
                    if step_gap is not None:
                        added_support = max(added_support, step_gap.matched_pixels)
                        step_score = step_gap.score
                        step_level = step_gap.evidence_level
                        new_gap_steps = state_info.gap_steps + 1
                    else:
                        step_score = 1.0 if best_adaptive is None else best_adaptive
                        step_level = (
                            "automatic" if not has_incident or any_automatic else "review"
                        )
                        new_gap_steps = state_info.gap_steps
                    new_support = state_info.support + added_support
                    new_evidence_score = min(state_info.evidence_score, step_score)
                    new_evidence_level = (
                        "automatic"
                        if state_info.evidence_level == "automatic"
                        and step_level == "automatic"
                        else "review"
                    )

                    raw_coverage = new_union_count / float(total_area)
                    if raw_coverage >= coverage_threshold - 0.15:
                        dilate.append(len(passed))
                    passed.append(
                        (
                            neighbour,
                            child,
                            new_labels,
                            new_area_sum,
                            new_union_count,
                            raw_coverage,
                            new_support,
                            new_evidence_score,
                            new_evidence_level,
                            new_gap_steps,
                        )
                    )

                coverages = [entry[5] for entry in passed]
                if dilate:
                    dilated = geometry.dilated_union_counts(
                        state_info, [passed[slot][0] for slot in dilate]
                    )
                    for slot, pixels in zip(dilate, dilated):
                        coverages[slot] = pixels / float(total_area)
                for entry, coverage in zip(passed, coverages):
                    (
                        neighbour,
                        child,
                        new_labels,
                        new_area_sum,
                        new_union_count,
                        raw_coverage,
                        new_support,
                        new_evidence_score,
                        new_evidence_level,
                        new_gap_steps,
                    ) = entry
                    base_score = (
                        coverage * 10_000.0
                        + new_support
                        + 100.0 * len(new_labels)
                        + 250.0 * new_evidence_score
                        - 25.0 * new_gap_steps
                    )
                    preference_bonus = (
                        _pair_preference_bonus(
                            frozenset(child), preferred_pair_indices, preferred_pair_bonus
                        )
                        if preferred_pair_indices
                        else 0.0
                    )
                    score = base_score + preference_bonus

                    existing = next_frontier.get(child)
                    if existing is None or score > existing[0]:
                        next_frontier[child] = (
                            score,
                            state_info,
                            neighbour,
                            new_area_sum,
                            new_labels,
                            new_support,
                            raw_coverage,
                            coverage,
                            new_evidence_score,
                            new_evidence_level,
                            new_gap_steps,
                            new_union_count,
                        )

                    if (
                        coverage >= coverage_threshold
                        or raw_coverage >= record_raw_threshold
                    ):
                        ids = tuple(sorted(fragments[index].id for index in child))
                        if ids in forbidden_candidate_keys:
                            continue
                        if forced_pair_indices and not _forced_pairs_satisfied(
                            frozenset(child), forced_pair_indices
                        ):
                            continue
                        existing_candidate = candidates.get(ids)
                        candidate = AssemblyCandidate(
                            fragment_ids=ids,
                            coverage=coverage,
                            raw_coverage=raw_coverage,
                            score=score,
                            support_pixels=new_support,
                            labels=tuple(sorted(new_labels)),
                            base_score=base_score,
                            constraint_bonus=preference_bonus,
                            evidence_score=new_evidence_score,
                            evidence_level=new_evidence_level,
                            gap_steps=new_gap_steps,
                        )
                        if (
                            existing_candidate is None
                            or candidate.score > existing_candidate.score
                        ):
                            candidates[ids] = candidate
                seen_states.add(members)
            if not next_frontier:
                break
            # A survivor's frozenset is built from the parent that won its
            # record, in the same element order as an eager build would use.
            frontier = [
                StateInfo(
                    state=frozenset((*record[1].state, record[2])),
                    union=None,
                    area_sum=record[3],
                    labels=record[4],
                    support=record[5],
                    score=record[0],
                    raw_coverage=record[6],
                    coverage=record[7],
                    evidence_score=record[8],
                    evidence_level=record[9],
                    gap_steps=record[10],
                    union_count=record[11],
                    parent=record[1],
                    added_index=record[2],
                    added_mask=fragments[record[2]].mask,
                    members=child_members,
                    blocked=record[1].blocked | geometry.overlap_bits(record[2]),
                )
                for child_members, record in _beam_records(next_frontier, beam_width)
            ]

    if search_stats is not None:
        search_stats.update(
            expanded_states=expanded_states,
            state_limit_reached=state_limit_reached,
            time_limit_reached=time_limit_reached,
        )
    return sorted(
        candidates.values(), key=lambda item: (-item.score, item.fragment_ids)
    )


def select_group_gap_bases(
    candidates: Iterable[AssemblyCandidate],
    *,
    max_base_candidates: int,
    strategy: str = "global",
    presorted: bool = False,
) -> tuple[list[AssemblyCandidate], int]:
    """Select expansion bases under a fixed budget without simulator truth.

    ``global`` preserves the historical top-K behavior. The deterministic
    ``disjoint_round_robin`` strategy repeatedly takes one greedy set-packing
    round from the same ranking. Candidates selected in one round cannot share
    fragments, so near-duplicate candidates cannot consume the entire budget;
    a fresh round then permits another candidate from each fragment region.
    This uses the exact-cover disjointness contract and has no similarity
    threshold or access to ``note_id``.
    """

    if max_base_candidates < 1:
        raise ValueError("max_base_candidates must be positive")
    if strategy not in TEARFIT_BASE_SELECTION_STRATEGIES:
        raise ValueError(
            "strategy must be one of: "
            + ", ".join(TEARFIT_BASE_SELECTION_STRATEGIES)
        )
    ordered = list(candidates)
    if not presorted:
        ordered.sort(
            key=lambda item: (
                -len(item.fragment_ids),
                -item.raw_coverage,
                -item.evidence_score,
                -item.score,
                item.fragment_ids,
            )
        )
    limit = min(max_base_candidates, len(ordered))
    if strategy == "global":
        return ordered[:limit], int(limit > 0)

    selected: list[AssemblyCandidate] = []
    remaining = ordered
    rounds = 0
    while remaining and len(selected) < limit:
        used_fragments: set[str] = set()
        deferred: list[AssemblyCandidate] = []
        selected_this_round = 0
        for candidate in remaining:
            if len(selected) >= limit:
                deferred.append(candidate)
                continue
            if used_fragments.isdisjoint(candidate.fragment_ids):
                selected.append(candidate)
                used_fragments.update(candidate.fragment_ids)
                selected_this_round += 1
            else:
                deferred.append(candidate)
        if selected_this_round == 0:
            break
        rounds += 1
        remaining = deferred
    return selected, rounds


def augment_candidates_with_group_gap(
    fragments: list[Fragment],
    candidates: list[AssemblyCandidate],
    gap_edges: list[TearFitEdge],
    *,
    tolerance: int = 2,
    coverage_threshold: float = 0.93,
    gap_fill_radius: int = 2,
    max_pieces: int = 12,
    core_min_pieces: int = 2,
    beam_width: int = 8,
    max_base_candidates: int = 512,
    base_selection_strategy: str = "global",
    proposal_pool: str = "weak_pair",
    min_group_gap_score: float = 0.35,
    automatic_group_gap_score: float = 0.55,
    max_group_overlap_pixels: int = 0,
    time_limit_seconds: float | None = 20.0,
    max_expanded_states: int | None = None,
    search_stats: dict[str, int | bool] | None = None,
    geometry: _AssemblyGeometry | None = None,
) -> list[AssemblyCandidate]:
    """Add weak pairwise fragments only after a high-confidence core exists.

    Every input candidate that already reaches ``coverage_threshold`` remains
    in the output. Lower-coverage cores are expansion seeds only. The second
    stage searches for fragments whose joint fit against the whole core is
    stronger than any individual weak seam, so enabling it cannot erase a
    complete core-only solution. ``weak_pair`` preserves the v4.3 proposal
    gate. ``boundary_contact`` lets every non-overlapping boundary-contact pair
    enter the same unchanged whole-assembly scorer, including pairs whose
    adaptive tear evidence was insufficient. ``geometry`` may carry
    per-fragment caches from an earlier search over the same ``fragments``
    list; it is used only when built for ``tolerance`` and ``gap_fill_radius``.
    """

    if base_selection_strategy not in TEARFIT_BASE_SELECTION_STRATEGIES:
        raise ValueError(
            "base_selection_strategy must be one of: "
            + ", ".join(TEARFIT_BASE_SELECTION_STRATEGIES)
        )
    if not candidates:
        if search_stats is not None:
            search_stats.update(
                expanded_states=0,
                state_limit_reached=False,
                time_limit_reached=False,
                selected_base_candidates=0,
                base_selection_rounds=0,
            )
        return []
    if max_expanded_states is not None and max_expanded_states < 1:
        raise ValueError("max_expanded_states must be positive")
    if proposal_pool not in TEARFIT_GAP_PROPOSAL_POOLS:
        raise ValueError(
            "proposal_pool must be one of: "
            + ", ".join(TEARFIT_GAP_PROPOSAL_POOLS)
        )
    id_to_index = {fragment.id: index for index, fragment in enumerate(fragments)}
    eligible_gap_edges = [
        edge
        for edge in gap_edges
        if edge.contiguous_pixels >= 2 and edge.effectiveness > 0.0
    ]
    gap_graph = _edge_graph(eligible_gap_edges, len(fragments))[0]
    proposal_edges = (
        eligible_gap_edges
        if proposal_pool == "weak_pair"
        else [edge for edge in gap_edges if edge.overlap_pixels > 0]
    )
    proposal_graph = _edge_graph(proposal_edges, len(fragments))[0]
    if geometry is None or not geometry.serves(
        fragments, tolerance=tolerance, gap_fill_radius=gap_fill_radius
    ):
        geometry = _AssemblyGeometry(
            fragments, tolerance=tolerance, gap_fill_radius=gap_fill_radius
        )
    total_area = fragments[0].mask.size
    deadline = None if time_limit_seconds is None else monotonic() + time_limit_seconds
    expanded_states = 0
    state_limit_reached = False
    time_limit_reached = False
    proposal_evaluations = 0
    accepted_proposals = 0
    no_weak_edge_evaluations = 0
    accepted_no_weak_edge_proposals = 0

    def budget_exhausted() -> bool:
        nonlocal state_limit_reached, time_limit_reached
        if max_expanded_states is not None and expanded_states >= max_expanded_states:
            state_limit_reached = True
            return True
        if deadline is not None and monotonic() >= deadline:
            time_limit_reached = True
            return True
        return False

    output: dict[tuple[str, ...], AssemblyCandidate] = {
        _candidate_key(candidate.fragment_ids): candidate
        for candidate in candidates
        if candidate.coverage >= coverage_threshold
    }

    base_order = sorted(
        candidates,
        key=lambda item: (
            -len(item.fragment_ids),
            -item.raw_coverage,
            -item.evidence_score,
            -item.score,
            item.fragment_ids,
        ),
    )
    selected_bases, selection_rounds = select_group_gap_bases(
        base_order,
        max_base_candidates=max_base_candidates,
        strategy=base_selection_strategy,
        presorted=True,
    )
    for base in selected_bases:
        if budget_exhausted():
            break
        try:
            start_state = frozenset(
                id_to_index[fragment_id] for fragment_id in base.fragment_ids
            )
        except KeyError:
            continue
        if len(start_state) < core_min_pieces or len(start_state) >= max_pieces:
            continue
        start_union = np.any(
            np.stack([fragments[index].mask for index in start_state]), axis=0
        )
        start_area_sum = sum(geometry.areas[index] for index in start_state)
        start_labels = set(base.labels)
        frontier = [
            StateInfo(
                state=start_state,
                union=start_union,
                area_sum=start_area_sum,
                labels=start_labels,
                support=base.support_pixels,
                score=base.score,
                raw_coverage=base.raw_coverage,
                coverage=base.coverage,
                evidence_score=base.evidence_score,
                evidence_level=base.evidence_level,
                gap_steps=base.gap_steps,
            )
        ]
        seen = {start_state}
        largest_seen = len(start_state)
        for _depth in range(max(0, max_pieces - len(start_state))):
            if budget_exhausted():
                break
            next_frontier: dict[frozenset[int], StateInfo] = {}
            for state_info in frontier:
                if budget_exhausted():
                    break
                expanded_states += 1
                state = state_info.state
                pool: set[int] = set()
                for member in state:
                    pool.update(proposal_graph[member])
                pool.difference_update(state)
                # Seams from the members to every candidate, built once per
                # state: each list follows state order (the order a scan of
                # the members gives) and comes with its strongest seam ratio.
                seams: dict[int, list[TearFitEdge]] = {}
                strongest: dict[int, float] = {}
                for member in state:
                    for other, edge in gap_graph[member].items():
                        ratio = edge.effectiveness / (1.0 + edge.effectiveness)
                        found = seams.get(other)
                        if found is None:
                            seams[other] = [edge]
                            strongest[other] = ratio
                        else:
                            found.append(edge)
                            if ratio > strongest[other]:
                                strongest[other] = ratio
                # Frontier states share one size and every seen state is no
                # larger, so a child can only be seen once that changes.
                child_size = len(state) + 1
                proposals: list[int] = []
                for neighbour in pool:
                    if child_size <= largest_seen and (
                        frozenset((*state, neighbour)) in seen
                    ):
                        continue
                    neighbour_label = fragments[neighbour].label
                    if (
                        neighbour_label
                        and state_info.labels
                        and neighbour_label not in state_info.labels
                    ):
                        continue
                    proposals.append(neighbour)
                proposal_evaluations += len(proposals)
                # Every gap_graph edge passes the support filter, so a weak
                # edge exists exactly when the candidate has a seam list.
                no_weak_edge_evaluations += sum(
                    1 for neighbour in proposals if neighbour not in seams
                )
                for neighbour, evidence in geometry.score_proposals(
                    state_info,
                    proposals,
                    seams,
                    strongest,
                    min_score=min_group_gap_score,
                    automatic_score=automatic_group_gap_score,
                ):
                    neighbour_label = fragments[neighbour].label
                    new_state = frozenset((*state, neighbour))
                    accepted_proposals += 1
                    if neighbour not in seams:
                        accepted_no_weak_edge_proposals += 1
                    # evidence.overlap_pixels is |U & M|, so |U | M| follows.
                    new_area_sum = state_info.area_sum + geometry.areas[neighbour]
                    new_union_count = (
                        state_info.union_count
                        + geometry.areas[neighbour]
                        - evidence.overlap_pixels
                    )
                    overlap = new_area_sum - new_union_count
                    if overlap > max_group_overlap_pixels:
                        continue
                    new_labels = set(state_info.labels)
                    if neighbour_label:
                        new_labels.add(neighbour_label)
                    raw_coverage = new_union_count / float(total_area)
                    coverage = (
                        geometry.dilated_union_pixels(state_info, neighbour)
                        / float(total_area)
                        if raw_coverage >= coverage_threshold - 0.15
                        else raw_coverage
                    )
                    evidence_score = min(state_info.evidence_score, evidence.score)
                    evidence_level = (
                        "automatic"
                        if state_info.evidence_level == "automatic"
                        and evidence.evidence_level == "automatic"
                        else "review"
                    )
                    gap_steps = state_info.gap_steps + 1
                    support = state_info.support + evidence.matched_pixels
                    base_score = (
                        coverage * 10_000.0
                        + support
                        + 100.0 * len(new_labels)
                        + 250.0 * evidence_score
                        - 25.0 * gap_steps
                    )
                    info = StateInfo(
                        state=new_state,
                        union=None,
                        area_sum=new_area_sum,
                        labels=new_labels,
                        support=support,
                        score=base_score,
                        raw_coverage=raw_coverage,
                        coverage=coverage,
                        evidence_score=evidence_score,
                        evidence_level=evidence_level,
                        gap_steps=gap_steps,
                        union_count=new_union_count,
                        parent=state_info,
                        added_index=neighbour,
                        added_mask=fragments[neighbour].mask,
                    )
                    existing = next_frontier.get(new_state)
                    if existing is None or info.score > existing.score:
                        next_frontier[new_state] = info
                    if coverage >= coverage_threshold:
                        ids = tuple(sorted(fragments[index].id for index in new_state))
                        candidate = AssemblyCandidate(
                            fragment_ids=ids,
                            coverage=coverage,
                            raw_coverage=raw_coverage,
                            score=base_score,
                            support_pixels=support,
                            labels=tuple(sorted(new_labels)),
                            base_score=base_score,
                            evidence_score=evidence_score,
                            evidence_level=evidence_level,
                            gap_steps=gap_steps,
                        )
                        previous = output.get(ids)
                        if previous is None or candidate.score > previous.score:
                            output[ids] = candidate
                seen.add(state)
                largest_seen = max(largest_seen, len(state))
            if not next_frontier:
                break
            frontier = _beam(next_frontier.values(), beam_width)
    if search_stats is not None:
        search_stats.update(
            expanded_states=expanded_states,
            state_limit_reached=state_limit_reached,
            time_limit_reached=time_limit_reached,
            weak_pair_edges=len(eligible_gap_edges),
            proposal_edges=len(proposal_edges),
            proposal_evaluations=proposal_evaluations,
            accepted_proposals=accepted_proposals,
            no_weak_edge_evaluations=no_weak_edge_evaluations,
            accepted_no_weak_edge_proposals=accepted_no_weak_edge_proposals,
            selected_base_candidates=len(selected_bases),
            base_selection_rounds=selection_rounds,
        )
    return sorted(output.values(), key=lambda item: (-item.score, item.fragment_ids))


def _connected_components(
    mask: np.ndarray, *, connectivity: int = 4
) -> list[np.ndarray]:
    """Return the connected components of a boolean mask via numpy flood-fill.

    Deliberately scipy-free: an explicit iterative stack walk keeps the kernel
    dependency-light and mirrors :func:`_largest_connected_support`.
    """

    mask = mask.astype(bool)
    if mask.ndim != 2 or not mask.any():
        return []
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    offsets: tuple[tuple[int, int], ...]
    if connectivity == 8:
        offsets = (
            (-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1),
        )
    else:
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    components: list[np.ndarray] = []
    for seed in np.argwhere(mask):
        sy = int(seed[0])
        sx = int(seed[1])
        if visited[sy, sx]:
            continue
        component = np.zeros(mask.shape, dtype=bool)
        stack = [(sy, sx)]
        visited[sy, sx] = True
        while stack:
            y, x = stack.pop()
            component[y, x] = True
            for dy, dx in offsets:
                ny = y + dy
                nx = x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and mask[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        components.append(component)
    return components


def _exposed_perimeter(mask: np.ndarray) -> int:
    """Count placed tear-boundary pixels (clean note-frame edges excluded)."""

    return int(np.count_nonzero(tear_boundary(mask)))


def _region_boundary_pixels(mask: np.ndarray) -> int:
    """Count all boundary pixels of a region, including note-frame contact."""

    mask = mask.astype(bool)
    if not mask.any():
        return 0
    padded = np.pad(mask, 1, constant_values=False)
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    interior = up & down & left & right
    boundary = mask & ~interior
    return int(np.count_nonzero(boundary))


def _min_informative_scale(tolerance: int, *, fray_layers: int = 0) -> int:
    """Minimum informative gap scale derived from tear tolerance and fray.

    A proposal thinner than one tolerance-diameter neighbourhood (optionally
    enlarged by the simulated fray depth) carries no reliable tear evidence.
    This is a physical-tolerance floor and deliberately ignores locator error.
    """

    span = 2 * max(0, int(tolerance)) + 1
    return int(max(1, span * span + max(0, int(fray_layers)) * span))


def _gap_complexity_score(region: ResidualGapRegion) -> float:
    """Continuous difficulty proxy from area, perimeter ratio and neighbours."""

    area = max(1, region.area)
    perimeter_ratio = region.perimeter_pixels / float(area)
    neighbours = len(region.adjacent_fragment_indices)
    return float(0.5 * perimeter_ratio + 0.15 * neighbours + area / 500.0)


def classify_gap_complexity(
    region: ResidualGapRegion,
    *,
    simple_max_area: int = 48,
    complex_min_area: int = 220,
    simple_max_neighbors: int = 2,
    complex_min_neighbors: int = 4,
    complex_perimeter_ratio: float = 1.6,
) -> str:
    """Route a residual gap to ``simple`` / ``moderate`` / ``complex`` work."""

    area = max(1, region.area)
    perimeter_ratio = region.perimeter_pixels / float(area)
    neighbours = len(region.adjacent_fragment_indices)
    if area >= complex_min_area or neighbours >= complex_min_neighbors:
        return "complex"
    if (
        area <= simple_max_area
        and neighbours <= simple_max_neighbors
        and perimeter_ratio <= complex_perimeter_ratio
    ):
        return "simple"
    return "moderate"


def compute_residual_gap_components(
    fragments: list[Fragment],
    core_candidates: list[AssemblyCandidate],
    *,
    coverage_threshold: float = 0.93,
    gap_fill_radius: int = 2,
    max_core_candidates: int = 8,
    boundary_touch_radius: int = 1,
    connectivity: int = 4,
    min_region_area: int = 1,
) -> list[ResidualGapRegion]:
    """Represent the residual gaps left by the highest-confidence cores.

    The complement of a core union within the shared note frame is decomposed
    into connected components. Each component is an explicit place where real
    fragment evidence is still missing; no tear geometry is ever synthesised.
    """

    if not fragments or not core_candidates:
        return []
    id_to_index = {fragment.id: index for index, fragment in enumerate(fragments)}
    frame_shape = fragments[0].mask.shape
    ranked = sorted(
        core_candidates,
        key=lambda item: (-item.raw_coverage, -item.score, -len(item.fragment_ids)),
    )
    dilated_boundaries = [
        _dilate(tear_boundary(fragment.mask), boundary_touch_radius)
        for fragment in fragments
    ]
    regions: list[ResidualGapRegion] = []
    seen_signatures: set[bytes] = set()
    component_id = 0
    for core in ranked[: max(1, max_core_candidates)]:
        try:
            indices = [id_to_index[fragment_id] for fragment_id in core.fragment_ids]
        except KeyError:
            continue
        union = np.zeros(frame_shape, dtype=bool)
        for index in indices:
            union |= fragments[index].mask
        complement = ~union
        for component in _connected_components(complement, connectivity=connectivity):
            area = int(np.count_nonzero(component))
            if area < min_region_area:
                continue
            signature = np.packbits(component.reshape(-1)).tobytes()
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            dilated_component = _dilate(component, boundary_touch_radius)
            adjacent = tuple(
                sorted(
                    index
                    for index in range(len(fragments))
                    if np.any(dilated_boundaries[index] & component)
                    or np.any(dilated_component & fragments[index].mask)
                )
            )
            region = ResidualGapRegion(
                mask=component,
                area=area,
                perimeter_pixels=_region_boundary_pixels(component),
                component_id=component_id,
                adjacent_fragment_indices=adjacent,
            )
            region = replace(
                region,
                complexity=_gap_complexity_score(region),
                routing_class=classify_gap_complexity(region),
            )
            regions.append(region)
            component_id += 1
    return regions


def evaluate_gap_proposal(
    fragments: list[Fragment],
    base_state: frozenset[int],
    base_union: np.ndarray,
    added_indices: tuple[int, ...],
    region: ResidualGapRegion,
    *,
    gap_lookup: dict[tuple[int, int], TearFitEdge],
    boundary_evidence: list[TearBoundaryEvidence],
    tolerance: int = 2,
    coverage_threshold: float = 0.93,
    gap_fill_radius: int = 2,
    min_informative_scale: int | None = None,
    max_group_overlap_pixels: int = 0,
    base_support: int = 0,
    base_evidence_score: float = 1.0,
    base_evidence_level: str = "automatic",
    min_group_gap_score: float = 0.35,
    automatic_group_gap_score: float = 0.55,
    alpha: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    eps: float = 1.0,
    zeta: float = 1.0,
    denom_epsilon: float = 1.0,
    min_proposal_effectiveness: float = 0.0,
) -> GapProposalEvaluation:
    """Score a whole-assembly fragment-set proposal against a residual gap.

    Insufficient-evidence seams are admitted as *soft* evidence: they are not
    hard-rejected, they only weaken ``E_proposal``. Sub-tolerance slivers and
    overlapping placements are rejected on physical grounds.
    """

    if min_informative_scale is None:
        min_informative_scale = _min_informative_scale(tolerance)
    total_area = float(fragments[0].mask.size)
    base_area = int(np.count_nonzero(base_union))
    base_perimeter = _exposed_perimeter(base_union)
    state = set(base_state)
    union = base_union.copy()
    seam_total = 0
    matched_pixels_total = 0
    unmatched_perimeter_total = 0.0
    unmatched_ratio_total = 0.0
    pose_total = 0.0
    sliver_area = 0.0
    sum_added_area = 0
    evidence_scores: list[float] = []
    levels: list[str] = []
    for index in added_indices:
        fragment = fragments[index]
        evidence = score_fragment_against_assembly(
            fragments,
            frozenset(state),
            union,
            index,
            gap_lookup,
            boundary_evidence,
            tolerance=tolerance,
            min_score=min_group_gap_score,
            automatic_score=automatic_group_gap_score,
        )
        seam_total += evidence.seam_count
        matched_pixels_total += evidence.matched_pixels
        boundary_pixels = int(np.count_nonzero(boundary_evidence[index].boundary_mask))
        unmatched_perimeter_total += evidence.unmatched_perimeter_ratio * boundary_pixels
        unmatched_ratio_total += evidence.unmatched_perimeter_ratio
        pose_total += evidence.pose_uncertainty
        evidence_scores.append(evidence.score)
        levels.append(evidence.evidence_level)
        if fragment.area < min_informative_scale:
            sliver_area += float(fragment.area)
        sum_added_area += fragment.area
        state.add(index)
        union = union | fragment.mask
    new_area = int(np.count_nonzero(union))
    net_new_area = new_area - base_area
    overlap_pixels = (base_area + sum_added_area) - new_area
    closed_perimeter_delta = float(base_perimeter - _exposed_perimeter(union))
    independent_seam_support = float(seam_total)
    tolerance_instability = float(pose_total + unmatched_ratio_total)
    residual_gap_delta = float(int(np.count_nonzero(region.mask & ~union)))
    newly_exposed_perimeter = float(unmatched_perimeter_total)
    numerator = closed_perimeter_delta + alpha * independent_seam_support
    denominator = (
        gamma * sliver_area
        + delta * tolerance_instability
        + eps * residual_gap_delta
        + zeta * newly_exposed_perimeter
        + denom_epsilon
    )
    effectiveness = numerator / denominator if denominator > 0.0 else 0.0
    raw_coverage = new_area / total_area
    coverage = (
        int(_dilate(union, gap_fill_radius).sum()) / total_area
        if raw_coverage >= coverage_threshold - 0.15
        else raw_coverage
    )
    support_pixels = base_support + matched_pixels_total
    evidence_score = (
        min([base_evidence_score, *evidence_scores])
        if evidence_scores
        else base_evidence_score
    )
    if base_evidence_level == "automatic" and all(
        level == "automatic" for level in levels
    ):
        evidence_level = "automatic"
    else:
        evidence_level = "review"
    reason = "accepted"
    accepted = True
    if not added_indices:
        accepted = False
        reason = "empty_proposal"
    elif overlap_pixels > max_group_overlap_pixels:
        accepted = False
        reason = "overlap"
    elif net_new_area < min_informative_scale:
        accepted = False
        reason = "sub_tolerance_sliver"
    elif effectiveness < min_proposal_effectiveness:
        accepted = False
        reason = "low_effectiveness"
    return GapProposalEvaluation(
        accepted=accepted,
        reason=reason,
        effectiveness=float(effectiveness),
        closed_perimeter_delta=closed_perimeter_delta,
        independent_seam_support=independent_seam_support,
        sliver_area=float(sliver_area),
        tolerance_instability=tolerance_instability,
        residual_gap_delta=residual_gap_delta,
        newly_exposed_perimeter=newly_exposed_perimeter,
        net_new_area=int(net_new_area),
        overlap_pixels=int(overlap_pixels),
        coverage=float(coverage),
        raw_coverage=float(raw_coverage),
        support_pixels=int(support_pixels),
        evidence_score=float(evidence_score),
        evidence_level=evidence_level,
    )


def augment_candidates_gap_first(
    fragments: list[Fragment],
    core_candidates: list[AssemblyCandidate],
    edges: list[TearFitEdge],
    gap_edges: list[TearFitEdge],
    residual_regions: list[ResidualGapRegion],
    *,
    tolerance: int = 2,
    coverage_threshold: float = 0.93,
    gap_fill_radius: int = 2,
    max_pieces: int = 12,
    core_min_pieces: int = 2,
    max_base_candidates: int = 512,
    proposal_pool: str = "boundary_contact",
    gap_routing: str = "complexity",
    min_group_gap_score: float = 0.35,
    automatic_group_gap_score: float = 0.55,
    max_group_overlap_pixels: int = 0,
    fray_layers: int = 0,
    min_informative_scale: int | None = None,
    min_proposal_effectiveness: float = 0.0,
    simple_top_k: int = 4,
    moderate_beam_width: int = 6,
    moderate_max_set_size: int = 3,
    complex_window: int = 6,
    complex_max_set_size: int = 3,
    max_universe: int = 12,
    alpha: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    eps: float = 1.0,
    zeta: float = 1.0,
    denom_epsilon: float = 1.0,
    time_limit_seconds: float | None = 20.0,
    max_expanded_states: int | None = None,
    search_stats: dict[str, int | bool] | None = None,
) -> list[AssemblyCandidate]:
    """Gap-first candidate construction over explicit residual regions.

    High-confidence complete cores are always preserved. Each residual gap is
    attached to the most-complete core whose complement still contains it, and
    per-gap routing proposes whole-assembly fragment sets: single fragments for
    ``simple`` gaps, a small subset beam for ``moderate`` gaps, and a windowed
    local exact cover for ``complex`` gaps. Every proposal is filtered by
    :func:`evaluate_gap_proposal` (proposal effectiveness + tolerance floor)
    before entering the same candidate list consumed by the exact-cover pass.
    """

    if proposal_pool not in TEARFIT_GAP_PROPOSAL_POOLS:
        raise ValueError(
            "proposal_pool must be one of: " + ", ".join(TEARFIT_GAP_PROPOSAL_POOLS)
        )
    if gap_routing not in TEARFIT_GAP_ROUTING:
        raise ValueError(
            "gap_routing must be one of: " + ", ".join(TEARFIT_GAP_ROUTING)
        )
    if max_expanded_states is not None and max_expanded_states < 1:
        raise ValueError("max_expanded_states must be positive")

    routing_counts = {"simple": 0, "moderate": 0, "complex": 0}
    for region in residual_regions:
        routing_counts[region.routing_class] = (
            routing_counts.get(region.routing_class, 0) + 1
        )
    output: dict[tuple[str, ...], AssemblyCandidate] = {
        _candidate_key(candidate.fragment_ids): candidate
        for candidate in core_candidates
        if candidate.coverage >= coverage_threshold
    }
    if not core_candidates or not residual_regions:
        if search_stats is not None:
            search_stats.update(
                expanded_states=0,
                state_limit_reached=False,
                time_limit_reached=False,
                gap_regions=len(residual_regions),
                gap_proposals_made=0,
                accepted_proposals=0,
                proposal_pool_size=0,
                routing_simple=routing_counts.get("simple", 0),
                routing_moderate=routing_counts.get("moderate", 0),
                routing_complex=routing_counts.get("complex", 0),
            )
        return sorted(
            output.values(), key=lambda item: (-item.score, item.fragment_ids)
        )

    if min_informative_scale is None:
        min_informative_scale = _min_informative_scale(
            tolerance, fray_layers=fray_layers
        )
    id_to_index = {fragment.id: index for index, fragment in enumerate(fragments)}
    boundary_evidence = tear_boundary_evidence(fragments, tolerance=tolerance)
    gap_lookup = _edge_graph(list(edges) + list(gap_edges), len(fragments))[1]

    viable_bases: list[tuple[AssemblyCandidate, frozenset[int], np.ndarray]] = []
    base_order = sorted(
        core_candidates,
        key=lambda item: (-item.raw_coverage, -item.score, -len(item.fragment_ids)),
    )[: max(1, max_base_candidates)]
    for base in base_order:
        try:
            base_state = frozenset(
                id_to_index[fragment_id] for fragment_id in base.fragment_ids
            )
        except KeyError:
            continue
        if base.coverage >= coverage_threshold:
            continue
        if len(base_state) < core_min_pieces or len(base_state) >= max_pieces:
            continue
        base_union = np.zeros(fragments[0].mask.shape, dtype=bool)
        for index in base_state:
            base_union |= fragments[index].mask
        viable_bases.append((base, base_state, base_union))

    deadline = None if time_limit_seconds is None else monotonic() + time_limit_seconds
    expanded_states = 0
    state_limit_reached = False
    time_limit_reached = False
    proposals_made = 0
    accepted_proposals = 0
    proposal_pool_size = 0

    def budget_exhausted() -> bool:
        nonlocal state_limit_reached, time_limit_reached
        if max_expanded_states is not None and expanded_states >= max_expanded_states:
            state_limit_reached = True
            return True
        if deadline is not None and monotonic() >= deadline:
            time_limit_reached = True
            return True
        return False

    def emit(
        base: AssemblyCandidate,
        base_state: frozenset[int],
        added: tuple[int, ...],
        evaluation: GapProposalEvaluation,
    ) -> None:
        if evaluation.coverage < coverage_threshold:
            return
        combined = frozenset(base_state) | set(added)
        if len(combined) > max_pieces:
            return
        ids = tuple(sorted(fragments[index].id for index in combined))
        labels = set(base.labels)
        for index in added:
            label = fragments[index].label
            if label:
                labels.add(label)
        gap_steps = base.gap_steps + len(added)
        score = (
            evaluation.coverage * 10_000.0
            + evaluation.support_pixels
            + 100.0 * len(labels)
            + 250.0 * evaluation.evidence_score
            - 25.0 * gap_steps
        )
        candidate = AssemblyCandidate(
            fragment_ids=ids,
            coverage=evaluation.coverage,
            raw_coverage=evaluation.raw_coverage,
            score=score,
            support_pixels=evaluation.support_pixels,
            labels=tuple(sorted(labels)),
            base_score=score,
            evidence_score=evaluation.evidence_score,
            evidence_level=evaluation.evidence_level,
            gap_steps=gap_steps,
        )
        previous = output.get(ids)
        if previous is None or candidate.score > previous.score:
            output[ids] = candidate

    def build_universe(
        base_state: frozenset[int], region: ResidualGapRegion
    ) -> list[int]:
        base_labels = {
            fragments[index].label for index in base_state if fragments[index].label
        }
        pool: dict[int, int] = {}
        candidate_indices = set(region.adjacent_fragment_indices)
        for index in range(len(fragments)):
            if index in base_state:
                continue
            overlap = int(np.count_nonzero(fragments[index].mask & region.mask))
            touches = index in candidate_indices
            if overlap <= 0 and not touches:
                continue
            label = fragments[index].label
            if label and base_labels and label not in base_labels:
                continue
            pool[index] = overlap
        ranked = sorted(pool, key=lambda index: (-pool[index], index))
        return ranked[: max(1, max_universe)]

    def evaluate(
        base: AssemblyCandidate,
        base_state: frozenset[int],
        base_union: np.ndarray,
        region: ResidualGapRegion,
        added: tuple[int, ...],
    ) -> GapProposalEvaluation:
        nonlocal expanded_states, proposals_made, accepted_proposals
        expanded_states += 1
        proposals_made += 1
        evaluation = evaluate_gap_proposal(
            fragments,
            base_state,
            base_union,
            added,
            region,
            gap_lookup=gap_lookup,
            boundary_evidence=boundary_evidence,
            tolerance=tolerance,
            coverage_threshold=coverage_threshold,
            gap_fill_radius=gap_fill_radius,
            min_informative_scale=min_informative_scale,
            max_group_overlap_pixels=max_group_overlap_pixels,
            base_support=base.support_pixels,
            base_evidence_score=base.evidence_score,
            base_evidence_level=base.evidence_level,
            min_group_gap_score=min_group_gap_score,
            automatic_group_gap_score=automatic_group_gap_score,
            alpha=alpha,
            gamma=gamma,
            delta=delta,
            eps=eps,
            zeta=zeta,
            denom_epsilon=denom_epsilon,
            min_proposal_effectiveness=min_proposal_effectiveness,
        )
        if evaluation.accepted:
            accepted_proposals += 1
        return evaluation

    for region in residual_regions:
        if budget_exhausted():
            break
        base_entry = None
        for base, base_state, base_union in viable_bases:
            if int(np.count_nonzero(region.mask & base_union)) == 0:
                base_entry = (base, base_state, base_union)
                break
        if base_entry is None:
            continue
        base, base_state, base_union = base_entry
        universe = build_universe(base_state, region)
        proposal_pool_size += len(universe)
        if not universe:
            continue
        remaining = max_pieces - len(base_state)
        if remaining < 1:
            continue
        routing = region.routing_class if gap_routing == "complexity" else "moderate"

        accepted_singles: list[tuple[float, int]] = []
        for index in universe:
            if budget_exhausted():
                break
            evaluation = evaluate(base, base_state, base_union, region, (index,))
            if evaluation.accepted:
                accepted_singles.append((evaluation.effectiveness, index))
                emit(base, base_state, (index,), evaluation)
        accepted_singles.sort(key=lambda item: (-item[0], item[1]))

        if routing == "simple":
            continue

        if routing == "moderate":
            beam = [
                (frozenset({index}), effectiveness)
                for effectiveness, index in accepted_singles[: max(1, moderate_beam_width)]
            ]
            max_set = min(moderate_max_set_size, remaining)
            for _size in range(2, max_set + 1):
                if budget_exhausted():
                    break
                next_beam: dict[frozenset[int], float] = {}
                for current, _score in beam:
                    for index in universe:
                        if index in current:
                            continue
                        if budget_exhausted():
                            break
                        added = tuple(sorted(current | {index}))
                        evaluation = evaluate(
                            base, base_state, base_union, region, added
                        )
                        if not evaluation.accepted:
                            continue
                        emit(base, base_state, added, evaluation)
                        key = frozenset(added)
                        if (
                            key not in next_beam
                            or evaluation.effectiveness > next_beam[key]
                        ):
                            next_beam[key] = evaluation.effectiveness
                if not next_beam:
                    break
                beam = sorted(
                    next_beam.items(), key=lambda item: -item[1]
                )[: max(1, moderate_beam_width)]
            continue

        # complex: windowed local exact cover over the near-neighbour universe
        window = universe[: max(1, complex_window)]
        max_set = min(complex_max_set_size, remaining)
        restricted: dict[tuple[str, ...], AssemblyCandidate] = {}
        for size in range(1, max_set + 1):
            if budget_exhausted():
                break
            for combo in combinations(window, size):
                if budget_exhausted():
                    break
                evaluation = evaluate(base, base_state, base_union, region, combo)
                if not evaluation.accepted:
                    continue
                if evaluation.coverage < coverage_threshold:
                    continue
                combined = frozenset(base_state) | set(combo)
                ids = tuple(sorted(fragments[index].id for index in combined))
                labels = set(base.labels)
                for index in combo:
                    combo_label = fragments[index].label
                    if combo_label:
                        labels.add(combo_label)
                gap_steps = base.gap_steps + len(combo)
                score = (
                    evaluation.coverage * 10_000.0
                    + evaluation.support_pixels
                    + 100.0 * len(labels)
                    + 250.0 * evaluation.evidence_score
                    - 25.0 * gap_steps
                )
                restricted[ids] = AssemblyCandidate(
                    fragment_ids=ids,
                    coverage=evaluation.coverage,
                    raw_coverage=evaluation.raw_coverage,
                    score=score,
                    support_pixels=evaluation.support_pixels,
                    labels=tuple(sorted(labels)),
                    base_score=score,
                    evidence_score=evaluation.evidence_score,
                    evidence_level=evaluation.evidence_level,
                    gap_steps=gap_steps,
                )
        if restricted:
            selected_local = select_exact_cover_candidates(
                list(restricted.values()),
                time_limit_seconds=None,
                objective="score_then_count",
            )
            for candidate in selected_local:
                previous = output.get(candidate.fragment_ids)
                if previous is None or candidate.score > previous.score:
                    output[candidate.fragment_ids] = candidate

    if search_stats is not None:
        search_stats.update(
            expanded_states=expanded_states,
            state_limit_reached=state_limit_reached,
            time_limit_reached=time_limit_reached,
            gap_regions=len(residual_regions),
            gap_proposals_made=proposals_made,
            accepted_proposals=accepted_proposals,
            proposal_pool_size=proposal_pool_size,
            routing_simple=routing_counts.get("simple", 0),
            routing_moderate=routing_counts.get("moderate", 0),
            routing_complex=routing_counts.get("complex", 0),
        )
    return sorted(output.values(), key=lambda item: (-item.score, item.fragment_ids))


def diagnose_gap_candidate_funnel(
    fragments: list[Fragment],
    generated_candidates: list[AssemblyCandidate],
    final_candidates: list[AssemblyCandidate],
    gap_edges: list[TearFitEdge],
    *,
    tolerance: int = 2,
    coverage_threshold: float = 0.93,
    gap_fill_radius: int = 2,
    core_min_pieces: int = 2,
    min_group_gap_score: float = 0.35,
    automatic_group_gap_score: float = 0.55,
    max_partial_core_candidates: int = 128,
    complete_base_limit: int = 512,
    base_selection_strategy: str = "global",
    beam_width: int = 8,
    max_bases_per_note: int = 4,
) -> dict:
    """Audit where simulated truth falls out of group-gap construction.

    Ground-truth ``note_id`` is used only to restrict a diagnostic replay to the
    correct note. Production scoring, thresholds, and candidate generation do
    not consume this output.
    """

    if max_bases_per_note < 1:
        raise ValueError("max_bases_per_note must be positive")
    missing_truth = [
        fragment.id for fragment in fragments if "note_id" not in fragment.meta
    ]
    if missing_truth:
        raise ValueError("candidate funnel requires simulated note_id ground truth")

    truth_by_key = _true_note_candidate_keys(fragments)
    note_indices: dict[str, list[int]] = {}
    note_by_fragment_id: dict[str, str] = {}
    for index, fragment in enumerate(fragments):
        note_id = str(fragment.meta["note_id"])
        note_indices.setdefault(note_id, []).append(index)
        note_by_fragment_id[fragment.id] = note_id

    complete_cores = [
        candidate
        for candidate in generated_candidates
        if candidate.coverage >= coverage_threshold
    ]
    partial_cores = [
        candidate
        for candidate in generated_candidates
        if candidate.coverage < coverage_threshold
    ]

    def base_order(candidates: list[AssemblyCandidate]) -> list[AssemblyCandidate]:
        return sorted(
            candidates,
            key=lambda item: (
                -len(item.fragment_ids),
                -item.raw_coverage,
                -item.evidence_score,
                -item.score,
                item.fragment_ids,
            ),
        )

    selected_complete, complete_selection_rounds = select_group_gap_bases(
        base_order(complete_cores),
        max_base_candidates=complete_base_limit,
        strategy=base_selection_strategy,
        presorted=True,
    )
    selected_partial, partial_selection_rounds = select_group_gap_bases(
        base_order(partial_cores),
        max_base_candidates=max_partial_core_candidates,
        strategy=base_selection_strategy,
        presorted=True,
    )
    selected_bases = [
        ("complete", rank + 1, candidate)
        for rank, candidate in enumerate(selected_complete)
    ] + [
        ("partial", rank + 1, candidate)
        for rank, candidate in enumerate(selected_partial)
    ]
    all_pure_bases: dict[str, list[tuple[str, int, AssemblyCandidate]]] = {}
    selected_pure_bases: dict[str, list[tuple[str, int, AssemblyCandidate]]] = {}

    def candidate_note(candidate: AssemblyCandidate) -> str | None:
        notes = {
            note_by_fragment_id[fragment_id]
            for fragment_id in candidate.fragment_ids
        }
        return next(iter(notes)) if len(notes) == 1 else None

    for source, candidates in (
        ("complete", base_order(complete_cores)),
        ("partial", base_order(partial_cores)),
    ):
        for rank, candidate in enumerate(candidates, start=1):
            pure_note = candidate_note(candidate)
            if pure_note is not None:
                all_pure_bases.setdefault(pure_note, []).append(
                    (source, rank, candidate)
                )
    for source, rank, candidate in selected_bases:
        pure_note = candidate_note(candidate)
        if pure_note is not None:
            selected_pure_bases.setdefault(pure_note, []).append(
                (source, rank, candidate)
            )

    final_keys = {_candidate_key(item.fragment_ids) for item in final_candidates}
    core_keys = {_candidate_key(item.fragment_ids) for item in complete_cores}

    def truth_restricted_recovery(
        note_id: str,
        bases: list[tuple[str, int, AssemblyCandidate]],
        proposal_pool: str,
    ) -> tuple[bool, int, int]:
        indices = note_indices[note_id]
        global_to_local = {
            global_index: local_index
            for local_index, global_index in enumerate(indices)
        }
        local_fragments = [fragments[index] for index in indices]
        local_edges = [
            replace(
                edge,
                left=global_to_local[edge.left],
                right=global_to_local[edge.right],
            )
            for edge in gap_edges
            if edge.left in global_to_local and edge.right in global_to_local
        ]
        truth_key = _candidate_key(fragment.id for fragment in local_fragments)
        attempted = 0
        expanded = 0
        for _source, _rank, base in bases[:max_bases_per_note]:
            attempted += 1
            stats: dict[str, int | bool] = {}
            candidates = augment_candidates_with_group_gap(
                local_fragments,
                [base],
                local_edges,
                tolerance=tolerance,
                coverage_threshold=coverage_threshold,
                gap_fill_radius=gap_fill_radius,
                max_pieces=len(local_fragments),
                core_min_pieces=core_min_pieces,
                beam_width=beam_width,
                max_base_candidates=1,
                proposal_pool=proposal_pool,
                min_group_gap_score=min_group_gap_score,
                automatic_group_gap_score=automatic_group_gap_score,
                time_limit_seconds=None,
                max_expanded_states=None,
                search_stats=stats,
            )
            expanded += int(stats.get("expanded_states", 0))
            if any(_candidate_key(item.fragment_ids) == truth_key for item in candidates):
                return True, attempted, expanded
        return False, attempted, expanded

    rows: list[dict] = []
    for truth_key, note_id in sorted(truth_by_key.items(), key=lambda item: item[1]):
        if truth_key in core_keys:
            rows.append({"note_id": note_id, "category": "core_exact"})
            continue
        if truth_key in final_keys:
            rows.append({"note_id": note_id, "category": "production_gap_exact"})
            continue

        all_bases = all_pure_bases.get(note_id, [])
        selected = selected_pure_bases.get(note_id, [])
        row = {
            "note_id": note_id,
            "pure_base_count": len(all_bases),
            "selected_pure_base_count": len(selected),
        }
        if all_bases:
            source, rank, best = all_bases[0]
            row.update(
                best_base_source=source,
                best_base_rank=rank,
                best_base_pieces=len(best.fragment_ids),
                best_base_raw_coverage=best.raw_coverage,
            )
        if not all_bases:
            row["category"] = "no_pure_core_base"
        elif not selected:
            row["category"] = "pure_core_base_not_selected"
        else:
            weak_recovered, weak_attempts, weak_expanded = truth_restricted_recovery(
                note_id, selected, "weak_pair"
            )
            contact_recovered, contact_attempts, contact_expanded = (
                truth_restricted_recovery(note_id, selected, "boundary_contact")
            )
            row.update(
                weak_pair_truth_recovered=weak_recovered,
                weak_pair_bases_attempted=weak_attempts,
                weak_pair_expanded_states=weak_expanded,
                boundary_contact_truth_recovered=contact_recovered,
                boundary_contact_bases_attempted=contact_attempts,
                boundary_contact_expanded_states=contact_expanded,
            )
            if weak_recovered:
                row["category"] = "cross_note_beam_or_ranking"
            elif contact_recovered:
                row["category"] = "weak_pair_proposal_gate"
            else:
                row["category"] = "whole_assembly_scorer_or_local_beam"
        rows.append(row)

    category_counts: dict[str, int] = {}
    for row in rows:
        category = str(row["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "simulation_truth_only": True,
        "complete_base_limit": complete_base_limit,
        "partial_base_limit": max_partial_core_candidates,
        "base_selection_strategy": base_selection_strategy,
        "complete_selection_rounds": complete_selection_rounds,
        "partial_selection_rounds": partial_selection_rounds,
        "beam_width": beam_width,
        "max_bases_per_note": max_bases_per_note,
        "category_counts": category_counts,
        "notes": rows,
    }


def select_exact_cover_candidates(
    candidates: list[AssemblyCandidate],
    *,
    time_limit_seconds: float | None = 10.0,
    objective: str = "score_then_count",
    forbidden_candidates: Iterable[Iterable[str]] | None = None,
    locked_candidates: Iterable[Iterable[str]] | None = None,
    preferred_candidates: Iterable[Iterable[str]] | None = None,
    preferred_candidate_bonus: float = 50_000.0,
    max_search_nodes: int | None = None,
    search_stats: dict[str, int | bool] | None = None,
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
        raise ValueError(
            f"objective must be one of: {', '.join(TEARFIT_COVER_OBJECTIVES)}"
        )
    if max_search_nodes is not None and max_search_nodes < 1:
        raise ValueError("max_search_nodes must be positive")
    forbidden_keys = _normalise_candidate_keys(forbidden_candidates)
    locked_keys = _normalise_candidate_keys(locked_candidates)
    preferred_keys = _normalise_candidate_keys(preferred_candidates)

    filtered = [
        candidate
        for candidate in candidates
        if _candidate_key(candidate.fragment_ids) not in forbidden_keys
    ]

    def adjusted_score(item: AssemblyCandidate) -> float:
        score = item.score
        if _candidate_key(item.fragment_ids) in preferred_keys:
            score += preferred_candidate_bonus
        return score

    ordered = sorted(
        filtered,
        key=lambda item: (
            -adjusted_score(item),
            -len(item.fragment_ids),
            item.fragment_ids,
        ),
    )
    locked: list[AssemblyCandidate] = []
    remaining: list[AssemblyCandidate] = []
    used_locked_ids: set[str] = set()
    used_locked_labels: set[str] = set()
    locked_score = 0.0
    available_locked_keys = {
        _candidate_key(candidate.fragment_ids) for candidate in ordered
    }
    missing_locked = locked_keys - available_locked_keys
    if missing_locked:
        raise ValueError(f"locked candidates are unavailable: {sorted(missing_locked)}")
    for candidate in ordered:
        key = _candidate_key(candidate.fragment_ids)
        if key in locked_keys:
            item_ids = set(candidate.fragment_ids)
            item_labels = set(candidate.labels)
            if item_ids & used_locked_ids or item_labels & used_locked_labels:
                raise ValueError(
                    "locked candidates conflict by fragment id or serial label"
                )
            locked.append(candidate)
            used_locked_ids.update(item_ids)
            used_locked_labels.update(item_labels)
            locked_score += adjusted_score(candidate)
        else:
            remaining.append(candidate)

    # Pre-compute optimistic suffix scores for score_then_count DFS pruning
    suffix_score = [0.0] * (len(remaining) + 1)
    for i in range(len(remaining) - 1, -1, -1):
        suffix_score[i] = suffix_score[i + 1] + max(0.0, adjusted_score(remaining[i]))

    deadline = None if time_limit_seconds is None else monotonic() + time_limit_seconds
    locked_count = len(locked)
    best_count = locked_count
    best_score = locked_score

    def better(count: int, score: float) -> bool:
        if objective == "count_then_score":
            return count > best_count or (count == best_count and score > best_score)
        return score > best_score or (score == best_score and count > best_count)

    # Iterative branch-and-bound. The recursive form recursed to depth equal to
    # the candidate count, so large pools (N=200+) overflowed Python's recursion
    # limit and crashed mid-search instead of returning the best packing found so
    # far. An explicit LIFO stack keeps the identical include-first DFS order and
    # both pruning bounds, with no depth limit. Fragment ids and serial labels
    # are interned as bits of Python ints, so a conflict test is one AND and a
    # child's used set one OR; chosen positions form a shared linked chain
    # instead of a tuple copied at every node.
    remaining_count = len(remaining)
    id_bits: dict[str, int] = {}
    label_bits: dict[str, int] = {}

    def as_bits(values: Iterable[str], table: dict[str, int]) -> int:
        mask = 0
        for value in values:
            bit = table.get(value)
            if bit is None:
                bit = 1 << len(table)
                table[value] = bit
            mask |= bit
        return mask

    locked_id_bits = as_bits(used_locked_ids, id_bits)
    locked_label_bits = as_bits(used_locked_labels, label_bits)
    item_id_bits = [as_bits(item.fragment_ids, id_bits) for item in remaining]
    item_label_bits = [as_bits(item.labels, label_bits) for item in remaining]
    item_scores = [adjusted_score(item) for item in remaining]

    # (position, used id bits, used label bits, chosen chain, chosen count, score)
    stack: list[tuple[int, int, int, Any, int, float]] = [
        (0, locked_id_bits, locked_label_bits, None, 0, locked_score)
    ]
    best_chain: Any = None
    search_nodes = 0
    node_limit_reached = False
    time_limit_reached = False
    while stack:
        if max_search_nodes is not None and search_nodes >= max_search_nodes:
            node_limit_reached = True
            break
        if deadline is not None and monotonic() >= deadline:
            time_limit_reached = True
            break
        pos, used_ids, used_labels, chosen, chosen_count, score = stack.pop()
        search_nodes += 1
        if (
            objective == "count_then_score"
            and locked_count + chosen_count + (remaining_count - pos) < best_count
        ):
            continue
        if objective == "score_then_count" and score + suffix_score[pos] < best_score:
            continue
        if pos >= remaining_count:
            count = locked_count + chosen_count
            if better(count, score):
                best_count = count
                best_score = score
                best_chain = chosen
            continue

        # Push skip-branch first so the include-branch is popped first, matching
        # the recursive preorder (a good packing found early tightens the bound).
        stack.append((pos + 1, used_ids, used_labels, chosen, chosen_count, score))
        if not (item_id_bits[pos] & used_ids) and not (item_label_bits[pos] & used_labels):
            stack.append(
                (
                    pos + 1,
                    used_ids | item_id_bits[pos],
                    used_labels | item_label_bits[pos],
                    (pos, chosen),
                    chosen_count + 1,
                    score + item_scores[pos],
                )
            )
    chosen_positions: list[int] = []
    while best_chain is not None:
        chosen_positions.append(best_chain[0])
        best_chain = best_chain[1]
    best_choice = tuple(reversed(chosen_positions))
    selected_out = locked + [remaining[index] for index in best_choice]
    final_out = []
    for candidate in selected_out:
        key = _candidate_key(candidate.fragment_ids)
        if key in preferred_keys:
            candidate = AssemblyCandidate(
                fragment_ids=candidate.fragment_ids,
                coverage=candidate.coverage,
                raw_coverage=candidate.raw_coverage,
                score=candidate.score + preferred_candidate_bonus,
                support_pixels=candidate.support_pixels,
                labels=candidate.labels,
                base_score=candidate.base_score,
                constraint_bonus=candidate.constraint_bonus + preferred_candidate_bonus,
                evidence_score=candidate.evidence_score,
                evidence_level=candidate.evidence_level,
                gap_steps=candidate.gap_steps,
            )
        final_out.append(candidate)
    if search_stats is not None:
        search_stats.update(
            search_nodes=search_nodes,
            node_limit_reached=node_limit_reached,
            time_limit_reached=time_limit_reached,
        )
    return final_out


def diagnose_confirmed_candidates(
    candidates: list[AssemblyCandidate], fragments: list[Fragment]
) -> TearFitDiagnostics:
    lookup = {fragment.id: fragment for fragment in fragments}
    true_notes: dict[str, set[str]] = {}
    for fragment in fragments:
        note_id = fragment.meta.get("note_id")
        if note_id:
            true_notes.setdefault(note_id, set()).add(fragment.id)

    exact = 0
    pure = 0
    chimeras = 0
    automatic_exact = 0
    automatic_exact_notes: set[str] = set()
    for candidate in candidates:
        ids = set(candidate.fragment_ids)
        notes = {
            lookup[fid].meta.get("note_id")
            for fid in ids
            if fid in lookup and lookup[fid].meta.get("note_id")
        }
        if len(notes) > 1:
            chimeras += 1
            continue
        if len(notes) == 1:
            pure += 1
            note_id = cast(str, next(iter(notes)))
            if ids == true_notes.get(note_id, set()):
                exact += 1
                if candidate.evidence_level == "automatic":
                    automatic_exact += 1
                    automatic_exact_notes.add(note_id)
    confirmed = len(candidates)
    true_count = len(true_notes)
    automatic_candidates = sum(
        candidate.evidence_level == "automatic" for candidate in candidates
    )
    review_candidates = sum(
        candidate.evidence_level == "review" for candidate in candidates
    )
    return TearFitDiagnostics(
        confirmed=confirmed,
        exact_confirmed=exact,
        pure_confirmed=pure,
        chimeras=chimeras,
        true_notes=true_count,
        exact_yield=exact / true_count if true_count else 0.0,
        exact_precision=exact / confirmed if confirmed else 0.0,
        pure_precision=pure / confirmed if confirmed else 0.0,
        manual_notes_remaining=true_count - len(automatic_exact_notes),
        confirmed_candidates=tuple(candidates),
        automatic_candidates=automatic_candidates,
        review_candidates=review_candidates,
        automatic_exact_confirmed=automatic_exact,
        automatic_exact_yield=automatic_exact / true_count if true_count else 0.0,
        automatic_exact_precision=(
            automatic_exact / automatic_candidates if automatic_candidates else 0.0
        ),
    )


def observed_fragment_view(fragments: Iterable[Fragment]) -> list[Fragment]:
    """Return fragments as the tear-fit core may see them.

    ``meta`` keeps only the locator pose-uncertainty keys listed in
    ``TEARFIT_OBSERVED_META_KEYS``; simulator truth such as ``note_id`` and
    evaluation annotations are dropped. A fragment whose ``meta["provenance"]``
    is anything other than ``"observed"`` (for example generated or inpainted
    content) is rejected.
    """

    observed: list[Fragment] = []
    for fragment in fragments:
        provenance = fragment.meta.get("provenance", "observed")
        if provenance != "observed":
            raise ValueError(
                "tear-fit core accepts observed fragments only; "
                f"{fragment.id} has provenance {provenance!r}"
            )
        observed.append(
            replace(
                fragment,
                meta={
                    key: fragment.meta[key]
                    for key in TEARFIT_OBSERVED_META_KEYS
                    if key in fragment.meta
                },
            )
        )
    return observed


def _candidate_source(
    key: tuple[str, ...],
    *,
    complete_core_keys: AbstractSet[tuple[str, ...]],
    complete_gap_keys: AbstractSet[tuple[str, ...]],
    partial_gap_keys: AbstractSet[tuple[str, ...]],
) -> str:
    if key in partial_gap_keys:
        return "partial_gap"
    if key in complete_gap_keys:
        return "complete_gap"
    if key in complete_core_keys:
        return "core"
    return "other"


def reconstruct_placed_fragments(
    fragments: list[Fragment],
    config: TearFitCoreConfig,
    *,
    max_pieces: int,
    expected_notes: int,
    edge_intervention: Callable[[TearFitEdge], bool] | None = None,
) -> TearFitReconstruction:
    """Run the tear-fit core on fragments already placed in the canonical frame.

    Pair scoring, candidate construction, and exact cover see only
    :func:`observed_fragment_view`, so simulator truth and evaluation
    annotations cannot influence the result. ``max_pieces`` caps the fragments
    in one assembled note; ``expected_notes`` only normalises
    ``cover_nodes_per_note``.

    ``edge_intervention`` is a simulation-only counterfactual hook: accepted
    edges for which it returns ``False`` are removed after pair scoring.
    Reconstruction of real inputs must leave it unset.
    """

    if not fragments:
        raise ValueError("at least one fragment is required")
    if max_pieces < 1:
        raise ValueError("max_pieces must be positive")
    if expected_notes < 1:
        raise ValueError("expected_notes must be positive")
    observed = observed_fragment_view(fragments)
    stage_timings: dict[str, float] = {}
    median_fragment_fraction = float(
        np.median([fragment.area for fragment in observed])
    ) / float(observed[0].mask.size)
    resolved_algorithm = config.algorithm
    if config.algorithm == "v43_routed":
        resolved_algorithm = (
            "baseline"
            if median_fragment_fraction >= config.route_fragment_fraction_threshold
            else "effectiveness_gap"
        )
    scoring = "overlap" if resolved_algorithm == "baseline" else "effectiveness"
    pair_scoring_started = monotonic()
    all_scores, raw_edges = score_absolute_tear_pairs(
        observed,
        tolerance=config.tolerance,
        min_overlap_pixels=config.min_overlap_pixels,
        use_labels=False,
        scoring=scoring,
        min_effectiveness=config.min_effectiveness,
        automatic_effectiveness=config.automatic_effectiveness,
        min_contiguous_pixels=config.min_contiguous_pixels,
        automatic_contiguous_pixels=config.automatic_contiguous_pixels,
    )
    if config.use_labels:
        # Each pair score is independent of the others, so the label-aware
        # pass is exactly the label-compatible subset of the unfiltered one.
        label_filtered_scores = [
            edge
            for edge in all_scores
            if _labels_compatible(observed[edge.left].label, observed[edge.right].label)
        ]
        edges = [
            edge
            for edge in label_filtered_scores
            if edge.evidence_level != "insufficient-evidence"
        ]
    else:
        label_filtered_scores = all_scores
        edges = raw_edges
    stage_timings["pair_scoring"] = monotonic() - pair_scoring_started
    intervention_removed_edges = 0
    intervention_removed_core_edges = 0
    if edge_intervention is not None:
        preintervention_core_edges = (
            edges
            if resolved_algorithm == "baseline"
            else [edge for edge in edges if edge.evidence_level == "automatic"]
        )
        intervention_removed_edges = sum(
            not edge_intervention(edge) for edge in edges
        )
        intervention_removed_core_edges = sum(
            not edge_intervention(edge) for edge in preintervention_core_edges
        )
        edges = [edge for edge in edges if edge_intervention(edge)]
        label_filtered_scores = [
            edge
            for edge in label_filtered_scores
            if edge.evidence_level == "insufficient-evidence" or edge_intervention(edge)
        ]
    resolved_candidate_state_limit = _resolve_workload_budget(
        config.candidate_state_limit,
        len(all_scores),
        config.candidate_states_per_pair_score,
        name="candidate_state_limit",
    )
    resolved_gap_state_limit = _resolve_workload_budget(
        config.gap_state_limit,
        len(observed),
        config.gap_states_per_fragment,
        name="gap_state_limit",
    )
    resolved_partial_gap_state_limit = _resolve_workload_budget(
        config.partial_gap_state_limit,
        len(observed),
        config.partial_gap_states_per_fragment,
        name="partial_gap_state_limit",
    )
    resolved_cover_node_limit = _resolve_workload_budget(
        config.cover_node_limit,
        expected_notes,
        config.cover_nodes_per_note,
        name="cover_node_limit",
    )
    core_edges = (
        edges
        if resolved_algorithm == "baseline"
        else [edge for edge in edges if edge.evidence_level == "automatic"]
    )
    core_search_stats: dict[str, int | bool] = {}
    core_search_started = monotonic()
    generated_candidates = generate_assembly_candidates(
        observed,
        core_edges,
        coverage_threshold=config.coverage_threshold,
        minimum_candidate_raw_coverage=(
            config.resolved_core_raw_coverage_threshold
            if resolved_algorithm in ("effectiveness_gap", "v44_gap_first")
            else None
        ),
        gap_fill_radius=config.gap_fill_radius,
        max_pieces=max_pieces,
        beam_width=config.beam_width,
        seed_strategy=config.seed_strategy,
        time_limit_seconds=config.candidate_time_limit_seconds,
        max_expanded_states=resolved_candidate_state_limit,
        search_stats=core_search_stats,
    )
    stage_timings["core_search"] = monotonic() - core_search_started
    complete_core_candidates = [
        candidate
        for candidate in generated_candidates
        if candidate.coverage >= config.coverage_threshold
    ]
    partial_core_candidates = [
        candidate
        for candidate in generated_candidates
        if candidate.coverage < config.coverage_threshold
    ]
    candidates = complete_core_candidates
    complete_core_keys = {
        _candidate_key(candidate.fragment_ids) for candidate in complete_core_candidates
    }
    gap_candidate_keys: set[tuple[str, ...]] = set()
    complete_gap_keys: set[tuple[str, ...]] = set()
    partial_gap_keys: set[tuple[str, ...]] = set()
    complete_gap_search_stats: dict[str, int | bool] = {
        "expanded_states": 0,
        "state_limit_reached": False,
        "time_limit_reached": False,
    }
    partial_gap_search_stats = dict(complete_gap_search_stats)
    gap_first_search_stats: dict[str, int | bool] = {}
    proposal_efficiency = 0.0
    stage_timings["complete_gap_search"] = 0.0
    stage_timings["partial_gap_search"] = 0.0
    if resolved_algorithm in ("effectiveness_gap", "v44_gap_first"):
        complete_gap_started = monotonic()
        # Both gap searches read the same per-fragment pixel caches.
        geometry = _AssemblyGeometry(
            observed, tolerance=config.tolerance, gap_fill_radius=config.gap_fill_radius
        )
        from_complete = augment_candidates_with_group_gap(
            observed,
            complete_core_candidates,
            label_filtered_scores,
            tolerance=config.tolerance,
            coverage_threshold=config.coverage_threshold,
            gap_fill_radius=config.gap_fill_radius,
            max_pieces=max_pieces,
            core_min_pieces=config.core_min_pieces,
            max_base_candidates=config.max_complete_core_candidates,
            base_selection_strategy=config.base_selection_strategy,
            proposal_pool=config.gap_proposal_pool,
            min_group_gap_score=config.min_group_gap_score,
            automatic_group_gap_score=config.automatic_group_gap_score,
            time_limit_seconds=config.candidate_time_limit_seconds,
            max_expanded_states=resolved_gap_state_limit,
            search_stats=complete_gap_search_stats,
            geometry=geometry,
        )
        stage_timings["complete_gap_search"] = monotonic() - complete_gap_started
        partial_gap_started = monotonic()
        from_partial = augment_candidates_with_group_gap(
            observed,
            partial_core_candidates,
            label_filtered_scores,
            tolerance=config.tolerance,
            coverage_threshold=config.coverage_threshold,
            gap_fill_radius=config.gap_fill_radius,
            max_pieces=max_pieces,
            core_min_pieces=config.core_min_pieces,
            max_base_candidates=config.max_partial_core_candidates,
            base_selection_strategy=config.base_selection_strategy,
            proposal_pool=config.gap_proposal_pool,
            min_group_gap_score=config.min_group_gap_score,
            automatic_group_gap_score=config.automatic_group_gap_score,
            time_limit_seconds=config.partial_gap_time_limit_seconds,
            max_expanded_states=resolved_partial_gap_state_limit,
            search_stats=partial_gap_search_stats,
            geometry=geometry,
        )
        stage_timings["partial_gap_search"] = monotonic() - partial_gap_started
        merged: dict[tuple[str, ...], AssemblyCandidate] = {}
        for candidate in (*from_complete, *from_partial):
            key = _candidate_key(candidate.fragment_ids)
            previous = merged.get(key)
            if previous is None or candidate.score > previous.score:
                merged[key] = candidate
        candidates = sorted(
            merged.values(), key=lambda item: (-item.score, item.fragment_ids)
        )
        from_complete_keys = {
            _candidate_key(candidate.fragment_ids) for candidate in from_complete
        }
        from_partial_keys = {
            _candidate_key(candidate.fragment_ids) for candidate in from_partial
        }
        gap_candidate_keys = set(merged) - complete_core_keys
        partial_gap_keys = (
            from_partial_keys - from_complete_keys - complete_core_keys
        )
        complete_gap_keys = gap_candidate_keys - partial_gap_keys
    if resolved_algorithm == "v44_gap_first" and config.enable_residual_gap_proposals:
        # Residual-gap-first proposals are layered ON TOP of the v4.3 group-gap
        # pool: v44 is a strict superset of effectiveness_gap, so getting the
        # true fragment set into the pool can only raise oracle recall.
        gap_first_started = monotonic()
        residual_regions = compute_residual_gap_components(
            observed,
            generated_candidates,
            coverage_threshold=config.coverage_threshold,
            gap_fill_radius=config.gap_fill_radius,
            max_core_candidates=config.gap_region_max_core_candidates,
        )
        gap_first_candidates = augment_candidates_gap_first(
            observed,
            generated_candidates,
            core_edges,
            label_filtered_scores,
            residual_regions,
            tolerance=config.tolerance,
            coverage_threshold=config.coverage_threshold,
            gap_fill_radius=config.gap_fill_radius,
            max_pieces=max_pieces,
            core_min_pieces=config.core_min_pieces,
            max_base_candidates=config.max_complete_core_candidates,
            proposal_pool="boundary_contact",
            gap_routing=config.gap_routing,
            min_group_gap_score=config.min_group_gap_score,
            automatic_group_gap_score=config.automatic_group_gap_score,
            fray_layers=config.fray_layers,
            min_proposal_effectiveness=config.gap_proposal_min_effectiveness,
            alpha=config.gap_proposal_alpha,
            gamma=config.gap_proposal_gamma,
            delta=config.gap_proposal_delta,
            eps=config.gap_proposal_eps,
            zeta=config.gap_proposal_zeta,
            time_limit_seconds=config.candidate_time_limit_seconds,
            max_expanded_states=resolved_gap_state_limit,
            search_stats=gap_first_search_stats,
        )
        stage_timings["gap_first"] = monotonic() - gap_first_started
        pool_v44: dict[tuple[str, ...], AssemblyCandidate] = {
            _candidate_key(candidate.fragment_ids): candidate
            for candidate in candidates
        }
        gap_first_only_keys: set[tuple[str, ...]] = set()
        for candidate in gap_first_candidates:
            key = _candidate_key(candidate.fragment_ids)
            if key not in complete_core_keys:
                gap_first_only_keys.add(key)
            previous = pool_v44.get(key)
            if previous is None or candidate.score > previous.score:
                pool_v44[key] = candidate
        candidates = sorted(
            pool_v44.values(), key=lambda item: (-item.score, item.fragment_ids)
        )
        gap_candidate_keys = set(pool_v44) - complete_core_keys
        complete_gap_keys = gap_candidate_keys - partial_gap_keys
        expanded = int(gap_first_search_stats.get("expanded_states", 0) or 0)
        proposals_made = int(
            gap_first_search_stats.get("gap_proposals_made", 0) or 0
        )
        budget_units = expanded if expanded > 0 else proposals_made
        proposal_efficiency = (
            len(gap_first_only_keys) / float(budget_units)
            if budget_units > 0
            else 0.0
        )
    cover_search_stats: dict[str, int | bool] = {}
    exact_cover_started = monotonic()
    selected = select_exact_cover_candidates(
        candidates,
        time_limit_seconds=config.cover_time_limit_seconds,
        max_search_nodes=resolved_cover_node_limit,
        search_stats=cover_search_stats,
        objective=config.cover_objective,
    )
    stage_timings["exact_cover"] = monotonic() - exact_cover_started
    selected_keys = {
        _candidate_key(candidate.fragment_ids) for candidate in selected
    }
    candidate_provenance_fingerprint = _fingerprint_records(
        f"{','.join(key)}|{source}|{candidate.evidence_level}|{candidate.gap_steps}"
        for candidate in candidates
        for key in (_candidate_key(candidate.fragment_ids),)
        for source in (
            _candidate_source(
                key,
                complete_core_keys=complete_core_keys,
                complete_gap_keys=complete_gap_keys,
                partial_gap_keys=partial_gap_keys,
            ),
        )
    )
    selected_solution_fingerprint = _fingerprint_records(
        ",".join(_candidate_key(candidate.fragment_ids)) for candidate in selected
    )
    edge_decisions = {
        level: sum(edge.evidence_level == level for edge in label_filtered_scores)
        for level in TEARFIT_EVIDENCE_LEVELS
    }
    candidate_decisions = {
        level: sum(candidate.evidence_level == level for candidate in candidates)
        for level in ("automatic", "review")
    }
    return TearFitReconstruction(
        config=config,
        resolved_algorithm=resolved_algorithm,
        median_fragment_fraction=median_fragment_fraction,
        max_pieces=max_pieces,
        expected_notes=expected_notes,
        resolved_candidate_state_limit=resolved_candidate_state_limit,
        resolved_gap_state_limit=resolved_gap_state_limit,
        resolved_partial_gap_state_limit=resolved_partial_gap_state_limit,
        resolved_cover_node_limit=resolved_cover_node_limit,
        all_scores=all_scores,
        label_filtered_scores=label_filtered_scores,
        core_edges=core_edges,
        generated_candidates=generated_candidates,
        partial_core_candidates=partial_core_candidates,
        candidates=candidates,
        selected=selected,
        complete_core_keys=frozenset(complete_core_keys),
        gap_candidate_keys=frozenset(gap_candidate_keys),
        complete_gap_keys=frozenset(complete_gap_keys),
        partial_gap_keys=frozenset(partial_gap_keys),
        selected_keys=frozenset(selected_keys),
        search_stats={
            "core": core_search_stats,
            "complete_gap": complete_gap_search_stats,
            "partial_gap": partial_gap_search_stats,
            "gap_first": gap_first_search_stats,
            "exact_cover": cover_search_stats,
        },
        stage_timings=stage_timings,
        edge_decisions=edge_decisions,
        candidate_decisions=candidate_decisions,
        proposal_efficiency=proposal_efficiency,
        selected_solution_fingerprint=selected_solution_fingerprint,
        candidate_provenance_fingerprint=candidate_provenance_fingerprint,
        edge_intervention_applied=edge_intervention is not None,
        intervention_removed_edges=intervention_removed_edges,
        intervention_removed_core_edges=intervention_removed_core_edges,
    )


def run_tearfit_reconstruction(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    max_pieces: int,
    expected_notes: int | None = None,
) -> dict:
    """Reconstruct a placed-fragment dataset with the frozen v4.4.1 core.

    The dataset is a ``save_dataset`` archive whose fragments already share the
    canonical frame, such as a ``reality-bridge`` handoff. When
    ``expected_notes`` is omitted it is estimated from total fragment area.
    Writes ``reconstruction_report.json`` and returns the same payload.
    Core evidence labels are shadow diagnostics, never release authorization.
    """

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    _template, fragments = load_dataset(dataset_path)
    if not fragments:
        raise ValueError(f"{dataset_path} contains no fragments")
    if expected_notes is None:
        canvas_pixels = float(fragments[0].mask.size)
        expected_notes = max(
            1, round(sum(fragment.area for fragment in fragments) / canvas_pixels)
        )
        expected_notes_source = "estimated_from_fragment_area"
    else:
        expected_notes_source = "argument"
    reconstruction = reconstruct_placed_fragments(
        fragments,
        TearFitCoreConfig.v4_4_1(),
        max_pieces=max_pieces,
        expected_notes=expected_notes,
    )
    assigned = {
        fragment_id
        for candidate in reconstruction.selected
        for fragment_id in candidate.fragment_ids
    }
    automatic_blockers = [
        "physical_resolution_unqualified",
        "native_seam_verifier_unqualified",
    ]
    for stage, stats in reconstruction.search_stats.items():
        if any(stats.get(flag, False) for flag in (
            "node_limit_reached", "state_limit_reached", "time_limit_reached",
        )):
            automatic_blockers.append(f"search_truncated:{stage}")
    report_path = output_dir / "reconstruction_report.json"
    report = {
        "tool": "moneyrepair",
        "stage": "tearfit_core_reconstruction",
        "core_preset": "v4.4.1",
        "claim_boundary": (
            "Runs the frozen v4.4.1 tear-fit core on fragments already placed in the "
            "canonical frame. Its pixel thresholds were calibrated on 180x90 simulation "
            "canvases and no physical resolution mapping is validated, so results are "
            "diagnostic only. Core automatic evidence labels are shadow classifications, "
            "not authorization for physical automatic confirmation."
        ),
        "inputs": {
            "dataset": str(dataset_path),
            "dataset_sha256": sha256(dataset_path.read_bytes()).hexdigest(),
            "fragments": len(fragments),
            "canvas_shape": list(fragments[0].mask.shape),
            "observed_meta_keys": list(TEARFIT_OBSERVED_META_KEYS),
        },
        "expected_notes_source": expected_notes_source,
        "reconstruction": reconstruction.to_jsonable(),
        "routing": {
            "release_mode": "diagnostic_only",
            "automatic_confirmation_enabled": False,
            "automatic_blockers": automatic_blockers,
            "automatic": [],
            "shadow_automatic": [
                list(candidate.fragment_ids)
                for candidate in reconstruction.selected
                if candidate.evidence_level == "automatic"
            ],
            "review": [
                list(candidate.fragment_ids)
                for candidate in reconstruction.selected
            ],
            "unassigned_fragment_ids": [
                fragment.id for fragment in fragments if fragment.id not in assigned
            ],
        },
        "outputs": {"report": str(report_path)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_tearfit_trial(
    config: FractalTearConfig,
    *,
    algorithm: str = "baseline",
    route_fragment_fraction_threshold: float = TEARFIT_V43_FINE_FRACTION,
    tolerance: int = 2,
    min_overlap_pixels: int = 14,
    min_effectiveness: float = 1.0,
    automatic_effectiveness: float = 2.0,
    min_contiguous_pixels: int = 3,
    automatic_contiguous_pixels: int = 5,
    core_min_pieces: int = 2,
    min_group_gap_score: float = 0.35,
    automatic_group_gap_score: float = 0.55,
    coverage_threshold: float = 0.93,
    core_raw_coverage_threshold: float | None = None,
    gap_fill_radius: int = 2,
    max_pieces: int | None = None,
    beam_width: int = 64,
    max_complete_core_candidates: int = 512,
    max_partial_core_candidates: int = 128,
    base_selection_strategy: str = "global",
    gap_proposal_pool: str = "weak_pair",
    use_labels: bool = True,
    seed_strategy: str = "anchor_priority",
    require_anchor: bool | None = None,
    serial_ocr_rate: float | None = None,
    candidate_time_limit_seconds: float | None = 20.0,
    candidate_state_limit: int | None = None,
    candidate_states_per_pair_score: float | None = None,
    partial_gap_time_limit_seconds: float | None = 5.0,
    gap_state_limit: int | None = None,
    gap_states_per_fragment: float | None = None,
    partial_gap_state_limit: int | None = None,
    partial_gap_states_per_fragment: float | None = None,
    cover_time_limit_seconds: float | None = 10.0,
    cover_node_limit: int | None = None,
    cover_nodes_per_note: float | None = None,
    cover_objective: str = "score_then_count",
    enable_residual_gap_proposals: bool = True,
    gap_routing: str = "complexity",
    gap_region_max_core_candidates: int = 8,
    gap_proposal_alpha: float = 1.0,
    gap_proposal_gamma: float = 1.0,
    gap_proposal_delta: float = 1.0,
    gap_proposal_eps: float = 1.0,
    gap_proposal_zeta: float = 1.0,
    gap_proposal_min_effectiveness: float = 0.0,
    diagnostic_oracle_drop_false_accepted_edges: bool = False,
    diagnostic_candidate_funnel: bool = False,
) -> TearFitTrialResult:
    """Run one labelled exact-cover tear-fit trial.

    The trial simulates fragments, reconstructs them with
    :func:`reconstruct_placed_fragments`, and only then scores the result
    against simulator truth.

    ``diagnostic_oracle_drop_false_accepted_edges`` is a simulation-only
    counterfactual. It reads ``note_id`` ground truth after pair scoring and
    removes accepted cross-note edges before candidate generation. It must
    never be used as a reconstruction feature or on real inputs.
    """

    trial_started = monotonic()
    core_config = TearFitCoreConfig(
        algorithm=algorithm,
        route_fragment_fraction_threshold=route_fragment_fraction_threshold,
        tolerance=tolerance,
        min_overlap_pixels=min_overlap_pixels,
        min_effectiveness=min_effectiveness,
        automatic_effectiveness=automatic_effectiveness,
        min_contiguous_pixels=min_contiguous_pixels,
        automatic_contiguous_pixels=automatic_contiguous_pixels,
        core_min_pieces=core_min_pieces,
        min_group_gap_score=min_group_gap_score,
        automatic_group_gap_score=automatic_group_gap_score,
        coverage_threshold=coverage_threshold,
        core_raw_coverage_threshold=core_raw_coverage_threshold,
        gap_fill_radius=gap_fill_radius,
        beam_width=beam_width,
        max_complete_core_candidates=max_complete_core_candidates,
        max_partial_core_candidates=max_partial_core_candidates,
        base_selection_strategy=base_selection_strategy,
        gap_proposal_pool=gap_proposal_pool,
        use_labels=use_labels,
        seed_strategy="anchor_priority" if require_anchor is not None else seed_strategy,
        candidate_time_limit_seconds=candidate_time_limit_seconds,
        candidate_state_limit=candidate_state_limit,
        candidate_states_per_pair_score=candidate_states_per_pair_score,
        partial_gap_time_limit_seconds=partial_gap_time_limit_seconds,
        gap_state_limit=gap_state_limit,
        gap_states_per_fragment=gap_states_per_fragment,
        partial_gap_state_limit=partial_gap_state_limit,
        partial_gap_states_per_fragment=partial_gap_states_per_fragment,
        cover_time_limit_seconds=cover_time_limit_seconds,
        cover_node_limit=cover_node_limit,
        cover_nodes_per_note=cover_nodes_per_note,
        cover_objective=cover_objective,
        enable_residual_gap_proposals=enable_residual_gap_proposals,
        gap_routing=gap_routing,
        gap_region_max_core_candidates=gap_region_max_core_candidates,
        gap_proposal_alpha=gap_proposal_alpha,
        gap_proposal_gamma=gap_proposal_gamma,
        gap_proposal_delta=gap_proposal_delta,
        gap_proposal_eps=gap_proposal_eps,
        gap_proposal_zeta=gap_proposal_zeta,
        gap_proposal_min_effectiveness=gap_proposal_min_effectiveness,
        fray_layers=config.fray_layers,
    )
    if serial_ocr_rate is not None:
        config = FractalTearConfig(
            **{**config.__dict__, "serial_ocr_rate": serial_ocr_rate}
        )
    simulation_started = monotonic()
    _template, fragments = make_fractal_tear_fragments(config)
    simulation_seconds = monotonic() - simulation_started
    edge_intervention: Callable[[TearFitEdge], bool] | None = None
    if diagnostic_oracle_drop_false_accepted_edges:
        missing_note_ids = [
            fragment.id for fragment in fragments if "note_id" not in fragment.meta
        ]
        if missing_note_ids:
            raise ValueError(
                "oracle false-edge deletion requires simulated note_id ground truth"
            )

        def is_true_pair(edge: TearFitEdge) -> bool:
            return (
                fragments[edge.left].meta["note_id"]
                == fragments[edge.right].meta["note_id"]
            )

        edge_intervention = is_true_pair
    reconstruction = reconstruct_placed_fragments(
        fragments,
        core_config,
        max_pieces=max_pieces or config.pieces_per_note + 2,
        expected_notes=config.notes,
        edge_intervention=edge_intervention,
    )
    stage_timings = {"simulation": simulation_seconds, **reconstruction.stage_timings}
    resolved_algorithm = reconstruction.resolved_algorithm
    all_scores = reconstruction.all_scores
    label_filtered_scores = reconstruction.label_filtered_scores
    core_edges = reconstruction.core_edges
    generated_candidates = reconstruction.generated_candidates
    candidates = reconstruction.candidates
    selected = reconstruction.selected
    gap_candidate_keys = reconstruction.gap_candidate_keys
    selected_keys = reconstruction.selected_keys
    diagnostics_started = monotonic()
    diagnostics = diagnose_confirmed_candidates(selected, fragments)
    stage_timings["diagnostics"] = monotonic() - diagnostics_started

    true_scored_edges = [
        edge
        for edge in all_scores
        if fragments[edge.left].meta.get("note_id")
        == fragments[edge.right].meta.get("note_id")
    ]
    true_scores = [
        edge.overlap_pixels if resolved_algorithm == "baseline" else edge.effectiveness
        for edge in true_scored_edges
    ]
    false_scores = [
        edge.overlap_pixels if resolved_algorithm == "baseline" else edge.effectiveness
        for edge in all_scores
        if fragments[edge.left].meta.get("note_id")
        != fragments[edge.right].meta.get("note_id")
    ]
    false_edges = [
        edge
        for edge in core_edges
        if fragments[edge.left].meta.get("note_id")
        != fragments[edge.right].meta.get("note_id")
    ]
    true_edges = [
        edge
        for edge in core_edges
        if fragments[edge.left].meta.get("note_id")
        == fragments[edge.right].meta.get("note_id")
    ]
    truth_by_key = _true_note_candidate_keys(fragments)
    truth_keys = set(truth_by_key)
    candidate_keys = {
        _candidate_key(candidate.fragment_ids) for candidate in candidates
    }
    candidate_funnel = (
        diagnose_gap_candidate_funnel(
            fragments,
            generated_candidates,
            candidates,
            label_filtered_scores,
            tolerance=tolerance,
            coverage_threshold=coverage_threshold,
            gap_fill_radius=gap_fill_radius,
            core_min_pieces=core_min_pieces,
            min_group_gap_score=min_group_gap_score,
            automatic_group_gap_score=automatic_group_gap_score,
            max_partial_core_candidates=max_partial_core_candidates,
            complete_base_limit=max_complete_core_candidates,
            base_selection_strategy=base_selection_strategy,
        )
        if diagnostic_candidate_funnel
        else {}
    )
    oracle_note_ids = {
        truth_by_key[key] for key in candidate_keys & truth_keys
    }
    true_gap_keys = gap_candidate_keys & truth_keys
    selected_gap_keys = selected_keys & gap_candidate_keys
    selected_true_gap_keys = selected_gap_keys & truth_keys
    stage_timings["total"] = monotonic() - trial_started
    return TearFitTrialResult(
        config={
            **config.__dict__,
            "algorithm": algorithm,
            "resolved_algorithm": resolved_algorithm,
            "median_fragment_fraction": reconstruction.median_fragment_fraction,
            "route_fragment_fraction_threshold": route_fragment_fraction_threshold,
            "tolerance": tolerance,
            "min_overlap_pixels": min_overlap_pixels,
            "min_effectiveness": min_effectiveness,
            "automatic_effectiveness": automatic_effectiveness,
            "min_contiguous_pixels": min_contiguous_pixels,
            "automatic_contiguous_pixels": automatic_contiguous_pixels,
            "core_min_pieces": core_min_pieces,
            "min_group_gap_score": min_group_gap_score,
            "automatic_group_gap_score": automatic_group_gap_score,
            "coverage_threshold": coverage_threshold,
            "core_raw_coverage_threshold": core_config.resolved_core_raw_coverage_threshold,
            "gap_fill_radius": gap_fill_radius,
            "beam_width": beam_width,
            "max_complete_core_candidates": max_complete_core_candidates,
            "max_partial_core_candidates": max_partial_core_candidates,
            "base_selection_strategy": base_selection_strategy,
            "gap_proposal_pool": gap_proposal_pool,
            "use_labels": use_labels,
            "seed_strategy": core_config.seed_strategy,
            "candidate_time_limit_seconds": candidate_time_limit_seconds,
            "candidate_state_limit": candidate_state_limit,
            "candidate_states_per_pair_score": candidate_states_per_pair_score,
            "resolved_candidate_state_limit": reconstruction.resolved_candidate_state_limit,
            "partial_gap_time_limit_seconds": partial_gap_time_limit_seconds,
            "gap_state_limit": gap_state_limit,
            "gap_states_per_fragment": gap_states_per_fragment,
            "resolved_gap_state_limit": reconstruction.resolved_gap_state_limit,
            "partial_gap_state_limit": partial_gap_state_limit,
            "partial_gap_states_per_fragment": partial_gap_states_per_fragment,
            "resolved_partial_gap_state_limit": reconstruction.resolved_partial_gap_state_limit,
            "cover_time_limit_seconds": cover_time_limit_seconds,
            "cover_node_limit": cover_node_limit,
            "cover_nodes_per_note": cover_nodes_per_note,
            "resolved_cover_node_limit": reconstruction.resolved_cover_node_limit,
            "cover_objective": cover_objective,
            "diagnostic_oracle_drop_false_accepted_edges": (
                diagnostic_oracle_drop_false_accepted_edges
            ),
            "diagnostic_candidate_funnel": diagnostic_candidate_funnel,
        },
        fragments=len(fragments),
        pair_scores=len(all_scores),
        accepted_edges=len(core_edges),
        true_possible_edges=len(true_scored_edges),
        true_accepted_edges=len(true_edges),
        false_accepted_edges=len(false_edges),
        false_edge_rate=len(false_edges) / len(core_edges) if core_edges else 0.0,
        true_edge_recall=(
            len(true_edges) / len(true_scored_edges) if true_scored_edges else 0.0
        ),
        true_edge_median=float(np.median(true_scores)) if true_scores else 0.0,
        false_edge_median=float(np.median(false_scores)) if false_scores else 0.0,
        candidates=len(candidates),
        diagnostics=diagnostics,
        edge_decisions=reconstruction.edge_decisions,
        candidate_decisions=reconstruction.candidate_decisions,
        search_stats=reconstruction.search_stats,
        core_candidates=len(generated_candidates),
        partial_core_candidates=len(reconstruction.partial_core_candidates),
        gap_candidates=len(gap_candidate_keys),
        partial_gap_candidates=len(reconstruction.partial_gap_keys),
        selected_core_candidates=len(selected_keys & reconstruction.complete_core_keys),
        selected_gap_candidates=len(selected_keys & gap_candidate_keys),
        selected_complete_gap_candidates=len(
            selected_keys & reconstruction.complete_gap_keys
        ),
        selected_partial_gap_candidates=len(
            selected_keys & reconstruction.partial_gap_keys
        ),
        true_gap_candidates=len(true_gap_keys),
        false_gap_candidates=len(gap_candidate_keys - truth_keys),
        selected_true_gap_candidates=len(selected_true_gap_keys),
        selected_false_gap_candidates=len(selected_gap_keys - truth_keys),
        oracle_candidate_notes=len(oracle_note_ids),
        oracle_candidate_recall=(
            len(oracle_note_ids) / len(truth_keys) if truth_keys else 0.0
        ),
        selected_score_total=float(sum(candidate.score for candidate in selected)),
        selected_solution_fingerprint=reconstruction.selected_solution_fingerprint,
        candidate_provenance_fingerprint=reconstruction.candidate_provenance_fingerprint,
        stage_timings=stage_timings,
        diagnostic_removed_false_accepted_edges=(
            reconstruction.intervention_removed_edges
        ),
        diagnostic_removed_false_core_edges=reconstruction.intervention_removed_core_edges,
        candidate_funnel=candidate_funnel,
        proposal_efficiency=reconstruction.proposal_efficiency,
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
    serial_ocr_rate: float = 0.6,
    seed_strategy: str = "anchor_priority",
    require_anchor: bool | None = None,
    ensure_serial_anchor: bool = False,
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


def tearfit_comparison_cases(
    profile: str = "smoke",
) -> tuple[TearFitComparisonCase, ...]:
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
            TearFitComparisonCase(
                "fray_n50_p8", notes=50, pieces_per_note=8, fray_probability=0.40
            ),
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
            raise ValueError(
                f"cover_objective must be one of: {', '.join(TEARFIT_COVER_OBJECTIVES)}"
            )
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
    summary: list[dict[str, Any]] = []
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


def run_tearfit_v43_ablation(
    *,
    notes: int = 20,
    pieces_list: Iterable[int] = (8, 16, 24),
    seeds: Iterable[int] = (7, 8, 9),
    algorithms: Iterable[str] = TEARFIT_ALGORITHMS,
    route_fragment_fraction_threshold: float = TEARFIT_V43_FINE_FRACTION,
    width: int = 180,
    height: int = 90,
    tolerance: int = 2,
    min_overlap_pixels: int = 14,
    min_effectiveness: float = 1.0,
    automatic_effectiveness: float = 2.0,
    min_contiguous_pixels: int = 3,
    automatic_contiguous_pixels: int = 5,
    coverage_threshold: float = 0.93,
    core_raw_coverage_threshold: float | None = None,
    gap_fill_radius: int = 2,
    beam_width: int = 32,
    max_partial_core_candidates: int = 128,
    candidate_time_limit_seconds: float | None = 15.0,
    candidate_state_limit: int | None = 100_000,
    candidate_states_per_pair_score: float | None = None,
    partial_gap_time_limit_seconds: float | None = 5.0,
    gap_state_limit: int | None = 20_000,
    gap_states_per_fragment: float | None = None,
    partial_gap_state_limit: int | None = 5_000,
    partial_gap_states_per_fragment: float | None = None,
    cover_time_limit_seconds: float | None = 5.0,
    cover_node_limit: int | None = 250_000,
    cover_nodes_per_note: float | None = None,
    enable_residual_gap_proposals: bool = True,
    gap_routing: str = "complexity",
    gap_region_max_core_candidates: int = 8,
    gap_proposal_alpha: float = 1.0,
    gap_proposal_gamma: float = 1.0,
    gap_proposal_delta: float = 1.0,
    gap_proposal_eps: float = 1.0,
    gap_proposal_zeta: float = 1.0,
    gap_proposal_min_effectiveness: float = 0.0,
) -> dict:
    """Run same-seed baseline -> Etear -> group-gap head-to-head trials."""

    algorithms = tuple(algorithms)
    pieces_values = tuple(int(value) for value in pieces_list)
    seed_values = tuple(int(value) for value in seeds)
    for algorithm in algorithms:
        if algorithm not in TEARFIT_ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of: {', '.join(TEARFIT_ALGORITHMS)}"
            )
    rows: list[dict] = []
    for pieces in pieces_values:
        for seed in seed_values:
            cached_trials: dict[str, tuple[TearFitTrialResult, float, str]] = {}
            median_fragment_fraction: float | None = None
            for algorithm in algorithms:
                reused_from_algorithm: str | None = None
                cached: tuple[TearFitTrialResult, float, str] | None = None
                if algorithm == "v43_routed" and median_fragment_fraction is not None:
                    route = (
                        "baseline"
                        if median_fragment_fraction >= route_fragment_fraction_threshold
                        else "effectiveness_gap"
                    )
                    cached = cached_trials.get(route)
                if cached is not None:
                    result, elapsed_seconds, reused_from_algorithm = cached
                else:
                    started = monotonic()
                    result = run_tearfit_trial(
                        FractalTearConfig(
                            notes=notes,
                            pieces_per_note=pieces,
                            width=width,
                            height=height,
                            seed=seed,
                            serial_ocr_rate=0.0,
                        ),
                        algorithm=algorithm,
                        route_fragment_fraction_threshold=route_fragment_fraction_threshold,
                        tolerance=tolerance,
                        min_overlap_pixels=min_overlap_pixels,
                        min_effectiveness=min_effectiveness,
                        automatic_effectiveness=automatic_effectiveness,
                        min_contiguous_pixels=min_contiguous_pixels,
                        automatic_contiguous_pixels=automatic_contiguous_pixels,
                        coverage_threshold=coverage_threshold,
                        core_raw_coverage_threshold=core_raw_coverage_threshold,
                        gap_fill_radius=gap_fill_radius,
                        beam_width=beam_width,
                        max_partial_core_candidates=max_partial_core_candidates,
                        use_labels=False,
                        candidate_time_limit_seconds=candidate_time_limit_seconds,
                        candidate_state_limit=candidate_state_limit,
                        candidate_states_per_pair_score=candidate_states_per_pair_score,
                        partial_gap_time_limit_seconds=partial_gap_time_limit_seconds,
                        gap_state_limit=gap_state_limit,
                        gap_states_per_fragment=gap_states_per_fragment,
                        partial_gap_state_limit=partial_gap_state_limit,
                        partial_gap_states_per_fragment=partial_gap_states_per_fragment,
                        cover_time_limit_seconds=cover_time_limit_seconds,
                        cover_node_limit=cover_node_limit,
                        cover_nodes_per_note=cover_nodes_per_note,
                        enable_residual_gap_proposals=enable_residual_gap_proposals,
                        gap_routing=gap_routing,
                        gap_region_max_core_candidates=gap_region_max_core_candidates,
                        gap_proposal_alpha=gap_proposal_alpha,
                        gap_proposal_gamma=gap_proposal_gamma,
                        gap_proposal_delta=gap_proposal_delta,
                        gap_proposal_eps=gap_proposal_eps,
                        gap_proposal_zeta=gap_proposal_zeta,
                        gap_proposal_min_effectiveness=gap_proposal_min_effectiveness,
                    )
                    elapsed_seconds = monotonic() - started
                    resolved = str(result.config["resolved_algorithm"])
                    cached_trials.setdefault(
                        resolved, (result, elapsed_seconds, algorithm)
                    )
                    median_fragment_fraction = float(
                        result.config["median_fragment_fraction"]
                    )
                diagnostics = result.diagnostics
                rows.append(
                    {
                        "algorithm": algorithm,
                        "resolved_algorithm": result.config["resolved_algorithm"],
                        "notes": notes,
                        "pieces_per_note": pieces,
                        "seed": seed,
                        "elapsed_seconds": elapsed_seconds,
                        "reused_from_algorithm": reused_from_algorithm,
                        "fragments": result.fragments,
                        "pair_scores": result.pair_scores,
                        "accepted_edges": result.accepted_edges,
                        "true_possible_edges": result.true_possible_edges,
                        "true_accepted_edges": result.true_accepted_edges,
                        "false_accepted_edges": result.false_accepted_edges,
                        "false_edge_rate": result.false_edge_rate,
                        "true_edge_recall": result.true_edge_recall,
                        "edge_decisions": result.edge_decisions,
                        "candidates": result.candidates,
                        "core_candidates": result.core_candidates,
                        "partial_core_candidates": result.partial_core_candidates,
                        "gap_candidates": result.gap_candidates,
                        "partial_gap_candidates": result.partial_gap_candidates,
                        "selected_core_candidates": result.selected_core_candidates,
                        "selected_gap_candidates": result.selected_gap_candidates,
                        "selected_complete_gap_candidates": result.selected_complete_gap_candidates,
                        "selected_partial_gap_candidates": result.selected_partial_gap_candidates,
                        "true_gap_candidates": result.true_gap_candidates,
                        "false_gap_candidates": result.false_gap_candidates,
                        "selected_true_gap_candidates": result.selected_true_gap_candidates,
                        "selected_false_gap_candidates": result.selected_false_gap_candidates,
                        "oracle_candidate_notes": result.oracle_candidate_notes,
                        "oracle_candidate_recall": result.oracle_candidate_recall,
                        "proposal_efficiency": result.proposal_efficiency,
                        "selected_score_total": result.selected_score_total,
                        "selected_solution_fingerprint": result.selected_solution_fingerprint,
                        "candidate_provenance_fingerprint": result.candidate_provenance_fingerprint,
                        "stage_timings": result.stage_timings,
                        "budgets": {
                            "candidate_state_limit": result.config["resolved_candidate_state_limit"],
                            "gap_state_limit": result.config["resolved_gap_state_limit"],
                            "partial_gap_state_limit": result.config[
                                "resolved_partial_gap_state_limit"
                            ],
                            "cover_node_limit": result.config["resolved_cover_node_limit"],
                        },
                        "candidate_decisions": result.candidate_decisions,
                        "search_stats": result.search_stats,
                        "confirmed": diagnostics.confirmed,
                        "exact_confirmed": diagnostics.exact_confirmed,
                        "exact_yield": diagnostics.exact_yield,
                        "exact_precision": diagnostics.exact_precision,
                        "automatic_exact_yield": diagnostics.automatic_exact_yield,
                        "automatic_exact_precision": diagnostics.automatic_exact_precision,
                        "pure_precision": diagnostics.pure_precision,
                        "manual_notes_remaining": diagnostics.manual_notes_remaining,
                        "automatic_candidates": diagnostics.automatic_candidates,
                        "review_candidates": diagnostics.review_candidates,
                    }
                )

    summary: list[dict] = []
    for pieces in pieces_values:
        for algorithm in algorithms:
            selected = [
                row
                for row in rows
                if row["pieces_per_note"] == pieces and row["algorithm"] == algorithm
            ]
            if not selected:
                continue
            summary.append(
                {
                    "algorithm": algorithm,
                    "pieces_per_note": pieces,
                    "runs": len(selected),
                    "mean_exact_yield": float(
                        np.mean([row["exact_yield"] for row in selected])
                    ),
                    "mean_exact_precision": float(
                        np.mean([row["exact_precision"] for row in selected])
                    ),
                    "min_exact_precision": float(
                        np.min([row["exact_precision"] for row in selected])
                    ),
                    "mean_automatic_exact_yield": float(
                        np.mean([row["automatic_exact_yield"] for row in selected])
                    ),
                    "mean_automatic_exact_precision": float(
                        np.mean([row["automatic_exact_precision"] for row in selected])
                    ),
                    "mean_pure_precision": float(
                        np.mean([row["pure_precision"] for row in selected])
                    ),
                    "mean_false_edge_rate": float(
                        np.mean([row["false_edge_rate"] for row in selected])
                    ),
                    "mean_pair_scores": float(
                        np.mean([row["pair_scores"] for row in selected])
                    ),
                    "mean_accepted_edges": float(
                        np.mean([row["accepted_edges"] for row in selected])
                    ),
                    "mean_true_accepted_edges": float(
                        np.mean([row["true_accepted_edges"] for row in selected])
                    ),
                    "mean_true_possible_edges": float(
                        np.mean([row["true_possible_edges"] for row in selected])
                    ),
                    "mean_true_edge_recall": float(
                        np.mean([row["true_edge_recall"] for row in selected])
                    ),
                    "mean_false_accepted_edges": float(
                        np.mean([row["false_accepted_edges"] for row in selected])
                    ),
                    "mean_candidates": float(
                        np.mean([row["candidates"] for row in selected])
                    ),
                    "mean_core_candidates": float(
                        np.mean([row["core_candidates"] for row in selected])
                    ),
                    "mean_partial_core_candidates": float(
                        np.mean([row["partial_core_candidates"] for row in selected])
                    ),
                    "mean_gap_candidates": float(
                        np.mean([row["gap_candidates"] for row in selected])
                    ),
                    "mean_partial_gap_candidates": float(
                        np.mean([row["partial_gap_candidates"] for row in selected])
                    ),
                    "mean_selected_core_candidates": float(
                        np.mean(
                            [row["selected_core_candidates"] for row in selected]
                        )
                    ),
                    "mean_selected_gap_candidates": float(
                        np.mean(
                            [row["selected_gap_candidates"] for row in selected]
                        )
                    ),
                    "mean_selected_complete_gap_candidates": float(
                        np.mean(
                            [
                                row["selected_complete_gap_candidates"]
                                for row in selected
                            ]
                        )
                    ),
                    "mean_selected_partial_gap_candidates": float(
                        np.mean(
                            [row["selected_partial_gap_candidates"] for row in selected]
                        )
                    ),
                    "mean_true_gap_candidates": float(
                        np.mean([row["true_gap_candidates"] for row in selected])
                    ),
                    "mean_false_gap_candidates": float(
                        np.mean([row["false_gap_candidates"] for row in selected])
                    ),
                    "mean_selected_true_gap_candidates": float(
                        np.mean(
                            [row["selected_true_gap_candidates"] for row in selected]
                        )
                    ),
                    "mean_selected_false_gap_candidates": float(
                        np.mean(
                            [row["selected_false_gap_candidates"] for row in selected]
                        )
                    ),
                    "mean_oracle_candidate_recall": float(
                        np.mean([row["oracle_candidate_recall"] for row in selected])
                    ),
                    "mean_proposal_efficiency": float(
                        np.mean([row["proposal_efficiency"] for row in selected])
                    ),
                    "mean_manual_notes_remaining": float(
                        np.mean([row["manual_notes_remaining"] for row in selected])
                    ),
                    "mean_elapsed_seconds": float(
                        np.mean([row["elapsed_seconds"] for row in selected])
                    ),
                }
            )

    baseline_by_pieces = {
        row["pieces_per_note"]: row for row in summary if row["algorithm"] == "baseline"
    }
    for row in summary:
        baseline = baseline_by_pieces.get(row["pieces_per_note"])
        if baseline is None:
            continue
        row["delta_exact_yield_vs_baseline"] = (
            row["mean_exact_yield"] - baseline["mean_exact_yield"]
        )
        row["delta_exact_precision_vs_baseline"] = (
            row["mean_exact_precision"] - baseline["mean_exact_precision"]
        )
        row["delta_automatic_exact_yield_vs_baseline"] = (
            row["mean_automatic_exact_yield"] - baseline["mean_automatic_exact_yield"]
        )
        row["delta_automatic_exact_precision_vs_baseline"] = (
            row["mean_automatic_exact_precision"]
            - baseline["mean_automatic_exact_precision"]
        )
        row["delta_false_edge_rate_vs_baseline"] = (
            row["mean_false_edge_rate"] - baseline["mean_false_edge_rate"]
        )
        row["delta_manual_notes_vs_baseline"] = (
            row["mean_manual_notes_remaining"] - baseline["mean_manual_notes_remaining"]
        )

    summary_by_key = {
        (row["pieces_per_note"], row["algorithm"]): row for row in summary
    }
    comparison_metrics = {
        "automatic_exact_yield": "mean_automatic_exact_yield",
        "automatic_exact_precision": "mean_automatic_exact_precision",
        "false_edge_rate": "mean_false_edge_rate",
        "accepted_edges": "mean_accepted_edges",
        "false_accepted_edges": "mean_false_accepted_edges",
        "candidates": "mean_candidates",
        "gap_candidates": "mean_gap_candidates",
        "selected_gap_candidates": "mean_selected_gap_candidates",
        "selected_partial_gap_candidates": "mean_selected_partial_gap_candidates",
        "oracle_candidate_recall": "mean_oracle_candidate_recall",
        "proposal_efficiency": "mean_proposal_efficiency",
        "manual_notes_remaining": "mean_manual_notes_remaining",
        "elapsed_seconds": "mean_elapsed_seconds",
    }
    mechanism_comparisons: list[dict] = []
    for pieces in pieces_values:
        for stage, previous_algorithm, current_algorithm in (
            ("adaptive_edge", "baseline", "effectiveness"),
            ("group_gap", "effectiveness", "effectiveness_gap"),
            ("routing", "effectiveness_gap", "v43_routed"),
            ("gap_first", "effectiveness_gap", "v44_gap_first"),
        ):
            previous = summary_by_key.get((pieces, previous_algorithm))
            current = summary_by_key.get((pieces, current_algorithm))
            if previous is None or current is None:
                continue
            previous_metrics = {
                name: float(previous[field])
                for name, field in comparison_metrics.items()
            }
            current_metrics = {
                name: float(current[field])
                for name, field in comparison_metrics.items()
            }
            item = {
                "stage": stage,
                "pieces_per_note": pieces,
                "from_algorithm": previous_algorithm,
                "to_algorithm": current_algorithm,
                "from": previous_metrics,
                "to": current_metrics,
                "delta": {
                    name: current_metrics[name] - previous_metrics[name]
                    for name in comparison_metrics
                },
            }
            if stage == "routing":
                routed_rows = [
                    row
                    for row in rows
                    if row["pieces_per_note"] == pieces
                    and row["algorithm"] == "v43_routed"
                ]
                item["resolved_algorithms"] = sorted(
                    {str(row["resolved_algorithm"]) for row in routed_rows}
                )
                item["reused_identical_trials"] = all(
                    row["reused_from_algorithm"] is not None for row in routed_rows
                )
            mechanism_comparisons.append(item)
    return {
        "config": {
            "schema_version": "4.3.1",
            "notes": notes,
            "pieces_list": pieces_values,
            "seeds": seed_values,
            "algorithms": algorithms,
            "route_fragment_fraction_threshold": route_fragment_fraction_threshold,
            "width": width,
            "height": height,
            "tolerance": tolerance,
            "min_overlap_pixels": min_overlap_pixels,
            "min_effectiveness": min_effectiveness,
            "automatic_effectiveness": automatic_effectiveness,
            "min_contiguous_pixels": min_contiguous_pixels,
            "automatic_contiguous_pixels": automatic_contiguous_pixels,
            "coverage_threshold": coverage_threshold,
            "core_raw_coverage_threshold": (
                max(0.05, coverage_threshold - 0.15)
                if core_raw_coverage_threshold is None
                else core_raw_coverage_threshold
            ),
            "gap_fill_radius": gap_fill_radius,
            "beam_width": beam_width,
            "max_partial_core_candidates": max_partial_core_candidates,
            "candidate_time_limit_seconds": candidate_time_limit_seconds,
            "candidate_state_limit": candidate_state_limit,
            "candidate_states_per_pair_score": candidate_states_per_pair_score,
            "partial_gap_time_limit_seconds": partial_gap_time_limit_seconds,
            "gap_state_limit": gap_state_limit,
            "gap_states_per_fragment": gap_states_per_fragment,
            "partial_gap_state_limit": partial_gap_state_limit,
            "partial_gap_states_per_fragment": partial_gap_states_per_fragment,
            "cover_time_limit_seconds": cover_time_limit_seconds,
            "cover_node_limit": cover_node_limit,
            "cover_nodes_per_note": cover_nodes_per_note,
            "serial_ocr_rate": 0.0,
            "use_labels": False,
        },
        "rows": rows,
        "summary": summary,
        "mechanism_comparisons": mechanism_comparisons,
    }


def _parse_notes_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the placed tear-fit research sandbox"
    )
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
    parser.add_argument("--serial-ocr-rate", type=float, default=0.6)
    parser.add_argument(
        "--seed-strategy", choices=TEARFIT_SEED_STRATEGIES, default="anchor_priority"
    )
    parser.add_argument("--ensure-serial-anchor", action="store_true")
    parser.add_argument("--ideal-serial-upper-bound", action="store_true")
    parser.add_argument("--candidate-time-limit", type=float, default=20.0)
    parser.add_argument("--cover-time-limit", type=float, default=10.0)
    parser.add_argument(
        "--cover-objective",
        choices=TEARFIT_COVER_OBJECTIVES,
        default="score_then_count",
    )
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
        serial_ocr_rate=1.0 if args.ideal_serial_upper_bound else args.serial_ocr_rate,
        seed_strategy=args.seed_strategy,
        ensure_serial_anchor=True
        if args.ideal_serial_upper_bound
        else args.ensure_serial_anchor,
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
