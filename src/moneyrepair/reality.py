from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numba
import numpy as np
from PIL import Image

from moneyrepair.ingest import load_mask, load_rgb, raw_fragments_from_manifest
from moneyrepair.locator import (
    CandidatePose,
    CoarseGridSpec,
    PoseSearchAudit,
    _crop_foreground,
    _rotate_image_and_mask,
    locate_fragment_poses,
)
from moneyrepair.quality import segmentation_confidence
from moneyrepair.scan import connected_components
from moneyrepair.simulate import make_synthetic_fragments, save_dataset, synthetic_banknote
from moneyrepair.types import Fragment


REALITY_ORIENTATION_MODES = ("cardinal", "free")


@dataclass(frozen=True)
class EvaluationAnnotation:
    """Ground truth kept outside production fragment objects."""

    fragment_id: str
    side: str
    crop_to_canonical_transform: np.ndarray
    ground_truth_mask: Path | None = None
    parent_id: str | None = None
    physical_fragment_id: str | None = None
    observation_pixels_per_mm: float | None = None
    canonical_pixels_per_mm: float | None = None
    effective_radius_mm: float | None = None

    def __post_init__(self) -> None:
        transform = np.asarray(self.crop_to_canonical_transform, dtype=np.float64)
        if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
            raise ValueError("crop_to_canonical_transform must be a finite 3x3 matrix")
        if abs(float(np.linalg.det(transform))) < 1e-12:
            raise ValueError("crop_to_canonical_transform must be invertible")
        calibration_values = (
            self.observation_pixels_per_mm,
            self.canonical_pixels_per_mm,
            self.effective_radius_mm,
        )
        if any(
            value is not None and (not math.isfinite(value) or value <= 0.0)
            for value in calibration_values
        ):
            raise ValueError("annotation calibration values must be positive when provided")
        object.__setattr__(self, "crop_to_canonical_transform", transform)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "ground_truth_side": self.side,
            "crop_to_canonical_transform": self.crop_to_canonical_transform.tolist(),
            "ground_truth_mask": str(self.ground_truth_mask) if self.ground_truth_mask else None,
            "parent_id": self.parent_id,
            "physical_fragment_id": self.physical_fragment_id,
            "observation_pixels_per_mm": self.observation_pixels_per_mm,
            "canonical_pixels_per_mm": self.canonical_pixels_per_mm,
            "effective_radius_mm": self.effective_radius_mm,
        }


@dataclass(frozen=True)
class PhysicalToleranceModel:
    """Convert declared acquisition tolerances into pixel-space gates."""

    scan_dpi: float = 300.0
    segmentation_tolerance_mm: float = 0.085
    registration_tolerance_mm: float = 0.085
    minimum_fragment_span_mm: float = 20.0
    effectiveness_ratio: float = 1.0
    overlay_height_mm: float = 0.0

    def __post_init__(self) -> None:
        if self.scan_dpi <= 0.0:
            raise ValueError("scan_dpi must be positive")
        if self.segmentation_tolerance_mm < 0.0 or self.registration_tolerance_mm < 0.0:
            raise ValueError("physical tolerances must be non-negative")
        if self.minimum_fragment_span_mm <= 0.0:
            raise ValueError("minimum_fragment_span_mm must be positive")
        if self.effectiveness_ratio < 0.0 or self.overlay_height_mm < 0.0:
            raise ValueError("effectiveness_ratio and overlay_height_mm must be non-negative")

    @property
    def pixels_per_mm(self) -> float:
        return self.scan_dpi / 25.4

    @property
    def segmentation_tolerance_pixels(self) -> float:
        return self.segmentation_tolerance_mm * self.pixels_per_mm

    def segmentation_tolerance_pixels_for(self, pixels_per_mm: float | None = None) -> float:
        resolved_pixels_per_mm = self.pixels_per_mm if pixels_per_mm is None else pixels_per_mm
        return self.segmentation_tolerance_mm * resolved_pixels_per_mm

    @property
    def combined_alignment_tolerance_mm(self) -> float:
        return self.segmentation_tolerance_mm + self.registration_tolerance_mm

    @property
    def combined_alignment_tolerance_pixels(self) -> float:
        return self.combined_alignment_tolerance_mm * self.pixels_per_mm

    def combined_alignment_tolerance_pixels_for(self, pixels_per_mm: float | None = None) -> float:
        resolved_pixels_per_mm = self.pixels_per_mm if pixels_per_mm is None else pixels_per_mm
        return self.combined_alignment_tolerance_mm * resolved_pixels_per_mm

    @property
    def angular_tolerance_degrees(self) -> float:
        radius = max(self.minimum_fragment_span_mm / 2.0, 1e-9)
        return math.degrees(math.atan(self.combined_alignment_tolerance_mm / radius))

    def angular_tolerance_degrees_for(self, effective_radius_mm: float | None = None) -> float:
        radius = (
            self.minimum_fragment_span_mm / 2.0
            if effective_radius_mm is None
            else effective_radius_mm
        )
        return math.degrees(math.atan(self.combined_alignment_tolerance_mm / max(radius, 1e-9)))

    @property
    def minimum_effective_feature_mm(self) -> float:
        combined = self.combined_alignment_tolerance_mm
        return (
            self.effectiveness_ratio * (2.0 * combined + 0.25 * self.overlay_height_mm)
            + combined
        )

    @property
    def minimum_effective_feature_pixels(self) -> float:
        return self.minimum_effective_feature_mm * self.pixels_per_mm

    def to_dict(self) -> dict[str, float]:
        return {
            "scan_dpi": self.scan_dpi,
            "segmentation_tolerance_mm": self.segmentation_tolerance_mm,
            "registration_tolerance_mm": self.registration_tolerance_mm,
            "minimum_fragment_span_mm": self.minimum_fragment_span_mm,
            "effectiveness_ratio": self.effectiveness_ratio,
            "overlay_height_mm": self.overlay_height_mm,
            "pixels_per_mm": self.pixels_per_mm,
            "segmentation_tolerance_pixels": self.segmentation_tolerance_pixels,
            "combined_alignment_tolerance_mm": self.combined_alignment_tolerance_mm,
            "combined_alignment_tolerance_pixels": self.combined_alignment_tolerance_pixels,
            "angular_tolerance_degrees": self.angular_tolerance_degrees,
            "minimum_effective_feature_mm": self.minimum_effective_feature_mm,
            "minimum_effective_feature_pixels": self.minimum_effective_feature_pixels,
        }


@dataclass(frozen=True)
class RealityBridgeThresholds:
    """Predeclared gates for the acquisition-to-pose diagnostic funnel."""

    min_segmentation_confidence: float = 0.55
    max_translation_error: float | None = None
    max_angle_error_degrees: float | None = None
    min_pose_score: float = 0.70
    min_global_score_margin: float = 0.01
    max_translation_sigma: float = 2.5
    max_angle_sigma_degrees: float = 5.0
    uncertainty_interval_scale: float = 2.0
    max_interior_missing_fraction: float = 0.01
    max_interior_extraneous_fraction: float = 0.01
    max_extraneous_component_fraction: float = 0.01
    physical_tolerance: PhysicalToleranceModel = PhysicalToleranceModel()

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_segmentation_confidence <= 1.0):
            raise ValueError("min_segmentation_confidence must be in [0, 1]")
        if self.max_translation_error is not None and self.max_translation_error < 0.0:
            raise ValueError("max_translation_error must be non-negative")
        if self.max_angle_error_degrees is not None and self.max_angle_error_degrees < 0.0:
            raise ValueError("max_angle_error_degrees must be non-negative")
        if not (0.0 <= self.min_pose_score <= 1.0):
            raise ValueError("min_pose_score must be in [0, 1]")
        if self.min_global_score_margin < 0.0:
            raise ValueError("min_global_score_margin must be non-negative")
        if self.max_translation_sigma < 0.0 or self.max_angle_sigma_degrees < 0.0:
            raise ValueError("uncertainty limits must be non-negative")
        if self.uncertainty_interval_scale <= 0.0:
            raise ValueError("uncertainty_interval_scale must be positive")
        fractions = (
            self.max_interior_missing_fraction,
            self.max_interior_extraneous_fraction,
            self.max_extraneous_component_fraction,
        )
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise ValueError("mask area fractions must be in [0, 1]")

    @property
    def resolved_translation_error(self) -> float:
        if self.max_translation_error is not None:
            return self.max_translation_error
        return self.physical_tolerance.combined_alignment_tolerance_pixels

    @property
    def resolved_angle_error_degrees(self) -> float:
        if self.max_angle_error_degrees is not None:
            return self.max_angle_error_degrees
        return self.physical_tolerance.angular_tolerance_degrees

    def resolved_segmentation_tolerance_pixels(
        self,
        annotation: EvaluationAnnotation | None,
    ) -> float:
        pixels_per_mm = annotation.observation_pixels_per_mm if annotation is not None else None
        return self.physical_tolerance.segmentation_tolerance_pixels_for(pixels_per_mm)

    def resolved_translation_error_for(self, annotation: EvaluationAnnotation | None) -> float:
        if self.max_translation_error is not None:
            return self.max_translation_error
        pixels_per_mm = annotation.canonical_pixels_per_mm if annotation is not None else None
        return self.physical_tolerance.combined_alignment_tolerance_pixels_for(pixels_per_mm)

    def resolved_angle_error_degrees_for(self, annotation: EvaluationAnnotation | None) -> float:
        if self.max_angle_error_degrees is not None:
            return self.max_angle_error_degrees
        effective_radius_mm = annotation.effective_radius_mm if annotation is not None else None
        return self.physical_tolerance.angular_tolerance_degrees_for(effective_radius_mm)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_segmentation_confidence": self.min_segmentation_confidence,
            "max_translation_error": self.max_translation_error,
            "max_angle_error_degrees": self.max_angle_error_degrees,
            "resolved_translation_error": self.resolved_translation_error,
            "resolved_angle_error_degrees": self.resolved_angle_error_degrees,
            "min_pose_score": self.min_pose_score,
            "min_global_score_margin": self.min_global_score_margin,
            "max_translation_sigma": self.max_translation_sigma,
            "max_angle_sigma_degrees": self.max_angle_sigma_degrees,
            "uncertainty_interval_scale": self.uncertainty_interval_scale,
            "max_interior_missing_fraction": self.max_interior_missing_fraction,
            "max_interior_extraneous_fraction": self.max_interior_extraneous_fraction,
            "max_extraneous_component_fraction": self.max_extraneous_component_fraction,
            "physical_tolerance": self.physical_tolerance.to_dict(),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _resolve_path(base: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _load_reference(
    value: str | Path | np.ndarray | None,
    *,
    manifest_value: str | None,
    manifest_base: Path,
    name: str,
) -> tuple[np.ndarray, dict[str, str]]:
    resolved: str | Path | np.ndarray | None = value
    if resolved is None:
        resolved = _resolve_path(manifest_base, manifest_value)
    if resolved is None:
        raise ValueError(f"missing {name} reference; pass it explicitly or add references.{name} to the manifest")
    if isinstance(resolved, np.ndarray):
        return resolved, {"source": "numpy_array", "sha256": _array_sha256(resolved)}
    path = Path(resolved)
    return load_rgb(path), {"source": str(path), "sha256": _sha256(path)}


def _angle_error(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _translation_matrix(tx: float, ty: float) -> np.ndarray:
    return np.asarray([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def _image_rotation_transform(width: int, height: int, angle: float, output_shape: tuple[int, int]) -> np.ndarray:
    """Map input pixel centres to a PIL/NumPy-style expanded rotation."""

    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    linear = np.asarray(
        [[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    output_height, output_width = output_shape
    return (
        _translation_matrix((output_width - 1.0) / 2.0, (output_height - 1.0) / 2.0)
        @ linear
        @ _translation_matrix(-(width - 1.0) / 2.0, -(height - 1.0) / 2.0)
    )


def _cardinal_rotation_transform(width: int, height: int, angle: int) -> np.ndarray:
    normalized = angle % 360
    if normalized == 0:
        return np.eye(3, dtype=np.float64)
    if normalized == 90:
        return np.asarray([[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]], dtype=np.float64)
    if normalized == 180:
        return np.asarray([[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]], dtype=np.float64)
    if normalized == 270:
        return np.asarray([[0, -1, height - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    raise ValueError("candidate angles must be cardinal")


def _candidate_crop_to_canonical_transform(fragment: Fragment, pose: CandidatePose) -> np.ndarray:
    ys, xs = np.nonzero(fragment.mask)
    if len(xs) == 0:
        raise ValueError("cannot build a pose transform for an empty fragment mask")
    x0 = int(xs.min())
    y0 = int(ys.min())
    width = int(xs.max()) - x0 + 1
    height = int(ys.max()) - y0 + 1
    crop_to_tight = _translation_matrix(-x0, -y0)
    tight_to_rotated = _cardinal_rotation_transform(width, height, int(pose.angle))
    return _translation_matrix(float(pose.tx), float(pose.ty)) @ tight_to_rotated @ crop_to_tight


def _project_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = (transform @ homogeneous.T).T
    denominators = projected[:, 2]
    if np.any(np.abs(denominators) < 1e-12):
        raise ValueError("crop_to_canonical_transform projects a sample to infinity")
    return projected[:, :2] / denominators[:, None]


def _transform_angle_degrees(transform: np.ndarray, centre: np.ndarray) -> float:
    points = np.asarray([centre, centre + np.asarray([1.0, 0.0])], dtype=np.float64)
    projected = _project_points(transform, points)
    direction = projected[1] - projected[0]
    return math.degrees(math.atan2(-float(direction[1]), float(direction[0]))) % 360.0


def _pose_error(
    pose: CandidatePose,
    annotation: EvaluationAnnotation,
    fragment: Fragment,
) -> dict[str, Any]:
    ys, xs = np.nonzero(fragment.mask)
    if len(xs) == 0:
        raise ValueError("cannot evaluate a pose for an empty fragment mask")
    all_points = np.column_stack([xs, ys]).astype(np.float64)
    stride = max(1, int(math.ceil(len(all_points) / 2048)))
    sample_points = all_points[::stride]
    centre = np.asarray([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)
    candidate_transform = _candidate_crop_to_canonical_transform(fragment, pose)
    truth_transform = annotation.crop_to_canonical_transform
    candidate_points = _project_points(candidate_transform, sample_points)
    truth_points = _project_points(truth_transform, sample_points)
    point_errors = np.linalg.norm(candidate_points - truth_points, axis=1)
    candidate_centre = _project_points(candidate_transform, centre[None, :])[0]
    truth_centre = _project_points(truth_transform, centre[None, :])[0]
    delta = candidate_centre - truth_centre
    candidate_angle = _transform_angle_degrees(candidate_transform, centre)
    truth_angle = _transform_angle_degrees(truth_transform, centre)
    return {
        "side_match": pose.side == annotation.side,
        "dx": float(delta[0]),
        "dy": float(delta[1]),
        "translation_error": float(np.linalg.norm(delta)),
        "angle_error_degrees": _angle_error(candidate_angle, truth_angle),
        "surface_rms_error": float(np.sqrt(np.mean(point_errors**2))),
        "surface_p95_error": float(np.percentile(point_errors, 95.0)),
        "surface_max_error": float(np.max(point_errors)),
        "sampled_foreground_points": len(sample_points),
        "candidate_crop_to_canonical_transform": candidate_transform.tolist(),
    }


def _pose_matches(
    pose: CandidatePose,
    annotation: EvaluationAnnotation,
    fragment: Fragment,
    thresholds: RealityBridgeThresholds,
) -> bool:
    error = _pose_error(pose, annotation, fragment)
    return bool(
        error["side_match"]
        and float(error["surface_p95_error"]) <= thresholds.resolved_translation_error_for(annotation)
        and float(error["angle_error_degrees"])
        <= thresholds.resolved_angle_error_degrees_for(annotation)
    )


def _mask_iou(predicted: np.ndarray, truth: np.ndarray) -> float:
    if predicted.shape != truth.shape:
        raise ValueError("ground-truth mask must have the same shape as the fragment crop mask")
    union = int(np.count_nonzero(predicted | truth))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(predicted & truth) / union)


@numba.njit(cache=True)
def _outside_background(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    outside = np.zeros((height, width), dtype=np.bool_)
    queue_y = np.empty(height * width, dtype=np.int32)
    queue_x = np.empty(height * width, dtype=np.int32)
    head = 0
    tail = 0

    for x in range(width):
        if not mask[0, x] and not outside[0, x]:
            outside[0, x] = True
            queue_y[tail] = 0
            queue_x[tail] = x
            tail += 1
        if not mask[height - 1, x] and not outside[height - 1, x]:
            outside[height - 1, x] = True
            queue_y[tail] = height - 1
            queue_x[tail] = x
            tail += 1
    for y in range(height):
        if not mask[y, 0] and not outside[y, 0]:
            outside[y, 0] = True
            queue_y[tail] = y
            queue_x[tail] = 0
            tail += 1
        if not mask[y, width - 1] and not outside[y, width - 1]:
            outside[y, width - 1] = True
            queue_y[tail] = y
            queue_x[tail] = width - 1
            tail += 1

    while head < tail:
        y = queue_y[head]
        x = queue_x[head]
        head += 1
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny = y + dy
            nx = x + dx
            if 0 <= ny < height and 0 <= nx < width and not mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                queue_y[tail] = ny
                queue_x[tail] = nx
                tail += 1
    return outside


def _fill_holes(mask: np.ndarray) -> tuple[np.ndarray, int]:
    padded = np.pad(mask.astype(bool), 1, mode="constant")
    outside = _outside_background(padded)
    filled = (~outside)[1:-1, 1:-1]
    holes = filled & ~mask
    hole_count = len(connected_components(holes, min_area=1, connectivity=4))
    return filled, hole_count


def _external_boundary(filled_mask: np.ndarray) -> np.ndarray:
    padded = np.pad(filled_mask.astype(bool), 1, mode="constant")
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return filled_mask & ~eroded


@numba.njit(cache=True)
def _directed_boundary_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    distances: np.ndarray = np.empty(len(source), dtype=np.float64)
    for source_index in range(len(source)):
        best_squared = np.inf
        source_y = source[source_index, 0]
        source_x = source[source_index, 1]
        for target_index in range(len(target)):
            dy = source_y - target[target_index, 0]
            dx = source_x - target[target_index, 1]
            squared = float(dy * dy + dx * dx)
            if squared < best_squared:
                best_squared = squared
        distances[source_index] = math.sqrt(best_squared)
    return distances


def _mask_tolerance_audit(
    predicted: np.ndarray,
    truth: np.ndarray,
    tolerance_pixels: float,
    *,
    max_interior_missing_fraction: float,
    max_interior_extraneous_fraction: float,
    max_extraneous_component_fraction: float,
) -> dict[str, Any]:
    if predicted.shape != truth.shape:
        raise ValueError("ground-truth mask must have the same shape as the fragment crop mask")
    predicted = predicted.astype(bool)
    truth = truth.astype(bool)
    predicted_filled, predicted_holes = _fill_holes(predicted)
    truth_filled, truth_holes = _fill_holes(truth)
    predicted_boundary = np.argwhere(_external_boundary(predicted_filled)).astype(np.int32)
    truth_boundary = np.argwhere(_external_boundary(truth_filled)).astype(np.int32)
    boundary_max: float | None
    boundary_p95: float | None
    if len(predicted_boundary) and len(truth_boundary):
        predicted_distances = _directed_boundary_distances(predicted_boundary, truth_boundary)
        truth_distances = _directed_boundary_distances(truth_boundary, predicted_boundary)
        symmetric_distances = np.concatenate([predicted_distances, truth_distances])
        boundary_max = float(np.max(symmetric_distances))
        boundary_p95 = float(np.percentile(symmetric_distances, 95.0))
    elif not len(predicted_boundary) and not len(truth_boundary):
        boundary_max = 0.0
        boundary_p95 = 0.0
    else:
        boundary_max = None
        boundary_p95 = None

    shared_interior = predicted_filled & truth_filled
    interior_missing = truth & shared_interior & ~predicted
    interior_extraneous = predicted & shared_interior & ~truth
    truth_area = int(np.count_nonzero(truth))
    predicted_area = int(np.count_nonzero(predicted))
    missing_fraction = int(np.count_nonzero(interior_missing)) / max(truth_area, 1)
    extraneous_fraction = int(np.count_nonzero(interior_extraneous)) / max(predicted_area, 1)
    predicted_components = connected_components(predicted, min_area=1, connectivity=8)
    truth_components = connected_components(truth, min_area=1, connectivity=8)
    largest_component = max((component.area for component in predicted_components), default=0)
    extraneous_component_fraction = (predicted_area - largest_component) / max(predicted_area, 1)
    boundary_gate_distance = boundary_p95
    boundary_within_tolerance = (
        boundary_gate_distance is not None and boundary_gate_distance <= tolerance_pixels
    )
    connectivity_matches = len(predicted_components) == len(truth_components)
    ready = bool(
        boundary_within_tolerance
        and missing_fraction <= max_interior_missing_fraction
        and extraneous_fraction <= max_interior_extraneous_fraction
        and extraneous_component_fraction <= max_extraneous_component_fraction
        and connectivity_matches
    )
    return {
        "ready": ready,
        "distance_metric": "euclidean_pixel_center",
        "boundary_gate_statistic": "symmetric_surface_distance_p95",
        "boundary_gate_distance_pixels": boundary_gate_distance,
        "tolerance_pixels": float(tolerance_pixels),
        "boundary_within_tolerance": bool(boundary_within_tolerance),
        "boundary_max_distance_pixels": boundary_max,
        "boundary_p95_distance_pixels": boundary_p95,
        "interior_missing_pixels": int(np.count_nonzero(interior_missing)),
        "interior_missing_fraction": float(missing_fraction),
        "interior_extraneous_pixels": int(np.count_nonzero(interior_extraneous)),
        "interior_extraneous_fraction": float(extraneous_fraction),
        "extraneous_component_fraction": float(extraneous_component_fraction),
        "predicted_component_count": len(predicted_components),
        "truth_component_count": len(truth_components),
        "connectivity_matches": bool(connectivity_matches),
        "predicted_hole_count": int(predicted_holes),
        "truth_hole_count": int(truth_holes),
    }


def route_pose_candidates(
    fragment: Fragment,
    poses: list[CandidatePose],
    *,
    thresholds: RealityBridgeThresholds,
    orientation_mode: str,
) -> tuple[str, tuple[str, ...], float | None]:
    """Route using observable evidence only; simulator annotations are excluded."""

    if orientation_mode not in REALITY_ORIENTATION_MODES:
        raise ValueError(f"orientation_mode must be one of: {', '.join(REALITY_ORIENTATION_MODES)}")
    reasons: list[str] = []
    seg_confidence = segmentation_confidence(fragment.mask)
    if seg_confidence < thresholds.min_segmentation_confidence:
        reasons.append("weak_segmentation")
    if not poses:
        reasons.append("no_pose_candidates")
    if reasons:
        return "insufficient-evidence", tuple(reasons), None

    top = poses[0]
    global_margin = (
        float(top.score - poses[1].score)
        if len(poses) > 1
        else (float(top.score_margin) if top.basin_samples > 0 else None)
    )
    if top.score < thresholds.min_pose_score:
        reasons.append("low_pose_score")
    if global_margin is None:
        reasons.append("pose_margin_unavailable")
    elif global_margin < thresholds.min_global_score_margin:
        reasons.append("ambiguous_pose_ranking")
    translation_sigma = float(np.hypot(top.sigma_x, top.sigma_y))
    if translation_sigma > thresholds.max_translation_sigma:
        reasons.append("broad_translation_basin")
    if orientation_mode == "free":
        if top.sigma_theta is None:
            reasons.append("theta_uncertainty_unavailable")
        elif top.sigma_theta > thresholds.max_angle_sigma_degrees:
            reasons.append("broad_angle_basin")
    return ("review", tuple(reasons), global_margin) if reasons else ("automatic", (), global_margin)


def place_fragment_at_pose(
    fragment: Fragment,
    pose: CandidatePose,
    canvas_shape: tuple[int, int],
    *,
    route: str,
) -> Fragment:
    """Place one raw crop at an observable candidate pose in note coordinates."""

    if fragment.image is None:
        raise ValueError("fragment image is required for pose placement")
    raw_image, raw_mask = _crop_foreground(fragment.image, fragment.mask)
    crop_image, crop_mask = _rotate_image_and_mask(raw_image, raw_mask, int(pose.angle))
    height, width = canvas_shape
    crop_height, crop_width = crop_mask.shape
    if pose.tx < 0 or pose.ty < 0 or pose.tx + crop_width > width or pose.ty + crop_height > height:
        raise ValueError(f"pose {pose.pose_id} lies outside the reference canvas")
    placed_mask: np.ndarray = np.zeros((height, width), dtype=bool)
    placed_mask[pose.ty : pose.ty + crop_height, pose.tx : pose.tx + crop_width] = crop_mask
    placed_image: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
    target = placed_image[pose.ty : pose.ty + crop_height, pose.tx : pose.tx + crop_width]
    target[crop_mask] = crop_image[crop_mask]
    return Fragment(
        id=fragment.id,
        label=fragment.label,
        side=pose.side,
        mask=placed_mask,
        image=placed_image,
        tags=fragment.tags,
        meta={
            "original_id": fragment.id,
            "pose_id": pose.pose_id,
            "pose_tx": int(pose.tx),
            "pose_ty": int(pose.ty),
            "pose_angle": int(pose.angle),
            "pose_score": float(pose.score),
            "pose_sigma_x": float(pose.sigma_x),
            "pose_sigma_y": float(pose.sigma_y),
            "pose_sigma_theta": float(pose.sigma_theta) if pose.sigma_theta is not None else None,
            "reality_route": route,
            "source_image": fragment.meta.get("source_image"),
        },
    )


def load_evaluation_annotations(path: str | Path) -> dict[str, EvaluationAnnotation]:
    """Load truth from a file that is physically separate from observations."""

    annotations_path = Path(path)
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    annotations: dict[str, EvaluationAnnotation] = {}
    for index, item in enumerate(payload.get("fragments", [])):
        fragment_id = str(item.get("fragment_id", ""))
        if not fragment_id:
            raise ValueError(f"annotation {index} is missing fragment_id")
        if fragment_id in annotations:
            raise ValueError(f"duplicate evaluation annotation for {fragment_id}")
        mask_path = _resolve_path(annotations_path.parent, item.get("ground_truth_mask"))
        transform = item.get("crop_to_canonical_transform")
        if transform is None:
            raise ValueError(f"annotation {fragment_id} is missing crop_to_canonical_transform")
        annotations[fragment_id] = EvaluationAnnotation(
            fragment_id=fragment_id,
            side=str(item.get("ground_truth_side", "unknown")),
            crop_to_canonical_transform=np.asarray(transform, dtype=np.float64),
            ground_truth_mask=mask_path,
            parent_id=str(item["parent_id"]) if item.get("parent_id") is not None else None,
            physical_fragment_id=(
                str(item["physical_fragment_id"]) if item.get("physical_fragment_id") is not None else None
            ),
            observation_pixels_per_mm=(
                float(item["observation_pixels_per_mm"])
                if item.get("observation_pixels_per_mm") is not None
                else None
            ),
            canonical_pixels_per_mm=(
                float(item["canonical_pixels_per_mm"])
                if item.get("canonical_pixels_per_mm") is not None
                else None
            ),
            effective_radius_mm=(
                float(item["effective_radius_mm"])
                if item.get("effective_radius_mm") is not None
                else None
            ),
        )
    return annotations


def _ground_truth_mask(annotation: EvaluationAnnotation | None) -> np.ndarray | None:
    if annotation is None or annotation.ground_truth_mask is None:
        return None
    return load_mask(annotation.ground_truth_mask)


def _nearest_coarse_grid_position(
    grid: CoarseGridSpec,
    target_tx: float,
    target_ty: float,
) -> tuple[int, int]:
    if grid.count_x < 1 or grid.count_y < 1 or grid.step_tx < 1 or grid.step_ty < 1:
        raise ValueError("coarse grid metadata must have positive counts and steps")
    x_index = int(round((target_tx - grid.origin_tx) / grid.step_tx))
    y_index = int(round((target_ty - grid.origin_ty) / grid.step_ty))
    x_index = min(max(x_index, 0), grid.count_x - 1)
    y_index = min(max(y_index, 0), grid.count_y - 1)
    return (
        grid.origin_tx + x_index * grid.step_tx,
        grid.origin_ty + y_index * grid.step_ty,
    )


def _pose_stage_audit(
    fragment: Fragment,
    annotation: EvaluationAnnotation,
    poses: list[CandidatePose],
    search_audit: PoseSearchAudit,
    thresholds: RealityBridgeThresholds,
) -> dict[str, Any]:
    translation_limit = thresholds.resolved_translation_error_for(annotation)
    angle_limit = thresholds.resolved_angle_error_degrees_for(annotation)
    coarse_translation_limit = translation_limit + search_audit.fine_search_radius

    ys, xs = np.nonzero(fragment.mask)
    if len(xs) == 0:
        return {
            "transform_family_covered": False,
            "best_cardinal_family_fit": None,
            "coarse_grid_reachable": False,
            "coarse_grid_best_fit": None,
            "coarse_neighborhood_hit": False,
            "coarse_shortlist_neighborhood_hit": False,
            "refined_candidate_hit": False,
            "returned_candidate_hit": False,
            "terminal_stage": "empty_mask",
            "miss_stage": "empty_mask",
        }
    all_points = np.column_stack([xs, ys]).astype(np.float64)
    stride = max(1, int(math.ceil(len(all_points) / 2048)))
    sample_points = all_points[::stride]
    centre = np.asarray([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)
    truth_points = _project_points(annotation.crop_to_canonical_transform, sample_points)
    truth_centre = _project_points(annotation.crop_to_canonical_transform, centre[None, :])[0]
    truth_angle = _transform_angle_degrees(annotation.crop_to_canonical_transform, centre)
    x0 = int(xs.min())
    y0 = int(ys.min())
    width = int(xs.max()) - x0 + 1
    height = int(ys.max()) - y0 + 1
    crop_to_tight = _translation_matrix(-x0, -y0)
    family_trials: list[dict[str, float | int]] = []
    for angle in (0, 90, 180, 270):
        base_transform = _cardinal_rotation_transform(width, height, angle) @ crop_to_tight
        base_points = _project_points(base_transform, sample_points)
        base_centre = _project_points(base_transform, centre[None, :])[0]
        translation = truth_centre - base_centre
        aligned_points = base_points + translation[None, :]
        errors = np.linalg.norm(aligned_points - truth_points, axis=1)
        family_trials.append(
            {
                "angle": angle,
                "tx": float(translation[0]),
                "ty": float(translation[1]),
                "surface_p95_error": float(np.percentile(errors, 95.0)),
                "angle_error_degrees": _angle_error(float(angle), truth_angle),
            }
        )
    best_family = min(
        family_trials,
        key=lambda item: (float(item["surface_p95_error"]), float(item["angle_error_degrees"])),
    )
    transform_family_covered = bool(
        float(best_family["surface_p95_error"]) <= translation_limit
        and float(best_family["angle_error_degrees"]) <= angle_limit
    )
    family_by_angle = {int(item["angle"]): item for item in family_trials}

    coarse_grid_fits: list[dict[str, Any]] = []
    for grid in search_audit.coarse_grids:
        if grid.side != annotation.side:
            continue
        family_fit = family_by_angle.get(int(grid.angle))
        if family_fit is None:
            continue
        tx, ty = _nearest_coarse_grid_position(
            grid,
            float(family_fit["tx"]),
            float(family_fit["ty"]),
        )
        grid_pose = CandidatePose(
            fragment_id=fragment.id,
            pose_id="",
            side=grid.side,
            tx=tx,
            ty=ty,
            angle=grid.angle,
            score=0.0,
        )
        grid_error = _pose_error(grid_pose, annotation, fragment)
        coarse_grid_fits.append(
            {
                "side": grid.side,
                "angle": int(grid.angle),
                "tx": int(tx),
                "ty": int(ty),
                "surface_p95_error": float(grid_error["surface_p95_error"]),
                "angle_error_degrees": float(grid_error["angle_error_degrees"]),
            }
        )
    coarse_grid_best_fit = (
        min(
            coarse_grid_fits,
            key=lambda item: (
                float(item["surface_p95_error"]),
                float(item["angle_error_degrees"]),
            ),
        )
        if coarse_grid_fits
        else None
    )
    coarse_grid_reachable = bool(
        coarse_grid_best_fit is not None
        and float(coarse_grid_best_fit["surface_p95_error"]) <= coarse_translation_limit
        and float(coarse_grid_best_fit["angle_error_degrees"]) <= angle_limit
    )

    def coarse_matches(pose: CandidatePose) -> bool:
        error = _pose_error(pose, annotation, fragment)
        return bool(
            error["side_match"]
            and float(error["surface_p95_error"]) <= coarse_translation_limit
            and float(error["angle_error_degrees"]) <= angle_limit
        )

    coarse_hit = any(coarse_matches(pose) for pose in search_audit.coarse_shortlist)
    refined_hit = any(_pose_matches(pose, annotation, fragment, thresholds) for pose in search_audit.refined_candidates)
    returned_hit = any(_pose_matches(pose, annotation, fragment, thresholds) for pose in poses)
    if returned_hit:
        miss_stage = None
    elif refined_hit:
        miss_stage = "output_filter_or_dedup"
    elif coarse_hit:
        miss_stage = "fine_refinement"
    elif coarse_grid_reachable:
        miss_stage = "coarse_top10_ranking"
    elif transform_family_covered:
        miss_stage = "coarse_grid_coverage"
    else:
        miss_stage = "transform_family"
    return {
        "transform_family_covered": transform_family_covered,
        "best_cardinal_family_fit": best_family,
        "coarse_grid_reachable": coarse_grid_reachable,
        "coarse_grid_best_fit": coarse_grid_best_fit,
        "coarse_neighborhood_hit": bool(coarse_hit),
        "coarse_shortlist_neighborhood_hit": bool(coarse_hit),
        "refined_candidate_hit": bool(refined_hit),
        "returned_candidate_hit": bool(returned_hit),
        "terminal_stage": miss_stage or "success",
        "miss_stage": miss_stage,
    }


def _fragment_record(
    fragment: Fragment,
    poses: list[CandidatePose],
    *,
    annotation: EvaluationAnnotation | None,
    search_audit: PoseSearchAudit,
    thresholds: RealityBridgeThresholds,
    orientation_mode: str,
) -> dict[str, Any]:
    route, route_reasons, global_margin = route_pose_candidates(
        fragment,
        poses,
        thresholds=thresholds,
        orientation_mode=orientation_mode,
    )
    truth_mask = _ground_truth_mask(annotation)
    mask_iou = _mask_iou(fragment.mask, truth_mask) if truth_mask is not None else None
    segmentation_tolerance_pixels = thresholds.resolved_segmentation_tolerance_pixels(annotation)
    translation_tolerance_pixels = thresholds.resolved_translation_error_for(annotation)
    angular_tolerance_degrees = thresholds.resolved_angle_error_degrees_for(annotation)
    mask_audit = None
    if truth_mask is not None:
        mask_audit = _mask_tolerance_audit(
            fragment.mask,
            truth_mask,
            segmentation_tolerance_pixels,
            max_interior_missing_fraction=thresholds.max_interior_missing_fraction,
            max_interior_extraneous_fraction=thresholds.max_interior_extraneous_fraction,
            max_extraneous_component_fraction=thresholds.max_extraneous_component_fraction,
        )
    seg_confidence = segmentation_confidence(fragment.mask)
    evaluation_segmentation_ready = seg_confidence >= thresholds.min_segmentation_confidence
    if mask_audit is not None:
        evaluation_segmentation_ready = evaluation_segmentation_ready and bool(mask_audit["ready"])

    top_k_rank = None
    top_k_hit = None
    top1_hit = None
    top1_error = None
    translation_interval_hit = None
    pose_stage = None
    if annotation is not None:
        for index, pose in enumerate(poses):
            if _pose_matches(pose, annotation, fragment, thresholds):
                top_k_rank = index + 1
                break
        top_k_hit = top_k_rank is not None
        if poses:
            top1_hit = _pose_matches(poses[0], annotation, fragment, thresholds)
            top1_error = _pose_error(poses[0], annotation, fragment)
            if (
                bool(top1_error["side_match"])
                and float(top1_error["angle_error_degrees"]) <= angular_tolerance_degrees
            ):
                x_radius = max(
                    translation_tolerance_pixels,
                    thresholds.uncertainty_interval_scale * float(poses[0].sigma_x),
                )
                y_radius = max(
                    translation_tolerance_pixels,
                    thresholds.uncertainty_interval_scale * float(poses[0].sigma_y),
                )
                translation_interval_hit = bool(
                    abs(float(top1_error["dx"])) <= x_radius
                    and abs(float(top1_error["dy"])) <= y_radius
                )
        pose_stage = _pose_stage_audit(fragment, annotation, poses, search_audit, thresholds)

    return {
        "fragment_id": fragment.id,
        "route": route,
        "route_reasons": list(route_reasons),
        "segmentation_confidence": seg_confidence,
        "mask_iou": mask_iou,
        "mask_within_physical_tolerance": mask_audit["ready"] if mask_audit is not None else None,
        "mask_audit": mask_audit,
        "evaluation_segmentation_ready": evaluation_segmentation_ready,
        "has_pose_annotation": annotation is not None,
        "evaluation_annotation": annotation.to_dict() if annotation is not None else None,
        "evaluation_tolerances": {
            "observation_pixels_per_mm": (
                annotation.observation_pixels_per_mm
                if annotation is not None and annotation.observation_pixels_per_mm is not None
                else thresholds.physical_tolerance.pixels_per_mm
            ),
            "canonical_pixels_per_mm": (
                annotation.canonical_pixels_per_mm
                if annotation is not None and annotation.canonical_pixels_per_mm is not None
                else thresholds.physical_tolerance.pixels_per_mm
            ),
            "effective_radius_mm": (
                annotation.effective_radius_mm
                if annotation is not None and annotation.effective_radius_mm is not None
                else thresholds.physical_tolerance.minimum_fragment_span_mm / 2.0
            ),
            "segmentation_tolerance_pixels": segmentation_tolerance_pixels,
            "translation_tolerance_pixels": translation_tolerance_pixels,
            "angular_tolerance_degrees": angular_tolerance_degrees,
        },
        "pose_candidates": [pose.to_dict() for pose in poses],
        "pose_search_audit": search_audit.to_dict(),
        "pose_recall_stage": pose_stage,
        "effective_pose_margin": global_margin,
        "pose_margin_source": "global_candidate" if len(poses) > 1 else "local_score_basin",
        "top_k_hit": top_k_hit,
        "top_k_rank": top_k_rank,
        "top1_hit": top1_hit,
        "top1_error": top1_error,
        "translation_interval_hit": translation_interval_hit,
        "theta_uncertainty_available": bool(poses and poses[0].sigma_theta is not None),
    }


def _funnel_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    annotated = [record for record in records if record["has_pose_annotation"]]
    segmentation_ready = [record for record in annotated if record["evaluation_segmentation_ready"]]
    top_k_recalled = [record for record in segmentation_ready if record["top_k_hit"]]
    top1_correct = [record for record in top_k_recalled if record["top1_hit"]]
    automatic_ready_correct = [record for record in top1_correct if record["route"] == "automatic"]
    automatic_annotated = [record for record in annotated if record["route"] == "automatic"]
    automatic_correct = [record for record in automatic_annotated if record["top1_hit"]]
    false_automatic = [record for record in automatic_annotated if not record["top1_hit"]]
    automatic_mask_gate_fail = [
        record for record in automatic_annotated if not record["evaluation_segmentation_ready"]
    ]
    automatic_release_ready = [
        record
        for record in automatic_annotated
        if record["evaluation_segmentation_ready"] and record["top1_hit"]
    ]

    if not annotated:
        first_bottleneck = "annotations_required"
    elif len(annotated) < len(records):
        first_bottleneck = "annotations_incomplete"
    elif len(segmentation_ready) < len(annotated):
        first_bottleneck = "segmentation"
    elif len(top_k_recalled) < len(segmentation_ready):
        first_bottleneck = "pose_recall"
    elif len(top1_correct) < len(top_k_recalled):
        first_bottleneck = "pose_ranking"
    elif len(automatic_ready_correct) < len(top1_correct):
        first_bottleneck = "uncertainty_routing"
    else:
        first_bottleneck = "downstream_core_unmeasured"

    interval_rows = [record for record in annotated if record["translation_interval_hit"] is not None]
    mask_rows = [record for record in annotated if record["mask_iou"] is not None]
    mask_audits = [record["mask_audit"] for record in annotated if record["mask_audit"] is not None]
    recall_miss_stages: dict[str, int] = {}
    for record in segmentation_ready:
        stage = record["pose_recall_stage"]
        if stage is None or stage["miss_stage"] is None:
            continue
        key = str(stage["miss_stage"])
        recall_miss_stages[key] = recall_miss_stages.get(key, 0) + 1
    return {
        "input_fragments": len(records),
        "annotated_fragments": len(annotated),
        "segmentation_ready": len(segmentation_ready),
        "pose_top_k_recalled": len(top_k_recalled),
        "pose_top1_correct": len(top1_correct),
        "automatic_top1_correct": len(automatic_correct),
        "automatic": sum(record["route"] == "automatic" for record in records),
        "automatic_annotated": len(automatic_annotated),
        "false_automatic_count": len(false_automatic),
        "automatic_mask_gate_fail_count": len(automatic_mask_gate_fail),
        "review": sum(record["route"] == "review" for record in records),
        "insufficient_evidence": sum(record["route"] == "insufficient-evidence" for record in records),
        "top_k_recall": len(top_k_recalled) / len(segmentation_ready) if segmentation_ready else None,
        "top1_accuracy": len(top1_correct) / len(segmentation_ready) if segmentation_ready else None,
        "automatic_handoff_recall": (
            len(automatic_ready_correct) / len(segmentation_ready) if segmentation_ready else None
        ),
        "automatic_precision": (
            len(automatic_correct) / len(automatic_annotated) if automatic_annotated else None
        ),
        "automatic_pose_precision": (
            len(automatic_correct) / len(automatic_annotated) if automatic_annotated else None
        ),
        "automatic_release_precision": (
            len(automatic_release_ready) / len(automatic_annotated) if automatic_annotated else None
        ),
        "mean_mask_iou": float(np.mean([record["mask_iou"] for record in mask_rows])) if mask_rows else None,
        "mask_boundary_pass_rate": (
            sum(bool(audit["boundary_within_tolerance"]) for audit in mask_audits) / len(mask_audits)
            if mask_audits
            else None
        ),
        "mean_interior_missing_fraction": (
            float(np.mean([audit["interior_missing_fraction"] for audit in mask_audits]))
            if mask_audits
            else None
        ),
        "mean_interior_extraneous_fraction": (
            float(np.mean([audit["interior_extraneous_fraction"] for audit in mask_audits]))
            if mask_audits
            else None
        ),
        "translation_interval_coverage": (
            sum(bool(record["translation_interval_hit"]) for record in interval_rows) / len(interval_rows)
            if interval_rows
            else None
        ),
        "theta_uncertainty_available_rate": (
            sum(record["theta_uncertainty_available"] for record in records) / len(records) if records else None
        ),
        "pose_recall_miss_stages": recall_miss_stages,
        "first_observed_bottleneck": first_bottleneck,
        "downstream_core_status": "not_run",
    }


def run_reality_bridge_diagnostic(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    annotations_path: str | Path | None = None,
    reference_front: str | Path | np.ndarray | None = None,
    reference_back: str | Path | np.ndarray | None = None,
    orientation_mode: str | None = None,
    top_k: int = 3,
    coarse_step: int = 8,
    score_margin: float | None = None,
    min_score: float | None = None,
    thresholds: RealityBridgeThresholds | None = None,
) -> dict[str, Any]:
    """Measure the real-input funnel and write auditable pose handoff artifacts."""

    run_started = perf_counter()
    thresholds = thresholds or RealityBridgeThresholds()
    if top_k < 1 or coarse_step < 1:
        raise ValueError("top_k and coarse_step must be positive")
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    load_started = perf_counter()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_annotations_path = _resolve_path(
        manifest_path.parent,
        annotations_path if annotations_path is not None else manifest.get("evaluation_annotations"),
    )
    annotations = (
        load_evaluation_annotations(resolved_annotations_path) if resolved_annotations_path is not None else {}
    )
    acquisition = manifest.get("acquisition") or {}
    resolved_orientation_mode = orientation_mode or str(acquisition.get("orientation_mode", "free"))
    if resolved_orientation_mode not in REALITY_ORIENTATION_MODES:
        raise ValueError(f"orientation_mode must be one of: {', '.join(REALITY_ORIENTATION_MODES)}")
    references = manifest.get("references") or {}
    ref_front, front_info = _load_reference(
        reference_front,
        manifest_value=references.get("front"),
        manifest_base=manifest_path.parent,
        name="front",
    )
    ref_back, back_info = _load_reference(
        reference_back,
        manifest_value=references.get("back"),
        manifest_base=manifest_path.parent,
        name="back",
    )
    if ref_front.shape != ref_back.shape:
        raise ValueError("front and back references must have the same shape")

    fragments = raw_fragments_from_manifest(manifest_path)
    fragment_ids = {fragment.id for fragment in fragments}
    missing_annotation_ids = sorted(fragment_ids - set(annotations))
    orphan_annotation_ids = sorted(set(annotations) - fragment_ids)
    load_seconds = perf_counter() - load_started
    records: list[dict[str, Any]] = []
    automatic_by_side: dict[str, list[Fragment]] = {"front": [], "back": []}
    locator_seconds: list[float] = []
    for fragment in fragments:
        locate_started = perf_counter()
        search_audit = PoseSearchAudit()
        poses = locate_fragment_poses(
            fragment,
            ref_front,
            ref_back,
            top_k=top_k,
            coarse_step=coarse_step,
            score_margin=score_margin,
            min_score=min_score,
            audit=search_audit,
        )
        locator_seconds.append(perf_counter() - locate_started)
        record = _fragment_record(
            fragment,
            poses,
            annotation=annotations.get(fragment.id),
            search_audit=search_audit,
            thresholds=thresholds,
            orientation_mode=resolved_orientation_mode,
        )
        records.append(record)
        if record["route"] == "automatic" and poses:
            automatic_by_side[poses[0].side].append(
                place_fragment_at_pose(fragment, poses[0], ref_front.shape[:2], route="automatic")
            )

    pose_path = output_dir / "pose_candidates.json"
    pose_path.write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
    handoff_outputs: dict[str, str | None] = {}
    for side, reference in (("front", ref_front), ("back", ref_back)):
        placed = automatic_by_side[side]
        if not placed:
            handoff_outputs[side] = None
            continue
        handoff_path = output_dir / f"handoff_{side}.npz"
        save_dataset(handoff_path, reference, placed)
        handoff_outputs[side] = str(handoff_path)

    report_path = output_dir / "reality_bridge_report.json"
    report = {
        "tool": "moneyrepair",
        "stage": "v5_reality_bridge_diagnostic",
        "claim_boundary": (
            "This diagnostic measures acquisition, segmentation, pose recall, and uncertainty routing. "
            "It does not validate real-banknote reconstruction and does not run the frozen v4.4.1 core."
        ),
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "evaluation_annotations": (
                {
                    "path": str(resolved_annotations_path),
                    "sha256": _sha256(resolved_annotations_path),
                    "count": len(annotations),
                    "missing_observation_ids": missing_annotation_ids,
                    "orphan_annotation_ids": orphan_annotation_ids,
                }
                if resolved_annotations_path is not None
                else None
            ),
            "references": {"front": front_info, "back": back_info},
        },
        "parameters": {
            "orientation_mode": resolved_orientation_mode,
            "top_k": top_k,
            "coarse_step": coarse_step,
            "score_margin": score_margin,
            "min_score": min_score,
            "thresholds": thresholds.to_dict(),
        },
        "funnel": _funnel_summary(records),
        "timings_seconds": {
            "load": load_seconds,
            "locator_total": float(sum(locator_seconds)),
            "locator_mean_per_fragment": float(np.mean(locator_seconds)) if locator_seconds else 0.0,
            "total_before_report_write": perf_counter() - run_started,
        },
        "outputs": {
            "pose_candidates": str(pose_path),
            "handoff_datasets": handoff_outputs,
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def _rotate_capture(image: np.ndarray, mask: np.ndarray, angle: int) -> tuple[np.ndarray, np.ndarray]:
    if angle % 90 == 0:
        return _rotate_image_and_mask(image, mask, angle % 360)
    resampling = getattr(Image, "Resampling", Image)
    rotated_image = Image.fromarray(image, mode="RGB").rotate(
        angle,
        resample=resampling.BILINEAR,
        expand=True,
        fillcolor=(0, 0, 0),
    )
    rotated_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").rotate(
        angle,
        resample=resampling.NEAREST,
        expand=True,
        fillcolor=0,
    )
    mask_array = np.asarray(rotated_mask) > 127
    image_array = np.asarray(rotated_image, dtype=np.uint8)
    return np.where(mask_array[..., None], image_array, 0), mask_array


def _drop_interior_mask_pixels(mask: np.ndarray, rng: np.random.Generator, fraction: float) -> np.ndarray:
    if fraction <= 0.0:
        return mask.copy()
    if not (0.0 <= fraction < 1.0):
        raise ValueError("mask_dropout_fraction must be in [0, 1)")
    interior = np.zeros_like(mask)
    if mask.shape[0] >= 3 and mask.shape[1] >= 3:
        interior[1:-1, 1:-1] = (
            mask[1:-1, 1:-1]
            & mask[:-2, 1:-1]
            & mask[2:, 1:-1]
            & mask[1:-1, :-2]
            & mask[1:-1, 2:]
        )
    dropped = interior & (rng.random(mask.shape) < fraction)
    return mask & ~dropped


def write_synthetic_capture_manifest(
    output_dir: str | Path,
    *,
    pieces: int = 6,
    width: int = 160,
    height: int = 72,
    seed: int = 7,
    orientation_mode: str = "cardinal",
    noise_sigma: float = 0.0,
    mask_dropout_fraction: float = 0.0,
) -> Path:
    """Write an annotated proxy capture for plumbing tests, never as real evidence."""

    if orientation_mode not in REALITY_ORIENTATION_MODES:
        raise ValueError(f"orientation_mode must be one of: {', '.join(REALITY_ORIENTATION_MODES)}")
    output_dir = Path(output_dir)
    fragments_dir = output_dir / "fragments"
    masks_dir = output_dir / "masks"
    truth_dir = output_dir / "truth_masks"
    fragments_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    front, fragments = make_synthetic_fragments(
        pieces=pieces,
        width=width,
        height=height,
        seed=seed,
        side="front",
    )
    back = synthetic_banknote(width=width, height=height, seed=seed + 1000)
    Image.fromarray(front, mode="RGB").save(output_dir / "reference_front.png")
    Image.fromarray(back, mode="RGB").save(output_dir / "reference_back.png")
    rng = np.random.default_rng(seed + 2000)
    cardinal_angles = (0, 90, 180, 270)
    free_angles = (15, 30, 45, 60, 75, 105, 120, 135)
    capture_angles = cardinal_angles if orientation_mode == "cardinal" else free_angles
    items: list[dict[str, Any]] = []
    annotation_items: list[dict[str, Any]] = []

    for index, fragment in enumerate(fragments):
        if fragment.image is None:
            continue
        tx, ty, _x1, _y1 = fragment.bbox
        crop_image, crop_mask = _crop_foreground(fragment.image, fragment.mask)
        capture_angle = int(capture_angles[index % len(capture_angles)])
        observed_image, truth_mask = _rotate_capture(crop_image, crop_mask, capture_angle)
        predicted_mask = _drop_interior_mask_pixels(truth_mask, rng, mask_dropout_fraction)
        if noise_sigma > 0.0:
            noise = rng.normal(0.0, noise_sigma, observed_image.shape)
            observed_image = np.clip(observed_image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        observed_image = np.where(predicted_mask[..., None], observed_image, 0)
        image_name = f"f{index:05d}.png"
        mask_name = f"f{index:05d}_mask.png"
        truth_name = f"f{index:05d}_truth.png"
        Image.fromarray(observed_image, mode="RGB").save(fragments_dir / image_name)
        Image.fromarray(predicted_mask.astype(np.uint8) * 255, mode="L").save(masks_dir / mask_name)
        Image.fromarray(truth_mask.astype(np.uint8) * 255, mode="L").save(truth_dir / truth_name)
        local_to_capture = _image_rotation_transform(
            crop_mask.shape[1],
            crop_mask.shape[0],
            capture_angle,
            (int(observed_image.shape[0]), int(observed_image.shape[1])),
        )
        crop_to_canonical = _translation_matrix(float(tx), float(ty)) @ np.linalg.inv(local_to_capture)
        items.append(
            {
                "id": fragment.id,
                "label": fragment.label,
                "side": "front",
                "image": str((Path("fragments") / image_name).as_posix()),
                "mask": str((Path("masks") / mask_name).as_posix()),
                "observation_id": f"proxy-{seed}-{index:05d}",
                "acquisition_id": f"proxy-{seed}",
                "session_id": f"proxy-{seed}-session-0",
                "repeat_id": "0",
            }
        )
        annotation_items.append(
            {
                "fragment_id": fragment.id,
                "parent_id": f"proxy-note-{seed}",
                "physical_fragment_id": fragment.id,
                "ground_truth_side": "front",
                "ground_truth_mask": str((Path("truth_masks") / truth_name).as_posix()),
                "capture_angle_degrees": capture_angle,
                "crop_to_canonical_transform": crop_to_canonical.tolist(),
                "observation_pixels_per_mm": 300.0 / 25.4,
                "canonical_pixels_per_mm": 300.0 / 25.4,
                "effective_radius_mm": 10.0,
            }
        )

    manifest = {
        "schema": "moneyrepair-reality-bridge-v1",
        "claim_boundary": "Annotated synthetic capture proxy; not real-data evidence.",
        "note": {"width": width, "height": height},
        "references": {
            "front": "reference_front.png",
            "back": "reference_back.png",
        },
        "acquisition": {
            "orientation_mode": orientation_mode,
            "noise_sigma": noise_sigma,
            "mask_dropout_fraction": mask_dropout_fraction,
        },
        "fragments": items,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    annotations_payload = {
        "schema": "moneyrepair-evaluation-annotations-v2",
        "claim_boundary": "Evaluation truth only; never load into production Fragment objects.",
        "fragments": annotation_items,
    }
    (output_dir / "annotations.json").write_text(
        json.dumps(annotations_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest_path
