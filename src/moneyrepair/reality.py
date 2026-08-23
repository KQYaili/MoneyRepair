from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from moneyrepair.ingest import load_mask, load_rgb, raw_fragments_from_manifest
from moneyrepair.locator import CandidatePose, _crop_foreground, _rotate_image_and_mask, locate_fragment_poses
from moneyrepair.quality import segmentation_confidence
from moneyrepair.simulate import make_synthetic_fragments, save_dataset, synthetic_banknote
from moneyrepair.types import Fragment


REALITY_ORIENTATION_MODES = ("cardinal", "free")


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

    @property
    def combined_alignment_tolerance_mm(self) -> float:
        return self.segmentation_tolerance_mm + self.registration_tolerance_mm

    @property
    def combined_alignment_tolerance_pixels(self) -> float:
        return self.combined_alignment_tolerance_mm * self.pixels_per_mm

    @property
    def angular_tolerance_degrees(self) -> float:
        radius = max(self.minimum_fragment_span_mm / 2.0, 1e-9)
        return math.degrees(math.atan(self.combined_alignment_tolerance_mm / radius))

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


def _pose_error(pose: CandidatePose, truth: dict[str, Any]) -> dict[str, float | bool]:
    dx = float(pose.tx) - float(truth["tx"])
    dy = float(pose.ty) - float(truth["ty"])
    return {
        "side_match": pose.side == str(truth["side"]),
        "dx": dx,
        "dy": dy,
        "translation_error": float(np.hypot(dx, dy)),
        "angle_error_degrees": _angle_error(float(pose.angle), float(truth["angle"])),
    }


def _pose_matches(
    pose: CandidatePose,
    truth: dict[str, Any],
    thresholds: RealityBridgeThresholds,
) -> bool:
    error = _pose_error(pose, truth)
    return bool(
        error["side_match"]
        and float(error["translation_error"]) <= thresholds.resolved_translation_error
        and float(error["angle_error_degrees"]) <= thresholds.resolved_angle_error_degrees
    )


def _mask_iou(predicted: np.ndarray, truth: np.ndarray) -> float:
    if predicted.shape != truth.shape:
        raise ValueError("ground-truth mask must have the same shape as the fragment crop mask")
    union = int(np.count_nonzero(predicted | truth))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(predicted & truth) / union)


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result: np.ndarray = mask.astype(bool).copy()
    for _ in range(max(radius, 0)):
        padded = np.pad(result, 1, mode="constant")
        expanded = np.zeros_like(result)
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
        result = expanded
    return result


def _mask_tolerance_audit(
    predicted: np.ndarray,
    truth: np.ndarray,
    tolerance_pixels: float,
) -> tuple[bool, int, int]:
    if predicted.shape != truth.shape:
        raise ValueError("ground-truth mask must have the same shape as the fragment crop mask")
    radius = max(0, int(math.ceil(tolerance_pixels)))
    predicted_violation = predicted & ~_binary_dilate(truth, radius)
    truth_violation = truth & ~_binary_dilate(predicted, radius)
    violations = int(np.count_nonzero(predicted_violation | truth_violation))
    return violations == 0, violations, radius


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


def _ground_truth_mask(fragment: Fragment, manifest_base: Path) -> np.ndarray | None:
    value = fragment.meta.get("ground_truth_mask")
    if value is None:
        return None
    path = _resolve_path(manifest_base, str(value))
    if path is None:
        return None
    return load_mask(path)


def _fragment_record(
    fragment: Fragment,
    poses: list[CandidatePose],
    *,
    thresholds: RealityBridgeThresholds,
    orientation_mode: str,
    manifest_base: Path,
) -> dict[str, Any]:
    route, route_reasons, global_margin = route_pose_candidates(
        fragment,
        poses,
        thresholds=thresholds,
        orientation_mode=orientation_mode,
    )
    truth = fragment.meta.get("ground_truth_pose")
    truth_mask = _ground_truth_mask(fragment, manifest_base)
    mask_iou = _mask_iou(fragment.mask, truth_mask) if truth_mask is not None else None
    mask_within_tolerance = None
    mask_tolerance_violations = None
    mask_tolerance_radius = None
    if truth_mask is not None:
        mask_within_tolerance, mask_tolerance_violations, mask_tolerance_radius = _mask_tolerance_audit(
            fragment.mask,
            truth_mask,
            thresholds.physical_tolerance.segmentation_tolerance_pixels,
        )
    seg_confidence = segmentation_confidence(fragment.mask)
    evaluation_segmentation_ready = seg_confidence >= thresholds.min_segmentation_confidence
    if mask_within_tolerance is not None:
        evaluation_segmentation_ready = evaluation_segmentation_ready and mask_within_tolerance

    top_k_rank = None
    top_k_hit = None
    top1_hit = None
    top1_error = None
    translation_interval_hit = None
    if truth is not None:
        for index, pose in enumerate(poses):
            if _pose_matches(pose, truth, thresholds):
                top_k_rank = index + 1
                break
        top_k_hit = top_k_rank is not None
        if poses:
            top1_hit = _pose_matches(poses[0], truth, thresholds)
            top1_error = _pose_error(poses[0], truth)
            if (
                bool(top1_error["side_match"])
                and float(top1_error["angle_error_degrees"]) <= thresholds.resolved_angle_error_degrees
            ):
                x_radius = max(
                    thresholds.resolved_translation_error,
                    thresholds.uncertainty_interval_scale * float(poses[0].sigma_x),
                )
                y_radius = max(
                    thresholds.resolved_translation_error,
                    thresholds.uncertainty_interval_scale * float(poses[0].sigma_y),
                )
                translation_interval_hit = bool(
                    abs(float(top1_error["dx"])) <= x_radius
                    and abs(float(top1_error["dy"])) <= y_radius
                )

    return {
        "fragment_id": fragment.id,
        "route": route,
        "route_reasons": list(route_reasons),
        "segmentation_confidence": seg_confidence,
        "mask_iou": mask_iou,
        "mask_within_physical_tolerance": mask_within_tolerance,
        "mask_tolerance_violations": mask_tolerance_violations,
        "mask_tolerance_radius_pixels": mask_tolerance_radius,
        "evaluation_segmentation_ready": evaluation_segmentation_ready,
        "ground_truth_pose": truth,
        "pose_candidates": [pose.to_dict() for pose in poses],
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
    annotated = [record for record in records if record["ground_truth_pose"] is not None]
    segmentation_ready = [record for record in annotated if record["evaluation_segmentation_ready"]]
    top_k_recalled = [record for record in segmentation_ready if record["top_k_hit"]]
    top1_correct = [record for record in top_k_recalled if record["top1_hit"]]
    automatic_correct = [record for record in top1_correct if record["route"] == "automatic"]

    if not annotated:
        first_bottleneck = "annotations_required"
    elif len(segmentation_ready) < len(annotated):
        first_bottleneck = "segmentation"
    elif len(top_k_recalled) < len(segmentation_ready):
        first_bottleneck = "pose_recall"
    elif len(top1_correct) < len(top_k_recalled):
        first_bottleneck = "pose_ranking"
    elif len(automatic_correct) < len(top1_correct):
        first_bottleneck = "uncertainty_routing"
    else:
        first_bottleneck = "downstream_core_unmeasured"

    interval_rows = [record for record in annotated if record["translation_interval_hit"] is not None]
    mask_rows = [record for record in annotated if record["mask_iou"] is not None]
    return {
        "input_fragments": len(records),
        "annotated_fragments": len(annotated),
        "segmentation_ready": len(segmentation_ready),
        "pose_top_k_recalled": len(top_k_recalled),
        "pose_top1_correct": len(top1_correct),
        "automatic_top1_correct": len(automatic_correct),
        "automatic": sum(record["route"] == "automatic" for record in records),
        "review": sum(record["route"] == "review" for record in records),
        "insufficient_evidence": sum(record["route"] == "insufficient-evidence" for record in records),
        "top_k_recall": len(top_k_recalled) / len(segmentation_ready) if segmentation_ready else None,
        "top1_accuracy": len(top1_correct) / len(segmentation_ready) if segmentation_ready else None,
        "automatic_handoff_recall": len(automatic_correct) / len(segmentation_ready) if segmentation_ready else None,
        "mean_mask_iou": float(np.mean([record["mask_iou"] for record in mask_rows])) if mask_rows else None,
        "translation_interval_coverage": (
            sum(bool(record["translation_interval_hit"]) for record in interval_rows) / len(interval_rows)
            if interval_rows
            else None
        ),
        "theta_uncertainty_available_rate": (
            sum(record["theta_uncertainty_available"] for record in records) / len(records) if records else None
        ),
        "first_observed_bottleneck": first_bottleneck,
        "downstream_core_status": "not_run",
    }


def run_reality_bridge_diagnostic(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
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
    load_seconds = perf_counter() - load_started
    records: list[dict[str, Any]] = []
    automatic_by_side: dict[str, list[Fragment]] = {"front": [], "back": []}
    locator_seconds: list[float] = []
    for fragment in fragments:
        locate_started = perf_counter()
        poses = locate_fragment_poses(
            fragment,
            ref_front,
            ref_back,
            top_k=top_k,
            coarse_step=coarse_step,
            score_margin=score_margin,
            min_score=min_score,
        )
        locator_seconds.append(perf_counter() - locate_started)
        record = _fragment_record(
            fragment,
            poses,
            thresholds=thresholds,
            orientation_mode=resolved_orientation_mode,
            manifest_base=manifest_path.parent,
        )
        records.append(record)
        if record["route"] == "automatic" and poses:
            automatic_by_side[poses[0].side].append(
                place_fragment_at_pose(fragment, poses[0], ref_front.shape[:2], route="automatic")
            )

    pose_path = output_dir / "pose_candidates.json"
    pose_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
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
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
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
        items.append(
            {
                "id": fragment.id,
                "label": fragment.label,
                "side": "front",
                "image": str((Path("fragments") / image_name).as_posix()),
                "mask": str((Path("masks") / mask_name).as_posix()),
                "ground_truth_mask": str((Path("truth_masks") / truth_name).as_posix()),
                "capture_angle_degrees": capture_angle,
                "ground_truth_pose": {
                    "side": "front",
                    "tx": tx,
                    "ty": ty,
                    "angle": int((-capture_angle) % 360),
                },
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
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
