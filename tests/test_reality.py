import json

import numpy as np
import pytest

from moneyrepair.ingest import raw_fragments_from_manifest
from moneyrepair.locator import CandidatePose
from moneyrepair.reality import (
    PhysicalToleranceModel,
    RealityBridgeThresholds,
    _mask_tolerance_audit,
    load_evaluation_annotations,
    route_pose_candidates,
    run_reality_bridge_diagnostic,
    write_synthetic_capture_manifest,
)
from moneyrepair.types import Fragment


def _permissive_thresholds() -> RealityBridgeThresholds:
    return RealityBridgeThresholds(
        min_segmentation_confidence=0.0,
        max_translation_error=2.0,
        max_angle_error_degrees=1.0,
        min_pose_score=0.0,
        min_global_score_margin=0.0,
        max_translation_sigma=100.0,
    )


def test_physical_tolerance_model_derives_pixel_and_feature_gates():
    model = PhysicalToleranceModel()

    assert 2.0 < model.combined_alignment_tolerance_pixels < 2.1
    assert 0.9 < model.angular_tolerance_degrees < 1.1
    assert 6.0 < model.minimum_effective_feature_pixels < 6.1

    with pytest.raises(ValueError, match="scan_dpi"):
        PhysicalToleranceModel(scan_dpi=0.0)


def test_raw_manifest_loader_preserves_local_crop_and_strips_annotations(tmp_path):
    manifest = write_synthetic_capture_manifest(
        tmp_path / "capture",
        pieces=2,
        width=80,
        height=36,
        seed=4,
        orientation_mode="cardinal",
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["fragments"][0]["ground_truth_pose"] = {"side": "back", "tx": 99, "ty": 99, "angle": 180}
    payload["fragments"][0].setdefault("meta", {})["ground_truth_mask"] = "should-not-propagate.png"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fragments = raw_fragments_from_manifest(manifest)
    annotations = load_evaluation_annotations(manifest.parent / "annotations.json")

    assert len(fragments) == 2
    assert all(fragment.mask.shape != (36, 80) for fragment in fragments)
    assert all("ground_truth_pose" not in fragment.meta for fragment in fragments)
    assert all("ground_truth_mask" not in fragment.meta for fragment in fragments)
    assert set(annotations) == {fragment.id for fragment in fragments}
    assert all(annotation.crop_to_canonical_transform.shape == (3, 3) for annotation in annotations.values())
    assert all(fragment.image is not None for fragment in fragments)


def test_mask_audit_uses_continuous_pixel_distance_and_separate_interior_gate():
    truth = np.zeros((11, 11), dtype=bool)
    truth[3:8, 3:8] = True
    shifted_horizontal = np.zeros_like(truth)
    shifted_horizontal[3:8, 4:9] = True
    shifted_diagonal = np.zeros_like(truth)
    shifted_diagonal[4:9, 4:9] = True

    horizontal = _mask_tolerance_audit(
        shifted_horizontal,
        truth,
        1.01,
        max_interior_missing_fraction=1.0,
        max_interior_extraneous_fraction=1.0,
        max_extraneous_component_fraction=1.0,
    )
    diagonal = _mask_tolerance_audit(
        shifted_diagonal,
        truth,
        1.01,
        max_interior_missing_fraction=1.0,
        max_interior_extraneous_fraction=1.0,
        max_extraneous_component_fraction=1.0,
    )
    with_hole = truth.copy()
    with_hole[5, 5] = False
    hole = _mask_tolerance_audit(
        with_hole,
        truth,
        1.01,
        max_interior_missing_fraction=0.01,
        max_interior_extraneous_fraction=0.01,
        max_extraneous_component_fraction=0.01,
    )
    empty = _mask_tolerance_audit(
        np.zeros_like(truth),
        truth,
        1.01,
        max_interior_missing_fraction=1.0,
        max_interior_extraneous_fraction=1.0,
        max_extraneous_component_fraction=1.0,
    )

    assert horizontal["boundary_within_tolerance"] is True
    assert diagonal["boundary_max_distance_pixels"] == pytest.approx(np.sqrt(2.0))
    assert diagonal["boundary_within_tolerance"] is False
    assert hole["boundary_within_tolerance"] is True
    assert hole["interior_missing_fraction"] == pytest.approx(1.0 / 25.0)
    assert hole["predicted_hole_count"] == 1
    assert hole["ready"] is False
    assert empty["boundary_max_distance_pixels"] is None
    assert empty["ready"] is False
    json.dumps(empty, allow_nan=False)


def test_reality_bridge_cardinal_proxy_measures_pose_recall_and_writes_handoff(tmp_path):
    manifest = write_synthetic_capture_manifest(
        tmp_path / "capture",
        pieces=3,
        width=96,
        height=42,
        seed=7,
        orientation_mode="cardinal",
    )

    report = run_reality_bridge_diagnostic(
        manifest,
        tmp_path / "run",
        top_k=3,
        coarse_step=4,
        thresholds=_permissive_thresholds(),
    )

    assert report["funnel"]["annotated_fragments"] == 3
    assert report["funnel"]["top_k_recall"] == 1.0
    assert report["funnel"]["top1_accuracy"] == 1.0
    assert report["funnel"]["automatic_precision"] == 1.0
    assert report["funnel"]["false_automatic_count"] == 0
    assert report["funnel"]["automatic_mask_gate_fail_count"] == 0
    assert report["funnel"]["automatic_release_precision"] == 1.0
    assert report["funnel"]["downstream_core_status"] == "not_run"
    assert (tmp_path / "run" / "pose_candidates.json").exists()
    assert (tmp_path / "run" / "reality_bridge_report.json").exists()
    assert report["outputs"]["handoff_datasets"]["front"] is not None
    records = json.loads((tmp_path / "run" / "pose_candidates.json").read_text(encoding="utf-8"))
    assert all(record["pose_search_audit"]["coarse_shortlist_count"] <= 10 for record in records)
    assert all(record["pose_search_audit"]["refined_candidate_count"] <= 10 for record in records)
    assert all(record["pose_recall_stage"]["returned_candidate_hit"] for record in records)


def test_reality_bridge_reports_false_automatic_precision(tmp_path):
    manifest = write_synthetic_capture_manifest(
        tmp_path / "capture",
        pieces=2,
        width=80,
        height=36,
        seed=5,
        orientation_mode="cardinal",
    )
    annotations_path = manifest.parent / "annotations.json"
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    for item in annotations["fragments"]:
        transform = np.asarray(item["crop_to_canonical_transform"], dtype=np.float64)
        transform[0, 2] += 20.0
        item["crop_to_canonical_transform"] = transform.tolist()
    annotations_path.write_text(json.dumps(annotations, indent=2), encoding="utf-8")

    report = run_reality_bridge_diagnostic(
        manifest,
        tmp_path / "run",
        top_k=3,
        coarse_step=4,
        thresholds=_permissive_thresholds(),
    )

    assert report["funnel"]["automatic_annotated"] > 0
    assert report["funnel"]["false_automatic_count"] == report["funnel"]["automatic_annotated"]
    assert report["funnel"]["automatic_precision"] == 0.0


def test_reality_bridge_free_angle_proxy_exposes_pose_recall_before_core(tmp_path):
    manifest = write_synthetic_capture_manifest(
        tmp_path / "capture",
        pieces=2,
        width=80,
        height=36,
        seed=9,
        orientation_mode="free",
    )

    report = run_reality_bridge_diagnostic(
        manifest,
        tmp_path / "run",
        top_k=3,
        coarse_step=4,
        thresholds=_permissive_thresholds(),
    )

    assert report["funnel"]["top_k_recall"] == 0.0
    assert report["funnel"]["automatic"] == 0
    assert report["funnel"]["theta_uncertainty_available_rate"] == 0.0
    assert report["funnel"]["first_observed_bottleneck"] == "pose_recall"
    records = json.loads((tmp_path / "run" / "pose_candidates.json").read_text(encoding="utf-8"))
    assert all(record["route"] != "automatic" for record in records)
    assert all(record["pose_recall_stage"]["miss_stage"] == "transform_family" for record in records)
    assert all(record["pose_recall_stage"]["transform_family_covered"] is False for record in records)
    assert all(
        "no_pose_candidates" in record["route_reasons"]
        or "theta_uncertainty_unavailable" in record["route_reasons"]
        for record in records
    )


def test_production_route_uses_observable_pose_evidence_only():
    mask = np.ones((8, 8), dtype=bool)
    fragment = Fragment(
        id="f0",
        mask=mask,
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        meta={"ground_truth_pose": {"side": "back", "tx": 99, "ty": 99, "angle": 180}},
    )
    poses = [
        CandidatePose("f0", "f0_pose0", "front", 2, 3, 0, 0.95, sigma_x=0.2, sigma_y=0.2),
        CandidatePose("f0", "f0_pose1", "front", 8, 9, 0, 0.80, sigma_x=0.2, sigma_y=0.2),
    ]

    route, reasons, margin = route_pose_candidates(
        fragment,
        poses,
        thresholds=_permissive_thresholds(),
        orientation_mode="cardinal",
    )

    assert route == "automatic"
    assert reasons == ()
    assert margin is not None and margin > 0.1
