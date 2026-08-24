import json

import numpy as np
import pytest

from moneyrepair.ingest import raw_fragments_from_manifest
from moneyrepair.locator import CandidatePose, CoarseGridSpec, PoseSearchAudit
from moneyrepair.reality import (
    EvaluationAnnotation,
    PhysicalToleranceModel,
    RealityBridgeThresholds,
    _mask_tolerance_audit,
    _pose_stage_audit,
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


def test_evaluation_tolerances_use_distinct_observation_and_canonical_scales():
    annotation = EvaluationAnnotation(
        fragment_id="f0",
        side="front",
        crop_to_canonical_transform=np.eye(3),
        observation_pixels_per_mm=10.0,
        canonical_pixels_per_mm=20.0,
        effective_radius_mm=5.0,
    )
    thresholds = RealityBridgeThresholds(
        physical_tolerance=PhysicalToleranceModel(
            segmentation_tolerance_mm=0.1,
            registration_tolerance_mm=0.1,
        )
    )

    assert thresholds.resolved_segmentation_tolerance_pixels(annotation) == pytest.approx(1.0)
    assert thresholds.resolved_translation_error_for(annotation) == pytest.approx(4.0)


def test_evaluation_angular_tolerance_uses_each_fragment_effective_radius():
    thresholds = RealityBridgeThresholds(
        physical_tolerance=PhysicalToleranceModel(
            segmentation_tolerance_mm=0.1,
            registration_tolerance_mm=0.1,
        )
    )
    small = EvaluationAnnotation(
        fragment_id="small",
        side="front",
        crop_to_canonical_transform=np.eye(3),
        effective_radius_mm=5.0,
    )
    large = EvaluationAnnotation(
        fragment_id="large",
        side="front",
        crop_to_canonical_transform=np.eye(3),
        effective_radius_mm=20.0,
    )

    assert thresholds.resolved_angle_error_degrees_for(small) > thresholds.resolved_angle_error_degrees_for(large)


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
    assert "evaluation_annotations" not in payload
    payload["fragments"][0]["ground_truth_pose"] = {"side": "back", "tx": 99, "ty": 99, "angle": 180}
    payload["fragments"][0].setdefault("meta", {})["ground_truth_mask"] = "should-not-propagate.png"
    payload["fragments"][0]["meta"]["canonical_pixels_per_mm"] = 99.0
    payload["fragments"][0]["meta"]["effective_radius_mm"] = 99.0
    payload["fragments"][0]["meta"]["observation_pixels_per_mm"] = 99.0
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fragments = raw_fragments_from_manifest(manifest)
    annotations = load_evaluation_annotations(manifest.parent / "annotations.json")

    assert len(fragments) == 2
    assert all(fragment.mask.shape != (36, 80) for fragment in fragments)
    assert all("ground_truth_pose" not in fragment.meta for fragment in fragments)
    assert all("ground_truth_mask" not in fragment.meta for fragment in fragments)
    assert all("canonical_pixels_per_mm" not in fragment.meta for fragment in fragments)
    assert all("effective_radius_mm" not in fragment.meta for fragment in fragments)
    assert all("observation_pixels_per_mm" not in fragment.meta for fragment in fragments)
    assert set(annotations) == {fragment.id for fragment in fragments}
    assert all(annotation.crop_to_canonical_transform.shape == (3, 3) for annotation in annotations.values())
    assert all(annotation.observation_pixels_per_mm is not None for annotation in annotations.values())
    assert all(annotation.canonical_pixels_per_mm is not None for annotation in annotations.values())
    assert all(annotation.effective_radius_mm is not None for annotation in annotations.values())
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


def test_mask_boundary_gate_uses_the_preregistered_p95_statistic():
    truth = np.zeros((60, 60), dtype=bool)
    truth[15:45, 15:45] = True
    predicted = truth.copy()
    predicted[29:31, 45:51] = True

    audit = _mask_tolerance_audit(
        predicted,
        truth,
        1.01,
        max_interior_missing_fraction=1.0,
        max_interior_extraneous_fraction=1.0,
        max_extraneous_component_fraction=1.0,
    )

    assert audit["boundary_gate_statistic"] == "symmetric_surface_distance_p95"
    assert audit["boundary_p95_distance_pixels"] <= 1.01
    assert audit["boundary_max_distance_pixels"] > 1.01
    assert audit["boundary_within_tolerance"] is True


def _coarse_stage_fixture(truth_tx: float) -> tuple[Fragment, EvaluationAnnotation, PoseSearchAudit]:
    fragment = Fragment(
        id="f0",
        side="unknown",
        mask=np.ones((4, 4), dtype=bool),
        image=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    transform = np.eye(3)
    transform[0, 2] = truth_tx
    annotation = EvaluationAnnotation(
        fragment_id=fragment.id,
        side="front",
        crop_to_canonical_transform=transform,
    )
    wrong_shortlist = [
        CandidatePose(fragment.id, f"wrong-{index}", "back", 0, 0, 0, 1.0 - index * 0.01)
        for index in range(10)
    ]
    audit = PoseSearchAudit(
        coarse_grids=[CoarseGridSpec("front", 0, 0, 0, 8, 8, 2, 1)],
        coarse_shortlist=wrong_shortlist,
        fine_search_radius=1,
    )
    return fragment, annotation, audit


def test_pose_stage_distinguishes_coarse_grid_coverage_miss():
    fragment, annotation, search_audit = _coarse_stage_fixture(4.0)
    thresholds = RealityBridgeThresholds(max_translation_error=0.25, max_angle_error_degrees=0.1)

    stage = _pose_stage_audit(fragment, annotation, [], search_audit, thresholds)

    assert stage["transform_family_covered"] is True
    assert stage["coarse_grid_reachable"] is False
    assert stage["miss_stage"] == "coarse_grid_coverage"


def test_pose_stage_distinguishes_coarse_top10_ranking_miss():
    fragment, annotation, search_audit = _coarse_stage_fixture(8.0)
    thresholds = RealityBridgeThresholds(max_translation_error=0.25, max_angle_error_degrees=0.1)

    stage = _pose_stage_audit(fragment, annotation, [], search_audit, thresholds)

    assert stage["coarse_grid_reachable"] is True
    assert stage["coarse_shortlist_neighborhood_hit"] is False
    assert stage["miss_stage"] == "coarse_top10_ranking"


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
        annotations_path=manifest.parent / "annotations.json",
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
    assert all(0 < record["pose_search_audit"]["coarse_grid_count"] <= 8 for record in records)
    assert all(
        sum(grid["count_x"] * grid["count_y"] for grid in record["pose_search_audit"]["coarse_grids"])
        == record["pose_search_audit"]["coarse_positions_evaluated"]
        for record in records
    )
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
        annotations_path=annotations_path,
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
        annotations_path=manifest.parent / "annotations.json",
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
