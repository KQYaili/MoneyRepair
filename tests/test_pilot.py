from __future__ import annotations

import json

from PIL import Image

from moneyrepair.pilot import (
    EXPECTED_FRAGMENT_COUNT,
    EXPECTED_SCENE_COUNT,
    FROZEN_REALITY_BRIDGE_COMMIT,
    initialize_physical_pilot,
    register_physical_pilot_references,
    validate_physical_pilot,
)


def _write_reference(path, color):
    Image.new("RGB", (12, 6), color=color).save(path)


def _initialize(tmp_path):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen protocol\n", encoding="utf-8")
    front = tmp_path / "front.png"
    back = tmp_path / "back.png"
    _write_reference(front, (120, 30, 40))
    _write_reference(back, (40, 70, 120))
    root = tmp_path / "pilot"
    plan_path = initialize_physical_pilot(
        root,
        protocol_path=protocol,
        reference_front=front,
        reference_back=back,
    )
    return root, plan_path


def test_initialize_physical_pilot_writes_frozen_ledgers(tmp_path):
    root, plan_path = _initialize(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert plan["frozen_reality_bridge_commit"] == FROZEN_REALITY_BRIDGE_COMMIT
    assert len(plan["fragments"]) == EXPECTED_FRAGMENT_COUNT
    assert len(plan["scenes"]) == EXPECTED_SCENE_COUNT
    assert plan["split"]["calibration_proxy_ids"] == ["proxy-01", "proxy-02"]
    assert plan["split"]["evaluation_proxy_ids"] == [
        "proxy-03",
        "proxy-04",
        "proxy-05",
        "proxy-06",
        "proxy-07",
        "proxy-08",
    ]
    assert len({scene["scene_id"] for scene in plan["scenes"]}) == EXPECTED_SCENE_COUNT
    assert plan["protocol"]["path"] == "protocol/protocol.md"
    assert (root / plan["protocol"]["path"]).read_text(encoding="utf-8") == "frozen protocol\n"
    assert (root / "fragments.csv").is_file()
    assert (root / "scenes.csv").is_file()

    report = validate_physical_pilot(root, output_path="preflight.json")
    assert report["observed"]["reference_masters"] == 2
    assert report["observed"]["valid_production_manifests"] == 0
    assert report["readiness"]["contract_valid"] is True
    assert report["readiness"]["ready_for_gate_runs"] is False
    assert report["readiness"]["first_blocker"] == "physical_acquisition"
    assert (root / "preflight.json").is_file()


def test_generated_print_master_clears_reference_preflight(tmp_path):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen protocol\n", encoding="utf-8")
    root = tmp_path / "pilot"
    plan_path = initialize_physical_pilot(
        root,
        protocol_path=protocol,
        generate_reference_master=True,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    front = root / plan["references"]["front"]["path"]
    back = root / plan["references"]["back"]["path"]

    assert Image.open(front).size == (3685, 1819)
    assert Image.open(back).size == (3685, 1819)
    assert (root / "references" / "print_spec.json").is_file()
    report = validate_physical_pilot(root)
    assert report["observed"]["reference_masters"] == 2
    assert report["readiness"]["ready_for_physical_capture"] is True
    assert report["readiness"]["first_blocker"] == "physical_acquisition"


def test_register_references_is_one_time_and_preserves_ledgers(tmp_path):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen protocol\n", encoding="utf-8")
    root = tmp_path / "pilot"
    plan_path = initialize_physical_pilot(root, protocol_path=protocol)
    before = json.loads(plan_path.read_text(encoding="utf-8"))

    register_physical_pilot_references(root, generate_reference_master=True)
    after = json.loads(plan_path.read_text(encoding="utf-8"))

    assert after["fragments"] == before["fragments"]
    assert after["scenes"] == before["scenes"]
    assert after["references"]["front"]["sha256"]
    try:
        register_physical_pilot_references(root, generate_reference_master=True)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("registered references must be immutable")


def test_validator_detects_frozen_protocol_mutation(tmp_path):
    root, plan_path = _initialize(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    (root / plan["protocol"]["path"]).write_text("changed after freeze\n", encoding="utf-8")

    report = validate_physical_pilot(root)
    assert report["readiness"]["contract_valid"] is False
    assert "protocol_hash_mismatch" in {item["code"] for item in report["errors"]}


def test_pilot_validator_rejects_truth_in_production_manifest(tmp_path):
    root, plan_path = _initialize(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    scene = plan["scenes"][0]
    manifest_path = root / scene["production_manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "moneyrepair-reality-bridge-v1",
                "evaluation_annotations": "annotations.json",
                "fragments": [{"id": "f00000", "image": "fragment.png"}],
            }
        ),
        encoding="utf-8",
    )

    report = validate_physical_pilot(root)
    assert report["readiness"]["contract_valid"] is False
    assert report["readiness"]["first_blocker"] == "contract_validation"
    assert "truth_in_production_manifest" in {item["code"] for item in report["errors"]}


def test_pilot_validator_requires_per_observation_calibration_fields(tmp_path):
    root, plan_path = _initialize(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    scene = plan["scenes"][0]
    manifest_path = root / scene["production_manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"schema": "moneyrepair-reality-bridge-v1", "fragments": [{"id": "f00000"}]}),
        encoding="utf-8",
    )
    annotations_path = root / scene["annotations"]
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path.write_text(
        json.dumps(
            {
                "schema": "moneyrepair-evaluation-annotations-v2",
                "fragments": [
                    {
                        "fragment_id": "f00000",
                        "physical_fragment_id": "proxy-01-fragment-01",
                        "parent_id": "proxy-01",
                        "ground_truth_side": "front",
                        "ground_truth_mask": "gold.png",
                        "crop_to_canonical_transform": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validate_physical_pilot(root)
    errors = [item for item in report["errors"] if item["code"] == "incomplete_annotations"]
    assert len(errors) == 1
    assert "observation_pixels_per_mm" in errors[0]["message"]
    assert "canonical_pixels_per_mm" in errors[0]["message"]
    assert "effective_radius_mm" in errors[0]["message"]


def test_pilot_validator_counts_a_complete_truth_separated_scene(tmp_path):
    root, plan_path = _initialize(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    scene = plan["scenes"][0]
    manifest_path = root / scene["production_manifest"]
    annotations_path = root / scene["annotations"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    fragments = []
    annotations = []
    for index in range(1, 9):
        image_path = manifest_path.parent / f"fragment-{index:02d}.png"
        mask_path = manifest_path.parent / f"fragment-{index:02d}-mask.png"
        gold_path = annotations_path.parent / f"fragment-{index:02d}-gold.png"
        _write_reference(image_path, (80 + index, 40, 20))
        Image.new("L", (12, 6), color=255).save(mask_path)
        Image.new("L", (12, 6), color=255).save(gold_path)
        fragments.append(
            {
                "id": f"observation-{index:02d}",
                "image": image_path.name,
                "mask": mask_path.name,
                "observation_id": f"observation-{index:02d}",
                "acquisition_id": scene["scene_id"],
                "session_id": "scanner-session-01",
                "repeat_id": "1",
            }
        )
        annotations.append(
            {
                "fragment_id": f"observation-{index:02d}",
                "physical_fragment_id": f"proxy-01-fragment-{index:02d}",
                "parent_id": "proxy-01",
                "ground_truth_side": "front",
                "ground_truth_mask": str(gold_path.relative_to(annotations_path.parent).as_posix()),
                "crop_to_canonical_transform": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "observation_pixels_per_mm": 11.8,
                "canonical_pixels_per_mm": 23.6,
                "effective_radius_mm": 12.0,
                "physical_area_mm2": 100.0,
                "physical_span_mm": 25.0,
                "annotator": "annotator-a",
                "annotation_repeat": "1",
                "annotation_uncertainty": {"boundary_p95_mm": 0.04},
            }
        )
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "moneyrepair-reality-bridge-v1",
                "acquisition": {
                    "capture_timestamp": "2026-08-24T00:00:00Z",
                    "device": "test-scanner",
                    "modality": "scanner",
                    "orientation_mode": "cardinal",
                    "resolution": [1200, 600],
                    "scanner_dpi": 300,
                    "segmentation_method": "manual-test-mask",
                    "segmentation_version": "test-v1",
                },
                "fragments": fragments,
            }
        ),
        encoding="utf-8",
    )
    annotations_path.write_text(
        json.dumps({"schema": "moneyrepair-evaluation-annotations-v2", "fragments": annotations}),
        encoding="utf-8",
    )

    report = validate_physical_pilot(root)
    assert report["observed"]["valid_production_manifests"] == 1
    assert report["observed"]["valid_independent_annotations"] == 1
    assert "truth_in_production_manifest" not in {item["code"] for item in report["errors"]}


def test_initialize_physical_pilot_refuses_nonempty_directory(tmp_path):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen protocol\n", encoding="utf-8")
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "keep.txt").write_text("operator data", encoding="utf-8")

    try:
        initialize_physical_pilot(root, protocol_path=protocol)
    except ValueError as exc:
        assert "not empty" in str(exc)
    else:
        raise AssertionError("non-empty pilot directories must not be overwritten")
