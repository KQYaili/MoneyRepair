from __future__ import annotations

import json
import os

from PIL import Image

from moneyrepair.pilot import (
    EXPECTED_FRAGMENT_COUNT,
    EXPECTED_SCENE_COUNT,
    FROZEN_REALITY_BRIDGE_COMMIT,
    freeze_physical_pilot_coordinate_contract,
    initialize_physical_pilot,
    register_physical_pilot_references,
    validate_physical_pilot,
)


def _write_reference(path, color):
    Image.new("RGB", (12, 6), color=color).save(path)


def _initialize(tmp_path, *, freeze_coordinates=False):
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
        physical_width_mm=12.0,
        physical_height_mm=6.0,
    )
    if freeze_coordinates:
        freeze_physical_pilot_coordinate_contract(
            root,
            scanner_dpi=25.4,
            phone_pixels_per_mm=1.0,
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
    assert plan["print_master"]["physical_size_mm"] == [12.0, 6.0]
    assert plan["locator_references"] is None
    assert (root / plan["protocol"]["path"]).read_text(encoding="utf-8") == "frozen protocol\n"
    assert (root / "fragments.csv").is_file()
    assert (root / "scenes.csv").is_file()

    report = validate_physical_pilot(root, output_path="preflight.json")
    assert report["observed"]["reference_masters"] == 2
    assert report["observed"]["valid_production_manifests"] == 0
    assert report["readiness"]["contract_valid"] is True
    assert report["readiness"]["ready_for_physical_capture"] is False
    assert report["readiness"]["ready_for_gate_runs"] is False
    assert report["readiness"]["first_blocker"] == "coordinate_contract"
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
    front = root / plan["print_master"]["front"]["path"]
    back = root / plan["print_master"]["back"]["path"]

    assert Image.open(front).size == (3685, 1819)
    assert Image.open(back).size == (3685, 1819)
    assert (root / "references" / "print_spec.json").is_file()
    report = validate_physical_pilot(root)
    assert report["observed"]["reference_masters"] == 2
    assert report["readiness"]["ready_for_physical_capture"] is False
    assert report["readiness"]["first_blocker"] == "coordinate_contract"


def test_coordinate_contract_derives_track_references_and_clears_capture_preflight(tmp_path):
    root, plan_path = _initialize(tmp_path)

    contract_path = freeze_physical_pilot_coordinate_contract(
        root,
        scanner_dpi=25.4,
        phone_pixels_per_mm=1.0,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert "references" not in plan
    assert plan["print_master"]["front"]["sha256"]
    assert plan["locator_references"]["sha256"]
    assert set(contract["tracks"]) == {"scanner-cardinal", "phone-cardinal", "phone-free"}
    assert contract["tracks"]["scanner-cardinal"]["front"]["parent_sha256"] == plan["print_master"]["front"]["sha256"]
    for track in contract["tracks"].values():
        assert track["canonical_pixels_per_mm"] == 1.0
        assert track["front"]["derivation"]["output_pixel_size"] == [12, 6]
        assert Image.open(root / track["front"]["path"]).size == (12, 6)

    report = validate_physical_pilot(root)
    assert report["observed"]["locator_reference_tracks"] == 3
    assert report["readiness"]["coordinate_contract_frozen"] is True
    assert report["readiness"]["ready_for_physical_capture"] is True
    assert report["readiness"]["first_blocker"] == "physical_acquisition"


def test_coordinate_contract_is_one_time_and_hash_protected(tmp_path):
    root, _plan_path = _initialize(tmp_path)
    contract_path = freeze_physical_pilot_coordinate_contract(
        root,
        scanner_dpi=25.4,
        phone_pixels_per_mm=1.0,
    )

    try:
        freeze_physical_pilot_coordinate_contract(root, scanner_dpi=25.4, phone_pixels_per_mm=1.0)
    except ValueError as exc:
        assert "already frozen" in str(exc)
    else:
        raise AssertionError("coordinate contract must be immutable")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    locator_front = root / contract["tracks"]["scanner-cardinal"]["front"]["path"]
    Image.new("RGB", (12, 6), color=(1, 2, 3)).save(locator_front)
    report = validate_physical_pilot(root)
    assert report["readiness"]["contract_valid"] is False
    assert "invalid_coordinate_contract" in {item["code"] for item in report["errors"]}


def test_coordinate_contract_refuses_post_capture_selection(tmp_path):
    root, plan_path = _initialize(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest_path = root / plan["scenes"][0]["production_manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    try:
        freeze_physical_pilot_coordinate_contract(root, scanner_dpi=25.4, phone_pixels_per_mm=1.0)
    except ValueError as exc:
        assert "before physical acquisition" in str(exc)
    else:
        raise AssertionError("coordinate scale may not be selected after capture starts")


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
    assert after["print_master"]["front"]["sha256"]
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
                "fragments": [
                    {
                        "id": "f00000",
                        "image": "fragment.png",
                        "meta": {"canonical_pixels_per_mm": 99.0},
                    }
                ],
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
    root, plan_path = _initialize(tmp_path, freeze_coordinates=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    contract_record = plan["locator_references"]
    contract = json.loads((root / contract_record["path"]).read_text(encoding="utf-8"))
    scene = plan["scenes"][0]
    track_contract = contract["tracks"][scene["track"]]
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
                "observation_pixels_per_mm": 1.0,
                "canonical_pixels_per_mm": 1.0,
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
                "references": {
                    side: os.path.relpath(root / track_contract[side]["path"], manifest_path.parent).replace("\\", "/")
                    for side in ("front", "back")
                },
                "acquisition": {
                    "capture_timestamp": "2026-08-24T00:00:00Z",
                    "device": "test-scanner",
                    "modality": "scanner",
                    "orientation_mode": "cardinal",
                    "resolution": [1200, 600],
                    "scanner_dpi": 25.4,
                    "segmentation_method": "manual-test-mask",
                    "segmentation_version": "test-v1",
                    "coordinate_normalization": {
                        "locator_reference_id": track_contract["locator_reference_id"],
                        "observation_pixels_per_mm": 1.0,
                        "canonical_pixels_per_mm": 1.0,
                        "rectification": {
                            "applied": False,
                            "method": "native-scanner-dpi",
                            "target_pixels_per_mm": 1.0,
                        },
                    },
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

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["acquisition"]["coordinate_normalization"]["observation_pixels_per_mm"] = 2.0
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    mismatch_report = validate_physical_pilot(root)
    manifest_errors = [item for item in mismatch_report["errors"] if item["code"] == "incomplete_production_manifest"]
    assert len(manifest_errors) == 1
    assert "observation_pixels_per_mm" in manifest_errors[0]["message"]


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
