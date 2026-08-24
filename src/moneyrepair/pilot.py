from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PILOT_SCHEMA = "moneyrepair-v5-physical-pilot-v1"
CALIBRATION_SCHEMA = "moneyrepair-v5-pilot-calibration-v1"
COORDINATE_CONTRACT_SCHEMA = "moneyrepair-v5-pilot-coordinate-contract-v1"
FROZEN_REALITY_BRIDGE_COMMIT = "41195bbd6ab4e558e1dbcedd7dd1bf7b8ad45180"
PROXY_IDS = tuple(f"proxy-{index:02d}" for index in range(1, 9))
CALIBRATION_PROXY_IDS = PROXY_IDS[:2]
EVALUATION_PROXY_IDS = PROXY_IDS[2:]
TRACKS = ("scanner-cardinal", "phone-cardinal", "phone-free")
REPEAT_IDS = ("1", "2", "3")
FRAGMENTS_PER_PROXY = 8
EXPECTED_FRAGMENT_COUNT = len(PROXY_IDS) * FRAGMENTS_PER_PROXY
EXPECTED_SCENE_COUNT = len(PROXY_IDS) * len(TRACKS) * len(REPEAT_IDS)
DEFAULT_PROXY_WIDTH_MM = 156.0
DEFAULT_PROXY_HEIGHT_MM = 77.0
DEFAULT_PRINT_DPI = 600
DEFAULT_SCANNER_DPI = 300.0
DEFAULT_PHONE_PIXELS_PER_MM = DEFAULT_SCANNER_DPI / 25.4

_ANNOTATION_REQUIRED_KEYS = {
    "annotation_repeat",
    "annotation_uncertainty",
    "annotator",
    "fragment_id",
    "physical_fragment_id",
    "parent_id",
    "ground_truth_side",
    "ground_truth_mask",
    "crop_to_canonical_transform",
    "observation_pixels_per_mm",
    "canonical_pixels_per_mm",
    "effective_radius_mm",
    "physical_area_mm2",
    "physical_span_mm",
}
_FORBIDDEN_PRODUCTION_KEYS = {
    "annotation_uncertainty",
    "canonical_pixels_per_mm",
    "capture_angle_degrees",
    "crop_to_canonical_transform",
    "effective_radius_mm",
    "evaluation_annotation",
    "evaluation_annotations",
    "ground_truth_mask",
    "ground_truth_pose",
    "ground_truth_side",
    "observation_pixels_per_mm",
    "parent_id",
    "physical_fragment_id",
}
_ALLOWED_PRODUCTION_CALIBRATION_PATHS = {
    "$.acquisition.coordinate_normalization.canonical_pixels_per_mm",
    "$.acquisition.coordinate_normalization.observation_pixels_per_mm",
}
_CALIBRATION_REQUIRED_VALUES = {
    "segmentation_boundary_p95_mm",
    "registration_surface_tolerance_mm",
    "max_interior_missing_fraction",
    "max_interior_extraneous_fraction",
    "max_extraneous_component_fraction",
}
_COMMON_ACQUISITION_KEYS = {
    "capture_timestamp",
    "device",
    "modality",
    "orientation_mode",
    "resolution",
    "segmentation_method",
    "segmentation_version",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_artifact_path(track: str, repeat_id: str, proxy_id: str, artifact: str) -> str:
    return str((Path(artifact) / track / f"repeat-{repeat_id}" / proxy_id).as_posix())


def _fragment_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for proxy_id in PROXY_IDS:
        split = "calibration" if proxy_id in CALIBRATION_PROXY_IDS else "evaluation"
        for index in range(1, FRAGMENTS_PER_PROXY + 1):
            rows.append(
                {
                    "proxy_id": proxy_id,
                    "physical_fragment_id": f"{proxy_id}-fragment-{index:02d}",
                    "split": split,
                    "size_class": "",
                    "physical_span_mm": "",
                    "effective_radius_mm": "",
                    "notes": "",
                }
            )
    return rows


def _scene_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for proxy_id in PROXY_IDS:
        split = "calibration" if proxy_id in CALIBRATION_PROXY_IDS else "evaluation"
        for track in TRACKS:
            orientation_mode = "free" if track == "phone-free" else "cardinal"
            for repeat_id in REPEAT_IDS:
                base = f"{proxy_id}-{track}-r{repeat_id}"
                rows.append(
                    {
                        "scene_id": base,
                        "proxy_id": proxy_id,
                        "split": split,
                        "track": track,
                        "repeat_id": repeat_id,
                        "orientation_mode": orientation_mode,
                        "production_manifest": f"{_relative_artifact_path(track, repeat_id, proxy_id, 'scenes')}/manifest.json",
                        "annotations": f"{_relative_artifact_path(track, repeat_id, proxy_id, 'annotations')}.json",
                        "report_k3": f"{_relative_artifact_path(track, repeat_id, proxy_id, 'reports')}/k3/reality_bridge_report.json",
                        "report_k10": f"{_relative_artifact_path(track, repeat_id, proxy_id, 'reports')}/k10/reality_bridge_report.json",
                    }
                )
    return rows


def _copy_reference(source: str | Path | None, output_dir: Path, side: str) -> dict[str, Any] | None:
    if source is None:
        return None
    source_path = Path(source)
    if not source_path.is_file():
        raise ValueError(f"{side} reference does not exist: {source_path}")
    suffix = source_path.suffix.lower() or ".png"
    target = output_dir / "references" / f"print_master_{side}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target)
    with Image.open(target) as image:
        pixel_size = list(image.size)
    return {
        "path": str(target.relative_to(output_dir).as_posix()),
        "sha256": _sha256(target),
        "pixel_size": pixel_size,
    }


def _print_master_record(
    references: Mapping[str, dict[str, Any] | None],
    *,
    width_mm: float,
    height_mm: float,
) -> dict[str, Any]:
    return {
        "claim_boundary": "Printable common-template artwork; not a locator raster and not capture evidence.",
        "physical_size_mm": [float(width_mm), float(height_mm)],
        "front": references.get("front"),
        "back": references.get("back"),
    }


def _proxy_master_image(width: int, height: int, *, seed: int, side: str) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    phase = 0.0 if side == "front" else 1.7
    base = np.empty((height, width, 3), dtype=np.float64)
    base[..., 0] = 202 + 18 * np.sin(xx / 83.0 + phase) + 12 * np.cos((xx + yy) / 127.0)
    base[..., 1] = 218 + 16 * np.cos(yy / 71.0 + phase) + 10 * np.sin((2 * xx - yy) / 149.0)
    base[..., 2] = 224 + 15 * np.sin((xx + 2 * yy) / 109.0 + phase)
    base += rng.normal(0.0, 2.0, base.shape)
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, mode="RGBA")
    margin = max(18, height // 45)
    line = max(3, height // 300)
    draw.rectangle((margin, margin, width - margin - 1, height - margin - 1), outline=(25, 38, 55, 255), width=line)

    for index in range(1, 13):
        x = margin + index * (width - 2 * margin) / 13
        color = (25, 110, 125, 95) if side == "front" else (165, 65, 80, 95)
        draw.line((x, margin, width - x, height - margin), fill=color, width=max(2, line // 2))
    for index in range(8):
        radius = int(height * (0.05 + index * 0.035))
        cx = int(width * (0.24 if side == "front" else 0.76))
        cy = int(height * 0.50)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(45, 55, 75, 120), width=line)
    for _ in range(180):
        x = int(rng.integers(margin, width - margin))
        y = int(rng.integers(margin, height - margin))
        radius = int(rng.integers(max(2, line), max(3, line * 4)))
        color = (20, 95, 110, 90) if side == "front" else (155, 55, 75, 90)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    font = ImageFont.load_default(size=max(18, height // 15))
    small_font = ImageFont.load_default(size=max(14, height // 30))
    title = "RESEARCH PROXY"
    subtitle = "NOT CURRENCY  |  MONEYREPAIR PHYSICAL PILOT"
    title_box = draw.textbbox((0, 0), title, font=font)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=small_font)
    draw.rectangle(
        (
            width / 2 - (title_box[2] - title_box[0]) / 2 - margin,
            height * 0.40 - margin,
            width / 2 + (title_box[2] - title_box[0]) / 2 + margin,
            height * 0.60 + margin,
        ),
        fill=(245, 247, 244, 225),
        outline=(25, 38, 55, 230),
        width=line,
    )
    draw.text(
        (width / 2 - (title_box[2] - title_box[0]) / 2, height * 0.43),
        title,
        fill=(20, 28, 38, 255),
        font=font,
    )
    draw.text(
        (width / 2 - (subtitle_box[2] - subtitle_box[0]) / 2, height * 0.55),
        subtitle,
        fill=(20, 28, 38, 255),
        font=small_font,
    )
    draw.text((margin * 1.5, margin * 1.5), side.upper(), fill=(20, 28, 38, 255), font=small_font)
    return image


def write_printable_proxy_master(
    output_dir: str | Path,
    *,
    width_mm: float = DEFAULT_PROXY_WIDTH_MM,
    height_mm: float = DEFAULT_PROXY_HEIGHT_MM,
    dpi: int = DEFAULT_PRINT_DPI,
    seed: int = 5103,
) -> dict[str, dict[str, Any]]:
    """Write a deterministic, clearly non-currency duplex print master."""

    if width_mm <= 0 or height_mm <= 0 or dpi <= 0:
        raise ValueError("proxy dimensions and dpi must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width = int(round(width_mm / 25.4 * dpi))
    height = int(round(height_mm / 25.4 * dpi))
    records: dict[str, dict[str, Any]] = {}
    for offset, side in enumerate(("front", "back")):
        path = output_dir / f"print_master_{side}.png"
        image = _proxy_master_image(width, height, seed=seed + offset, side=side)
        image.save(path, dpi=(dpi, dpi))
        records[side] = {"path": str(path), "sha256": _sha256(path), "width": width, "height": height}
    specification = {
        "schema": "moneyrepair-v5-print-master-v1",
        "claim_boundary": "Common paper-proxy artwork; not a capture and not Gate 1-4 evidence.",
        "physical_size_mm": [width_mm, height_mm],
        "dpi": dpi,
        "pixel_size": [width, height],
        "copies": 8,
        "duplex": True,
        "print_scale_percent": 100,
        "references": records,
    }
    (output_dir / "print_spec.json").write_text(
        json.dumps(specification, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return records


def initialize_physical_pilot(
    output_dir: str | Path,
    *,
    protocol_path: str | Path,
    reference_front: str | Path | None = None,
    reference_back: str | Path | None = None,
    generate_reference_master: bool = False,
    physical_width_mm: float = DEFAULT_PROXY_WIDTH_MM,
    physical_height_mm: float = DEFAULT_PROXY_HEIGHT_MM,
) -> Path:
    """Create an empty, truth-separated ledger for the frozen physical pilot."""

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"pilot directory is not empty: {output_dir}")
    protocol_path = Path(protocol_path)
    if not protocol_path.is_file():
        raise ValueError(f"pilot protocol does not exist: {protocol_path}")
    if generate_reference_master and (reference_front is not None or reference_back is not None):
        raise ValueError("generated and supplied reference masters are mutually exclusive")
    if physical_width_mm <= 0 or physical_height_mm <= 0:
        raise ValueError("physical proxy dimensions must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_target = output_dir / "protocol" / protocol_path.name
    protocol_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(protocol_path, protocol_target)
    references: dict[str, dict[str, Any] | None]
    if generate_reference_master:
        generated = write_printable_proxy_master(
            output_dir / "references",
            width_mm=physical_width_mm,
            height_mm=physical_height_mm,
        )
        references = {
            side: {
                "path": str(Path(record["path"]).relative_to(output_dir).as_posix()),
                "sha256": record["sha256"],
                "pixel_size": [record["width"], record["height"]],
            }
            for side, record in generated.items()
        }
    else:
        references = {
            "front": _copy_reference(reference_front, output_dir, "front"),
            "back": _copy_reference(reference_back, output_dir, "back"),
        }
    fragments = _fragment_rows()
    scenes = _scene_rows()
    plan = {
        "schema": PILOT_SCHEMA,
        "claim_boundary": (
            "Physical acquisition ledger only. No real-data result exists until independent observations, "
            "annotations, calibration, and reality-bridge reports are present."
        ),
        "frozen_reality_bridge_commit": FROZEN_REALITY_BRIDGE_COMMIT,
        "protocol": {
            "path": str(protocol_target.relative_to(output_dir).as_posix()),
            "sha256": _sha256(protocol_target),
        },
        "print_master": _print_master_record(
            references,
            width_mm=physical_width_mm,
            height_mm=physical_height_mm,
        ),
        "locator_references": None,
        "split": {
            "calibration_proxy_ids": list(CALIBRATION_PROXY_IDS),
            "evaluation_proxy_ids": list(EVALUATION_PROXY_IDS),
        },
        "counts": {
            "proxies": len(PROXY_IDS),
            "physical_fragments": EXPECTED_FRAGMENT_COUNT,
            "scenes": EXPECTED_SCENE_COUNT,
            "tracks": len(TRACKS),
            "repeats_per_track": len(REPEAT_IDS),
        },
        "fragments": fragments,
        "scenes": scenes,
    }
    plan_path = output_dir / "pilot_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")

    with (output_dir / "fragments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fragments[0]))
        writer.writeheader()
        writer.writerows(fragments)
    with (output_dir / "scenes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scenes[0]))
        writer.writeheader()
        writer.writerows(scenes)

    calibration_dir = output_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration = {
        "schema": CALIBRATION_SCHEMA,
        "status": "pending",
        "calibration_proxy_ids": list(CALIBRATION_PROXY_IDS),
        "frozen_before_evaluation": False,
        "values": {key: None for key in sorted(_CALIBRATION_REQUIRED_VALUES)},
    }
    (calibration_dir / "frozen_thresholds.json").write_text(
        json.dumps(calibration, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# v5 Physical Pilot Workspace\n\n"
        "This directory contains acquisition ledgers and generated run evidence, not source code.\n"
        "Populate scene manifests and separate annotation files using `scenes.csv`, freeze calibration using only "
        "proxy-01/02, then run K=3 and K=10 diagnostics for proxy-03..08.\n\n"
        "Before capture, run `moneyrepair pilot-freeze-coordinate-contract --pilot-dir <this-directory>` to "
        "derive and hash track-specific locator references from the print master.\n\n"
        "Run `moneyrepair pilot-validate --pilot-dir <this-directory> --output preflight_report.json` after each "
        "collection stage. Missing artifacts are reported as pending rather than converted into synthetic evidence.\n",
        encoding="utf-8",
    )
    return plan_path


def register_physical_pilot_references(
    pilot_dir: str | Path,
    *,
    reference_front: str | Path | None = None,
    reference_back: str | Path | None = None,
    generate_reference_master: bool = False,
    physical_width_mm: float = DEFAULT_PROXY_WIDTH_MM,
    physical_height_mm: float = DEFAULT_PROXY_HEIGHT_MM,
) -> Path:
    """Register the common duplex master once without rebuilding operator data."""

    pilot_dir = Path(pilot_dir)
    plan_path = pilot_dir / "pilot_plan.json"
    if not plan_path.is_file():
        raise ValueError(f"missing pilot plan: {plan_path}")
    plan = _load_json(plan_path)
    if plan.get("schema") != PILOT_SCHEMA:
        raise ValueError(f"pilot plan schema must be {PILOT_SCHEMA}")
    if plan.get("frozen_reality_bridge_commit") != FROZEN_REALITY_BRIDGE_COMMIT:
        raise ValueError("pilot plan is not pinned to the alpha-3 freeze")
    existing = plan.get("print_master") or plan.get("references") or {}
    if existing.get("front") is not None or existing.get("back") is not None:
        raise ValueError("reference masters are already registered; refusing to overwrite them")
    if generate_reference_master and (reference_front is not None or reference_back is not None):
        raise ValueError("generated and supplied reference masters are mutually exclusive")
    if not generate_reference_master and (reference_front is None or reference_back is None):
        raise ValueError("supply both reference masters or use generate_reference_master")
    if physical_width_mm <= 0 or physical_height_mm <= 0:
        raise ValueError("physical proxy dimensions must be positive")

    references: dict[str, dict[str, Any] | None]
    if generate_reference_master:
        generated = write_printable_proxy_master(
            pilot_dir / "references",
            width_mm=physical_width_mm,
            height_mm=physical_height_mm,
        )
        references = {
            side: {
                "path": str(Path(record["path"]).relative_to(pilot_dir).as_posix()),
                "sha256": record["sha256"],
                "pixel_size": [record["width"], record["height"]],
            }
            for side, record in generated.items()
        }
    else:
        references = {
            "front": _copy_reference(reference_front, pilot_dir, "front"),
            "back": _copy_reference(reference_back, pilot_dir, "back"),
        }
    plan.pop("references", None)
    plan["print_master"] = _print_master_record(
        references,
        width_mm=physical_width_mm,
        height_mm=physical_height_mm,
    )
    plan["locator_references"] = None
    plan_path.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
    return plan_path


def _print_master_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    print_master = plan.get("print_master")
    if isinstance(print_master, dict):
        return print_master
    legacy = plan.get("references")
    if isinstance(legacy, dict):
        return _print_master_record(
            legacy,
            width_mm=DEFAULT_PROXY_WIDTH_MM,
            height_mm=DEFAULT_PROXY_HEIGHT_MM,
        )
    return {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def freeze_physical_pilot_coordinate_contract(
    pilot_dir: str | Path,
    *,
    scanner_dpi: float = DEFAULT_SCANNER_DPI,
    phone_pixels_per_mm: float = DEFAULT_PHONE_PIXELS_PER_MM,
    physical_width_mm: float | None = None,
    physical_height_mm: float | None = None,
) -> Path:
    """Freeze track-specific locator rasters before any physical capture exists."""

    if scanner_dpi <= 0 or phone_pixels_per_mm <= 0:
        raise ValueError("scanner DPI and phone pixels/mm must be positive")
    pilot_dir = Path(pilot_dir)
    plan_path = pilot_dir / "pilot_plan.json"
    if not plan_path.is_file():
        raise ValueError(f"missing pilot plan: {plan_path}")
    plan = _load_json(plan_path)
    if plan.get("schema") != PILOT_SCHEMA:
        raise ValueError(f"pilot plan schema must be {PILOT_SCHEMA}")
    if plan.get("frozen_reality_bridge_commit") != FROZEN_REALITY_BRIDGE_COMMIT:
        raise ValueError("pilot plan is not pinned to the alpha-3 freeze")
    if plan.get("locator_references") is not None:
        raise ValueError("coordinate contract is already frozen; refusing to overwrite it")
    evidence_fields = ("production_manifest", "annotations", "report_k3", "report_k10")
    if any(
        (pilot_dir / str(scene[field])).is_file()
        for scene in plan.get("scenes", [])
        for field in evidence_fields
    ):
        raise ValueError("coordinate contract must be frozen before physical acquisition begins")

    print_master = _print_master_from_plan(plan)
    recorded_size = print_master.get("physical_size_mm")
    if isinstance(recorded_size, list) and len(recorded_size) == 2:
        recorded_width = float(recorded_size[0])
        recorded_height = float(recorded_size[1])
    else:
        recorded_width = DEFAULT_PROXY_WIDTH_MM
        recorded_height = DEFAULT_PROXY_HEIGHT_MM
    width_mm = recorded_width if physical_width_mm is None else float(physical_width_mm)
    height_mm = recorded_height if physical_height_mm is None else float(physical_height_mm)
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("physical proxy dimensions must be positive")
    if not math.isclose(width_mm, recorded_width, rel_tol=1e-9, abs_tol=1e-9) or not math.isclose(
        height_mm,
        recorded_height,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("coordinate dimensions must match the preregistered print-master physical size")

    source_records: dict[str, dict[str, Any]] = {}
    for side in ("front", "back"):
        record = print_master.get(side)
        if not isinstance(record, dict) or not record.get("path"):
            raise ValueError(f"missing {side} print master")
        source_path = pilot_dir / str(record["path"])
        if not source_path.is_file():
            raise ValueError(f"missing {side} print master: {source_path}")
        source_hash = _sha256(source_path)
        if record.get("sha256") != source_hash:
            raise ValueError(f"{side} print-master hash changed before coordinate freeze")
        with Image.open(source_path) as source_image:
            source_pixel_size = list(source_image.size)
        source_records[side] = {
            "path": str(source_path.relative_to(pilot_dir).as_posix()),
            "sha256": source_hash,
            "pixel_size": source_pixel_size,
        }

    locator_root = pilot_dir / "references" / "locator"
    if locator_root.exists() and any(locator_root.iterdir()):
        raise ValueError(f"locator reference directory is not empty: {locator_root}")
    locator_root.mkdir(parents=True, exist_ok=True)
    target_scales = {
        "scanner-cardinal": float(scanner_dpi / 25.4),
        "phone-cardinal": float(phone_pixels_per_mm),
        "phone-free": float(phone_pixels_per_mm),
    }
    tracks: dict[str, dict[str, Any]] = {}
    resampling = getattr(Image, "Resampling", Image)
    for track in TRACKS:
        target_pixels_per_mm = target_scales[track]
        target_size = (
            int(round(width_mm * target_pixels_per_mm)),
            int(round(height_mm * target_pixels_per_mm)),
        )
        if min(target_size) < 1:
            raise ValueError(f"{track} canonical raster would be empty")
        track_dir = locator_root / track
        track_dir.mkdir(parents=True, exist_ok=True)
        side_records: dict[str, dict[str, Any]] = {}
        for side in ("front", "back"):
            source_path = pilot_dir / source_records[side]["path"]
            target_path = track_dir / f"locator_reference_{side}.png"
            with Image.open(source_path) as source_image:
                resized = source_image.convert("RGB").resize(target_size, resample=resampling.LANCZOS)
                resized.save(target_path, format="PNG", optimize=False, compress_level=9)
            side_records[side] = {
                "path": str(target_path.relative_to(pilot_dir).as_posix()),
                "sha256": _sha256(target_path),
                "parent_path": source_records[side]["path"],
                "parent_sha256": source_records[side]["sha256"],
                "derivation": {
                    "operation": "resize",
                    "resampling": "lanczos",
                    "source_pixel_size": source_records[side]["pixel_size"],
                    "output_pixel_size": target_size,
                    "physical_size_mm": [width_mm, height_mm],
                    "target_pixels_per_mm": target_pixels_per_mm,
                },
            }
        rectification_required = track.startswith("phone-")
        tracks[track] = {
            "locator_reference_id": f"{track}-canonical-v1",
            "canonical_pixels_per_mm": target_pixels_per_mm,
            "expected_observation_pixels_per_mm": target_pixels_per_mm,
            "acquisition_preprocessing": {
                "rectification_required": rectification_required,
                "method": "fiducial-homography" if rectification_required else "native-scanner-dpi",
                "scanner_dpi": float(scanner_dpi) if track == "scanner-cardinal" else None,
                "target_pixels_per_mm": target_pixels_per_mm,
            },
            **side_records,
        }

    contract = {
        "schema": COORDINATE_CONTRACT_SCHEMA,
        "claim_boundary": "Acquisition coordinate normalization only; no locator score, truth, or Gate result used.",
        "frozen_before_capture": True,
        "selection_policy": "Targets are preregistered and may not be selected from Gate 1-4 outcomes.",
        "physical_size_mm": [width_mm, height_mm],
        "print_master": source_records,
        "tracks": tracks,
    }
    contract_path = locator_root / "coordinate_contract.json"
    _write_json_atomic(contract_path, contract)
    plan.pop("references", None)
    plan["print_master"] = _print_master_record(source_records, width_mm=width_mm, height_mm=height_mm)
    plan["locator_references"] = {
        "schema": COORDINATE_CONTRACT_SCHEMA,
        "path": str(contract_path.relative_to(pilot_dir).as_posix()),
        "sha256": _sha256(contract_path),
    }
    _write_json_atomic(plan_path, plan)
    return contract_path


def _find_forbidden_keys(value: Any, prefix: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if (
                key in _FORBIDDEN_PRODUCTION_KEYS
                and child_path not in _ALLOWED_PRODUCTION_CALIBRATION_PATHS
            ) or key.startswith("ground_truth_"):
                hits.append(child_path)
            hits.extend(_find_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(_find_forbidden_keys(child, f"{prefix}[{index}]"))
    return hits


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve_path(base: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _scale_matches(value: Any, expected: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
        and math.isclose(float(value), expected, rel_tol=1e-6, abs_tol=1e-9)
    )


def _coordinate_contract_errors(
    contract: dict[str, Any],
    pilot_dir: Path,
    print_master: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if contract.get("schema") != COORDINATE_CONTRACT_SCHEMA:
        errors.append(f"schema must be {COORDINATE_CONTRACT_SCHEMA}")
    if contract.get("frozen_before_capture") is not True:
        errors.append("frozen_before_capture must be true")
    physical_size = contract.get("physical_size_mm")
    if not (
        isinstance(physical_size, list)
        and len(physical_size) == 2
        and all(isinstance(value, (int, float)) and value > 0 for value in physical_size)
    ):
        errors.append("physical_size_mm must contain two positive values")
        physical_size = None
    recorded_size = print_master.get("physical_size_mm")
    if physical_size is not None and recorded_size != physical_size:
        errors.append("physical_size_mm must match the print-master contract")

    tracks = contract.get("tracks")
    if not isinstance(tracks, dict) or set(tracks) != set(TRACKS):
        errors.append("tracks must contain exactly scanner-cardinal, phone-cardinal, and phone-free")
        return errors
    reference_ids: list[str] = []
    for track in TRACKS:
        track_record = tracks.get(track)
        if not isinstance(track_record, dict):
            errors.append(f"tracks.{track} must be an object")
            continue
        reference_id = track_record.get("locator_reference_id")
        if not isinstance(reference_id, str) or not reference_id:
            errors.append(f"tracks.{track}.locator_reference_id is required")
        else:
            reference_ids.append(reference_id)
        canonical_scale = track_record.get("canonical_pixels_per_mm")
        observation_scale = track_record.get("expected_observation_pixels_per_mm")
        if not isinstance(canonical_scale, (int, float)) or canonical_scale <= 0:
            errors.append(f"tracks.{track}.canonical_pixels_per_mm must be positive")
            continue
        if not _scale_matches(observation_scale, float(canonical_scale)):
            errors.append(f"tracks.{track} observation and canonical pixels/mm must match after preprocessing")
        preprocessing = track_record.get("acquisition_preprocessing")
        if not isinstance(preprocessing, dict):
            errors.append(f"tracks.{track}.acquisition_preprocessing must be an object")
        else:
            expected_rectification = track.startswith("phone-")
            if preprocessing.get("rectification_required") is not expected_rectification:
                errors.append(f"tracks.{track}.rectification_required must be {str(expected_rectification).lower()}")
            if not _scale_matches(preprocessing.get("target_pixels_per_mm"), float(canonical_scale)):
                errors.append(f"tracks.{track}.target_pixels_per_mm must match the canonical scale")
            if track == "scanner-cardinal" and not _scale_matches(
                preprocessing.get("scanner_dpi"),
                float(canonical_scale) * 25.4,
            ):
                errors.append("tracks.scanner-cardinal scanner_dpi must match the canonical scale")

        for side in ("front", "back"):
            record = track_record.get(side)
            if not isinstance(record, dict) or not record.get("path"):
                errors.append(f"tracks.{track}.{side} locator reference is missing")
                continue
            artifact = _resolve_path(pilot_dir, record.get("path"))
            if artifact is None or not artifact.is_file():
                errors.append(f"tracks.{track}.{side} locator reference does not resolve to a file")
                continue
            if record.get("sha256") != _sha256(artifact):
                errors.append(f"tracks.{track}.{side} locator-reference hash changed")
            parent = print_master.get(side)
            if not isinstance(parent, dict) or record.get("parent_sha256") != parent.get("sha256"):
                errors.append(f"tracks.{track}.{side} parent SHA must match the print master")
            elif record.get("parent_path") != parent.get("path"):
                errors.append(f"tracks.{track}.{side} parent path must match the print master")
            derivation = record.get("derivation")
            if not isinstance(derivation, dict):
                errors.append(f"tracks.{track}.{side}.derivation must be an object")
                continue
            expected_size = derivation.get("output_pixel_size")
            if not (
                isinstance(expected_size, list)
                and len(expected_size) == 2
                and all(isinstance(value, int) and value > 0 for value in expected_size)
            ):
                errors.append(f"tracks.{track}.{side}.output_pixel_size must contain two positive integers")
            else:
                with Image.open(artifact) as image:
                    if list(image.size) != expected_size:
                        errors.append(f"tracks.{track}.{side} raster size does not match its derivation")
                if physical_size is not None:
                    derived_size = [
                        int(round(float(physical_size[0]) * float(canonical_scale))),
                        int(round(float(physical_size[1]) * float(canonical_scale))),
                    ]
                    if expected_size != derived_size:
                        errors.append(f"tracks.{track}.{side} raster size does not match physical size and scale")
            if derivation.get("resampling") != "lanczos":
                errors.append(f"tracks.{track}.{side}.resampling must be lanczos")
            if not _scale_matches(derivation.get("target_pixels_per_mm"), float(canonical_scale)):
                errors.append(f"tracks.{track}.{side} derivation scale must match the track contract")
    if len(reference_ids) != len(set(reference_ids)):
        errors.append("locator_reference_id values must be unique by track")
    return errors


def _manifest_errors(
    payload: dict[str, Any],
    path: Path,
    scene: dict[str, Any],
    track_contract: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != "moneyrepair-reality-bridge-v1":
        errors.append("schema must be moneyrepair-reality-bridge-v1")
    acquisition = payload.get("acquisition")
    if not isinstance(acquisition, dict):
        errors.append("acquisition must be an object")
    else:
        missing = sorted(_COMMON_ACQUISITION_KEYS - set(acquisition))
        if missing:
            errors.append(f"acquisition missing: {', '.join(missing)}")
        if acquisition.get("orientation_mode") != scene.get("orientation_mode"):
            errors.append(f"acquisition.orientation_mode must be {scene.get('orientation_mode')}")
        if scene.get("track") == "scanner-cardinal" and acquisition.get("scanner_dpi") is None:
            errors.append("scanner-cardinal acquisition requires scanner_dpi")
        if str(scene.get("track", "")).startswith("phone-"):
            phone_missing = sorted(
                key
                for key in ("capture_height", "exposure", "fiducial_calibration", "focus", "lens_calibration_version")
                if acquisition.get(key) is None
            )
            if phone_missing:
                errors.append(f"phone acquisition missing: {', '.join(phone_missing)}")
        if track_contract is not None:
            target_scale = float(track_contract["canonical_pixels_per_mm"])
            if scene.get("track") == "scanner-cardinal":
                expected_dpi = float(
                    track_contract["acquisition_preprocessing"]["scanner_dpi"]
                )
                if not _scale_matches(acquisition.get("scanner_dpi"), expected_dpi):
                    errors.append(f"scanner_dpi must equal the frozen coordinate contract ({expected_dpi:g})")
            normalization = acquisition.get("coordinate_normalization")
            if not isinstance(normalization, dict):
                errors.append("acquisition.coordinate_normalization must be an object")
            else:
                expected_reference_id = track_contract.get("locator_reference_id")
                if normalization.get("locator_reference_id") != expected_reference_id:
                    errors.append(f"coordinate_normalization.locator_reference_id must be {expected_reference_id}")
                for key in ("observation_pixels_per_mm", "canonical_pixels_per_mm"):
                    if not _scale_matches(normalization.get(key), target_scale):
                        errors.append(f"coordinate_normalization.{key} must equal {target_scale:g}")
                rectification = normalization.get("rectification")
                if not isinstance(rectification, dict):
                    errors.append("coordinate_normalization.rectification must be an object")
                else:
                    expected_preprocessing = track_contract["acquisition_preprocessing"]
                    expected_applied = bool(expected_preprocessing["rectification_required"])
                    if rectification.get("applied") is not expected_applied:
                        errors.append(f"coordinate_normalization.rectification.applied must be {expected_applied}")
                    if rectification.get("method") != expected_preprocessing.get("method"):
                        errors.append(
                            "coordinate_normalization.rectification.method must match the frozen coordinate contract"
                        )
                    if not _scale_matches(rectification.get("target_pixels_per_mm"), target_scale):
                        errors.append(
                            f"coordinate_normalization.rectification.target_pixels_per_mm must equal {target_scale:g}"
                        )

    if track_contract is not None:
        references = payload.get("references")
        if not isinstance(references, dict):
            errors.append("references must contain the frozen track-specific locator rasters")
        else:
            for side in ("front", "back"):
                artifact = _resolve_path(path.parent, references.get(side))
                if artifact is None or not artifact.is_file():
                    errors.append(f"references.{side} does not resolve to a locator-reference file")
                elif _sha256(artifact) != track_contract[side].get("sha256"):
                    errors.append(f"references.{side} hash does not match the frozen {scene.get('track')} raster")

    fragments = payload.get("fragments")
    if not isinstance(fragments, list) or not fragments:
        return errors + ["fragments must be a non-empty list"]
    ids: list[str] = []
    for index, item in enumerate(fragments):
        if not isinstance(item, dict):
            errors.append(f"fragments[{index}] must be an object")
            continue
        fragment_id = item.get("id")
        if not isinstance(fragment_id, str) or not fragment_id:
            errors.append(f"fragments[{index}].id is required")
        else:
            ids.append(fragment_id)
        for key in ("observation_id", "acquisition_id", "session_id", "repeat_id"):
            if item.get(key) in (None, ""):
                errors.append(f"fragments[{index}].{key} is required")
        if str(item.get("repeat_id")) != str(scene.get("repeat_id")):
            errors.append(f"fragments[{index}].repeat_id must be {scene.get('repeat_id')}")
        for key in ("image", "mask"):
            artifact_path = _resolve_path(path.parent, item.get(key))
            if artifact_path is None or not artifact_path.is_file():
                errors.append(f"fragments[{index}].{key} does not resolve to a file")
    if len(ids) != len(set(ids)):
        errors.append("fragment observation IDs must be unique within a scene")
    return errors


def _annotation_errors(
    payload: dict[str, Any],
    path: Path,
    scene: dict[str, Any],
    track_contract: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    fragments = payload.get("fragments")
    if not isinstance(fragments, list) or not fragments:
        return ["annotations.fragments must be a non-empty list"]
    physical_ids: set[str] = set()
    for index, item in enumerate(fragments):
        if not isinstance(item, dict):
            errors.append(f"fragments[{index}] must be an object")
            continue
        missing = sorted(_ANNOTATION_REQUIRED_KEYS - set(item))
        if missing:
            errors.append(f"fragments[{index}] missing: {', '.join(missing)}")
            continue
        physical_id = str(item.get("physical_fragment_id"))
        physical_ids.add(physical_id)
        if item.get("parent_id") != scene.get("proxy_id"):
            errors.append(f"fragments[{index}].parent_id must be {scene.get('proxy_id')}")
        gold_mask = _resolve_path(path.parent, item.get("ground_truth_mask"))
        if gold_mask is None or not gold_mask.is_file():
            errors.append(f"fragments[{index}].ground_truth_mask does not resolve to a file")
        for key in ("observation_pixels_per_mm", "canonical_pixels_per_mm", "effective_radius_mm"):
            value = item.get(key)
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"fragments[{index}].{key} must be positive")
        if track_contract is not None:
            target_scale = float(track_contract["canonical_pixels_per_mm"])
            for key in ("observation_pixels_per_mm", "canonical_pixels_per_mm"):
                if not _scale_matches(item.get(key), target_scale):
                    errors.append(f"fragments[{index}].{key} must equal the frozen track scale {target_scale:g}")
        transform = item.get("crop_to_canonical_transform")
        if not (
            isinstance(transform, list)
            and len(transform) == 3
            and all(isinstance(row, list) and len(row) == 3 for row in transform)
        ):
            errors.append(f"fragments[{index}].crop_to_canonical_transform must be 3x3")
    expected_physical_ids = {
        f"{scene.get('proxy_id')}-fragment-{index:02d}" for index in range(1, FRAGMENTS_PER_PROXY + 1)
    }
    if physical_ids != expected_physical_ids:
        errors.append("physical_fragment_id set must match the eight frozen IDs for this proxy")
    return errors


def _report_error(report: dict[str, Any], expected_top_k: int, manifest: Path, annotations: Path) -> str | None:
    if int((report.get("parameters") or {}).get("top_k", -1)) != expected_top_k:
        return f"parameters.top_k must equal {expected_top_k}"
    inputs = report.get("inputs") or {}
    manifest_hash = inputs.get("manifest_sha256")
    if manifest_hash != _sha256(manifest):
        return "manifest SHA256 does not match the current production manifest"
    annotation_input = inputs.get("evaluation_annotations") or {}
    if annotation_input.get("sha256") != _sha256(annotations):
        return "annotation SHA256 does not match the current evaluation annotation file"
    return None


def validate_physical_pilot(pilot_dir: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
    """Inventory physical-pilot evidence without estimating or changing any gate."""

    pilot_dir = Path(pilot_dir)
    plan_path = pilot_dir / "pilot_plan.json"
    if not plan_path.is_file():
        raise ValueError(f"missing pilot plan: {plan_path}")
    plan = _load_json(plan_path)
    errors: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def add_error(code: str, message: str, path: str | None = None) -> None:
        errors.append({"code": code, "message": message, "path": path})

    def add_pending(code: str, message: str, paths: list[str]) -> None:
        pending.append({"code": code, "message": message, "count": len(paths), "paths": paths})

    if plan.get("schema") != PILOT_SCHEMA:
        add_error("invalid_schema", f"expected {PILOT_SCHEMA}", str(plan_path))
    if plan.get("frozen_reality_bridge_commit") != FROZEN_REALITY_BRIDGE_COMMIT:
        add_error("frozen_commit_changed", "pilot must remain pinned to the alpha-3 freeze", str(plan_path))

    protocol_record = plan.get("protocol") or {}
    protocol_artifact = _resolve_path(pilot_dir, protocol_record.get("path"))
    if protocol_artifact is None or not protocol_artifact.is_file():
        add_error("missing_frozen_protocol", "pilot protocol copy is missing", str(plan_path))
    elif protocol_record.get("sha256") != _sha256(protocol_artifact):
        add_error("protocol_hash_mismatch", "frozen pilot protocol changed after initialization", str(protocol_artifact))

    fragment_ids = [item.get("physical_fragment_id") for item in plan.get("fragments", []) if isinstance(item, dict)]
    expected_fragment_ids = [row["physical_fragment_id"] for row in _fragment_rows()]
    if fragment_ids != expected_fragment_ids:
        add_error("fragment_ledger_changed", "fragment ledger must contain the frozen 64 ordered IDs", str(plan_path))

    scenes = [item for item in plan.get("scenes", []) if isinstance(item, dict)]
    scene_ids = [item.get("scene_id") for item in scenes]
    expected_scene_ids = [row["scene_id"] for row in _scene_rows()]
    if scene_ids != expected_scene_ids:
        add_error("scene_ledger_changed", "scene ledger must contain the frozen 72 ordered IDs", str(plan_path))

    print_master = _print_master_from_plan(plan)
    missing_references: list[str] = []
    for side in ("front", "back"):
        record = print_master.get(side)
        if not isinstance(record, dict) or not record.get("path"):
            missing_references.append(side)
            continue
        reference_path = pilot_dir / str(record["path"])
        if not reference_path.is_file():
            missing_references.append(side)
        elif record.get("sha256") != _sha256(reference_path):
            add_error("reference_hash_mismatch", f"{side} reference hash changed", str(reference_path))
    if missing_references:
        add_pending("missing_print_master", "front/back printable common-template masters are required", missing_references)

    coordinate_contract: dict[str, Any] | None = None
    coordinate_contract_ready = False
    locator_record = plan.get("locator_references")
    if not isinstance(locator_record, dict) or not locator_record.get("path"):
        add_pending(
            "missing_coordinate_contract",
            "freeze track-specific locator references before physical capture",
            [str(pilot_dir / "references" / "locator" / "coordinate_contract.json")],
        )
    else:
        contract_path = _resolve_path(pilot_dir, locator_record.get("path"))
        if contract_path is None or not contract_path.is_file():
            add_error(
                "missing_frozen_coordinate_contract",
                "the registered coordinate-contract file is missing",
                str(contract_path) if contract_path is not None else str(plan_path),
            )
        elif locator_record.get("sha256") != _sha256(contract_path):
            add_error(
                "coordinate_contract_hash_mismatch",
                "the coordinate contract changed after it was frozen",
                str(contract_path),
            )
        else:
            try:
                coordinate_contract = _load_json(contract_path)
            except (ValueError, json.JSONDecodeError) as exc:
                add_error("invalid_coordinate_contract", str(exc), str(contract_path))
            if coordinate_contract is not None:
                contract_errors = _coordinate_contract_errors(coordinate_contract, pilot_dir, print_master)
                if contract_errors:
                    add_error("invalid_coordinate_contract", "; ".join(contract_errors), str(contract_path))
                else:
                    coordinate_contract_ready = True

    calibration_path = pilot_dir / "calibration" / "frozen_thresholds.json"
    calibration_frozen = False
    if not calibration_path.is_file():
        add_pending("missing_calibration", "calibration threshold file is absent", [str(calibration_path)])
    else:
        calibration = _load_json(calibration_path)
        values = calibration.get("values") or {}
        scalar_values_valid = all(
            isinstance(values.get(key), (int, float)) for key in _CALIBRATION_REQUIRED_VALUES
        )
        positive_tolerances = all(
            isinstance(values.get(key), (int, float)) and values[key] > 0
            for key in ("segmentation_boundary_p95_mm", "registration_surface_tolerance_mm")
        )
        bounded_fractions = all(
            isinstance(values.get(key), (int, float)) and 0 <= values[key] <= 1
            for key in (
                "max_interior_missing_fraction",
                "max_interior_extraneous_fraction",
                "max_extraneous_component_fraction",
            )
        )
        calibration_frozen = (
            calibration.get("schema") == CALIBRATION_SCHEMA
            and calibration.get("status") == "frozen"
            and calibration.get("frozen_before_evaluation") is True
            and calibration.get("calibration_proxy_ids") == list(CALIBRATION_PROXY_IDS)
            and scalar_values_valid
            and positive_tolerances
            and bounded_fractions
        )
        if not calibration_frozen:
            add_pending(
                "calibration_not_frozen",
                "freeze all preregistered values from proxy-01/02 before evaluating proxy-03..08",
                [str(calibration_path)],
            )

    missing_manifests: list[str] = []
    missing_annotations: list[str] = []
    missing_k3: list[str] = []
    missing_k10: list[str] = []
    valid_manifest_count = 0
    valid_annotation_count = 0
    valid_k3_count = 0
    valid_k10_count = 0

    existing_manifest_paths = [
        pilot_dir / str(scene["production_manifest"])
        for scene in scenes
        if (pilot_dir / str(scene["production_manifest"])).is_file()
    ]
    if existing_manifest_paths and not coordinate_contract_ready:
        add_error(
            "capture_without_coordinate_contract",
            "physical acquisition exists without a valid preregistered coordinate contract",
            str(existing_manifest_paths[0]),
        )

    for scene in scenes:
        manifest_path = pilot_dir / str(scene["production_manifest"])
        annotations_path = pilot_dir / str(scene["annotations"])
        report_paths = {
            3: pilot_dir / str(scene["report_k3"]),
            10: pilot_dir / str(scene["report_k10"]),
        }
        manifest_payload: dict[str, Any] | None = None
        annotation_payload: dict[str, Any] | None = None
        track_contract = None
        if coordinate_contract_ready and coordinate_contract is not None:
            track_contract = coordinate_contract["tracks"][str(scene["track"])]

        if not manifest_path.is_file():
            missing_manifests.append(str(manifest_path.relative_to(pilot_dir)))
        else:
            try:
                manifest_payload = _load_json(manifest_path)
            except (ValueError, json.JSONDecodeError) as exc:
                add_error("invalid_production_manifest", str(exc), str(manifest_path))
            if manifest_payload is not None:
                forbidden = _find_forbidden_keys(manifest_payload)
                if forbidden:
                    add_error(
                        "truth_in_production_manifest",
                        f"evaluation-only keys found: {', '.join(forbidden)}",
                        str(manifest_path),
                    )
                manifest_errors = _manifest_errors(manifest_payload, manifest_path, scene, track_contract)
                if manifest_errors:
                    add_error("incomplete_production_manifest", "; ".join(manifest_errors), str(manifest_path))
                if not forbidden and not manifest_errors:
                    valid_manifest_count += 1

        if not annotations_path.is_file():
            missing_annotations.append(str(annotations_path.relative_to(pilot_dir)))
        else:
            try:
                annotation_payload = _load_json(annotations_path)
            except (ValueError, json.JSONDecodeError) as exc:
                add_error("invalid_annotations", str(exc), str(annotations_path))
            if annotation_payload is not None:
                annotation_errors = _annotation_errors(annotation_payload, annotations_path, scene, track_contract)
                if annotation_errors:
                    add_error("incomplete_annotations", "; ".join(annotation_errors), str(annotations_path))
                else:
                    valid_annotation_count += 1

        for top_k, report_path in report_paths.items():
            missing_bucket = missing_k3 if top_k == 3 else missing_k10
            if not report_path.is_file():
                missing_bucket.append(str(report_path.relative_to(pilot_dir)))
                continue
            if not manifest_path.is_file() or not annotations_path.is_file():
                add_error("orphan_reality_report", "report exists without its manifest and annotations", str(report_path))
                continue
            try:
                report = _load_json(report_path)
                error = _report_error(report, top_k, manifest_path, annotations_path)
            except (ValueError, json.JSONDecodeError) as exc:
                error = str(exc)
            if error is not None:
                add_error("stale_or_invalid_reality_report", error, str(report_path))
            elif top_k == 3:
                valid_k3_count += 1
            else:
                valid_k10_count += 1

    if missing_manifests:
        add_pending("missing_production_manifests", "physical scene manifests have not been collected", missing_manifests)
    if missing_annotations:
        add_pending("missing_independent_annotations", "truth-isolated scene annotations are incomplete", missing_annotations)
    if missing_k3:
        add_pending("missing_k3_reports", "K=3 reality-bridge diagnostics have not all run", missing_k3)
    if missing_k10:
        add_pending("missing_k10_reports", "K=10 diagnostics have not all run", missing_k10)

    print_master_ready = not missing_references
    manifests_complete = valid_manifest_count == EXPECTED_SCENE_COUNT and not missing_manifests
    annotations_complete = valid_annotation_count == EXPECTED_SCENE_COUNT and not missing_annotations
    reports_complete = valid_k3_count == EXPECTED_SCENE_COUNT and valid_k10_count == EXPECTED_SCENE_COUNT
    structural_ok = not errors
    ready_for_gate_runs = (
        structural_ok
        and print_master_ready
        and coordinate_contract_ready
        and manifests_complete
        and annotations_complete
        and calibration_frozen
    )

    if not structural_ok:
        first_blocker = "contract_validation"
    elif not print_master_ready:
        first_blocker = "print_master"
    elif not coordinate_contract_ready:
        first_blocker = "coordinate_contract"
    elif not manifests_complete:
        first_blocker = "physical_acquisition"
    elif not annotations_complete:
        first_blocker = "independent_annotations"
    elif not calibration_frozen:
        first_blocker = "calibration_freeze"
    elif not reports_complete:
        first_blocker = "reality_bridge_execution"
    else:
        first_blocker = "none"

    report = {
        "schema": "moneyrepair-v5-pilot-preflight-v1",
        "claim_boundary": (
            "Inventory and contract validation only. This report contains no Gate 1-4 outcome and is not real-data evidence."
        ),
        "pilot_plan": str(plan_path),
        "frozen_reality_bridge_commit": plan.get("frozen_reality_bridge_commit"),
        "expected": {
            "physical_fragments": EXPECTED_FRAGMENT_COUNT,
            "scenes": EXPECTED_SCENE_COUNT,
            "k3_reports": EXPECTED_SCENE_COUNT,
            "k10_reports": EXPECTED_SCENE_COUNT,
        },
        "observed": {
            "reference_masters": 2 - len(missing_references),
            "print_master_sides": 2 - len(missing_references),
            "locator_reference_tracks": len((coordinate_contract or {}).get("tracks", {}))
            if coordinate_contract_ready
            else 0,
            "valid_production_manifests": valid_manifest_count,
            "valid_independent_annotations": valid_annotation_count,
            "valid_k3_reports": valid_k3_count,
            "valid_k10_reports": valid_k10_count,
            "calibration_frozen": calibration_frozen,
        },
        "readiness": {
            "contract_valid": structural_ok,
            "coordinate_contract_frozen": coordinate_contract_ready,
            "ready_for_physical_capture": structural_ok and print_master_ready and coordinate_contract_ready,
            "ready_for_gate_runs": ready_for_gate_runs,
            "gate_reports_complete": structural_ok and reports_complete,
            "first_blocker": first_blocker,
        },
        "errors": errors,
        "pending": pending,
    }
    if output_path is not None:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = pilot_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        report["output"] = str(output_path)
    return report
