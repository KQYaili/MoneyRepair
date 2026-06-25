from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from moneyrepair.ingest import infer_foreground_mask


@dataclass(frozen=True)
class ScanComponent:
    index: int
    bbox: tuple[int, int, int, int]
    area: int
    pixels_y: np.ndarray
    pixels_x: np.ndarray


def connected_components(mask: np.ndarray, min_area: int = 50, connectivity: int = 8) -> list[ScanComponent]:
    """Find connected foreground components in a binary scan mask."""

    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    mask = mask.astype(bool)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[ScanComponent] = []
    neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        neighbors += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    seeds_y, seeds_x = np.nonzero(mask)
    for seed_y, seed_x in zip(seeds_y.tolist(), seeds_x.tolist()):
        if visited[seed_y, seed_x]:
            continue
        stack = [(seed_y, seed_x)]
        visited[seed_y, seed_x] = True
        pixels_y: list[int] = []
        pixels_x: list[int] = []

        while stack:
            y, x = stack.pop()
            pixels_y.append(y)
            pixels_x.append(x)
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if ny < 0 or nx < 0 or ny >= height or nx >= width:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))

        area = len(pixels_y)
        if area < min_area:
            continue
        ys = np.array(pixels_y, dtype=np.int32)
        xs = np.array(pixels_x, dtype=np.int32)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        components.append(ScanComponent(len(components), bbox, area, ys, xs))

    components.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.bbox[3], item.bbox[2]))
    return [
        ScanComponent(index=index, bbox=item.bbox, area=item.area, pixels_y=item.pixels_y, pixels_x=item.pixels_x)
        for index, item in enumerate(components)
    ]


def load_label_overrides(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    overrides: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = next(csv.reader([line])) if "," in line else line.split()
        if len(parts) < 2:
            continue
        key = parts[0].strip()
        value = parts[1].strip()
        if key.lower() in {"index", "id", "fragment_id"}:
            continue
        overrides[key] = value
    return overrides


def _component_crop_mask(component: ScanComponent, padding: int, shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = shape
    x0, y0, x1, y1 = component.bbox
    x0p = max(0, x0 - padding)
    y0p = max(0, y0 - padding)
    x1p = min(width, x1 + padding)
    y1p = min(height, y1 + padding)
    crop_mask = np.zeros((y1p - y0p, x1p - x0p), dtype=bool)
    crop_mask[component.pixels_y - y0p, component.pixels_x - x0p] = True
    return crop_mask, (x0p, y0p, x1p, y1p)


def segment_scan_to_manifest(
    image_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    threshold: float = 22.0,
    min_area: int = 50,
    padding: int = 4,
    side: str = "front",
    id_prefix: str = "f",
    label_prefix: str = "",
    labels_file: str | Path | None = None,
    note_width: int | None = None,
    note_height: int | None = None,
    preserve_scan_coordinates: bool = True,
) -> dict:
    """Segment one scan/photo into fragment crops and a manifest."""

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    fragment_dir = output_dir / "fragments"
    mask_dir = output_dir / "masks"
    fragment_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    raw = np.asarray(Image.open(image_path).convert("RGBA"), dtype=np.uint8)
    rgb = raw[..., :3]
    mask_source = raw if np.any(raw[..., 3] < 255) else rgb
    foreground = infer_foreground_mask(mask_source, threshold=threshold)
    components = connected_components(foreground, min_area=min_area)
    overrides = load_label_overrides(labels_file)
    height, width = foreground.shape
    manifest = {
        "note": {
            "width": int(note_width or width),
            "height": int(note_height or height),
        },
        "source_scan": str(image_path),
        "fragments": [],
    }

    for component in components:
        fragment_id = f"{id_prefix}{component.index:05d}"
        default_label = f"{label_prefix}{component.index:05d}" if label_prefix else fragment_id
        label = overrides.get(fragment_id) or overrides.get(str(component.index)) or default_label
        crop_mask, padded_bbox = _component_crop_mask(component, padding=padding, shape=foreground.shape)
        x0, y0, x1, y1 = padded_bbox
        crop_rgb = rgb[y0:y1, x0:x1]
        crop_rgba = np.dstack([crop_rgb, (crop_mask.astype(np.uint8) * 255)])

        image_name = f"{fragment_id}.png"
        mask_name = f"{fragment_id}_mask.png"
        Image.fromarray(crop_rgba, mode="RGBA").save(fragment_dir / image_name)
        Image.fromarray((crop_mask.astype(np.uint8) * 255), mode="L").save(mask_dir / mask_name)

        affine = [[1, 0, int(x0)], [0, 1, int(y0)]] if preserve_scan_coordinates else [[1, 0, 0], [0, 1, 0]]
        manifest["fragments"].append(
            {
                "id": fragment_id,
                "label": label,
                "side": side,
                "image": str((Path("fragments") / image_name).as_posix()),
                "mask": str((Path("masks") / mask_name).as_posix()),
                "affine_to_note": affine,
                "scan_bbox": [int(value) for value in padded_bbox],
                "area": int(component.area),
            }
        )

    target_manifest = Path(manifest_path) if manifest_path else output_dir / "manifest.json"
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
