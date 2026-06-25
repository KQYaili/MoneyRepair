from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from moneyrepair.types import Fragment


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_mask(path: str | Path, threshold: int = 127) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > threshold


def infer_foreground_mask(image: np.ndarray, threshold: float = 22.0) -> np.ndarray:
    """Infer a foreground mask from alpha or corner-background color."""

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image must be an RGB or RGBA array")
    if image.shape[2] == 4:
        return image[..., 3] > 0

    rgb = image.astype(np.float32)
    corners = np.array(
        [
            rgb[0, 0],
            rgb[0, -1],
            rgb[-1, 0],
            rgb[-1, -1],
        ]
    )
    background = np.median(corners, axis=0)
    distance = np.linalg.norm(rgb - background, axis=2)
    return distance > threshold


def _inverse_affine(matrix: np.ndarray) -> tuple[float, float, float, float, float, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise ValueError("affine_to_note must have shape (2, 3)")
    hom = np.eye(3, dtype=np.float64)
    hom[:2, :] = matrix
    inv = np.linalg.inv(hom)
    return tuple(float(value) for value in inv[:2, :].reshape(-1))


def warp_fragment_to_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    affine_to_note: np.ndarray,
    canvas_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a local fragment image/mask into note coordinates."""

    height, width = canvas_shape
    coeffs = _inverse_affine(affine_to_note)
    pil_image = Image.fromarray(image[..., :3].astype(np.uint8), mode="RGB")
    pil_mask = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    resampling = getattr(Image, "Resampling", Image)
    warped_image = pil_image.transform(
        (width, height),
        Image.Transform.AFFINE,
        coeffs,
        resample=resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )
    warped_mask = pil_mask.transform(
        (width, height),
        Image.Transform.AFFINE,
        coeffs,
        resample=resampling.NEAREST,
        fillcolor=0,
    )
    mask_arr = np.asarray(warped_mask) > 127
    image_arr = np.asarray(warped_image, dtype=np.uint8)
    image_arr = np.where(mask_arr[..., None], image_arr, 0)
    return image_arr, mask_arr


def _resolve_path(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def label_from_filename(path: str | Path) -> str:
    """Best-effort label fallback for already-numbered input images."""

    stem = Path(path).stem
    match = re.search(r"[A-Za-z]*\d+[A-Za-z0-9_-]*", stem)
    return match.group(0) if match else stem


def _canvas_shape(manifest: dict[str, Any], reference: np.ndarray | None) -> tuple[int, int]:
    if reference is not None:
        return reference.shape[:2]
    note = manifest.get("note") or {}
    if "height" in note and "width" in note:
        return int(note["height"]), int(note["width"])
    raise ValueError("manifest needs note.height/note.width when no reference image is supplied")


def fragments_from_manifest(path: str | Path, reference: np.ndarray | None = None) -> list[Fragment]:
    """Load fragment images from a JSON manifest and place them on the note."""

    manifest_path = Path(path)
    base = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canvas_shape = _canvas_shape(manifest, reference)
    fragments: list[Fragment] = []

    for index, item in enumerate(manifest.get("fragments", [])):
        image_path = _resolve_path(base, item.get("image"))
        if image_path is None:
            raise ValueError(f"fragment {index} is missing an image path")
        raw = np.asarray(Image.open(image_path).convert("RGBA"), dtype=np.uint8)
        rgb = raw[..., :3]
        mask_path = _resolve_path(base, item.get("mask"))
        local_mask = load_mask(mask_path) if mask_path else infer_foreground_mask(raw, threshold=float(item.get("threshold", 22.0)))
        affine = np.asarray(item.get("affine_to_note", [[1, 0, 0], [0, 1, 0]]), dtype=np.float32)
        placed_image, placed_mask = warp_fragment_to_canvas(rgb, local_mask, affine, canvas_shape)
        fragment_id = str(item.get("id", f"f{index:05d}"))
        fragments.append(
            Fragment(
                id=fragment_id,
                label=item.get("label") or label_from_filename(image_path),
                side=str(item.get("side", "front")),
                mask=placed_mask,
                image=placed_image,
                tags=tuple(item.get("tags", ())),
                meta={"affine_to_note": affine.tolist(), "source_image": str(image_path)},
            )
        )
    return fragments
