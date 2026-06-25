from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from moneyrepair.ingest import label_from_filename
from moneyrepair.scan import load_label_overrides

Recognizer = Callable[[Path], str]
DEFAULT_TESSERACT_CONFIG = (
    "--psm 7 -c "
    "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
)


def clean_label(text: str) -> str:
    """Normalize OCR text into a compact fragment label."""

    text = text.strip()
    text = re.sub(r"\s+", "", text)
    match = re.search(r"[A-Za-z0-9_-]+", text)
    return match.group(0) if match else ""


def parse_roi(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must have four comma-separated values: x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI must have positive width and height")
    return x0, y0, x1, y1


def crop_roi(image: Image.Image, roi: tuple[float, float, float, float] | None) -> Image.Image:
    if roi is None:
        return image
    width, height = image.size
    x0, y0, x1, y1 = roi
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
        box = (int(round(x0 * width)), int(round(y0 * height)), int(round(x1 * width)), int(round(y1 * height)))
    else:
        box = (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
    return image.crop(box)


def recognize_label_with_tesseract(
    image_path: str | Path,
    roi: tuple[float, float, float, float] | None = None,
    config: str = DEFAULT_TESSERACT_CONFIG,
) -> str:
    """OCR a label with optional pytesseract installed by the user."""

    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("OCR requires optional dependency pytesseract and a local Tesseract executable") from exc

    image = Image.open(image_path).convert("L")
    image = crop_roi(image, roi)
    image = ImageOps.autocontrast(image)
    text = pytesseract.image_to_string(image, config=config)
    return clean_label(text)


def _resolve_manifest_image(base: Path, item: dict[str, Any]) -> Path | None:
    value = item.get("image")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def label_for_item(
    item: dict[str, Any],
    base: Path,
    method: str,
    recognizer: Recognizer | None = None,
    roi: tuple[float, float, float, float] | None = None,
    tesseract_config: str | None = None,
) -> str:
    if method == "id":
        return clean_label(str(item.get("id", "")))

    image_path = _resolve_manifest_image(base, item)
    if method == "filename":
        if image_path is None:
            return ""
        return clean_label(label_from_filename(image_path))

    if method == "ocr":
        if image_path is None:
            return ""
        if recognizer is not None:
            return clean_label(recognizer(image_path))
        return recognize_label_with_tesseract(image_path, roi=roi, config=tesseract_config or DEFAULT_TESSERACT_CONFIG)

    raise ValueError("method must be one of: id, filename, ocr")


def update_manifest_labels(
    manifest_path: str | Path,
    output_path: str | Path | None = None,
    method: str = "filename",
    labels_file: str | Path | None = None,
    overwrite: bool = False,
    roi: tuple[float, float, float, float] | None = None,
    recognizer: Recognizer | None = None,
    tesseract_config: str | None = None,
) -> dict:
    """Update manifest fragment labels from overrides, filenames, ids, or OCR."""

    manifest_path = Path(manifest_path)
    base = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    overrides = load_label_overrides(labels_file)

    for index, item in enumerate(manifest.get("fragments", [])):
        current = str(item.get("label", "") or "")
        if current and not overwrite:
            continue
        fragment_id = str(item.get("id", f"f{index:05d}"))
        override = overrides.get(fragment_id) or overrides.get(str(index))
        label = clean_label(override) if override else label_for_item(
            item,
            base=base,
            method=method,
            recognizer=recognizer,
            roi=roi,
            tesseract_config=tesseract_config,
        )
        if label:
            item["label"] = label

    target = Path(output_path) if output_path else manifest_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
