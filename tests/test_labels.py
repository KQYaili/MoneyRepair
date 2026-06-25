import json

import numpy as np
from PIL import Image

from moneyrepair.labels import clean_label, crop_roi, parse_roi, update_manifest_labels


def test_clean_label_and_parse_roi():
    assert clean_label("  A 001\n") == "A001"
    assert parse_roi("0.1,0.2,0.9,0.8") == (0.1, 0.2, 0.9, 0.8)


def test_crop_roi_supports_fractional_coordinates():
    image = Image.fromarray(np.zeros((20, 40), dtype=np.uint8), mode="L")
    cropped = crop_roi(image, (0.25, 0.25, 0.75, 0.75))

    assert cropped.size == (20, 10)


def test_update_manifest_labels_uses_overrides_then_filename(tmp_path):
    image = np.zeros((8, 12, 4), dtype=np.uint8)
    image[..., 3] = 255
    Image.fromarray(image, mode="RGBA").save(tmp_path / "piece_0012.png")
    manifest = {
        "note": {"width": 12, "height": 8},
        "fragments": [
            {"id": "f00000", "image": "piece_0012.png"},
            {"id": "f00001", "image": "piece_0012.png"},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text("f00000,MANUAL-A\n", encoding="utf-8")

    updated = update_manifest_labels(manifest_path, output_path=tmp_path / "labeled.json", labels_file=labels_path)

    assert updated["fragments"][0]["label"] == "MANUAL-A"
    assert updated["fragments"][1]["label"] == "0012"


def test_update_manifest_labels_can_use_fake_ocr(tmp_path):
    image = np.zeros((8, 12, 4), dtype=np.uint8)
    image[..., 3] = 255
    Image.fromarray(image, mode="RGBA").save(tmp_path / "fragment.png")
    manifest = {
        "note": {"width": 12, "height": 8},
        "fragments": [{"id": "f00000", "image": "fragment.png", "label": "old"}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    updated = update_manifest_labels(
        manifest_path,
        method="ocr",
        overwrite=True,
        recognizer=lambda path: " OCR-77 ",
    )

    assert updated["fragments"][0]["label"] == "OCR-77"
