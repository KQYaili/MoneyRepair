import json

import numpy as np
from PIL import Image

from moneyrepair.ingest import fragments_from_manifest
from moneyrepair.scan import connected_components, load_label_overrides, segment_scan_to_manifest


def test_connected_components_filters_and_sorts_components():
    mask = np.zeros((12, 16), dtype=bool)
    mask[7:10, 10:14] = True
    mask[1:4, 2:6] = True
    mask[0, 15] = True

    components = connected_components(mask, min_area=4)

    assert len(components) == 2
    assert components[0].bbox == (2, 1, 6, 4)
    assert components[1].bbox == (10, 7, 14, 10)


def test_load_label_overrides_accepts_bom_csv(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("\ufeffid,label\nf00000,A-01\n1,B-02\n", encoding="utf-8")

    assert load_label_overrides(path) == {"f00000": "A-01", "1": "B-02"}


def test_segment_scan_to_manifest_outputs_crops_masks_and_ingestable_manifest(tmp_path):
    scan = np.full((32, 48, 3), 245, dtype=np.uint8)
    scan[4:12, 5:16] = [180, 30, 40]
    scan[18:29, 30:43] = [40, 70, 190]
    scan_path = tmp_path / "scan.png"
    Image.fromarray(scan, mode="RGB").save(scan_path)
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text("0,left-piece\nf00001,right-piece\n", encoding="utf-8")

    manifest = segment_scan_to_manifest(
        scan_path,
        tmp_path / "out",
        threshold=30.0,
        min_area=20,
        padding=2,
        labels_file=labels_path,
    )

    assert len(manifest["fragments"]) == 2
    assert manifest["fragments"][0]["label"] == "left-piece"
    assert manifest["fragments"][1]["label"] == "right-piece"
    assert (tmp_path / "out" / "fragments" / "f00000.png").exists()
    assert (tmp_path / "out" / "masks" / "f00000_mask.png").exists()

    manifest_path = tmp_path / "out" / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw["fragments"][0]["affine_to_note"] == [[1, 0, 3], [0, 1, 2]]

    template = np.zeros((32, 48, 3), dtype=np.uint8)
    fragments = fragments_from_manifest(manifest_path, reference=template)
    assert len(fragments) == 2
    assert fragments[0].mask.sum() == 88
