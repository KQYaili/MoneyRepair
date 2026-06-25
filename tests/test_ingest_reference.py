import json

import numpy as np
from PIL import Image

from moneyrepair.ingest import fragments_from_manifest, infer_foreground_mask, label_from_filename, warp_fragment_to_canvas
from moneyrepair.reference import score_best_reference_side, score_fragment_against_reference, score_fragments_by_side
from moneyrepair.simulate import load_dataset, save_dataset
from moneyrepair.types import Fragment


def test_infer_foreground_mask_uses_alpha():
    image = np.zeros((5, 5, 4), dtype=np.uint8)
    image[1:4, 2:4, 3] = 255

    mask = infer_foreground_mask(image)

    assert mask.sum() == 6
    assert mask[2, 3]


def test_warp_fragment_to_canvas_translates_image_and_mask():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[..., 0] = 200
    mask = np.ones((3, 4), dtype=bool)
    affine = np.array([[1, 0, 2], [0, 1, 4]], dtype=np.float32)

    placed_image, placed_mask = warp_fragment_to_canvas(image, mask, affine, canvas_shape=(10, 12))

    assert placed_mask[4:7, 2:6].all()
    assert placed_mask.sum() == 12
    assert int(placed_image[5, 3, 0]) == 200


def test_manifest_ingest_and_reference_score_roundtrip(tmp_path):
    fragment_image = np.zeros((3, 4, 4), dtype=np.uint8)
    fragment_image[..., 0] = 180
    fragment_image[..., 3] = 255
    image_path = tmp_path / "fragment.png"
    Image.fromarray(fragment_image, mode="RGBA").save(image_path)

    manifest = {
        "note": {"height": 10, "width": 12},
        "fragments": [
            {
                "id": "frag-a",
                "label": "A001",
                "side": "front",
                "image": "fragment.png",
                "affine_to_note": [[1, 0, 2], [0, 1, 4]],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    reference = np.zeros((10, 12, 3), dtype=np.uint8)
    reference[4:7, 2:6, 0] = 180

    fragments = fragments_from_manifest(manifest_path, reference=reference)
    score = score_fragment_against_reference(fragments[0], reference)

    assert fragments[0].id == "frag-a"
    assert fragments[0].label == "A001"
    assert score.rmse == 0.0

    dataset_path = tmp_path / "dataset.npz"
    save_dataset(dataset_path, reference, fragments)
    _, loaded = load_dataset(dataset_path)
    assert loaded[0].image is not None
    assert int(loaded[0].image[5, 3, 0]) == 180


def test_reference_scoring_by_declared_or_best_side():
    mask = np.zeros((3, 3), dtype=bool)
    mask[1:, 1:] = True
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    image[mask] = [10, 20, 30]
    fragment = Fragment("f", mask, side="back", image=image)
    front = np.zeros((3, 3, 3), dtype=np.uint8)
    back = np.zeros((3, 3, 3), dtype=np.uint8)
    back[mask] = [10, 20, 30]

    declared = score_fragments_by_side([fragment], {"front": front, "back": back})
    best = score_best_reference_side([fragment], {"front": front, "back": back})

    assert declared[0].side == "back"
    assert declared[0].rmse == 0.0
    assert best[0].side == "back"


def test_label_from_filename_prefers_numbered_stem():
    assert label_from_filename("scan/frag_0012_front.png") == "0012_front"
