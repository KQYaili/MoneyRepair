import json
from pathlib import Path

import moneyrepair
import numpy as np
import pytest
from moneyrepair.pipeline import run_production_pipeline
from moneyrepair.simulate import make_synthetic_fragments, save_dataset
from moneyrepair.types import Fragment


def test_run_production_pipeline_writes_auditable_manifest(tmp_path):
    template, fragments = make_synthetic_fragments(pieces=12, width=140, height=70, seed=5)
    dataset_path = tmp_path / "dataset.npz"
    save_dataset(dataset_path, template, fragments)

    output_dir = tmp_path / "run"
    manifest = run_production_pipeline(
        dataset_path,
        output_dir,
        target_coverage=0.9,
        max_solutions=3,
        time_limit_seconds=10,
    )

    assert manifest["version"] == moneyrepair.__version__
    assert manifest["inputs"]["dataset_sha256"]
    assert manifest["inputs"]["fragments_total"] == len(fragments)
    assert manifest["search"]["active_fragments"] >= 1
    assert "total" in manifest["timings_seconds"]
    assert "total_without_jit_warmup" in manifest["timings_seconds"]

    for key in ("matrix", "candidates", "quality_report", "report", "run_manifest"):
        assert Path(manifest["outputs"][key]).exists()

    written = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert written["tool"] == "moneyrepair"
    assert written["quality"]["accepted"] + written["quality"]["rejected"] == len(fragments)


def test_run_production_pipeline_auto_locate(tmp_path):
    template, fragments = make_synthetic_fragments(pieces=4, width=140, height=70, seed=5)
    for f in fragments:
        if "affine_to_note" in f.meta:
            del f.meta["affine_to_note"]
    
    dataset_path = tmp_path / "dataset_no_affine.npz"
    save_dataset(dataset_path, template, fragments)

    output_dir = tmp_path / "run_auto"
    manifest = run_production_pipeline(
        dataset_path,
        output_dir,
        target_coverage=0.6,
        max_solutions=3,
        time_limit_seconds=15,
        auto_locate=True,
    )
    assert manifest["version"] == moneyrepair.__version__
    assert manifest["outputs"]["run_manifest"]


def test_auto_locate_discriminate_appearance_requires_template_coordinates(tmp_path):
    template = np.zeros((20, 20, 3), dtype=np.uint8)
    fragment = Fragment(
        id="raw_crop",
        mask=np.ones((8, 8), dtype=bool),
        image=np.ones((8, 8, 3), dtype=np.uint8) * 128,
    )
    dataset_path = tmp_path / "raw_crop.npz"
    save_dataset(dataset_path, template, [fragment])

    with pytest.raises(ValueError, match="requires fragments already in template coordinates"):
        run_production_pipeline(
            dataset_path,
            tmp_path / "run_raw",
            auto_locate=True,
            discriminate_appearance=True,
        )


def test_run_production_pipeline_with_interlock(tmp_path):
    template, fragments = make_synthetic_fragments(pieces=6, width=140, height=70, seed=5)
    dataset_path = tmp_path / "dataset.npz"
    save_dataset(dataset_path, template, fragments)

    output_dir = tmp_path / "run_interlock"
    manifest = run_production_pipeline(
        dataset_path,
        output_dir,
        target_coverage=0.9,
        max_solutions=3,
        time_limit_seconds=10,
        include_interlock=True,
    )

    assert manifest["parameters"]["include_interlock"] is True
    assert manifest["search"]["interlock_stats"] is not None
    assert "bbox_candidate_pairs" in manifest["search"]["interlock_stats"]

