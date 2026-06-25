import json
from pathlib import Path

import moneyrepair
from moneyrepair.pipeline import run_production_pipeline
from moneyrepair.simulate import make_synthetic_fragments, save_dataset


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

