import json
from pathlib import Path

import pytest

from moneyrepair.figures import assemble_standard_panels, render_report_figure, validate_report


STRATEGY_RESULTS = [
    {
        "order_strategy": "area",
        "timings_seconds": {"solve": 0.02, "build_matrix": 0.01},
        "best_coverage": 0.981,
        "matrix_footprint": {"dense_bool_mb": 0.4, "packed_mb": 0.05},
    },
    {
        "order_strategy": "area_degree",
        "timings_seconds": {"solve": 0.012, "build_matrix": 0.011},
        "best_coverage": 0.985,
        "matrix_footprint": {"dense_bool_mb": 0.4, "packed_mb": 0.05},
    },
]

QUALITY_SUMMARIES = {
    "clean": {"accepted": 39, "rejected": 1},
    "degraded": {"accepted": 26, "rejected": 14},
}


def test_assemble_standard_panels_covers_evidence_chain():
    panels = assemble_standard_panels(strategy_results=STRATEGY_RESULTS, quality_summaries=QUALITY_SUMMARIES)
    keys = [panel.key for panel in panels]
    assert keys == ["quality", "algorithm", "footprint", "coverage"]


def test_render_report_figure_writes_exports_csv_and_provenance(tmp_path):
    pytest.importorskip("matplotlib")
    source = tmp_path / "strategy.json"
    source.write_text(json.dumps(STRATEGY_RESULTS), encoding="utf-8")

    panels = assemble_standard_panels(strategy_results=STRATEGY_RESULTS, quality_summaries=QUALITY_SUMMARIES)
    manifest = render_report_figure(
        panels,
        tmp_path / "report",
        title="Test report",
        claim="evidence chain",
        sources={"strategy_benchmark": str(source)},
        dpi=120,
    )

    exports = manifest["exports"]
    for key in ("svg", "pdf", "tiff", "data_csv", "manifest"):
        assert Path(exports[key]).exists()

    assert manifest["provenance"]["strategy_benchmark"]["sha256"]
    assert validate_report(manifest) == []

    csv_text = (tmp_path / "report_data.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0] == "panel,panel_title,series,category,value"
    assert "algorithm" in csv_text


def test_validate_report_flags_missing_export(tmp_path):
    pytest.importorskip("matplotlib")
    panels = assemble_standard_panels(strategy_results=STRATEGY_RESULTS)
    manifest = render_report_figure(panels, tmp_path / "r2", dpi=120)
    manifest["exports"]["svg"] = str(tmp_path / "does_not_exist.svg")
    problems = validate_report(manifest)
    assert any("svg" in problem for problem in problems)
