import pytest

from moneyrepair.reports import write_strategy_report


def test_write_strategy_report_exports_three_formats(tmp_path):
    pytest.importorskip("matplotlib")
    results = [
        {
            "order_strategy": "area",
            "timings_seconds": {"solve": 0.02, "build_matrix": 0.01},
            "best_coverage": 0.98,
        },
        {
            "order_strategy": "degree",
            "timings_seconds": {"solve": 0.015, "build_matrix": 0.01},
            "best_coverage": 0.981,
        },
    ]

    outputs = write_strategy_report(results, tmp_path / "strategy_report")

    assert set(outputs) == {"svg", "pdf", "tiff"}
    assert all((tmp_path / f"strategy_report.{suffix}").exists() for suffix in outputs)
