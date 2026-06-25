from moneyrepair.benchmark import estimate_matrix_footprint, run_synthetic_benchmark, write_matrix_footprint


def test_estimate_matrix_footprint_for_20000_fragments(tmp_path):
    footprint = estimate_matrix_footprint(20_000)

    assert footprint.dense_bool_bytes == 400_000_000
    assert footprint.packed_bytes == 50_000_000

    output = tmp_path / "footprint.json"
    write_matrix_footprint(output, 20_000)
    assert "packed_mb" in output.read_text(encoding="utf-8")


def test_run_synthetic_benchmark_reports_timings_and_solution():
    result = run_synthetic_benchmark(
        pieces=10,
        width=120,
        height=60,
        target_coverage=0.9,
        max_solutions=2,
        time_limit_seconds=5,
    )

    assert result.pieces_generated == 10
    assert result.matrix_footprint.fragments == 10
    assert result.timings_seconds["total"] >= 0
    assert result.solutions_found >= 1
