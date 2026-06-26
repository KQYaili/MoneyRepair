from __future__ import annotations

from moneyrepair.policy_compare import run_policy_controller_comparison


def test_policy_controller_comparison_reports_best_policy():
    payload = run_policy_controller_comparison(
        profile="smoke",
        policies=("static_score", "static_count", "llm_balanced"),
        serial_ocr_rates=(0.6,),
        width=90,
        height=48,
        min_overlap_pixels=6,
        beam_width=24,
        candidate_time_limit_seconds=3.0,
        cover_time_limit_seconds=1.0,
    )

    assert payload["best_policy"] in {"static_score", "static_count", "llm_balanced"}
    assert len(payload["rows"]) == 2 * 3
    assert [item["policy"] for item in payload["summary"]]
    assert all("mean_exact_yield" in item for item in payload["summary"])
