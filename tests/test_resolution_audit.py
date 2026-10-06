from __future__ import annotations

import json
from pathlib import Path
import runpy

import pytest

AUDIT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "docs/experiments/resolution_audit.py"))


def test_zero_error_does_not_mean_certified_precision():
    bound = AUDIT["zero_failure_lower_bound"]
    assert bound(0) is None
    assert bound(16) == pytest.approx(0.8292502770)
    assert AUDIT["zero_failure_sample_size"]() == 149
    assert bound(148) < 0.98 <= bound(149)
    with pytest.raises(ValueError):
        bound(-1)


def test_physical_grid_and_dense_resource_units():
    contract = json.loads(AUDIT["CONTRACT"].read_text())
    rows = {row["grid"]: row for row in AUDIT["sampling_rows"](contract)}
    assert (rows["300_dpi"]["width"], rows["300_dpi"]["height"]) == (1843, 909)
    assert rows["legacy"]["minimum_feature_pixels"] < 1
    assert not rows["legacy"]["sampling_policy_pass"]
    assert rows["300_dpi"]["sampling_policy_pass"]
    assert rows["300_dpi"]["dense_input_gib_320"] == pytest.approx(1843 * 909 * 320 * 4 / 2**30)


def test_sampling_probe_is_phase_sensitive_and_not_a_gate():
    rows = { (r["feature_fwhm_mm"], r["grid"]): r for r in AUDIT["notch_probe"]() }
    assert rows[(0.51, "legacy")]["peak_retention_min"] < 0.15
    assert rows[(0.51, "300_dpi")]["peak_retention_min"] > 0.97
    contract = json.loads(AUDIT["CONTRACT"].read_text())
    assert contract["resolution"]["physical_core_auto_confirmation_enabled"] is False


def test_physical_contract_keeps_repeats_blindness_and_safety_explicit():
    contract = json.loads(AUDIT["CONTRACT"].read_text())
    assert contract["primary"]["sheets"] == (
        contract["primary"]["calibration_sheets"] + contract["primary"]["blind_evaluation_sheets"]
    )
    acquisition = contract["acquisition_policy"]
    assert acquisition["primary_repeat"] == 1
    assert acquisition["stability_only_repeats"] == [2, 3]
    assert acquisition["best_of_repeats_allowed"] is False
    assert acquisition["production_may_read_evaluation_sidecar"] is False
    assert contract["gates"]["automatic_allowed_after_search_limit_hit"] is False
    assert contract["gates"]["native_seam_verifier_qualification_required"] is True
    assert contract["gates"]["preregistered_ab_tests_per_blind_cohort"] == 1
    assert contract["resolution"]["nominal_dpi_is_calibration"] is False
