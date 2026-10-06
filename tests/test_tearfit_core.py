from __future__ import annotations

import inspect
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from moneyrepair.cli import main
from moneyrepair.scale import run_v44_base_selection_diagnostic
from moneyrepair.simulate import save_dataset
from moneyrepair.tearfit import (
    TEARFIT_OBSERVED_META_KEYS,
    FractalTearConfig,
    TearFitCoreConfig,
    make_fractal_tear_fragments,
    observed_fragment_view,
    reconstruct_placed_fragments,
    run_tearfit_reconstruction,
    run_tearfit_trial,
)

BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "benchmarks"
    / "v4_4_1_base_selection_n100_seed7.json"
)
GOLDEN_SIMULATION = FractalTearConfig(
    notes=4, pieces_per_note=10, width=144, height=72, seed=2, serial_ocr_rate=0.0
)
GOLDEN_CORE = replace(
    TearFitCoreConfig.v4_4_1(),
    route_fragment_fraction_threshold=0.5,
    min_overlap_pixels=4,
    beam_width=8,
    max_complete_core_candidates=16,
    max_partial_core_candidates=8,
)
GOLDEN_SELECTED = "d6f353ed653cd59d1737993a92c67d6ae80999d4bbeb56b5a0813c5fdd0a2aab"
GOLDEN_PROVENANCE = "74c2eb180419d1082cac3f7123f9d619b70560b97c00d512977b465b5e72c0da"


def _golden_reconstruction(fragments):
    return reconstruct_placed_fragments(
        fragments,
        GOLDEN_CORE,
        max_pieces=GOLDEN_SIMULATION.pieces_per_note + 2,
        expected_notes=GOLDEN_SIMULATION.notes,
    )


def test_v441_preset_is_the_committed_base_selection_intervention():
    committed = json.loads(BENCHMARK.read_text(encoding="utf-8"))["config"]
    preset = TearFitCoreConfig.v4_4_1()

    assert committed["schema_version"] == "4.4.1-diagnostic"
    assert preset.algorithm == committed["algorithm"]
    assert preset.base_selection_strategy == "disjoint_round_robin"
    assert preset.max_complete_core_candidates == committed["max_complete_core_candidates"]
    assert preset.max_partial_core_candidates == committed["max_partial_core_candidates"]
    for name, rate in committed["normalized_rates"].items():
        assert getattr(preset, name) == rate

    diagnostic_defaults = inspect.signature(run_v44_base_selection_diagnostic).parameters
    for name in (
        "route_fragment_fraction_threshold",
        "tolerance",
        "min_overlap_pixels",
        "min_effectiveness",
        "automatic_effectiveness",
        "min_contiguous_pixels",
        "automatic_contiguous_pixels",
        "coverage_threshold",
        "core_raw_coverage_threshold",
        "gap_fill_radius",
        "beam_width",
    ):
        assert getattr(preset, name) == diagnostic_defaults[name].default, name

    trial_defaults = inspect.signature(run_tearfit_trial).parameters
    for name in (
        "core_min_pieces",
        "min_group_gap_score",
        "automatic_group_gap_score",
        "seed_strategy",
        "cover_objective",
    ):
        assert getattr(preset, name) == trial_defaults[name].default, name

    assert preset.gap_proposal_pool == "weak_pair"
    assert preset.use_labels is False
    assert preset.candidate_time_limit_seconds is None
    assert preset.partial_gap_time_limit_seconds is None
    assert preset.cover_time_limit_seconds is None
    assert preset.candidate_state_limit is None
    assert preset.gap_state_limit is None
    assert preset.partial_gap_state_limit is None
    assert preset.cover_node_limit is None


def test_frozen_core_reproduces_golden_fingerprints():
    """Any change to scoring, candidate construction, or exact cover moves these hashes."""

    _template, fragments = make_fractal_tear_fragments(GOLDEN_SIMULATION)
    reconstruction = _golden_reconstruction(fragments)

    assert reconstruction.resolved_algorithm == "effectiveness_gap"
    assert reconstruction.selected_solution_fingerprint == GOLDEN_SELECTED
    assert reconstruction.candidate_provenance_fingerprint == GOLDEN_PROVENANCE
    assert sorted(
        reconstruction.candidate_source(candidate.fragment_ids)
        for candidate in reconstruction.selected
    ) == ["core", "core", "core", "partial_gap"]
    assert reconstruction.edge_intervention_applied is False


def test_trial_delegates_to_the_same_core():
    trial_kwargs = asdict(GOLDEN_CORE)
    del trial_kwargs["fray_layers"]
    result = run_tearfit_trial(GOLDEN_SIMULATION, **trial_kwargs)

    assert result.selected_solution_fingerprint == GOLDEN_SELECTED
    assert result.candidate_provenance_fingerprint == GOLDEN_PROVENANCE
    assert result.diagnostics.exact_yield == 1.0
    assert result.selected_partial_gap_candidates == 1


def test_observed_view_keeps_only_pose_uncertainty():
    _template, fragments = make_fractal_tear_fragments(GOLDEN_SIMULATION)
    fragment = replace(
        fragments[0],
        meta={
            **fragments[0].meta,
            "pose_sigma_x": 1.5,
            "sigma_theta": 0.4,
            "ground_truth_pose": [0, 0, 0],
            "parent_id": "proxy-01",
        },
    )

    (observed,) = observed_fragment_view([fragment])

    assert observed.meta == {"pose_sigma_x": 1.5, "sigma_theta": 0.4}
    assert set(observed.meta) <= set(TEARFIT_OBSERVED_META_KEYS)
    assert observed.mask is fragment.mask
    assert "note_id" in fragment.meta


def test_core_ignores_truth_and_evaluation_metadata():
    _template, fragments = make_fractal_tear_fragments(GOLDEN_SIMULATION)
    poisoned = [
        replace(
            fragment,
            meta={
                **fragment.meta,
                "note_id": "note-000",
                "serial": "SN00000000",
                "ground_truth_pose": [0, 0, 0],
            },
        )
        for fragment in fragments
    ]

    reconstruction = _golden_reconstruction(poisoned)

    assert reconstruction.selected_solution_fingerprint == GOLDEN_SELECTED
    assert reconstruction.candidate_provenance_fingerprint == GOLDEN_PROVENANCE


def test_core_rejects_generated_fragments():
    _template, fragments = make_fractal_tear_fragments(GOLDEN_SIMULATION)
    fragments[3] = replace(
        fragments[3], meta={**fragments[3].meta, "provenance": "generated"}
    )

    with pytest.raises(ValueError, match="observed fragments only"):
        _golden_reconstruction(fragments)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"algorithm": "learned"}, "algorithm must be one of"),
        ({"base_selection_strategy": "random"}, "base_selection_strategy must be one of"),
        ({"seed_strategy": "nearest"}, "seed_strategy must be one of"),
        ({"core_raw_coverage_threshold": 0.95}, "core_raw_coverage_threshold"),
    ],
)
def test_core_config_rejects_unknown_choices(override, message):
    with pytest.raises(ValueError, match=message):
        replace(TearFitCoreConfig.v4_4_1(), **override)


def test_reconstruct_cli_writes_truth_blind_report(tmp_path):
    simulation = FractalTearConfig(
        notes=2, pieces_per_note=4, width=72, height=40, seed=29, serial_ocr_rate=0.0
    )
    template, fragments = make_fractal_tear_fragments(simulation)
    dataset = tmp_path / "placed.npz"
    save_dataset(dataset, template, fragments)
    output_dir = tmp_path / "core"

    exit_code = main(
        [
            "reconstruct",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--max-pieces",
            "6",
        ]
    )

    assert exit_code == 0
    report_text = (output_dir / "reconstruction_report.json").read_text(encoding="utf-8")
    report = json.loads(report_text)
    direct = reconstruct_placed_fragments(
        fragments, TearFitCoreConfig.v4_4_1(), max_pieces=6, expected_notes=2
    )
    assert report["core_preset"] == "v4.4.1"
    assert report["expected_notes_source"] == "estimated_from_fragment_area"
    assert report["reconstruction"]["expected_notes"] == 2
    assert report["reconstruction"]["edge_intervention_applied"] is False
    assert (
        report["reconstruction"]["selected_solution_fingerprint"]
        == direct.selected_solution_fingerprint
    )
    routing = report["routing"]
    assert routing["release_mode"] == "diagnostic_only"
    assert routing["automatic_confirmation_enabled"] is False
    assert routing["automatic"] == []
    assert routing["shadow_automatic"] == [
        list(candidate.fragment_ids) for candidate in direct.selected
        if candidate.evidence_level == "automatic"
    ]
    routed = [
        fragment_id
        for group in routing["automatic"] + routing["review"]
        for fragment_id in group
    ]
    assert sorted(routed + routing["unassigned_fragment_ids"]) == sorted(
        fragment.id for fragment in fragments
    )
    assert "note-00" not in report_text


@pytest.mark.parametrize(("stage", "flag"), [
    ("exact_cover", "node_limit_reached"),
    ("exact_cover", "time_limit_reached"),
    ("core", "state_limit_reached"),
])
def test_reconstruct_report_blocks_truncated_search(tmp_path, monkeypatch, stage, flag):
    simulation = FractalTearConfig(notes=2, pieces_per_note=4, width=72, height=40, seed=29)
    template, fragments = make_fractal_tear_fragments(simulation)
    dataset = tmp_path / "placed.npz"
    save_dataset(dataset, template, fragments)
    core = reconstruct_placed_fragments(
        fragments, TearFitCoreConfig.v4_4_1(), max_pieces=6, expected_notes=2,
    )
    assert core.selected
    core = replace(core, selected=[
        replace(candidate, evidence_level="automatic") for candidate in core.selected
    ], search_stats={stage: {flag: True}})
    monkeypatch.setattr(
        "moneyrepair.tearfit.reconstruct_placed_fragments", lambda *args, **kwargs: core,
    )

    report = run_tearfit_reconstruction(dataset, tmp_path / "report", max_pieces=6)
    routing = report["routing"]
    assert routing["automatic"] == []
    assert routing["shadow_automatic"] == routing["review"]
    assert f"search_truncated:{stage}" in routing["automatic_blockers"]
    assert "physical_resolution_unqualified" in routing["automatic_blockers"]
    assert "native_seam_verifier_unqualified" in routing["automatic_blockers"]
