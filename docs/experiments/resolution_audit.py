"""Measurement-only sampling/resource audit; never calibrates production gates."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from time import monotonic

import numpy as np

from moneyrepair.reality import PhysicalToleranceModel
from moneyrepair.tearfit import (
    FractalTearConfig,
    TearFitCoreConfig,
    diagnose_confirmed_candidates,
    make_fractal_tear_fragments,
    reconstruct_placed_fragments,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "benchmarks" / "physical_decision_contract.json"


def zero_failure_lower_bound(n: int, alpha: float = 0.05) -> float | None:
    """One-sided exact binomial bound, only for n independent zero-error trials."""
    if n < 0 or not 0 < alpha < 1:
        raise ValueError("n must be non-negative and alpha must be in (0, 1)")
    return alpha ** (1.0 / n) if n else None


def zero_failure_sample_size(target: float = 0.98, alpha: float = 0.05) -> int:
    if not 0 < target < 1 or not 0 < alpha < 1:
        raise ValueError("target and alpha must be in (0, 1)")
    return math.ceil(math.log(alpha) / math.log(target))


def sampling_rows(contract: dict) -> list[dict]:
    width, height = contract["sheet_size_mm"]
    tolerance = PhysicalToleranceModel(**contract["proxy_tolerance_model"])
    minimum = tolerance.minimum_effective_feature_mm
    grids = [("legacy", 180, 90)] + [
        (f"{dpi}_dpi", round(width * dpi / 25.4), round(height * dpi / 25.4))
        for dpi in (150, 300, 600, 1200)
    ]
    rows = []
    for name, nx, ny in grids:
        sx, sy = nx / width, ny / height
        pixels = nx * ny
        fragments = contract["primary"]["sheets"] * contract["primary"]["pieces_per_sheet"]
        # Bool mask + uint8 RGB, excluding all temporaries and boundary caches.
        rows.append({
            "grid": name, "width": nx, "height": ny,
            "pixels_per_mm_x": sx, "pixels_per_mm_y": sy,
            "minimum_feature_pixels": minimum * min(sx, sy),
            "proxy_translation_tolerance_pixels": tolerance.combined_alignment_tolerance_mm * max(sx, sy),
            "sampling_policy_pass": minimum * min(sx, sy) >= contract["resolution"]["minimum_samples_per_feature"],
            "dense_input_gib_320": pixels * fragments * 4 / 2**30,
            "dense_input_gib_20000": pixels * 20_000 * 4 / 2**30,
            "relative_pixel_work": pixels / (180 * 90),
        })
    return rows


def notch_probe() -> list[dict]:
    """Analytic Gaussian notches, not physical tears or a camera/MTF model.

    Pixel centres sample a continuous profile. Phases expose aliasing; no SR or
    upsampled edge is passed to a matcher. This is an optimistic sampling probe.
    """
    rows = []
    for width_mm in (0.1, 0.25, 0.51, 1.0):
        sigma = width_mm / (2 * math.sqrt(2 * math.log(2)))
        for name, ppm in (("legacy", 180 / 156), ("300_dpi", 300 / 25.4), ("600_dpi", 600 / 25.4)):
            ratios = []
            for phase in np.linspace(0, 1, 101):
                positions = (np.arange(-100, 101) + phase) / ppm
                ratios.append(float(np.exp(-0.5 * (positions / sigma)**2).max()))
            rows.append({
                "feature_fwhm_mm": width_mm, "grid": name,
                "peak_retention_min": min(ratios),
                "peak_retention_mean": float(np.mean(ratios)),
                "feature_samples": width_mm * ppm,
            })
    return rows


def replay_rows() -> list[dict]:
    """Same masks, fixed budgets, no new geometry: detect non-invariance only."""
    base = replace(
        TearFitCoreConfig.v4_4_1(),
        candidate_state_limit=40_000, gap_state_limit=3_000,
        partial_gap_state_limit=1_000, cover_node_limit=75_000,
        candidate_states_per_pair_score=None, gap_states_per_fragment=None,
        partial_gap_states_per_fragment=None, cover_nodes_per_note=None,
    )
    rows = []
    for seed in (101, 102, 103):
        sim = FractalTearConfig(notes=3, pieces_per_note=16, seed=seed, serial_ocr_rate=0.0)
        _, original = make_fractal_tear_fragments(sim)
        for scale in (1, 2, 4):
            fragments = [replace(
                f, mask=np.repeat(np.repeat(f.mask, scale, axis=0), scale, axis=1), image=None,
            ) for f in original]
            arms = ("fixed_pixels",) if scale == 1 else ("fixed_pixels", "linear_units")
            for arm in arms:
                config = base
                if arm == "linear_units":
                    # Hit/support counts count one-pixel boundary samples, not area.
                    config = replace(base, **{
                        key: getattr(base, key) * scale for key in (
                            "tolerance", "min_overlap_pixels", "min_contiguous_pixels",
                            "automatic_contiguous_pixels", "gap_fill_radius", "fray_layers",
                        )
                    })
                start = monotonic()
                core = reconstruct_placed_fragments(fragments, config, max_pieces=18, expected_notes=3)
                seconds = monotonic() - start
                diagnosis = diagnose_confirmed_candidates(candidates=core.selected, fragments=fragments)
                true_keys = {tuple(sorted(f.id for f in fragments if f.meta["note_id"] == note))
                             for note in {f.meta["note_id"] for f in fragments}}
                candidate_keys = {tuple(sorted(c.fragment_ids)) for c in core.candidates}
                row = {
                    "seed": seed, "scale": scale, "arm": arm, "simulation": asdict(sim),
                    "core_config": asdict(config), "resolved_algorithm": core.resolved_algorithm,
                    "scored_pairs": len(core.all_scores), "core_edges": len(core.core_edges),
                    "candidates": len(core.candidates), "selected": len(core.selected),
                    "exact_yield": diagnosis.exact_yield, "exact_precision": diagnosis.exact_precision,
                    "oracle_candidate_recall": len(true_keys & candidate_keys) / 3,
                    "selected_fingerprint": core.selected_solution_fingerprint,
                    "seconds": seconds, "search_stats": core.search_stats,
                }
                rows.append(row)
                print(json.dumps({key: row[key] for key in (
                    "seed", "scale", "arm", "core_edges", "exact_yield", "exact_precision", "seconds",
                )}), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay", action="store_true", help="also run 15 bounded same-geometry core diagnostics")
    args = parser.parse_args()
    contract_bytes = CONTRACT.read_bytes()
    contract = json.loads(contract_bytes)
    payload = {
        "schema": "moneyrepair-resolution-audit-v1",
        "claim_boundary": "Analytic sampling/resources and synthetic scale replay only. No physical gate measured.",
        "contract_sha256": sha256(contract_bytes).hexdigest(),
        "contract": contract,
        "execution": {
            "python": sys.version,
            "numpy": np.__version__,
            "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "tolerance_status": "Uncalibrated proxy; not production thresholds",
        "sampling": sampling_rows(contract), "notches": notch_probe(),
        "zero_error_statistics": {
            "alpha": 0.05, "precision_target": 0.98,
            "required_independent_successes": zero_failure_sample_size(),
            "lower_bound_13": zero_failure_lower_bound(13),
            "lower_bound_16": zero_failure_lower_bound(16),
        },
        "replay": replay_rows() if args.replay else [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
