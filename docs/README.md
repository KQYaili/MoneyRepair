# MoneyRepair Documentation Index

Welcome to the **MoneyRepair** documentation portal. This directory contains complete architectural notes, mathematical proofs, experimental logs, and benchmark reports for the MoneyRepair banknote reconstruction system.

---

## 🗺️ Documentation Sitemap

```
docs/
├── README.md                          # 👈 You are here: Master Documentation Index
├── pipeline.md                        # Production Pipeline Architecture & Schematics
│
├── 🔬 Version 4.3.2 Scale-Fineness Validation (Latest)
│   ├── v4_3_tear_effectiveness.md     # Adaptive Etear Score, Triage Routing & Gap Recovery
│   ├── v4_3_1_mechanism_validation.md # Canonical N=20 mechanism decomposition
│   ├── v4_3_2_scale_fineness.md        # Compute calibration and bottleneck protocol
│   └── v4_3_ab_benchmark.md           # Supplemental N=10 consistency audit
│
├── 📐 Auto-Locator & Candidate Pose Search
│   ├── v4_0_production_reconstruction.md # Hybrid Coarse-to-Fine Locator & Zero-Allocation Solver
│   ├── v4_0_algorithm_deduction.md       # Hardcore Mathematical Analysis & Proofs
│   └── stage4_convergence_report.md      # Performance Scaling & Convergence Curves
│
├── 🧪 Multi-Note Pool & Pressure Research
│   ├── tearfit_research.md            # Placed-Coordinate Fractal Tear-Fit Sandbox
│   ├── v3_0_chimera_discrimination.md  # Appearance Fingerprinting & DBSCAN Chimera Pruning
│   └── v4_1_pressure_realism.md       # Spatial Wear & Pressure Benchmark Sweeps
│
├── 📜 Historical Milestones & Specifications
│   ├── v1_5_experiments.md            # Realism Augmentation & Baseline DFS Orderings
│   ├── v2_0_industrial_algorithm.md   # Acquisition QA Contract & Packed Bit-Matrix
│   └── v2_5_scientific_reporting.md   # Publication Figures & Editable Visio Exports
│
└── 🚀 Operations & Deployment
    ├── github.md                      # GitHub CI Workflow & Package Publishing
    └── release.md                     # Release Verification Checklist
```

---

## 📑 Core Documentation Guide

### 1. System Architecture & Workflows
* **[pipeline.md](pipeline.md)**: Overall pipeline architecture, acquisition QA gate, branch-and-bound DFS solver logic, and operator interactive review loop. Includes editable Visio/SVG flowcharts.

### 2. v4.3.2 Adaptive Physical Evidence (Latest)
* **[v4_3_tear_effectiveness.md](v4_3_tear_effectiveness.md)**: Mathematical formulation of $E_{\text{tear}}$, contiguous seam length, normal opposition, curvature entropy, locator uncertainty, and whole-assembly gap recovery ($G$).
* **[v4_3_1_mechanism_validation.md](v4_3_1_mechanism_validation.md)**: Canonical N=20 mechanism decomposition comparing `baseline`, `effectiveness`, `effectiveness_gap`, and `v43_routed` under fixed search budgets.
* **[v4_3_2_scale_fineness.md](v4_3_2_scale_fineness.md)**: Preregistered anchor staircase, fixed and normalized compute tracks, checkpointing, and causal wall diagnostics.
* **[v4_3_ab_benchmark.md](v4_3_ab_benchmark.md)**: Supplemental N=10 same-seed audit retained for reproducibility; it is not used for headline claims.

### 3. Pose Locator & Solver Optimization
* **[v4_0_production_reconstruction.md](v4_0_production_reconstruction.md)**: Overview of the pyramid downsampling locator, candidate pose model, and dense vectorized solver.
* **[v4_0_algorithm_deduction.md](v4_0_algorithm_deduction.md)**: Mathematical deductions for JIT template matching, score-basin uncertainty, and time-complexity bounds.
* **[stage4_convergence_report.md](stage4_convergence_report.md)**: Empirical timing curves, memory footprints, and scalability benchmarks.

### 4. Multi-Note Pool & Chimera Hardening
* **[v3_0_chimera_discrimination.md](v3_0_chimera_discrimination.md)**: Tone gain fitting, DBSCAN appearance clustering, and elimination of cross-note chimeras.
* **[v4_1_pressure_realism.md](v4_1_pressure_realism.md)**: Stress tests under non-uniform spatial wear, local staining, and large banknote pools ($N \ge 50$).

---

## 🔍 Single Source of Truth

> [!IMPORTANT]
> All quantitative metrics, measured performance boundaries, and status claims are governed by **[STATUS.md](../STATUS.md)** in the repository root.
