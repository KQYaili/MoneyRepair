# MoneyRepair Final Cleanup Design

This document records the final cleanup plan for the repository. `STATUS.md` is
the authoritative project status; this file explains the code organization that
keeps that status honest.

## Goal

Collapse the exploration rounds into one honest, runnable, bounded system:

- keep the deterministic geometry pipeline as the supported core;
- demote superseded discriminators to comparison baselines;
- quarantine unverified ML/LLM work behind explicit experimental entry points;
- keep claims bounded by the measured simulation results in `STATUS.md`.

This is not a rewrite. It is a packaging, documentation, and test-boundary
cleanup so the repository does not quietly overclaim.

## Package Tiers

```text
src/moneyrepair/
  __init__.py            # exports CORE only
  types.py

  # CORE: supported pipeline, torch-free
  tearfit.py             # geometry kernel and exact-cover candidate selection
  locator.py             # registration and candidate pose search
  simulate.py            # synthetic testbeds
  pressure.py            # stress harness
  diagnostics.py         # yield, precision, chimera, and grouping metrics
  compat.py
  solver.py
  batch.py
  quality.py
  reference.py
  ingest.py
  scan.py
  labels.py
  pipeline.py
  cli.py
  reports.py
  figures.py
  diagrams.py
  visualize.py
  style.py
  benchmark.py
  realism.py

  baselines/             # superseded; comparison only
    fingerprint.py       # appearance-gain clustering
    interlock.py         # contact-count discriminator
    features.py
    vision.py            # whole-contour similarity helpers

  experimental/          # unverified; opt-in only
    v6_to_v10.py         # untrained DL scaffolds
    llm_control.py       # LLM search-policy controller
    policy_compare.py    # policy comparison harness
```

## Disposition

| Module | Tier | Reason |
|---|---|---|
| `tearfit.py` | CORE | Geometry-first tear discriminator plus generate-then-select exact cover is the real contribution. |
| `locator.py` | CORE | Registration is needed for position-unknown inputs; boundary-colour discrimination remains superseded. |
| `simulate.py` | CORE | Keeps both friendly and honest simulation regimes explicit. |
| `pressure.py`, `diagnostics.py` | CORE | They report collapse modes instead of hiding them. |
| `compat.py`, `solver.py` | CORE | General compatibility matrix and covering search. |
| `batch.py`, `quality.py`, `reference.py`, `ingest.py`, `scan.py`, `labels.py` | CORE | Acquisition, IO, and human-review infrastructure. |
| `pipeline.py`, `cli.py` | CORE | Default orchestration must not import experimental ML/LLM machinery. |
| `fingerprint.py` | baseline | Appearance gain fails under spatially non-uniform wear. |
| `interlock.py` | baseline | Contact count does not verify tear-profile mating. |
| `features.py`, `vision.py` | baseline | Whole-contour similarity is over-invariant and fails on jagged input. |
| `v6_to_v10.py` | experimental | Untrained DL stack; not benchmarked on the fine-fragment wall. |
| `llm_control.py`, `policy_compare.py` | experimental | Search-policy tools only; they do not change geometry evidence. |

## Test Boundary

Default `pytest` is the CORE test run. It must not collect experimental tests or
import torch on machines where torch happens to be installed. Experimental tests
are explicitly marked and require:

```bash
python -m pytest --run-experimental -m experimental
```

CI keeps the same split: the main matrix installs `.[dev]` and runs the core
suite; the ML job installs `.[dev,ml]` and opts into experimental tests.

## Definition of Done

- `pip install .` provides the supported pipeline without torch.
- `python -m moneyrepair.cli smoke ...` runs the honest synthetic pipeline.
- `python -m pytest` runs the core suite without collecting experimental tests.
- Baseline modules and experimental modules carry explicit honesty banners.
- `STATUS.md` is the single source of truth for capability claims and limits.
