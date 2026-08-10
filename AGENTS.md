# AGENTS.md

Routing guide for coding agents working in this repository. Human contributors
should also read [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Authoritative status

**[STATUS.md](STATUS.md) is the single source of truth** for what the system can
and cannot do, where the simulation wall is, and which paths are dead ends. No
change, claim, or comment anywhere in this repository may exceed what `STATUS.md`
supports. When behavior and `STATUS.md` disagree, treat it as a bug to surface,
not something to work around.

## Core boundary

- Production code lives under `src/moneyrepair/` (the core package).
- `src/moneyrepair/experimental/` is **UNVERIFIED research scaffold** (v6–v10,
  LLM control, policy compare). It is not trained, not benchmarked, and does not
  beat the deterministic exact-cover solver. Do not route production behavior
  through it, and keep it behind the `experimental` pytest marker.
- Tests live under `tests/`. Diagrams, reports, and measured results live under
  `docs/` and `runs/`.

## Canonical owners

| Concern | Owner |
| --- | --- |
| CLI entry point, sub-commands, argument parsing | `src/moneyrepair/cli.py` |
| Production reconstruction pipeline & run manifest | `src/moneyrepair/pipeline.py` |
| Branch-and-bound exact-cover search | `src/moneyrepair/solver.py` |
| Tear-fit scoring, candidate generation, exact cover | `src/moneyrepair/tearfit.py` |
| Compatibility matrix build | `src/moneyrepair/compat.py` |
| Pose location / auto-locate | `src/moneyrepair/locator.py` |
| Quality gating | `src/moneyrepair/quality.py` |

## Setup

Use the WSL Anaconda/Miniconda Python for this project.

```bash
conda env create -f environment.yml   # or: conda create -n moneyrepair python=3.11 -y
conda activate moneyrepair
pip install -e ".[dev]"
```

Optional dependency gates (install only what a task needs):

- `.[dev]` — pytest, ruff, mypy (required for the checks below)
- `.[ml]` — torch/torchvision (only for `experimental` v6–v10 tests)
- `.[ocr]` — pytesseract
- `.[reports]` — matplotlib

## Checks (run before committing)

```bash
python -m pytest -q            # test suite (core path)
python -m compileall -q src    # syntax gate
ruff check src tests           # lint gate (enforced in CI)
mypy src/moneyrepair --ignore-missing-imports   # type check (advisory; being tightened)
```

CI (`.github/workflows/ci.yml`) runs the pytest matrix (Python 3.10–3.13),
`compileall`, `ruff check` (blocking), and `mypy` (advisory). Keep `ruff check`
clean before pushing.

## Smoke run

```bash
moneyrepair smoke --output-dir runs/smoke --pieces 18 --coverage 0.98
```

This writes a synthetic dataset, packed compatibility matrix, solution JSON, PNG
visualizations, and an auditable `run_manifest.json` (input SHA256, parameters,
timings, quality/search stats). Pipeline logs are emitted via the
`moneyrepair.pipeline` logger, correlated by the dataset SHA256 prefix.

## Non-negotiable engineering principles

- **Physical realism first.** Never generate or complete missing tear edges
  (no hallucinated geometry). Do not reintroduce features already falsified under
  pressure testing (e.g. boundary color continuity, wear clustering).
- **Evidence-backed thresholds.** Tolerance gates must be grounded in a physical
  tolerance model (e.g. `r_e`), not tuned by hand.
- **Small, deterministic changes.** Prefer deterministic synthetic tests for
  algorithm changes; do not commit real banknote scans, private fragment images,
  or generated run outputs.
- **Document new commands** in `README.md` or `docs/pipeline.md`.
