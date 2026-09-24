# Contributing

Thanks for improving MoneyRepair. Read [STATUS.md](STATUS.md) before proposing
an algorithm change; it is the authority for current evidence and allowed next
work.

## Development Setup

Use the isolated WSL environment:

```bash
conda env create -f environment.yml
conda activate moneyrepair
pip install -e ".[dev,reports]"
```

Install `.[ocr]` or `.[ml]` only when the task requires it. Experimental ML
modules must remain behind the `experimental` pytest marker and outside the
production path.

## Branch And Commit Policy

`main` is the sole canonical long-lived branch. Short-lived pull-request
branches are acceptable while work is active, but they should be merged and
deleted after validation. Preserve release tags; do not recreate version
branches as parallel product lines.

Keep commits scoped and deterministic. Do not rewrite published history or
force-push `main`.

## Required Checks

```bash
python -m pytest -q
python -m compileall -q src
ruff check src tests docs/figures
mypy src/moneyrepair --ignore-missing-imports
```

Pytest, compileall, and Ruff are blocking. Full-package mypy is advisory while
the legacy type surface is reduced; changed modules should be clean.

## Evidence Rules

- Prefer same-seed, frozen-budget A/B tests for algorithm changes.
- Preserve raw JSON, seeds, limits, fingerprints, and manifests.
- Label simulator, synthetic-capture, and physical results separately.
- Never use evaluation truth to choose a production mask, pose, edge, or
  candidate.
- Do not tune another algorithm against the frozen seed-7 discovery case.
- Repair only the first failed physical stage.
- Do not generate or inpaint missing tear geometry.
- Do not revive appearance clustering, boundary colour, contact count, or
  naive whole-contour matching as note-identity evidence.

## Documentation And Figures

Document new commands in `README.md` or `docs/pipeline.md`. Update
`STATUS.md` only when a measured result changes the supported claim.

Draw.io is the canonical editable diagram format. Generate diagrams through
`moneyrepair export-diagram` so JSON, Draw.io, and SVG stay synchronized.
Scientific figures must be generated from committed source data and retain
editable text in SVG.

## Data Safety

Do not commit real banknote scans, private fragment images, OCR caches, local
labels, generated `runs/` output, or credentials. Commit synthetic examples
only when their protocol and provenance are explicit.
