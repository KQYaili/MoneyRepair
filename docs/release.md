# Release Checklist

Use this checklist before pushing the canonical `main` branch.

## Code Gates

Run in the isolated WSL environment:

```bash
python -m pytest -q
python -m compileall -q src
ruff check src tests docs/figures
mypy src/moneyrepair --ignore-missing-imports
```

Pytest, compileall, and Ruff are blocking. Record unchanged advisory mypy debt
instead of hiding it.

## Smoke And Artifact Gates

```bash
moneyrepair smoke \
  --output-dir runs/release_smoke \
  --pieces 18 \
  --coverage 0.98

moneyrepair export-diagram \
  --name production-pipeline \
  --output-prefix runs/release_pipeline

python docs/figures/make_research_summary.py
```

Verify that the smoke manifest records input hashes, parameters, timings,
limits, search/quality counts, and output paths. Parse every checked-in
`.drawio` file as XML and visually inspect the current pipeline and evidence
summary at desktop width.

## Claim Gates

- `STATUS.md` remains the authority and the first-screen decision is current.
- Simulation, synthetic-capture, and physical results are labelled separately.
- A single-seed mechanism result is not described as replicated evidence.
- Experimental modules are not routed into the supported pipeline.
- No physical success is claimed while the pilot remains uncollected.

## Data Safety

Do not commit:

- real banknote photos or scans;
- private fragment images or labels;
- OCR caches, credentials, or local calibration secrets;
- generated `runs/` or `data/` output;
- evaluation annotations that could identify a private dataset.

## Git Gates

```bash
git status --short
git branch --merged main
git branch -r --merged origin/main
git tag --list
git push origin main
```

`main` is the only canonical long-lived branch. After confirming every retired
branch head is an ancestor of `main`, delete those branch refs and preserve the
release tags. Never force-push `main`.

After the push, confirm the remote commit, default branch, CI matrix, and
rendering of README figures and Draw.io download links.
