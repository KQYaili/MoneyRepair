# Release Checklist

Use this before publishing the project to GitHub.

## Local checks

```bash
python -m pytest -q
python -m compileall -q src
moneyrepair benchmark-synthetic --pieces 80 --width 480 --height 210 --coverage 0.98 --output runs/release_benchmark.json
```

## Data safety

Do not commit:

- real banknote photos or scans;
- private fragment images;
- generated `runs/` or `data/` outputs;
- OCR caches or local labels that identify private datasets.

The repository should contain source code, documentation, tests, and synthetic
examples only.

## GitHub publish

```bash
git status --short
git remote add origin git@github.com:<owner>/MoneyRepair.git
git push -u origin main
```

After publishing, update `pyproject.toml` project URLs to the final repository
URL.

See [github.md](github.md) for SSH, HTTPS, and bundle-based publishing options.
