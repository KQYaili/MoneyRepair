# Contributing

Thanks for improving MoneyRepair.

## Development setup

```bash
conda env create -f environment.yml
conda activate moneyrepair
pip install -e ".[dev]"
```

## Checks

Before opening a pull request, run:

```bash
python -m pytest -q
python -m compileall -q src
```

## Scope

This project is currently a simulation-first reconstruction toolkit. Keep
changes small and evidence-backed:

- prefer deterministic synthetic tests for algorithm changes;
- keep optional OCR and heavy imaging tools behind optional dependencies;
- do not commit real banknote scans, private fragment images, or generated run
  outputs;
- document any new command in `README.md` or `docs/pipeline.md`.
