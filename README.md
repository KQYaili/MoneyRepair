# MoneyRepair

MoneyRepair is a geometry-first reconstruction system for hand-torn near-identical banknotes.

> [!IMPORTANT]
> **Project Status, Capabilities, and Limitations:**
> For the authoritative status of the system, including what works, where the simulation wall is, dead ends, and future resume-paths, please read **[STATUS.md](STATUS.md)**. No claim elsewhere in this repository may exceed what `STATUS.md` supports.

## Setup

With WSL Anaconda or Miniconda:

```bash
conda create -n moneyrepair python=3.11 -y
conda activate moneyrepair
pip install -e ".[dev]"
```

Or from the checked-in environment file:

```bash
conda env create -f environment.yml
conda activate moneyrepair
```

## Smoke run

```bash
moneyrepair smoke --output-dir runs/smoke --pieces 18 --coverage 0.98
```

The smoke command writes a synthetic dataset, packed compatibility matrix, solution JSON, and PNG visualizations.

For more detailed command descriptions, pipeline orchestration, quality assessment, and reporting, please refer to the files in the `docs/` directory.
