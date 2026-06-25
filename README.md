# MoneyRepair

MoneyRepair is a small, testable simulation project for reconstructing shredded
banknotes from fragment compatibility evidence.

The first implementation focuses on the workflow where each fragment already has
an approximate position on the note:

1. Generate or load fragment masks in note coordinates.
2. Build a pairwise compatibility matrix. Two fragments are incompatible when
   their occupied pixels overlap beyond a tolerance.
3. Store the matrix as packed bits.
4. Run a depth-first coverage search to find fragment sets covering a target
   fraction of the note.
5. Render candidate solutions for manual inspection.

It also includes early contour utilities: boundary extraction, coarse tags
(`edge`, `corner`, `center`), affine point transforms, and curve similarity.

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

The smoke command writes a synthetic dataset, packed compatibility matrix,
solution JSON, and PNG visualizations.

## Individual commands

```bash
moneyrepair simulate --output data/demo.npz --pieces 24 --width 420 --height 180
moneyrepair build-matrix --dataset data/demo.npz --output data/demo_matrix.npz
moneyrepair solve --dataset data/demo.npz --matrix data/demo_matrix.npz --output data/solutions.json --coverage 0.99
moneyrepair visualize --dataset data/demo.npz --solutions data/solutions.json --output-dir data/vis --report data/report.html
```

## Real image manifest

For photos or scans, start from a JSON manifest that tells the software where
each fragment image sits on the reference note after affine placement:

```json
{
  "note": {"width": 420, "height": 180},
  "fragments": [
    {
      "id": "frag-0001",
      "label": "0001",
      "side": "front",
      "image": "fragments/0001.png",
      "mask": "masks/0001.png",
      "affine_to_note": [[1, 0, 120], [0, 1, 36]]
    }
  ]
}
```

Then run:

```bash
moneyrepair ingest-manifest --manifest data/manifest.json --reference-front data/rmb100_front.png --output data/real_fragments.npz
moneyrepair score-reference --dataset data/real_fragments.npz --reference-front data/rmb100_front.png --reference-back data/rmb100_back.png --output data/reference_scores.json --best-side
moneyrepair build-matrix --dataset data/real_fragments.npz --reference-scores data/reference_scores.json --max-reference-rmse 35 --output data/real_matrix.npz
moneyrepair solve --dataset data/real_fragments.npz --matrix data/real_matrix.npz --coverage 0.99 --output data/real_solutions.json
moneyrepair visualize --dataset data/real_fragments.npz --solutions data/real_solutions.json --output-dir data/real_vis --report data/real_report.html
```

See [docs/pipeline.md](docs/pipeline.md) for the data model and pruning notes.

## Segment one scan/photo

For a clear scan or photo with separated fragments on a simple background:

```bash
moneyrepair segment-scan --image data/scan_001.png --output-dir data/scan_001_segments --threshold 28 --min-area 200 --padding 6
moneyrepair label-manifest --manifest data/scan_001_segments/manifest.json --labels-file data/labels.csv --overwrite
moneyrepair ingest-manifest --manifest data/scan_001_segments/manifest.json --output data/scan_001_fragments.npz
```

`segment-scan` writes RGBA crops, binary masks, and a manifest. Labels default
to generated ids such as `f00000`; pass `--labels-file labels.csv` to override
them with a two-column `id,label` or `index,label` file. Full OCR can be added
later behind the same manifest `label` field.

Optional OCR is available if `pytesseract` and a local Tesseract executable are
installed:

```bash
pip install -e ".[ocr]"
moneyrepair label-manifest --manifest data/scan_001_segments/manifest.json --method ocr --roi 0,0,1,0.25 --overwrite
```

## Precomputed pair records

If pairwise comparison has already been computed elsewhere, import it directly:

```bash
moneyrepair import-pairs --dataset data/real_fragments.npz --pairs data/incompatible_pairs.csv --relation incompatible --output data/imported_matrix.npz
moneyrepair solve --dataset data/real_fragments.npz --matrix data/imported_matrix.npz --coverage 0.99 --output data/imported_solutions.json
```

The pair file can be CSV, TSV, or whitespace-delimited text with two ids per
line. With `--relation incompatible`, all unspecified pairs are considered
compatible. With `--relation compatible`, all unspecified pairs are considered
incompatible.

To estimate matrix memory before a large run:

```bash
moneyrepair estimate-matrix --fragments 20000 --output data/matrix_footprint.json
moneyrepair benchmark-synthetic --pieces 120 --width 600 --height 260 --coverage 0.98 --output data/benchmark_120.json
moneyrepair benchmark-strategies --pieces 120 --width 600 --height 260 --coverage 0.98 --output data/strategy_benchmark.json
moneyrepair report-strategies --input data/strategy_benchmark.json --output-prefix data/strategy_report
```

`benchmark-synthetic` times synthetic data generation, compatibility matrix
construction, and DFS search. Use it for machine-local sanity checks before a
large WSL/Anaconda run. `report-strategies` requires matplotlib; install it with
`pip install -e ".[reports]"` if your environment does not already include it.

## Batch reconstruction

For the "search a candidate, inspect it, confirm it, then move faster" workflow:

```bash
moneyrepair batch-next --dataset data/real_fragments.npz --matrix data/real_matrix.npz --state data/batch_state.json --output-dir data/batch_0001 --coverage 0.99 --max-solutions 10
moneyrepair batch-confirm --state data/batch_state.json --candidates data/batch_0001/candidates.json --index 0 --note-id note-0001
moneyrepair batch-next --dataset data/real_fragments.npz --matrix data/real_matrix.npz --state data/batch_state.json --output-dir data/batch_0002 --coverage 0.99 --max-solutions 10
```

`batch-next` writes `candidates.json`, a `vis/` directory, and `report.html`.
`batch-confirm` records the accepted candidate and removes those fragments from
future searches. Use `batch-reject` when a visually bad candidate should not be
shown again. Both accept `--operator` and `--reason`, recorded in the batch
state audit log.

## Production pipeline (v2.0)

v2.0 makes production tradeoffs: gate capture quality, prune fast, search, and
keep everything auditable. See
[docs/v2.0 industrial algorithm](docs/v2_0_industrial_algorithm.md).

Score acquisition quality against the contract (focus, glare, segmentation
confidence, color drift):

```bash
moneyrepair assess-quality --dataset data/real_fragments.npz --use-reference --output data/quality.json
```

Build the compatibility matrix with the grid-accelerated engine for large
fragment counts, optionally streaming incompatible pairs without a dense matrix:

```bash
moneyrepair build-matrix --dataset data/real_fragments.npz --engine fast --pairs-out data/incompatible_pairs.csv --output data/real_matrix.npz
```

Run one auditable batch end to end. It gates frames on the quality contract,
prunes, searches with the `area_degree` strategy, renders candidates, and writes
`run_manifest.json` with input hashes, parameters, timings, and the QA summary:

```bash
moneyrepair run-pipeline --dataset data/real_fragments.npz --output-dir data/run_0001 --coverage 0.99 --max-solutions 10
```

## Current scope

This is intentionally a software simulation first. Real scan/photo ingestion can
plug into the same fragment model once masks, approximate affine placement, and
labels are available. The current real-input path assumes labels are provided by
the manifest or encoded in filenames; full OCR can be added later behind the same
manifest field.

## Contributing

Run `python -m pytest -q` and `python -m compileall -q src` before publishing
changes. See [CONTRIBUTING.md](CONTRIBUTING.md).

For GitHub publishing steps, see [docs/github.md](docs/github.md).

Version planning:
[v1.5 experiments](docs/v1_5_experiments.md),
[v2.0 industrial algorithm](docs/v2_0_industrial_algorithm.md), and
[v2.5 scientific reporting](docs/v2_5_scientific_reporting.md).
