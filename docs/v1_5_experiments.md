# v1.5 Experiments

> **Historical design note**. This document represents an earlier design round and is superseded by the authoritative [STATUS.md](../STATUS.md) in the repository root.

v1.5 extends the v1.0 simulation baseline in two directions: realistic data and
measurable search optimization.

## Dataset realism directions

1. Photometric capture noise
   - Adds brightness, contrast, color, Gaussian blur, JPEG compression, sensor
     noise, and illumination gradients.
   - Command:
     `moneyrepair simulate-realistic --output data/realistic.npz --pieces 80`
   - Test:
     `tests/test_realism.py` verifies masks are preserved while RGB values are
     degraded.

2. Scan/photo segmentation realism
   - Uses foreground thresholding and connected components to split one scan
     into crops, masks, and a manifest.
   - Command:
     `moneyrepair segment-scan --image data/scan.png --output-dir data/segments`
   - Test:
     `tests/test_scan.py` verifies component sorting, mask crops, label
     overrides, and manifest ingestion.

3. Label realism
   - Supports manual CSV labels, filename/id labels, and optional OCR.
   - Command:
     `moneyrepair label-manifest --manifest data/segments/manifest.json --labels-file data/labels.csv --overwrite`
   - Test:
     `tests/test_labels.py` verifies CSV, filename, and fake OCR paths.

## Algorithm optimization directions

1. Ordering by fragment area
   - Baseline strategy.
   - Best when fragment areas are uneven and large fragments cover most note
     area early.

2. Ordering by compatibility degree
   - Tries more constrained fragments first.
   - Best when pairwise incompatibility is strong and graph degree is
     informative.

3. Ordering by area then degree
   - Hybrid strategy.
   - Intended as a safer default when both area and graph constraints matter.

Benchmark command:

```bash
moneyrepair benchmark-strategies --pieces 120 --width 600 --height 260 --coverage 0.98 --output runs/v1_5_strategy_benchmark.json
```

Scientific report command:

```bash
moneyrepair report-strategies --input runs/v1_5_strategy_benchmark.json --output-prefix runs/v1_5_strategy_report
```

## Acceptance criteria

- `python -m pytest -q`
- `python -m compileall -q src`
- `moneyrepair simulate-realistic ...`
- `moneyrepair benchmark-strategies ...`
- `moneyrepair report-strategies ...`
