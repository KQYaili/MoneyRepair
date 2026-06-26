# Tear-Fit Research Track

This track tests the geometry claim that was not proven by the earlier contour
and contact-ratio experiments:

> Once fragments are placed in the standard banknote coordinate frame, torn
> edges should be compared by absolute coordinate coincidence, not by global
> rotation/translation-invariant shape similarity.

The module is intentionally a research sandbox. It does not replace the
production locator. It assumes the locator has already produced placed fragment
masks in one template coordinate frame.

## Implemented Kernel

`moneyrepair.tearfit` adds:

- Per-note recursive fractal tear partitions, so every note has independent
  jagged tear geometry instead of shared Voronoi straight edges.
- Edge fray, which removes a small random subset of internal tear-boundary
  pixels and creates the 1-2 pixel gaps that forced coverage checks to use a
  small dilation allowance.
- Tear-boundary extraction that excludes clean outer note-frame edges.
- Absolute-coordinate tear overlap: two pieces get an edge when their internal
  tear boundaries occupy the same note coordinates within a small tolerance.
- Serial-label hard anchors: groups cannot contain conflicting labels, and the
  final selection cannot confirm two candidates with the same serial. Labels
  are priority seeds and constraints, not the only legal search starts.
- A global candidate set-packing pass over full-note candidates, replacing the
  fragile "connected component equals one note" assumption. The default
  objective is weighted (`score_then_count`) because pressure sweeps favoured
  precision over blindly maximising candidate count.

## Run

```bash
moneyrepair tearfit-demo \
  --notes-list 20,50,100 \
  --pieces-per-note 8 \
  --min-overlap-pixels 14 \
  --seed-strategy anchor_priority \
  --cover-objective score_then_count \
  --serial-ocr-rate 0.6 \
  --output runs/tearfit_demo.json
```

For a quick smoke run:

```bash
moneyrepair tearfit-demo \
  --notes-list 3,8 \
  --pieces-per-note 5 \
  --width 90 \
  --height 48 \
  --min-overlap-pixels 6
```

Use `--serial-ocr-rate 1.0 --ensure-serial-anchor` for the ideal upper bound
where every note has a readable serial anchor. Use `--serial-ocr-rate 0` to
probe geometry-only assembly. The old `anchor_only` behaviour can be reproduced
with `--seed-strategy anchor_only`, but it makes OCR coverage a hard yield
ceiling and is kept mainly as a comparison baseline.

Compare seed and exact-cover objectives directly:

```bash
moneyrepair tearfit-compare \
  --profile pressure \
  --seed-strategies anchor_priority,all \
  --cover-objectives count_then_score,score_then_count \
  --serial-ocr-rates 0,0.6,1 \
  --output runs/tearfit_compare_pressure.json
```

## How To Read The Report

- `false_edge_rate`: fraction of accepted tear edges that join different true
  notes. This should be low; otherwise single-link chaining will return.
- `exact_precision`: fraction of automatically confirmed candidates that are
  exactly the true note fragment set.
- `exact_yield`: fraction of true notes automatically and exactly confirmed.
- `manual_notes_remaining`: notes that still need human review after automatic
  confirmation.

The intended production stance is high-precision automatic confirmation plus a
human queue for the residual, not full automation.

## Known Limits

- Pose search is not implemented in this sandbox. It starts from placed masks,
  which are the output expected from the locator.
- The exact-cover pass chooses among generated full-note candidates. Candidate
  generation is still a bounded beam search, not a complete ILP over every
  possible fragment subset.
- Labels are treated as correct hard anchors. Real OCR needs confidence gating
  before it can be used with the same authority.
- The simulator is more realistic than shared Voronoi partitions, but it is
  still synthetic and should be used to falsify ideas, not to claim field
  performance.
