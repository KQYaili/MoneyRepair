# v4.1 Chimera Pressure Realism

> **Historical design note**. This document represents an earlier design round and is superseded by the authoritative [STATUS.md](../STATUS.md) in the repository root.

v3.0 fixed the small friendly chimera test, but that was not enough. The
important question is whether the discrimination signal survives the two real
stressors:

1. many notes of the same denomination;
2. small appearance spread, like a stack of notes stored together.

This release keeps the old red/green tests, then adds a pressure harness that
reports when the apparent fix collapses.

## Spatial wear

`make_multi_note_fragments` now has two wear models:

- `global_gain` keeps the v3.0 baseline. Each note is `template * RGB_gain`;
  the appearance fingerprint fits that same global gain, so this is an easy
  model.
- `spatial` adds local low-frequency wear, gamma drift, vignetting, and stain
  fields before the global gain. The global-gain fingerprint can no longer
  invert the simulator perfectly, so residual structure appears across
  fragments from the same physical note.

Example:

```bash
moneyrepair simulate-multi-note \
  --output runs/spatial_pool.npz \
  --notes 30 \
  --pieces-per-note 8 \
  --appearance-spread 0.04 \
  --wear-model spatial \
  --local-wear-strength 0.12 \
  --gamma-spread 0.04 \
  --stain-count 3 \
  --stain-strength 0.08
```

## Pressure sweeps

Sweep note count at the old friendly spread:

```bash
moneyrepair pressure-chimeras \
  --mode n-sweep \
  --notes-list 3,8,20,40,80,150 \
  --appearance-spread 0.18 \
  --seeds 7,8,9 \
  --output runs/pressure_n_global.json
```

Sweep realistic appearance spread:

```bash
moneyrepair pressure-chimeras \
  --mode spread-sweep \
  --notes 30 \
  --spread-list 0.18,0.10,0.06,0.04,0.02 \
  --seeds 7,8,9 \
  --wear-model spatial \
  --local-wear-strength 0.12 \
  --gamma-spread 0.04 \
  --stain-count 3 \
  --stain-strength 0.08 \
  --output runs/pressure_spread_spatial.json
```

The command prints a compact table and writes JSON with all seed-level rows.

Use a short smoke run before launching the full table:

```bash
moneyrepair pressure-chimeras \
  --mode n-sweep \
  --notes-list 3,8,20 \
  --seeds 7 \
  --time-limit 5 \
  --output runs/pressure_smoke.json
```

The full N/spread sweeps above are long runs.

## Tear geometry

`partition_model=shared` is the adversarial identity test: every note is cut by
the same partition, so shape carries no identity signal. Real hand tearing is
closer to `partition_model=per_note`, where each physical note has independent
tear geometry. v4.2 adds a first raster interlock check:

```bash
moneyrepair pressure-chimeras \
  --mode n-sweep \
  --notes-list 3,8,20 \
  --appearance-spread 0.04 \
  --wear-model spatial \
  --partition-model per_note \
  --include-interlock \
  --include-disc-interlock \
  --seeds 7,8,9 \
  --output runs/pressure_interlock.json
```

For a dataset already placed in note coordinates, the same signal is available
as a matrix builder:

```bash
moneyrepair build-matrix \
  --dataset runs/per_note_pool.npz \
  --discriminate interlock \
  --min-interlock-contact 8 \
  --min-interlock-ratio 0.03 \
  --output runs/interlock_matrix.npz
```

This is intentionally a minimal geometric baseline for **placed fragments**,
not a full raw-crop torn-edge matcher. It does not search arbitrary
translation/rotation to discover a join. It assumes candidate poses have already
been placed into one note coordinate frame, then checks whether adjacent masks
share enough local boundary contact. Non-adjacent pairs are not penalised; a
1-2 px pose gap receives no help and no penalty from interlock. Use it as
placed-fragment local validation, not as edge reconstruction. The next stronger
version should compare contour fragments as negative shapes, not only count
raster contact edges.

## Metrics that matter

Top-k chimera counts are still reported, but they are not the primary large-N
metric. `solve_covering_sets(..., max_solutions=20)` can stop after the first
20 pure-looking candidates, hiding merged identities that would produce
chimeras later.

The pressure harness therefore reports grouping metrics before DFS:

- `cluster_count`: should be close to the true note count. If it is lower,
  identities were merged. If it is higher, individual notes were split.
- `mixed_note_count`: number of true notes that appear inside mixed clusters.
- `cluster_exact_recoverable_rate`: fraction of true notes whose discrimination
  group exactly equals that note's complete fragment set. This is not capped by
  `max_solutions`.
- `disc_uniquely_exact_recovered_rate`: the same idea after DFS, still useful
  for debugging but capped by the configured top-k search.
- `interlock_compatible_pairs` and `interlock_incompatible_pairs`: matrix
  sparsity after the placed-fragment tear interlock pass, useful for spotting
  thresholds that are too weak or too aggressive.
- `disc_interlock_*`: the combined constraint path, built by applying interlock
  after the appearance/serial discrimination matrix.

## Algorithm implication

Pure appearance clustering is a useful tie-breaker, not the industrial answer
for 2000 near-identical notes. The production direction is:

1. serial/OCR anchors dominate the identity graph;
2. appearance propagates within anchored neighborhoods and breaks ties;
3. edge-curve continuity is used only for the hard residual after pose and
   identity pruning;
4. evaluation reports unique correct recovery rate, not only top-k chimera
   count.

This turns the simulator from a proof that the fix works in a friendly case
into a testbed that can falsify the fix under realistic pressure.
