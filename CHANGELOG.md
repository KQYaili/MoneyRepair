# Changelog

## 4.4 — empirical validation (NULL result; bottleneck relocated)

Preregistered same-seed A/B validation of the v4.4 residual-gap-first
construction, without changing reconstruction logic (docs + benchmark artifacts
only):

- runs the decisive N=100, p=24, seed-7 A/B under the stabilized v4.3.3
  workload-normalized compute budgets (`candidate_states_per_pair_score`
  12.6919659855, `gap_states_per_fragment` 83.3333, `partial_gap_states_per_fragment`
  20.8333, `cover_nodes_per_note` 25000), with identical same-seed
  `FractalTearConfig` arms that exactly reproduce the v4.3.3 control;
- measures the NULL: residual-gap-first construction does **not** raise oracle
  candidate recall (`0.840 -> 0.840`, delta `0.000`), so the preregistered
  `+0.05` rescue gate (`>= 0.890`) is not cleared; precision holds (`0.9655`) and
  the only effect is `+33.28 s` (`+3.4%`) runtime;
- records that gap-first was inert here: `augment_candidates_gap_first` found
  2,716 residual gap regions, all routed `complex` (`routing_simple`=0,
  `routing_moderate`=0), yet `gap_proposals_made`=0, `accepted_proposals`=0,
  `expanded_states`=0, with neither state nor time limit reached;
- adds the truth-restricted candidate funnel (reproduces the control at
  `oracle recall = yield = 0.840`) that relocates the binding constraint
  **upstream** of the gap stage — `category_counts` = {`core_exact`:77,
  `production_gap_exact`:7, `no_pure_core_base`:10, `pure_core_base_not_selected`:6},
  dominant unresolved `no_pure_core_base`, with zero misses attributed to the
  weak-pair / gap-proposal gate;
- concludes that v4.4 does not solve the candidate wall; the measured seed-7 wall
  moves from *gap proposal / candidate construction* to **pure core-base
  construction & base selection**, and notes the secondary implementation
  follow-up (the `complex`-routing branch emits no proposals under normalized
  budgets) as a non-primary limiter;
- keeps all claims simulation-only and single-seed (`STATUS.md` remains the sole
  capability authority; no extrapolation to N=200 replication or real fragments);
- adds `docs/v4_4_empirical_validation.md`,
  `docs/benchmarks/v4_4_gap_first_n100_seed7.json`, and
  `docs/benchmarks/v4_4_candidate_funnel_n100_seed7.json`.

## 4.3.3

Simulation-only causal localization of the v4.3.2 candidate-evidence wall:

- adds an oracle false-edge deletion counterfactual that preserves every true
  accepted edge, removes every accepted cross-note edge after pair scoring,
  and leaves thresholds, search budgets, gap proposal, and exact cover fixed;
- adds `tearfit-v433-oracle-edges`, paired control/intervention reporting, and
  a preregistered `+0.05` oracle-candidate-recall falsification gate;
- exactly reproduces the N=100, p=24, seed-7 normalized control at
  `oracle recall = yield = 0.840`;
- removes all 466 false core edges and 2,064 accepted cross-note edges, but
  raises oracle recall only to `0.860` (`+0.020`), below the gate;
- records that candidate count falls from 30,828 to 24,243 and total runtime
  from 510.4 s to 396.8 s, while combined gap runtime remains effectively
  unchanged at about 275 s;
- therefore falsifies false-edge reduction as the next yield-quality route,
  retains it as a secondary precision/performance lever, and narrows the
  measured seed-7 bottleneck to gap proposal / candidate construction without
  extending the claim to other seeds or real fragments.

## 4.3.2

Preregistered scale-fineness diagnostics without changing reconstruction logic:

- calibrates an N=20 p=24 `1x/2x/4x/8x` compute staircase and refuses to emit
  normalized claims until adjacent solution and candidate-provenance
  fingerprints stabilize;
- separates fixed-budget operational capacity from workload-normalized
  mechanism scaling, using pre-stage pair scores, fragments, and note count
  rather than post-treatment candidate counts;
- records true-edge recall, true/false generated and selected gap candidates,
  oracle candidate recall, deterministic fingerprints, stage saturation, and
  stage timing;
- adds a preregistered N=20/50/100/200 matrix, JSONL checkpoint/resume, quality
  gates, and mechanism-retention diagnostics through `tearfit-v432-scale`;
- measures normalized N=50 p=24 at `0.880/0.985` yield/precision over three
  seeds and records the first N=100 seed diagnostic at `0.840/0.966`;
- identifies the current measured wall as missing correct candidates before
  exact cover, while keeping N=100 replication, N=200, and all real-data claims
  explicitly open.

## 4.3.1

Mechanism validation without a new reconstruction algorithm:

- records true/false accepted-edge counts and selected core, complete-gap, and
  partial-gap candidate provenance for every trial;
- emits machine-readable `adaptive_edge`, `group_gap`, and `routing` deltas in
  the canonical same-seed ablation JSON;
- changes the ablation CLI defaults to the documented N=20, seeds 7/8/9
  protocol, preventing accidental N=10 results from being presented as the
  headline benchmark;
- confirms that Etear primarily cleans the edge graph and relieves fixed-budget
  search congestion, while group-gap improves both recovery and final candidate
  selection at p=24;
- adds a JSON-backed mechanism figure and a limitations-first v4.3.1 report,
  including exact-cover and candidate-state budget saturation.

## 4.3.0

Tolerance-aware fine-tear evidence and reproducible geometry pressure:

- replaces the single fixed-overlap research rule with optional `Etear`, which
  rewards one continuous bidirectional seam, opposing normals, curvature
  specificity, and explained perimeter while penalizing scattered/one-sided
  hits, accidental coincidence, physical overlap, and locator uncertainty;
- adds `automatic`, `review`, and `insufficient-evidence` decisions for edges
  and assemblies, with automatic-only yield/precision and an honest manual queue;
- adds two-stage high-confidence core generation plus whole-assembly
  multi-boundary gap scoring, including separately audited partial-core output;
- propagates translation uncertainty, score margin, and score-basin width from
  the locator into placed fragment metadata; continuous-angle uncertainty stays
  explicitly unavailable;
- adds deterministic candidate-state and exact-cover-node budgets, per-stage
  stop diagnostics, same-seed p=8/16/24 ablation, and a complexity route that
  keeps fixed overlap above a 5% median fragment-area fraction;
- records the N=20, seeds 7/8/9 result: routed p=24 automatic yield/precision
  `0.917/0.981` versus fixed overlap `0.533/0.846`, with false-edge rate
  `0.036` versus `0.094`, at roughly 2.3x runtime;
- adds a JSON-backed scientific figure and a limitations-first v4.3 report.

## 4.2.1

Scale-axis crash fix and an honest disentangling of the two yield-collapse axes:

- fixes a `RecursionError` in `select_exact_cover_candidates`: the set-packing
  search recursed once per candidate, so large pools (N=200-scale runs generate
  well over 1000 candidates) overflowed Python's recursion limit and crashed
  mid-search. It is now an explicit-stack branch-and-bound with identical
  include-first ordering and both pruning bounds, plus precomputed per-candidate
  frozensets/scores;
- adds a regression test that packs 2500 disjoint candidates without crashing;
- documents the measured result that the **scale** collapse (large N) was
  compute/bug-bound — N=200 p=8 geometry-only recovers to 1.000 exact yield and
  precision once the crash is fixed and given budget — while the **fineness**
  collapse (many small pieces, e.g. p=16) is the real signal wall.

## 4.2.0

Tear-geometry pressure and final v4.1 review fixes:

- adds `partition_model=per_note` to multi-note simulation so each physical note can have independent tear geometry instead of one shared Voronoi partition;
- adds `interlock.py` with raster tear-contact scoring and `compute_interlock_compatibility`, exposed through `build-matrix --discriminate interlock`;
- extends `pressure-chimeras` with `--partition-model per_note` and `--include-interlock` for head-to-head overlap/appearance/interlock pressure tables;
- hardens interlock matrix building with sparse bbox-contact pair enumeration instead of a dense all-pairs Python scan;
- records complete pressure run config plus interlock compatible/incompatible pair counts in JSON reports;
- adds interlock stats to `build-matrix --discriminate interlock`;
- adds `pressure-chimeras --include-disc-interlock` for combined appearance/serial plus placed-fragment interlock constraints;
- exposes `pressure-chimeras --cell` for overlap and interlock candidate enumeration;
- adds `touch_priority` controls to the solver plus `--no-touch-priority` on solver/pipeline/pressure CLI paths;
- adds `--no-touch-priority` to `diagnose-chimeras` for CLI consistency;
- documents raw-crop auto-location separately from pre-aligned appearance discrimination and adds smoke vs long pressure profiles;
- keeps the existing auto-locate overlap tolerance forwarding, raw-crop appearance guard, and ordered DFS suffix fixes.

## 4.1.0

Chimera pressure realism:

- adds `spatial` wear to `make_multi_note_fragments`, with local low-frequency wear, gamma drift, vignetting, and stains so the global RGB-gain fingerprint no longer perfectly inverts the simulator;
- adds uncapped grouping diagnosis (`diagnose_groups`) with `cluster_exact_recoverable_rate`, `mixed_note_count`, and split-note counts;
- adds `pressure-chimeras` for N-sweeps and appearance-spread sweeps over overlap-only vs discriminative matrices;
- extends `diagnose_solutions` with count/rate fields and `uniquely_exact_recovered_rate`;
- documents why top-20 chimera counts can hide large-N identity merges, and why serial/OCR must anchor the industrial algorithm while appearance acts as a tie-breaker.

## 4.0.0

Industrialized banknote reconstruction with automated pose estimation, JIT-acceleration, and flowchart diagrams:

- **Auto-Locator & Candidate Pose Search** (`locator.py`): automatically estimates Top-K candidate placement poses (X, Y, rotation, side, score) for each fragment over front/back templates. Supports optional `--reference-front` and `--reference-back` template images.
- **Multi-Scale Pyramid & JIT-Acceleration**: downsamples crops and templates to Level 1 (0.5x) for coarse search, followed by Level 0 (1x) fine-tuning. Optimizes inner loops using `@numba.njit` with zero-allocation flat array indexing, reducing processing time per fragment to ~67ms. Added `numba` to core dependencies in `pyproject.toml`.
- **Two-Tier Scalar Bounding Solver** (`solver.py`): resolves memory allocation churn in recursive hot-paths by replacing copy-heavy geometric checks with $O(1)$ scalar bounding (sum of candidate areas), running precise vectorized mask unions only when candidate count is small ($< 5$).
- **Compatible Pose Solving**: extends the depth-first search (DFS) engine to support virtual placed fragments and candidate-pose-level mutual exclusion (selecting one pose of fragment $i$ excludes all other poses of $i$). Filters background pixels in placed images using the fragment mask to prevent template contamination.
- **CLI run-pipeline integration**: adds `--auto-locate` command-line argument to run the full pipeline without given approximate placements.
- **Process Diagram Exports**: supports generating process flowchart diagrams in native Microsoft Visio (`.vsdx`), editable vector graphic (`.svg`), and structured JSON (`.json`) formats. Added `export-diagram` command.

## 3.0.0

Turn the self-confirming demo into an honest testbed, then fix the failure it
exposes:

- `make_multi_note_fragments` + `simulate-multi-note`: N identical-denomination
  notes sharing one region partition, each with its own appearance and serial,
  mixed into one pool with the true `note_id` recorded for diagnosis.
- `diagnostics.diagnose_solutions` + `diagnose-chimeras`: count chimeras
  (solutions mixing notes) and recovered notes. The overlap-only matrix produces
  ~90% chimeras on a 5-note pool — the failure mode single-note simulation could
  never show.
- `fingerprint.py`: per-fragment appearance gain relative to the template and
  appearance clustering — the discrimination signal the matrix was missing.
- `compute_compatibility_clustered` + `build-matrix --discriminate
  appearance|serial`: compatible only when same note and non-overlapping;
  eliminates chimeras and recovers every note pure.
- fragment `meta` now round-trips through `save_dataset`/`load_dataset`.
- honest docs: the v3.0 note states the gap, the fix, and the residual hard tail.

## 2.5.0

Polished scientific reporting:

- shared publication style (`style.py`): one palette and rcParams reused by
  every Python panel;
- multi-panel evidence figure (`figures.py`) over the QA, algorithm, matrix
  footprint, and coverage panels, each with a per-panel claim;
- a source-data CSV and a figure manifest with panel claims, export paths, and
  SHA-256 provenance next to every figure, plus report-level QA validation;
- editable Visio-style diagrams (`diagrams.py`): a JSON node/edge spec plus an
  editable-text SVG of the production pipeline loop;
- `report-figures` and `export-diagram` commands.

## 2.0.0

Industrial production tradeoffs over open-ended research:

- acquisition quality contract (`quality.py`): focus, glare, segmentation
  confidence, and color-drift metrics with accept/reject thresholds and an
  `assess-quality` command;
- grid-accelerated compatibility build (`compute_compatibility_fast`) that
  writes packed bits directly for the ~20k fragment target, plus streaming
  incompatible-pair export (`build-matrix --engine fast --pairs-out`);
- packed-matrix helpers: `compatible_pair_count` and `restrict_packed_to_ids`;
- audit log on every confirm/reject with operator and reason;
- one auditable `run-pipeline` command that gates frames, prunes, searches,
  renders candidates, and writes a run manifest with input hashes, parameters,
  timings, and the QA summary.

## 1.5.0

Simulation realism and measurable algorithm comparison:

- realistic synthetic fragment degradation;
- solver ordering strategies: `area`, `degree`, and `area_degree`;
- strategy benchmark command;
- Python SVG/PDF/TIFF strategy report command;
- v1.5/v2.0/v2.5 planning documents.

## 1.0.0

Initial simulation-first release:

- synthetic banknote and fragment generation;
- scan/photo segmentation into fragment crops, masks, and manifests;
- manifest ingestion with affine placement into note coordinates;
- RGB reference scoring against front/back note images;
- contour tags, direction histograms, and curve similarity matching;
- packed compatibility matrix storage and precomputed pair import;
- DFS coverage search with candidate visualization reports;
- batch confirmation loop for manual reconstruction;
- benchmark and matrix footprint commands.
- GitHub CI, release checklist, and publishing guide.
