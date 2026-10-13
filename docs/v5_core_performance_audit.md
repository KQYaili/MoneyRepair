# v5: Core Performance and Exact-Cover Audit

> Simulation evidence only. Every number comes from placed fragments of the
> per-note fractal tear simulator, run through the frozen v4.4.1 core. This
> audit measures computation and solver optimality; it changes no core
> decision and says nothing about physical tears. `STATUS.md` remains the
> capability authority.

## Questions

1. Where does the core spend its time as the synthetic pool grows?
2. Why does its memory grow?
3. How far is the bounded exact-cover selection from the optimum of its fixed
   candidate pool?

Budgets are the v4.4.1 preset's preregistered workload-normalised rates:
12.69 core states per scored pair, 83.3 complete-gap and 20.8 partial-gap
states per fragment, and 25,000 exact-cover nodes per note. Wall-clock limits
are off, so every search trajectory depends only on those budgets.

## Output-identical speedup

The core was made faster without changing any decision:

- **Pair scoring.** Tear-boundary evidence is cropped to its dilated bounding
  box. Each fragment's mask overlaps and bidirectional boundary hits with all
  of its candidate partners come from one compiled pass. Normal opposition
  uses a table of the few distinct raster normals, and curvature-entropy
  binning uses binary search over the same bin edges as `np.histogram`.
- **Gap search.** All proposals of one state are scored as a batch. An upper
  bound (the strongest seam in place of the mean) rejects proposals that cannot
  reach a threshold. When the members do not overlap one another, a proposal's
  overlap with the union is the sum of its precomputed pairwise overlaps. The
  complete and partial gap searches share per-fragment caches.
- **Core beam search.** Expanded states are remembered as sorted member tuples,
  children stay plain records until they enter the beam, mask overlap is an
  integer bitset tested first, and dilated coverage is counted once per state.
- **Exact cover.** Fragment ids and serial labels are integer bitsets, and the
  chosen positions form a shared linked chain.

Decisions cannot move for four reasons. Every reordered filter is free of side
effects. Integer counts run over windows that contain every set pixel.
Floating-point expressions keep the same operations in the same order, except
where a test proves bit equality, such as the scalar clip. Bounds only
pre-filter proposals and keep a `1e-9` margin. Finally, beam ordering is a
total order over distinct states.

`docs/experiments/core_equivalence.py` checks this against the previous tree
(`da354f0`), each side in its own process. It compares every scored pair with
all of its evidence fields, every accepted edge, core candidate and pool
candidate with its source, the selection and its objective, fingerprints,
search counters and trial metrics. It also compares 126 direct calls of the
public functions under parameters that reconstruction never uses: labels,
pose uncertainty, another tolerance, overlap limits, pair and candidate
constraints, seed restrictions and node limits. Floats are compared bit for
bit, including the sign of zero. Source:
[v5_core_equivalence_aarch64.json](benchmarks/v5_core_equivalence_aarch64.json).

| configuration | result | before (s) | after (s) | before RSS (MB) | after RSS (MB) |
|---|---|---:|---:|---:|---:|
| N=100, p=24, seed 7, global | identical | 931.8 | 54.7 | 1208 | 738 |
| N=100, p=24, seed 7, disjoint rounds | identical | 1032.3 | 54.9 | 1206 | 731 |
| N=50, p=24, seed 8, disjoint rounds | identical | 469.9 | 23.2 | 608 | 407 |
| N=20, p=24, seed 9, global | identical | 154.3 | 8.3 | 290 | 246 |
| N=12, p=24, seed 8, disjoint rounds | identical | 111.7 | 5.4 | 208 | 205 |
| N=6, p=24, seed 7, global | identical | 51.3 | 3.1 | 151 | 179 |
| N=8, p=16, seed 9, global (routed baseline scoring) | identical | 6.5 | 1.2 | 142 | 169 |
| N=8, p=16, seed 9, disjoint rounds, serial labels | identical | 6.6 | 1.2 | 143 | 169 |
| N=6, p=24, seed 3, global, serial labels | identical | 57.2 | 3.3 | 154 | 180 |
| N=8, p=16, seed 9, effectiveness + gap scoring | identical | 18.9 | 2.0 | 140 | 173 |
| N=6, p=24, seed 8, v4.4 gap-first | identical | 70.0 | 5.3 | 165 | 215 |
| N=6, p=24, seed 7, boundary-contact proposals | identical | 55.4 | 3.2 | 151 | 179 |
| direct-call suite (126 entries) | identical | 210.4 | 24.6 | 252 | 284 |

Times are single-threaded on one 4-core aarch64 machine (Neoverse-N1, numpy
2.5.4) while other single-threaded jobs shared it, so they vary by about 10%.
Small runs use more memory after the change because compiled kernels add a
fixed footprint. The pull-request workflow
[core-equivalence.yml](../.github/workflows/core-equivalence.yml) repeats the
two N=100 arms and the direct-call suite on x86 runners for every change that
touches the core.

## Where the time goes

At N=100 (p=24, seed 7) the stages changed as follows:

| stage | before (s) | after (s) | speedup |
|---|---:|---:|---:|
| pair scoring | 90.5 | 15.9 | 5.7x |
| core beam search | 299.2 | 24.3 | 12.3x |
| complete-gap search | 355.4 | 6.2 | 57x |
| partial-gap search | 279.2 | 3.5 | 81x |
| exact cover | 4.3 | 1.3 | 3.3x |
| total | 1032.3 | 54.9 | 18.8x |

(Disjoint-round arm; the global arm is 931.7 s to 54.7 s.) Core search and
pair scoring now account for about 45% and 29% of the run.

### Pair scoring is quadratic by definition

Source: [v5_core_profile.json](benchmarks/v5_core_profile.json).

| N | fragments | all pairs | boxes intersect | share | scored | boundaries touch | review or automatic edges |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 480 | 114,960 | 36,192 | 31.5% | 15,758 | 4,329 | 894 |
| 50 | 1,200 | 719,400 | 226,996 | 31.6% | 95,344 | 23,327 | 2,481 |
| 100 | 2,400 | 2,878,800 | 906,887 | 31.5% | 374,773 | 87,411 | 5,936 |
| 200 | 4,800 | 11,517,600 | 3,633,838 | 31.6% | 1,494,373 | 339,067 | 16,113 |

All notes share one canonical frame, so a fixed 31.5% of all pairs pass the
bounding-box test at every N. The scored-pair list is part of the core's
output, and the core-state budget is 12.69 states per scored pair. A spatial
index therefore cannot shrink the scored set without changing results. It
could only speed up a test that is already a vectorised box check. The
remaining cost is per-pair work on the 23% of scored pairs whose boundaries
touch.

### Gap proposals are rejected for overlap

| N | proposals | overlap penalty | weak evidence | no reachable level | accepted, then pruned for overlap | accepted and kept |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 399,779 | 89.6% | 9.4% | 0.00% | 0.55% | 0.45% |
| 50 | 922,824 | 89.8% | 9.6% | 0.01% | 0.40% | 0.16% |
| 100 | 1,947,104 | 89.8% | 9.8% | 0.01% | 0.39% | 0.08% |
| 200 | 3,830,054 | 90.1% | 9.4% | 0.01% | 0.39% | 0.04% |

"Overlap penalty" means the same placement without overlapping the assembly
would have reached a threshold. About 90% of proposals fail for that reason, so
computing each proposal's exact overlap early is the prune that matters; the
batched member-sum overlap does it in one pass per state.

### Core search is spread across per-neighbour Python work

Expanded core states grow from 99,319 (N=20) to 555,117 (N=100) and 1,420,951
(N=200). No single line dominates the profile after the changes above: child
construction, seen-state lookup, support accumulation, coverage and candidate
records each take a few percent. Compiling the whole expansion loop is the
remaining large step; it is deferred until a scale target needs it.

## Scaling

The grid uses the v4.4.1 preset (disjoint-round arm) with seeds 7-9, run one
at a time on an otherwise idle core. Each row shows the mean over three seeds,
with the range in brackets when the seeds differ. Source:
[v5_core_scaling.jsonl](benchmarks/v5_core_scaling.jsonl).

| p | N | wall s | peak RSS MB | scored pairs | core states | pool candidates | oracle recall | exact yield | exact precision |
|---:|---:|---|---:|---:|---:|---:|---|---|---|
| 24 | 20 | 8.7 (8.5-8.8) | 245 | 15,606 | 98,246 | 5,713 | 0.967 (0.900-1.000) | 0.967 (0.900-1.000) | 1.000 |
| 24 | 50 | 22.5 (22.5-22.6) | 406 | 94,997 | 258,256 | 14,979 | 0.940 (0.920-0.960) | 0.940 (0.920-0.960) | 1.000 |
| 24 | 100 | 55.6 (54.4-56.5) | 734 | 375,327 | 572,564 | 32,961 | 0.913 (0.900-0.920) | 0.913 (0.900-0.920) | 1.000 |
| 24 | 200 | 162.5 (161.3-164.7) | 1674 | 1,492,957 | 1,429,717 | 82,977 | 0.890 (0.850-0.920) | 0.873 (0.820-0.915) | 0.977 (0.953-0.989) |
| 16 | 20 | 2.9 (2.8-2.9) | 193 | 8,114 | 55,269 | 1,108 | 1.000 | 1.000 | 1.000 |
| 16 | 50 | 8.3 (8.3-8.4) | 271 | 49,032 | 165,316 | 3,447 | 1.000 | 1.000 | 1.000 |
| 16 | 100 | 20.6 (20.5-20.6) | 418 | 194,185 | 376,065 | 9,232 | 0.993 (0.980-1.000) | 0.993 (0.980-1.000) | 1.000 |
| 16 | 200 | 54.7 (54.6-54.8) | 806 | 768,894 | 836,831 | 22,012 | 0.780 (0.750-0.810) | 0.780 (0.750-0.810) | 0.925 (0.909-0.934) |

Every one of the 24 runs exhausted its exact-cover node budget, and no core,
gap or time limit was reached. From N=20 to 200 at p=24, scored pairs grow as
N^1.98, expanded core states as N^1.16, wall time as N^1.27 (N^1.55 between
100 and 200) and peak memory as N^0.83 (N^1.19 between 100 and 200). The
quadratic pair set takes over as N grows: pair scoring is about 28% of the
N=100 runs and 35% of the N=200 runs.

Peak memory is reached in core search. The structures a seed-7 reconstruction
keeps measure as follows ([profile](benchmarks/v5_core_profile.json), objects
shared between structures counted once):

| N | fragment masks and images | scored-pair records | core candidates | expanded core states |
|---:|---:|---:|---:|---:|
| 100 | 150 MB | 129 MB (374,773) | 70 MB (134,672) | 555,117 |
| 200 | 300 MB | 512 MB (1,494,373) | 184 MB (356,956) | 1,420,951 |

Fragment data and candidates grow about linearly with N. The scored-pair
records, about 360 bytes each, grow quadratically and are already the largest
single structure at N=200. Core search adds its seen set, one sorted member
tuple of 48-250 bytes per expanded state, while it runs. No N=2000 estimate is
made from these curves.

## Exact-cover optimality

`docs/experiments/cover_optimality_audit.py` re-solves each final candidate pool
as a MILP with HiGHS. The model is the same weighted set packing: fragment ids
and serial labels are used at most once, and the objective is total score, then
candidate count. The depth-first search is run at 1x, 10x and 100x its node
budget. Source:
[v5_cover_optimality_audit.json](benchmarks/v5_cover_optimality_audit.json).

| N | seed | arm | pool | DFS nodes | DFS below optimum | selected DFS / MILP | exact yield DFS / MILP | exact precision DFS / MILP | 100x nodes helps |
|---:|---:|---|---:|---:|---:|---|---|---|---|
| 50 | 7 | disjoint | 14,971 | 1,250,000 | 0 | 46 / 46 | 0.920 / 0.920 | 1.000 / 1.000 | no |
| 50 | 8 | disjoint | 15,255 | 1,250,000 | 0 | 48 / 48 | 0.960 / 0.960 | 1.000 / 1.000 | no |
| 50 | 9 | disjoint | 14,712 | 1,250,000 | 0 | 47 / 47 | 0.940 / 0.940 | 1.000 / 1.000 | no |
| 100 | 7 | global | 30,828 | 2,500,000 | 0 | 87 / 87 | 0.840 / 0.840 | 0.966 / 0.966 | no |
| 100 | 7 | disjoint | 30,982 | 2,500,000 | 0 | 90 / 90 | 0.900 / 0.900 | 1.000 / 1.000 | no |
| 200 | 7 | disjoint | 82,478 | 5,000,000 | 1.94% | 179 / 183 | 0.885 / 0.885 | 0.989 / 0.967 | no |
| 200 | 8 | disjoint | 84,992 | 5,000,000 | 0.60% | 185 / 187 | 0.915 / 0.860 | 0.989 / 0.920 | no |
| 200 | 9 | disjoint | 81,460 | 5,000,000 | 4.66% | 172 / 183 | 0.820 / 0.730 | 0.953 / 0.798 | no |

Every node budget is exhausted, so each DFS result is an incumbent until the
MILP certifies it. Through N=100 the incumbent is the proven optimum of its
pool. At N=200 it is 0.6-4.7% below the optimum, and 100 times the nodes does
not change it. The score-optimal packing is not better on truth: it selects
more candidates, never raises exact yield, and lowers exact precision in all
three seeds. At this scale the candidate score and its objective, not the
bounded search, limit selection quality. HiGHS proves each optimum at the root
node in 3-35 s; the count tie-break takes longer.

## Reproduce

```bash
pip install -e ".[dev,audit]"
python docs/experiments/core_equivalence.py --reference-ref da354f0
python docs/experiments/core_equivalence.py --reference-ref da354f0 --direct-suite
python docs/experiments/core_profile.py --notes 100 --pieces 24 --seed 7
python docs/experiments/core_scaling.py --notes 20,50,100,200 --pieces 16,24 --seeds 7,8,9
python docs/experiments/cover_optimality_audit.py --notes 200 --pieces 24 --seed 9
```

## Limits

- Timings come from one aarch64 machine; x86 equivalence is checked in CI,
  x86 timing is not measured.
- At p=16 the routed preset uses overlap scoring and skips gap search, so the
  p=16 rows measure a different code path from p=24.
- Three seeds per N do not make a replicated headline result, and the N=200
  optimality gap is a seed-level observation.
- This audit opens no algorithm route. Changing scores, budgets or the
  objective remains an algorithm experiment under `STATUS.md`.
