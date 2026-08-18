# v4.4: Residual-Gap-First Empirical Validation (NULL result)

> Simulation evidence only. Every number below is measured on placed fragments
> under per-note fractal tears plus fraying, not on real torn notes. The
> simulator `note_id` is used only after production candidate generation to
> replay missing notes. `STATUS.md` remains the sole authority on capability
> limits; nothing here extrapolates to real fragments.

## Phase framing

- **Phase I — geometry-first mechanism discovery + bottleneck localization:
  COMPLETE.** v4.3/v4.3.1/v4.3.2/v4.3.3 established tear-boundary coincidence as
  the discriminator, shifted the p=24 wall through N=50 under normalized compute,
  and falsified false-edge removal as the yield route, narrowing the measured
  N=100 seed-7 wall to gap proposal / candidate construction.
- **v4.4 — residual-gap-first candidate construction: IMPLEMENTATION COMPLETE;
  EMPIRICAL VALIDATION = NULL at the measured wall.** The mechanism is built,
  unit-tested, and runs, but under the preregistered protocol it did not move the
  bottleneck it was designed to move.

The precise claim this experiment supports:

> Under the preregistered N=100 seed-7 normalized-compute protocol,
> residual-gap-first construction did not raise oracle candidate recall
> (`0.840 -> 0.840`, delta `0.000`; the `+0.05` gate was not cleared); precision
> held (`0.9655`).

## Question

v4.3.3 narrowed the measured N=100, p=24, seed-7 wall to gap proposal /
candidate construction. v4.4 introduces residual-gap-first candidate
construction specifically to close that gap. This experiment asks one
preregistered question:

> Under identical same-seed, workload-normalized compute, does the
> residual-gap-first arm (`v44_gap_first`) raise oracle candidate recall by at
> least `+0.05` over the `v43_routed` control?

The preregistered decision rule (rescue gate):

```text
oracle_candidate_recall >= 0.890  (control 0.840 + 0.050)  -> gap-first rescues the wall
oracle_candidate_recall <  0.890                            -> gap-first does not rescue the wall
```

## Protocol

Both arms use N=100, p=24, seed=7, the same-seed `FractalTearConfig`, and the
stabilized v4.3.3 workload-normalized compute budgets. The two arms are
identical except for the candidate-construction strategy; this exactly
reproduces the v4.3.3 control.

Normalized compute budgets (identical for both arms):

| budget | value |
| --- | ---: |
| `candidate_states_per_pair_score` | 12.6919659855 |
| `gap_states_per_fragment` | 83.3333 |
| `partial_gap_states_per_fragment` | 20.8333 |
| `cover_nodes_per_note` | 25000 |

## Result — same-seed A/B (authoritative, normalized budgets)

| metric | v43_routed (control) | v44_gap_first | delta |
| --- | ---: | ---: | ---: |
| oracle_candidate_recall | 0.840 | 0.840 | 0.000 |
| exact_yield | 0.840 | 0.840 | 0.000 |
| exact_precision | 0.9655 | 0.9655 | 0.000 |
| candidates | 30828 | 30828 | 0 |
| gap_candidates | 230 | 230 | 0 |
| manual_notes_remaining | 16 | 16 | 0 |
| elapsed_seconds | 965.87 | 999.15 | +33.28 (+3.4%) |

### Gate arithmetic

```text
control oracle_candidate_recall = 0.840
rescue gate                     = 0.840 + 0.050 = 0.890
observed v44_gap_first          = 0.840
delta                           = 0.000
0.840 < 0.890  ->  GATE FAILS
```

Precision held at `0.9655` (no drop). The only measurable effect of the
gap-first arm was `+33.28 s` (`+3.4%`) of runtime with zero recall or precision
change.

## Why gap-first was inert here

`augment_candidates_gap_first` found **2716 residual gap regions**, and every
one routed to the `complex` class:

| routing class | regions |
| --- | ---: |
| `routing_simple` | 0 |
| `routing_moderate` | 0 |
| `routing_complex` | 2716 |

But the arm emitted no viable candidates from that branch:

```text
gap_regions          = 2716
gap_proposals_made   = 0
accepted_proposals   = 0
expanded_states      = 0
state_limit_reached  = false
time_limit_reached   = false
```

Neither a state nor a time limit was reached — the complex-routing branch simply
emitted no viable proposals under the normalized budgets. The candidate pool,
gap-candidate count, and selected solution were therefore byte-for-byte the
control's.

## Independent funnel localization — the binding constraint is UPSTREAM

The truth-restricted candidate funnel (valid; it reproduces the control at
`oracle recall = yield = 0.840`) partitions all 100 notes by where the exact
candidate is lost:

| category | count | meaning |
| --- | ---: | --- |
| `core_exact` | 77 | exact candidate reached via pure core base |
| `production_gap_exact` | 7 | exact candidate reached via production gap recovery |
| `no_pure_core_base` | 10 | no pure core base was ever constructed |
| `pure_core_base_not_selected` | 6 | a pure base exists but was not selected |

Truth-in-production-pool = `(77 + 7) / 100 = 84/100 = 0.840`, matching the A/B
oracle recall exactly. The dominant unresolved category is `no_pure_core_base`.

Crucially, the 16 missing notes are dominated by **upstream core-base
construction (10)** and **base selection (6)** — *not* the gap-proposal stage
that v4.4 targets. **Zero** misses are attributed to the weak-pair / gap-proposal
gate.

This is a two-part NULL:

1. Gap-first is inert on this seed (0 proposals; all regions `complex`).
2. Even a fully-firing gap-first stage could not rescue the 16 missing notes,
   because 10 have no pure core base to extend and 6 have a pure base that is
   never selected. The gap stage operates *after* a pure core base exists; it
   cannot manufacture the base that is missing upstream.

## Conclusion — the measured wall has moved

The preregistered rescue gate fails: residual-gap-first construction does not
raise oracle candidate recall on the measured N=100 seed-7 point. The funnel
relocates the binding constraint from **"gap proposal / candidate construction"**
(the v4.3.3 endpoint) to **"pure core-base construction & base selection"**.

```text
candidate-evidence wall
        |
        +-- false accepted-edge contamination    : falsified in v4.3.3
        |
        +-- gap proposal / candidate construction : NOT the limiter here
        |                                           (gap-first NULL; 0 gap misses)
        |
        `-- pure core-base construction & selection : current measured wall
              +-- no_pure_core_base            (10 / 100)
              `-- pure_core_base_not_selected  ( 6 / 100)
```

There is a **secondary implementation follow-up**: the `complex`-routing branch
of `augment_candidates_gap_first` emits no proposals under normalized budgets
(2716 regions -> 0 proposals). This is worth fixing so the mechanism can fire,
but it is **not the primary limiter** — the funnel shows the primary limiter is
upstream of the gap stage entirely.

## Next direction

The next simulation iteration should target **pure core-base construction and
base selection**, not gap proposal:

- recover a pure core base for the 10 notes that currently have none;
- promote the pure base into the selected solution for the 6 notes that have one
  but do not select it (a selection / ranking problem, `best_base_rank` is deep
  for several of these);
- separately, make the `complex`-routing gap branch actually emit proposals, so
  gap-first can be re-measured once it is non-inert.

## Limits

- This is one deterministic seed, not an N=100 cross-seed claim.
- It does not test N=200, raw-crop localization, OCR errors, camera distortion,
  continuous-angle uncertainty, or real torn paper.
- The funnel uses simulator ground truth only to localize where the exact
  candidate is lost; it is not a reconstruction feature and must never appear in
  the production path.
- All numbers are placed / fractal-fray simulation. `STATUS.md` is the sole
  authority on capability limits.

## Measured records

- A/B (authoritative, normalized budgets):
  [`benchmarks/v4_4_gap_first_n100_seed7.json`](benchmarks/v4_4_gap_first_n100_seed7.json)
- Candidate funnel:
  [`benchmarks/v4_4_candidate_funnel_n100_seed7.json`](benchmarks/v4_4_candidate_funnel_n100_seed7.json)
- Full run artifacts (gitignored, reproducible): `runs/v4_4/v44_decisive_ab_n100_seed7.json`,
  `runs/v4_4/candidate_funnel_n100_seed7_full.json`
