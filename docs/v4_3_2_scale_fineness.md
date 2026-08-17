# v4.3.2: Scale-Fineness Protocol

> Simulation evidence only. Inputs are already placed in a common banknote
> coordinate frame. This protocol does not test raw-crop localization, OCR
> errors, camera distortion, continuous-angle uncertainty, or real paper.

## Question

v4.3.1 showed why adaptive tear evidence and whole-assembly gap recovery improve
the N=20, p=24 result under a fixed compute contract. v4.3.2 asks three separate
questions without changing the reconstruction algorithm:

1. What scale fits inside the same absolute CPU search budget?
2. Does reconstruction quality remain stable when compute is normalized by
   workload known before each search stage?
3. If quality fails, is the first dominant wall edge signal, candidate search,
   exact cover, or gap proposal?

## Anchor Calibration

The existing N=20 limits are known to saturate, so they cannot be assumed to be
an adequate normalized-compute baseline. The protocol first runs p=24,
seeds 7/8/9, and all four algorithms at `1x, 2x, 4x, 8x` state/node budgets.

The lower of two adjacent factors becomes the normalized anchor only when every
algorithm/seed pair has identical:

- selected assembly fingerprint;
- candidate-provenance fingerprint;
- exact yield and precision;
- selected objective score.

If no adjacent pair stabilizes, the normalized track is omitted and the report
states that N=20 remains computationally truncated. It does not extrapolate a
known-truncated budget.

## Two Compute Tracks

`fixed` keeps the same absolute candidate, gap, and exact-cover budgets at every
N. It measures operational capacity under one production compute contract.

`normalized` uses only workload available before the corresponding stage:

```text
core budget          = anchor core budget * pair_scores(N) / pair_scores(20)
gap budget           = anchor gap budget * fragments(N) / fragments(20)
partial-gap budget   = anchor partial budget * fragments(N) / fragments(20)
exact-cover budget   = anchor cover budget * notes(N) / 20
```

Candidate count and accepted-edge count are deliberately excluded because they
are outcomes of the tested algorithm. Scaling compute by either would reward an
algorithm for producing more low-quality candidates.

## Preregistered Matrix

- N=20/50/100: four algorithms and seeds 7/8/9.
- N=200: four algorithms at seed 7; `baseline` and `v43_routed` also use seeds
  8/9.
- p=24, no serial labels, no wall-clock cutoff.
- Completed seed/track cases are appended to a JSONL checkpoint.

```bash
moneyrepair tearfit-v432-scale \
  --notes-list 20,50,100,200 \
  --pieces-per-note 24 \
  --seeds 7,8,9 \
  --anchor-notes 20 \
  --anchor-budget-factors 1,2,4,8 \
  --candidate-state-limit 100000 \
  --gap-state-limit 20000 \
  --partial-gap-state-limit 5000 \
  --cover-node-limit 250000 \
  --checkpoint runs/v4_3_2/checkpoint.jsonl \
  --output runs/v4_3_2/report.json
```

## Diagnostic Metrics

Every row records:

- true available and accepted edges, true-edge recall, false-edge rate;
- true/false generated and selected gap candidates;
- oracle candidate recall before exact cover;
- candidate and selected-solution fingerprints;
- core, complete-gap, partial-gap, and exact-cover saturation;
- timing for simulation, pair scoring, each search stage, diagnostics, and total.

`oracle_candidate_recall` is the fraction of ground-truth notes for which an
exact assembly is already present before exact cover. It separates failure to
generate a correct candidate from failure to select an available candidate.

## Quality and Mechanism Gates

Against normalized N=20 `v43_routed`, quality is scale-stable only when:

```text
precision >= 0.97 and precision drop <= 0.02
yield     >= 0.80 and yield drop     <= 0.10
manual queue <= 20% of notes
```

The mechanism audit also reports whether Etear reduces false-edge rate by at
least 40%, and whether gap recovery adds at most 5% candidates while improving
yield by at least 0.05 or precision by at least 0.02.

## Measured Checkpoint

The N=20 anchor stabilized between 2x and 4x, so the lower 2x limits are the
normalized baseline. Results through N=50 are means over seeds 7/8/9. N=100 is
the preregistered first-seed diagnostic only; it is not a three-seed estimate.

| track | N | runs | routed yield | precision | oracle recall | true-edge recall | false-edge rate | total seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 20 | 3 | 0.917 | 0.981 | 0.917 | 0.377 | 0.036 | 92.2 |
| fixed | 50 | 3 | 0.300 | 0.764 | 0.333 | 0.373 | 0.071 | 249.0 |
| fixed | 100 | 1 | 0.130 | 0.812 | 0.140 | 0.369 | 0.135 | 459.0 |
| normalized | 20 | 3 | 0.917 | 0.981 | 0.917 | 0.377 | 0.036 | 102.5 |
| normalized | 50 | 3 | 0.880 | 0.985 | 0.880 | 0.373 | 0.071 | 217.8 |
| normalized | 100 | 1 | 0.840 | 0.966 | 0.840 | 0.369 | 0.135 | 528.8 |

N=50 passes every preregistered normalized quality and mechanism gate. The
fixed-budget collapse is therefore an operational capacity limit, not evidence
that v4.3 geometry has already failed at N=50.

The N=100 seed-7 diagnostic is the first failed quality point. Its routed row
has `yield == oracle_candidate_recall == 0.84`; core, complete-gap, and
partial-gap searches are all unsaturated. Exact cover reaches its node limit,
but selects every exact candidate already available and consumes only 3.2 of
528.8 seconds. The missing 16% of notes have no exact candidate before final
selection.

The edge signal changes in a more specific way: true-edge recall remains nearly
flat (`0.377 -> 0.373 -> 0.369`) while false-edge rate rises (`0.036 -> 0.071 ->
0.135`). At N=100, gap recovery generates 230 added candidates, only 7 of which
are exact; all 7 true gap candidates and no false gap candidates are selected.
The dominant runtime is gap search (307.4 s), followed by core search (163.3 s)
and pair scoring (53.2 s).

Under the preregistered rules this is a **candidate-evidence wall**: edge
discrimination and/or gap proposal fails to place enough exact assemblies in
the pool, while exact cover is excluded as the current quality limiter. The
N=100 conclusion remains a deterministic seed-7 diagnostic until seeds 8/9 are
completed. It does not establish behavior on real fragments.

The follow-up v4.3.3 oracle false-edge deletion test raises oracle recall only
from `0.840` to `0.860`, below its preregistered `+0.050` gate. The measured
seed-7 wall is therefore narrowed further to gap proposal / candidate
construction. See [the v4.3.3 report](v4_3_3_oracle_false_edges.md).

Machine-readable summaries are checked in as
[`v4_3_2_n50.json`](benchmarks/v4_3_2_n50.json) and
[`v4_3_2_n100_seed7.json`](benchmarks/v4_3_2_n100_seed7.json).

## Bottleneck Rules

The main sweep locates the suspicious stage; a later single-variable rescue
test is required before a causal wall claim:

- **Signal wall:** search budgets are not saturated, or a 4x stage rescue does
  not recover at least 0.05 yield/oracle recall, while edge true-recall falls or
  false-edge rate rises.
- **Candidate wall:** edge discrimination stays stable, candidate search
  saturates, and a candidate-stage-only 4x rescue raises yield or oracle recall
  by at least 0.05.
- **Exact-cover wall:** oracle candidate recall remains at least 0.95, exact
  cover saturates, and a cover-only 4x rescue raises yield by at least 0.05.
- **Gap-proposal wall:** correct gap candidates are absent despite adequate
  core/cover compute, and increasing only the gap budget does not recover them.

The report must retain the distinction between a measured operational capacity
limit and a mechanism failure. Neither is evidence for real banknotes.
