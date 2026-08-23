# v4.4.1: Fixed-Budget Base-Selection Validation

> Simulation evidence only. The experiment uses placed fragments under
> per-note fractal tears plus fraying. It does not test raw crops, locator
> error, OCR error, camera distortion, or real torn paper. `STATUS.md` remains
> the capability authority.

## Why this experiment exists

The v4.4 residual-gap-first A/B returned a NULL at N=100, p=24, seed 7. Its
truth-restricted funnel split the 16 missing notes into two upstream categories:

| category | notes |
| --- | ---: |
| no pure core base was constructed | 10 |
| a pure core base existed but global top-K did not select it | 6 |

Increasing the complete-base cap from 512 to 1024 recovered only three notes
(`0.840 -> 0.870`) while increasing total runtime from about 605 s to 816 s.
That result suggested a ranking problem, but did not establish that a better
fixed-budget selector could solve it.

## Single-variable intervention

The control preserves the historical global ranking:

```text
global = first K candidates ordered by pieces, raw coverage, evidence, score
```

The intervention keeps the same `K=512` complete and `K=128` partial bases but
selects them in deterministic disjoint rounds:

```text
repeat until K bases are selected:
    scan the unchanged global ranking
    take candidates whose fragment sets are disjoint within this round
    start a fresh round
```

This is production-valid: it uses only fragment IDs and the same disjointness
constraint enforced by final exact cover. It has no similarity threshold and
does not read simulator `note_id`.

Everything else is fixed between the two arms: simulator seed, Etear scoring,
core candidate generation, whole-assembly scorer, base limits, normalized
state/node budgets, and exact-cover objective.

## Preregistered gate

The intervention passes only if all of the following hold:

```text
oracle candidate recall delta >= +0.05
exact yield delta             >= +0.05
exact precision drop          <=  0.02
```

## Result: N=100, p=24, seed 7

| metric | global control | disjoint rounds | delta |
| --- | ---: | ---: | ---: |
| oracle candidate recall | 0.840 | **0.900** | **+0.060** |
| exact yield | 0.840 | **0.900** | **+0.060** |
| exact precision | 0.9655 | **1.0000** | +0.0345 |
| manual notes remaining | 16 | 11 | -5 |
| candidates | 30,828 | 30,982 | +154 |
| gap candidates | 230 | 384 | +154 |
| true gap candidates | 7 | 13 | **+6** |
| false gap candidates | 223 | 371 | +148 |
| selected false gap candidates | 0 | 0 | 0 |
| total runtime | 564.63 s | 600.08 s | +35.45 s (+6.3%) |

The intervention used exactly 512 complete and 128 partial bases, equal to the
control. It required seven complete-base disjoint rounds and two partial-base
rounds. Core, complete-gap, and partial-gap searches did not hit their state or
time limits. Exact cover reached its node limit in both arms, but selected every
available exact candidate (`oracle recall == exact yield`) in both arms.

The broader base distribution increased complete-gap proposal evaluations by
143,228 (+15.3%) and partial-gap evaluations by 125,093 (+16.8%). This produced
154 additional candidates. Six were new exact true-note candidates and none of
the 148 additional false gap candidates entered the selected solution.

## Causal conclusion

The preregistered gate passes. A fixed-budget selector recovers exactly six
additional true gap candidates, matching the six notes previously classified
as `pure_core_base_not_selected`. Therefore, on this measured seed:

> Global top-K base ranking was a real yield limiter, and exact-cover-aware
> disjoint rounds remove that limiter without increasing K or reducing
> precision.

This does not solve the full N=100 wall. The remaining ten notes are exactly the
upstream `no_pure_core_base` category. A separate truth-connectivity audit shows
that their automatic true-edge graphs split into 2-4 components and that no
single component reaches the 0.78 core record threshold. Base re-ranking cannot
select a base that was never constructed.

The next simulation experiment, if simulation continues, must therefore test
**multi-component core construction** as one isolated mechanism. It must not
increase global K, retune Etear on seed 7, or revive residual-gap-first as the
primary route.

## Limits and stop condition

- This is one deterministic discovery seed. N=100 seeds 8/9 are unmeasured.
- The selector remains opt-in; `global` is the compatibility default until
  cross-seed validation is complete.
- Precision 1.0 here does not imply real-data precision.
- If a production-valid multi-component bridge fails the same `+0.05` recall
  and yield gate, deterministic simulation v4 should freeze and effort should
  move to the v5 real-data acquisition/registration bridge.

## Reproduce

```bash
moneyrepair tearfit-v44-base-selection \
  --notes 100 \
  --pieces-per-note 24 \
  --seed 7 \
  --output runs/v4_4/base_selection_n100_seed7.json
```

Measured record:
[`benchmarks/v4_4_1_base_selection_n100_seed7.json`](benchmarks/v4_4_1_base_selection_n100_seed7.json)
