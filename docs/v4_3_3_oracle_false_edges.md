# v4.3.3: Oracle False-Edge Falsification

> Simulation evidence only. The fragments are already placed in one banknote
> coordinate frame, and the intervention reads simulator `note_id` truth. It
> is not a reconstruction feature and cannot be used on real fragments.

## Question

v4.3.2 localized the first N=100, p=24, seed-7 quality loss before exact cover:
`oracle_candidate_recall == exact_yield == 0.840`. True-edge recall stayed near
0.37 while false-edge rate increased with N, but that correlation did not show
whether false accepted edges caused exact candidates to disappear.

This experiment asks one counterfactual question:

> If every accepted cross-note edge is removed after scoring, with every true
> accepted edge and every downstream setting preserved, does oracle candidate
> recall improve by at least 0.05?

The preregistered decision rule is:

```text
delta oracle >= +0.05  -> false-edge contamination is the current limiter
delta oracle <  +0.05  -> stop that quality route; prioritize gap proposal /
                          candidate construction
```

## Intervention

Both arms use N=100, p=24, seed=7, `v43_routed`, the stabilized N=20 workload
normalization rates, and no wall-clock limits. The intervention changes only
the downstream pair evidence:

- keep every accepted same-note edge;
- delete every `automatic` or `review` cross-note edge;
- retain originally insufficient evidence, so simulator truth does not replace
  the real gap scorer;
- rerun core search, complete/partial gap recovery, and exact cover with the
  same thresholds and state/node budgets.

Run it with:

```bash
moneyrepair tearfit-v433-oracle-edges \
  --notes 100 \
  --pieces-per-note 24 \
  --seed 7 \
  --output runs/v4_3_3/oracle_false_edge_n100_seed7.json
```

## Result

| metric | control | false accepted edges deleted | delta |
| --- | ---: | ---: | ---: |
| oracle candidate recall | 0.840 | 0.860 | +0.020 |
| exact yield | 0.840 | 0.860 | +0.020 |
| exact precision | 0.966 | 0.989 | +0.023 |
| false core edges | 466 | 0 | -466 |
| total accepted cross-note edges removed | 0 | 2,064 | +2,064 |
| candidates | 30,828 | 24,243 | -6,585 |
| gap candidates (true / false) | 230 (7 / 223) | 189 (9 / 180) | -41 |
| core search seconds | 173.0 | 68.0 | -105.1 |
| complete + partial gap seconds | 275.2 | 275.6 | +0.5 |
| total seconds | 510.4 | 396.8 | -113.6 |

Core, complete-gap, and partial-gap search remain unsaturated in both arms.
Exact cover reaches its node limit in both arms, but still selects every exact
candidate available: `exact_yield == oracle_candidate_recall` in each arm.

## Conclusion

The intervention fails the `+0.05` rescue gate. Accepted false edges are a real
compute burden: deleting them removes 21.4% of candidates and 22.3% of total
runtime, mostly from core search. They are not the dominant cause of the
remaining exact-candidate deficit. Combined gap time is essentially unchanged,
and only two additional ground-truth notes enter the candidate pool.

For this measured N=100 seed-7 simulation, the causal diagnostic tree therefore
stops at:

```text
candidate-evidence wall
        |
        +-- false accepted-edge contamination: quality route falsified
        |
        `-- gap proposal / candidate construction: current measured wall
```

The next simulation algorithm iteration should improve deterministic
whole-assembly proposal and candidate construction. It should not treat lower
false-edge count alone as a quality objective. Faster pair/core filtering may
still be useful later as a performance optimization.

## Limits

- This is one deterministic seed, not an N=100 cross-seed claim.
- It does not test N=200, raw-crop localization, OCR errors, camera distortion,
  continuous-angle uncertainty, or real torn paper.
- The intervention uses ground truth only to falsify a mechanism. It is not
  available in production and must never appear in the reconstruction path.

The compact measured record is
[`benchmarks/v4_3_3_oracle_false_edge_n100_seed7.json`](benchmarks/v4_3_3_oracle_false_edge_n100_seed7.json).
