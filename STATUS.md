# MoneyRepair Status

This file is authoritative. It defines what MoneyRepair can claim, where the
measured wall is, which routes are closed, and what experiment is allowed next.

## Current Decision

**Freeze the deterministic simulation core and collect the preregistered
physical pilot.** The project has exhausted the current simulation and
literature-derived interventions. More seed-7 tuning, a learned descriptor,
RL, flow matching, or solver replacement is not justified before independent
physical masks and poses pass the acquisition gates.

The current release is a simulation-backed end-to-end research prototype with
physical-data measurement infrastructure. It is not a demonstrated industrial
banknote restoration system.

## Claim Boundary

| Question | Current answer |
|---|---|
| Does the deterministic geometry-first core work in its synthetic testbed? | Yes, within the measured regimes below. |
| Has the fine-fragment wall been removed? | No. v4.3 shifts it; v4.4.1 localizes the remaining seed-7 wall. |
| Can raw crops be audited without leaking evaluation truth? | Yes, the v5 reality bridge enforces this separation. |
| Has a qualifying physical pilot been collected? | No. |
| Are real mask, pose, seam, yield, and precision gates measured? | No. |
| Do v6-v10 learned modules beat the deterministic core? | No evidence; they are untrained/unverified scaffolds. |
| Is full automation at a 2,000-note fine-fragment scale supported? | No. |

The supported operating stance is **high-precision automatic confirmation of
the evidence-rich minority plus a human review queue**.

## Supported System

MoneyRepair keeps all fragments in a canonical note coordinate frame and uses
this evidence chain:

1. acquisition and segmentation quality;
2. truth-blind pose hypotheses with observable uncertainty;
3. placed tear-boundary coincidence, not whole-note appearance identity;
4. high-confidence core candidates plus bounded whole-assembly gap recovery;
5. globally consistent exact-cover selection;
6. serial/OCR anchors when legible, followed by automatic/review routing.

Serial numbers are hard anchors and deduplication constraints. They can prevent
cross-note chimeras, but they do not create missing geometric evidence.
Appearance may be used for localization or a final tie-break; it is not a
same-note identity key.

## Frozen Simulation Evidence

### v4.3 fine-fragment mechanism audit

The canonical experiment uses per-note fractal tears with fraying, `N=20`,
seeds `7/8/9`, fixed state/node budgets, and no serial labels.

| pieces per note | fixed overlap yield / precision | adaptive Etear yield / precision | Etear + gap yield / precision | routed v4.3 yield / precision |
|---:|---:|---:|---:|---:|
| 8 | 1.000 / 1.000 | 0.917 / 0.982 | 0.950 / 1.000 | 1.000 / 1.000 |
| 16 | 1.000 / 1.000 | 0.933 / 0.949 | 0.983 / 1.000 | 1.000 / 1.000 |
| 24 | 0.533 / 0.846 | 0.767 / 0.817 | **0.917 / 0.981** | **0.917 / 0.981** |

At `p=24`, Etear reduces accepted edges from `881.0` to `628.0` and false
accepted edges from `82.7` to `22.7`. Gap recovery adds only `57.7` candidates,
with `2.7` selected, and raises yield/precision from `0.767/0.817` to
`0.917/0.981`. The fixed-overlap false-edge rate is `0.094`; routed v4.3 is
`0.036`. Mean manual notes fall from `9.3` to `1.7` at about `2.3x` runtime.

Interpretation: adaptive evidence is mainly a precision/workload filter;
whole-assembly context supplies the recall gain. This is a simulation result,
not proof that the simulator reproduces real tear profiles.

### Scale and causal localization

- With workload-normalized compute, routed `p=24` reaches `0.880/0.985`
  yield/precision at `N=50` over seeds `7/8/9`. The historical fixed budget
  reaches only `0.300/0.764`.
- At `N=100`, `p=24`, seed 7, the control reaches oracle candidate recall and
  exact yield `0.840`, with precision `0.966`. Seeds 8/9 and `N=200` remain
  unmeasured under this protocol.
- Oracle deletion of every accepted cross-note edge raises recall only
  `0.840 -> 0.860`, below the preregistered `+0.050` gate. False edges remain a
  precision/runtime issue, not the dominant yield limiter.
- Residual-gap-first proposal returns a NULL result: `0.840 -> 0.840`. It finds
  2,716 complex regions and emits no viable proposal. The funnel localizes the
  16 misses to 10 `no_pure_core_base` plus 6
  `pure_core_base_not_selected` cases.

### v4.4.1 fixed-budget base selection

The single seed-7 intervention keeps the `512/128` complete/partial base
limits and all normalized budgets fixed. Only the base selector changes.

| selector | oracle candidate recall | exact yield | exact precision | manual notes | runtime |
|---|---:|---:|---:|---:|---:|
| global top-K | 0.840 | 0.840 | 0.9655 | 16 | 564.63 s |
| fragment-disjoint rounds | **0.900** | **0.900** | **1.0000** | **11** | 600.08 s |

The `+0.060` recall/yield gain clears the preregistered `+0.050` gate and
recovers all six ranking misses. The remaining ten misses have automatic true
edge graphs split into 2-4 components, with no pure component reaching the
frozen `0.78` core threshold.

The remaining measured simulation wall is therefore **multi-component pure
core-base construction**. This is a one-seed localization result, not a
cross-seed or physical-data conclusion.

### Literature-transfer audit

On the same `N=100`, `p=24`, seed-7 simulator case:

- all-scored same-source query hit rate is `1.0000`;
- first-positive MRR is `0.9931`, and Hit@1 is `0.9867`;
- automatic-only query hit rate is `0.9946`, Hit@1 is `0.9846`, and
  reciprocal-best-buddy precision is `1.0000`;
- only `90/100` notes form a true automatic component reaching the frozen core
  threshold;
- reciprocal-best-buddy predictions cover `741/2983` automatic true scored
  pairs under a filtered, conditional denominator.

This negative result matters: excellent local first-positive ranking does not
guarantee assembly-level component coverage. The metrics are same-source
proxies because the simulator does not annotate the exact pair of physical
boundary arcs. They are not physical seam-mate recall.

## Historical Stress Boundary

Older v4.2 stress tests establish that scale and fineness are different axes:

| stressor | geometry only | with ideal serial anchors |
|---|---:|---:|
| `N=100`, coarse pieces | yield 0.54, precision 0.96 | not measured |
| `N=200`, coarse pieces | yield 0.055, precision 0.55 | yield 0.26, precision 0.98 |
| 16 fine pieces per note | yield about 0.10, precision about 0.45 | yield 0.12, precision 1.00 |
| 24 fine pieces per note | yield about 0.02, precision about 0.20 | yield 0.00, precision 0.00 |

The old coarse-scale collapse was partly an implementation/budget problem and
was later repaired. Fineness in a large mixed-note pool remains a signal and
candidate-construction problem. Heavy fraying alone on coarse pieces is not the
same wall.

## v5 Reality Bridge

The v5 path measures the handoff without changing locator ranking, the fixed
coarse top-10, routing thresholds, Etear, candidate construction, or exact
cover. Production observations and evaluation annotations are stored in
separate files. Gold masks and transforms never select a production pose.

The default proxy tolerance contract derives a `2.008 px` combined translation
tolerance, `0.974 degree` angular tolerance, and `0.510 mm` minimum effective
feature from physical segmentation/registration tolerances. External boundary,
interior-area, component, and hole errors are measured separately.

On one annotated synthetic capture proxy (one note, eight fragments, seed 7):

| proxy | mask ready | top-k / top-1 | automatic | automatic pose precision | first failure |
|---|---:|---:|---:|---:|---|
| clean cardinal | 8/8 | 4/8 / 4/8 | 4 | 1.000 | fixed coarse-top-10 ranking |
| noise 5 + 8% interior dropout | 0/8 | not evaluated | 4 observable routes | 1.000 | segmentation gate |
| clean free angle | 8/8 | 0/8 / 0/8 | 0 | n/a | transform family |

These are synthetic diagnostics. The 64-fragment/72-scene scanner/phone pilot
ledger, coordinate freeze, hash checks, repeat aggregation, and zero-false-
automatic rule are implemented. No qualifying physical captures currently
exist, so every physical gate remains unmeasured.

## Physical Pilot Gates

Proceed in this order and repair only the first failed stage:

1. **Mask contract:** independent gold masks, calibrated pixel scale, boundary
   and topology tolerances.
2. **Pose handoff:** top-K recall, top-1 precision, uncertainty coverage, and
   zero false automatic observations.
3. **Failure localization:** candidate graph, component coverage, oracle
   candidate recall, exact yield/precision, and review queue.
4. **Conditional component A/B:** only if reliable physical handoff reproduces
   the multi-component core wall; freeze every other variable and require at
   least `+0.05` oracle recall or yield with no precision loss or false
   automatic assembly.
5. **Conditional learned seam descriptor:** only if the component A/B leaves a
   retrieval residual. Train on narrow physical boundary sequences/strips with
   same-position different-note and near-straight hard negatives. Final
   assembly outcomes, not classification accuracy, decide the gate.

The research-gate diagram is available as
[SVG](docs/research_gates.svg) and editable
[Draw.io](docs/research_gates.drawio).

## Closed Routes

Do not reintroduce these as production identity discriminators:

| route | measured failure |
|---|---|
| appearance or wear-gain clustering | spatially non-uniform wear destroys the assumed same-note gain identity |
| boundary-colour continuity | no stable threshold separates true and false seams under stains and wear |
| contact-count interlock | measures how much masks touch, not whether tear profiles are mates |
| whole-contour best-subsegment matching | straight paper edges dominate; angle-sorted contours break jagged paths; the naive matcher can invert true/false ordering |
| generated or inpainted tear geometry | hallucinated boundaries would create unsafe matching evidence |

Appearance and colour may support localization or tie-breaking after geometry;
they may not become a hidden note-identity key.

## Code Boundaries

- `src/moneyrepair/`: supported deterministic package.
- `src/moneyrepair/baselines/`: superseded comparison paths only.
- `src/moneyrepair/experimental/`: unverified v6-v10, LLM, and policy
  scaffolds. They require explicit opt-in and cannot feed production behavior.
- `docs/benchmarks/`: committed measurement sources.
- `docs/figures/`: deterministic plotting scripts and rendered outputs.
- `runs/`: local generated evidence; do not commit physical/private data.

## Validation Snapshot

The consolidation gate on 2026-09-24 reports:

- `159 passed` in the core pytest suite;
- `compileall` clean;
- Ruff clean across `src`, `tests`, and `docs/figures`;
- targeted mypy clean for the changed CLI and diagram modules;
- full-package mypy still advisory with 23 pre-existing errors in six unchanged
  modules (`compat`, `tearfit`, `locator`, `pressure`, `figures`, and
  `pipeline`).

## Next Action

The single highest-value step is operational, not algorithmic:

> Capture the frozen scanner/phone pilot, create independent evaluation masks
> and transforms, run `pilot-validate` and `reality-bridge`, and identify the
> first failed physical gate.

Until that evidence exists, the project is stage-complete and should remain
frozen.
