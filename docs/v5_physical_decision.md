# Physical Benchmark Decision Contract

Design date: 2026-10-06. This is a preregistration and a software-only resolution
audit, **not a physical result**. [STATUS](../STATUS.md) remains authoritative.
No learned model, dewarper, scoring threshold, or frozen pilot is changed.

## Four Decisions

| Question | Decision |
|---|---|
| Primary benchmark | 20 independently printed sheets x 16 hand-torn pieces = 320 physical pieces |
| Split | 4 sheets / 64 pieces for calibration; 16 sheets / 256 pieces held out |
| Material | 80 g/m2 white, uncoated cellulose copier paper from one ream |
| Coordinates | common duplex NOT CURRENCY artwork, 156 x 77 mm; same print job |
| Resolution | mm contract; 1200 DPI independent gold, 600 DPI raw seam evidence, 300 DPI locator/candidates |
| Expansion gate | automatic exact yield >= 0.80 on the blind scanner track, zero false automatic decisions |

This benchmark follows, rather than replaces, the existing
[8 x 8 acquisition pilot](v5_real_capture_pilot.md). The existing `pilot-init`
ledger remains fixed at 64 fragments; it does **not** create the new 320-piece
benchmark. Use a separate data directory and preregister its identity ledger.

The primary task is a **mixed pool of all 256 blind fragments**, not 16
parent-labelled subproblems. Parent labels, gold masks, gold transforms, capture
order, and unique printed IDs never enter production matching. Physical IDs are
linked through a separate handling/evaluation ledger. Disable serial anchors
for the primary geometry baseline. No currency artwork or real currency is
required. Copier-paper performance does not establish cotton-paper, banknote,
aged-paper, or polymer performance.

Use opaque production IDs and filenames. Randomize the entire mixed pool before
each reacquisition; do not group by parent in trays, directories, scan order, or
consecutive fragment IDs. Keep the randomization/parent mapping in a separate
evaluation sidecar that the production process cannot read. The observed-view
metadata allowlist is useful, but is not a substitute for this file/access
boundary. Audit the production input bundle for identity leakage before testing.

Keep brand, lot, print technology/settings, measured thickness, paper grain
direction, humidity/session, and print-master hashes in acquisition metadata.
Hold these fixed within the primary experiment. Future material transfer must
be a separate batch, not a pooled average. Hand-tear every sheet independently;
do not cut a stack with shared seams. Do not discard hard small pieces.

## Continue and Stop

**Safety gates precede throughput gates.** These are engineering investment
rules, not a claim of statistical certification or a learned probability.

1. Mask: >= 90% of blind physical fragments pass in >= 2 of 3 independent
   replacements; each parent >= 75%; interior missing/extraneous fractions
   <= 1% each and within calibration limits. Boundary/topology gates and
   calibration stay as specified in the original pilot.
2. Pose: on mask-ready fragments, K=3 recall >= 90% and top-1 >= 85%; K=10 is
   diagnostic, not permission to change the primary budget after testing.
   Report the all-fragment denominator alongside the conditional denominator.
3. Safety: any false automatic observation or assembly stops automatic release.
   Repeats/majority voting cannot erase a false automatic observation.
4. Reconstruction: automatic **exact** yield >= 80% on all 16 blind parents,
   at least 13 correct parents, with zero false automatic assemblies. Review
   candidates do not count as automatic successes; empty automatic output fails.
5. Only passing the primary scanner track permits a new N=50, p=16 study.
   Phone-cardinal and phone-free are separately reported transfer diagnostics;
   scanner success does not certify them. Free-angle cases without calibrated
   angle uncertainty remain review-only.

The primary 13/16 numerator uses **repeat 1**, the first valid scheduled scanner
mixed-pool acquisition, not best-of-3, majority voting, or any-success. Repeats
2/3 measure stability only; report each separately and require zero false-auto
on all repeats. Phone tracks use the same aggregation rule. Do not join partial
assemblies across repeats.

An acquisition may be invalidated only for an unreadable file, hardware capture
abort, or physically absent required scale target, recorded **before** examining
algorithm outputs. Allow at most one replacement per scheduled capture; retain
the original and reason in the ledger. A second technical failure leaves the
track incomplete and blocks expansion. Blur, difficult fragments, poor masks,
pose/solver failures, and low scores are task failures, never replacement reasons.

Until the native 600-DPI seam verifier is qualified, the pilot measures mask,
pose, candidate/component recall, and **shadow** exact yield. The final automatic
13/16 gate is not yet evaluable. Any search node/state/time limit hit makes the
result review-only; an accurate incumbent cannot bypass this rule without a
future, independently verified certificate covering unexplored states.

Repair only the first failed stage. A mask failure does not authorize solver
tuning; a pose miss does not authorize learned seam training. A core failure
after reliable handoff permits a separately preregistered component/proposal
experiment. A failure of a route means stop that route, not declare the entire
task impossible.

### A/B Rule

Freeze the same input, calibrated mm thresholds, candidate/state/node limits,
threads, and machine. Change one mechanism. On the 16-parent blind cohort:

- +1 rescued parent (6.25 percentage points): promising, confirmation only;
- >= +2 rescued parents (12.5 percentage points): route passes the screening
  gate, provided no parent regresses, no false automatic result occurs,
  automatic precision does not drop, and runtime is <= 1.25x;
- no rescue, any regression/false automatic, or excess cost: stop that route.

Choose oracle candidate recall or automatic exact yield before testing. Neither
gain proves population improvement on this small cohort. An independently
acquired replication is required before adoption. The historical +0.05
absolute-gain rule remains a design threshold for later, larger cohorts, not a
claim of statistical significance. No tuning/retesting on the same blind
parents is validation.

Allow at most one preregistered single-variable A/B on this blind cohort. Once
revealed for diagnosis or selection, the cohort becomes development evidence.
Trying further warps/descriptors/thresholds on it requires a new blind cohort;
do not select the best of multiple unregistered attempts.

Predeclare the metric appropriate to the failed stage: dewarping needs mask/
pose improvement, retrieval needs oracle candidate recall, and reconstruction
needs automatic exact yield. A proxy-score gain alone cannot pass the gate.

## Core Resolution Rule

Separate print/gold, acquisition, locator, proposal, and verification rasters.
The legacy `180 x 90` canvas is **simulation only**. The current `reconstruct`
CLI emits `release_mode=diagnostic_only`, an empty `routing.automatic` list, and
separate `shadow_automatic` classifications. Its frozen pixel thresholds have
no validated physical mapping. All selected assemblies remain in review, with
qualification and any search-limit blockers recorded. Internal core evidence
labels remain unchanged for frozen simulation comparisons, not release actions.

For the new physical study:

- Preserve native 600-DPI observations and 1200-DPI independently annotated gold.
  Require actual optical resolution; disable/record scanner sharpening, deskew,
  and enhancement. If 1200 DPI is interpolated, preregister native 600-DPI gold
  instead and retain that limitation.
- Normalize locator and candidate generation to 300 DPI, approximately 1843 x 909
  pixels for 156 x 77 mm, using recorded x/y scales and a shared transform.
- Use an independent external scale target to measure native pixels/mm_x and
  pixels/mm_y on every track/session, with uncertainty and transform recorded.
  Nominal scanner DPI and printer dimensions are planning values, not calibrated
  metrology. Convert physical thresholds with measured scales, including local
  scale after phone rectification. Do not infer scale from a matched tear.
- Phone input must have calibrated native local scale at least as fine as the
  target; enlarging a photograph is not new geometric evidence.
- Final seam verification targets native 600-DPI production boundary strips,
  approximately 3685 x 1819 for a full sheet, but not dense whole-sheet all-pair
  processing. This verifier is still pending implementation/qualification.
  Gold data are evaluation-only; never use independent gold masks in a decision.
- Derive tolerance from calibration in mm. Under the current **proxy** model,
  T = 0.085 + 0.085 = 0.170 mm and D_min = 3T = 0.510 mm (r_e=1, H=0).
- Require D_min * min(pixels/mm_x, pixels/mm_y) >= 6 at the verification grid.
  This is a conservative
  sampling design policy, **not a theorem of seam discriminability**. If it fails
  at native 600 DPI, preregister a finer native track before blind evaluation,
  or abstain. Coarse 300-DPI proposals do not certify a seam.
- Features below D_min stay insufficient-evidence even if a 600-DPI image looks
  sharp. Larger images do not reduce registration/annotation uncertainty.
- Do not copy legacy 2-pixel tolerance or 14-hit support into another raster.
  Geometry lengths use physical units; overlap area and boundary-sample counts
  have different scaling. Normal estimation, curvature proxy, constant terms,
  and candidate ranking also need a resolution qualification.
- Downsampled masks/keypoints may propose candidates, but cannot certify native
  fine tears. Never upsample/synthesize geometry to restore information.

A physically calibrated, resolution-qualified core is **not implemented by
this contract**. Until that qualification and physical gates pass, core outputs
on physical input are diagnostic/review only. The frozen simulation core remains
unchanged.

## Software Audit

Run from the WSL moneyrepair environment:

```bash
python docs/experiments/resolution_audit.py \
  --replay --output runs/decision_audit/resolution.json
```

The script records contract SHA256, exact grid units, dense input storage,
analytic phase-sensitive notch sampling, and 15 same-geometry core runs:
N=3, p=16, seeds 101/102/103, scales 1/2/4. Budgets are fixed at 40,000
candidate states, 3,000 gap states, 1,000 partial-gap states, and 75,000 cover
nodes. The two diagnostic arms keep legacy pixels fixed or linearly scale
length/support gates. The latter is **not** a calibrated production preset.
Nearest-neighbour replication adds no new tear information and is not physical
simulation. Runtime is machine-dependent; no linear extrapolation to N=2000 is
justified. JSON under `runs/` stays local.

### Measured software result

All 15 replay runs returned exact yield/precision 1.000/1.000 and the same
selected-fragment-set fingerprint. This small, position-known, synthetic case
is deliberately not the historical fine-fragment wall or a physical test.
Thirteen runs reached the 75,000-node cover limit; only the 4x fixed-pixel runs
for seeds 102 and 103 completed without that limit. The returned incumbents
match ground truth, but the truncated runs do **not** certify global optimality.
The common fingerprint identifies the selected fragment IDs, not independence
of acquisition or equivalence of the accepted edge graphs.

| Scale / arm | automatic core edges (seeds 101/102/103) | mean time, first / replay |
|---|---|---:|
| 1x / fixed pixels | 76 / 76 / 76 | 1.14 / 1.02 s |
| 2x / fixed pixels | 86 / 91 / 90 | 3.74 / 3.42 s |
| 2x / linear units | 85 / 86 / 88 | 4.00 / 3.69 s |
| 4x / fixed pixels | 93 / 95 / 93 | 15.89 / 13.73 s |
| 4x / linear units | 90 / 91 / 89 | 18.96 / 16.24 s |

The repeat (`runs/decision_audit/resolution_reviewed.json`) reproduces all edge
counts, selected sets, and node-limit flags. The final-contract audit
(`runs/decision_audit/resolution_final.json`) also preserves those metrics and
stores the complete contract snapshot, contract/script hashes, Python, and NumPy
versions. These are execution repeats, not new seeds or independent physical
samples. The timing variation is retained rather than hidden; the final replay
overlapped validation checks and is not used as an isolated timing comparison.

Thus neither fixed pixels nor naive length scaling preserves the edge graph.
There is no basis for declaring a resolution-invariant physical core. The replay
ends at 720 x 360, **not** at native 300/600 DPI; those grid resource and sampling
calculations below are analytic, not timed reconstruction measurements.

At the proxy D_min=0.510 mm, the legacy grid supplies 0.59 samples, 300 DPI 6.02,
and 600 DPI 12.05. In an optimistic analytic Gaussian-notch probe across 101
subpixel phases, the worst retained peak is 0.135 at legacy scale, 0.981 at
300 DPI, and 0.995 at 600 DPI. This does not model camera MTF, paper fibers, or
segmentation, and cannot establish seam precision.

Dense bool-mask + uint8-RGB inputs alone cost about 2.00 GiB at 300 DPI and
7.99 GiB at 600 DPI for 320 fragments. At 20,000 fragments the corresponding
numbers are 124.82 and 499.41 GiB, before caches/temporaries. Therefore the future
600-DPI verifier must use local strips/crops; dense full-canvas storage is not a
credible scale plan. These are storage formulas, not measured peak RSS.

## Statistical and Theoretical Boundaries

With n independent zero-error automatic confirmations, the one-sided 95% exact
binomial lower precision bound is 0.05^(1/n). Thus 13/13 gives about 0.794 and
16/16 about 0.829, not 0.98. Reaching a lower bound >= 0.98 requires at least
149 independent zero-error automatic confirmations under a fixed policy and
representative sampling. Correlated fragments and repeats do not supply n.
Even 149 successes within one narrowly controlled paper domain are not universal
industrial certification. See [NIST exact binomial bounds](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm).

The web discussion's mathematical ideas are conditional design tools:

- e < Delta/4 is sufficient only for corresponding physical seam arcs, a
  uniform geometric error bound, and a positive worst-case false-arc margin.
  A p95 error and a typical/quantile margin cannot yield a universal guarantee.
  Whole-fragment contours do not satisfy the same shared-arc assumption.
- Lipschitz quadrature bounds require actual arc-length coverage and valid
  weights. A learned selector need not satisfy an independent-sampling formula.
- Missing one true adjacency does not imply assembly impossibility if another
  path supports the same assembly. **Absence of the true complete candidate**
  from the final feasible candidate family is the valid recovery upper bound.
- Exact global optimality requires an exhaustive solve or an optimality
  certificate. Bounded beam/state/node search can be truncated; deterministic
  execution does not turn it into a certified global optimum.
- Template-conditioned warp must be estimated from each fragment and the public
  template alone, with deformation uncertainty retained. Pair-specific seam
  fitting and generated geometry cannot be physical evidence.

None of these assumptions is established on physical data yet. No domain
classifier, learned descriptor, or dewarper should be trained from this audit.

## Closeout

The final web review agreed to freeze the four numbers without adding a model,
after the repeat, truncation, qualification, truth-isolation, measured-scale,
and blind-cohort reuse rules above. Next: complete the existing 8x8 physical
pilot and mm/resolution qualification. Synthetic work is limited to unit tests,
resource estimates, and mechanical invariance/failure reproduction, not yield-
driven Etear/threshold/morphology/search-heuristic tuning.
