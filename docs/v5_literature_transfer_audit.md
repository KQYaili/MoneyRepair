# v5 Literature-Transfer Audit

Status: **measurement-only; production reconstruction remains frozen**.

This audit asks what can be transferred from three recent reconstruction
papers without turning MoneyRepair into a square-jigsaw benchmark optimizer or
reopening the spent v4.4.1 seed-7 tuning loop.

## Sources and transfer boundary

| source | useful mechanism | transferred now | deliberately not transferred |
|---|---|---|---|
| Song et al., *ERL-MPP* (AAAI 2025) | pair, small-group, and global evidence; moves over groups rather than only single pieces | component-level oracle accounting already available through the frozen core graph | actor-critic/evolutionary RL, fixed-grid puzzlets, RGB-semantic global discriminator |
| Shahar, Elkin, and Ben-Shahar, *The Missing GAP* (CVPR 2026) | validate synthetic shapes against geometry measured from real masks; preserve an explicit mask channel | deterministic fragment-mask distribution summary | treating learned single-fragment shape realism as proof of paired tear realism; ViT/flow solver |
| Rikaa et al., *A Generic Hybrid Framework for 2D Visual Reconstruction* (2025) | true-neighbour rank, Top-K retrieval, reciprocal best buddies, hard negatives | same-source first-positive MRR/Hit@K, query candidate-hit rate, and reciprocal-best-buddy diagnostics | claiming the proxy is seam-mate retrieval, whole-fragment appearance identity, per-edge min-max confidence, GA, best-of-many-run reporting |

The papers solve materially easier identity settings: square/grid topology,
small fragment counts, or visually distinct source images. MoneyRepair must
separate *canonical position* from *physical same-note origin*. A visually
coherent assembly made from two near-identical notes is still wrong.

## Command

```bash
moneyrepair paper-transfer-audit \
  --notes 20 \
  --pieces-per-note 24 \
  --seed 7 \
  --output runs/paper_transfer/audit_n20_p24_seed7.json
```

The command performs no candidate generation and no exact-cover selection. It
reports three post-hoc measurement groups:

1. `pair_retrieval`: same-source query candidate-hit rate, first-positive MRR,
   Hit@1/3/5/10, rank
   histogram, and reciprocal-best-buddy precision/recall, reported separately
   for all scored pairs, review-or-better pairs, and automatic-only pairs.
   A query scores zero when all of its same-source candidates are missing; loss
   of only some same-source candidates is not measured by this first-positive
   proxy. RBB predictions are built truth-blind over the complete scored graph;
   truth eligibility limits only MRR/Hit@K queries. Pairs with unknown truth are
   counted separately and excluded from the RBB precision denominator.
2. `fragment_geometry`: area, perimeter, normalized perimeter, bounding-box
   ratios, circularity, compactness, convex solidity, concavity, and hull-vertex
   distributions. Raw pixel quantities are comparable only in a common
   canonical scale; the dimensionless fields are the preferred cross-capture
   descriptors.
3. `component_oracle`: the existing simulator-truth audit of whether automatic
   true-edge components can reach the frozen `0.78` core threshold.

`note_id` is read only after edge scoring. In the current simulator it is a
same-source proxy, not a complete annotation of which two physical boundary
arcs came from the same tear. A physical pilot should add seam-pair ground truth
before calling these values true tear-mate retrieval metrics.

## Measured seed-7 audit

The new command was run at the frozen wall (`N=100`, `p=24`, seed 7) over
2,400 fragments and 374,773 scored pairs. This is one deterministic simulator
case, not a replicated or physical-data result.

| scored graph | same-source query hit | first-positive MRR | Hit@1 | reciprocal-best-buddy precision | conditional reciprocal coverage of same-note scored pairs |
|---|---:|---:|---:|---:|---:|
| all scored | 1.0000 | 0.9931 | 0.9867 | 1.0000 | 0.0916 |
| review or better | 1.0000 | 0.9931 | 0.9867 | 1.0000 | 0.1914 |
| automatic only | 0.9946 | 0.9894 | 0.9846 | 1.0000 | 0.2484 |

Despite those near-perfect local ranking values, the automatic true-edge graph
still records only `90/100` notes at the frozen `0.78` core threshold. The
remaining ten are the already-known multi-component misses. This is the useful
negative result from the hybrid-framework metrics: excellent nearest-neighbour
ranking is not equivalent to having enough accepted true edges to construct a
high-coverage component.

The reciprocal-coverage denominator is local to each filtered layer. Therefore
`0.0916 -> 0.1914 -> 0.2484` is **not** an end-to-end recall improvement: the
numerator remains 741 while the denominator shrinks from 8,086 to 3,872 to
2,983. Whole-fragment RBB also forms disjoint pairs (degree at most one), so it
cannot by itself represent a multi-piece assembly graph.

The same run establishes a synthetic shape baseline for the future physical
pilot:

| dimensionless mask feature | q05 | median | q95 |
|---|---:|---:|---:|
| area fraction | 0.0189 | 0.0412 | 0.0598 |
| perimeter / sqrt(area) | 5.5415 | 7.2334 | 10.1147 |
| convex solidity | 0.4084 | 0.6964 | 0.8871 |
| concavity ratio | 0.1129 | 0.3036 | 0.5916 |

These values are reference measurements, not realism gates. Thresholds for
real/synthetic agreement must be preregistered only after independent physical
calibration masks exist; fitting them to this simulator would repeat the same
benchmark-overfitting error the v5 freeze is designed to prevent.

The estimators are deliberately recorded rather than presented as exact
reproductions of The Missing GAP: perimeter is four-neighbour exposed
pixel-cell edge length; compactness is `P^2/(4*pi*A)`; concavity is
`1 - area/hull_area`; and hull vertices are not contour-simplified. The paper
uses different estimators for several of these quantities. Even dimensionless
values require the same canonical-frame canvas contract: `area_fraction` from
a tight raw crop is not comparable with the full-note-canvas baseline above.

## Preregistered decision ladder

The literature does not justify another production-core edit before physical
data exists. The next experiment is conditional:

### Gate 0: physical handoff first

Run the frozen v5 pilot. Do not study component bridges unless mask tolerance,
top-K pose recall, and automatic-pose precision pass the gates in
`v5_real_capture_pilot.md`. Repair only the first failed upstream stage.

### Gate 1: establish the real failure mechanism

On held-out, truth-annotated placed fragments, report:

- candidate-graph recall and recall@K for true physical seam mates;
- reciprocal-best-buddy precision;
- automatic/review/insufficient routing;
- pure-core component count and largest-component coverage;
- oracle candidate recall, exact yield, precision, and manual queue.

Only if real masks and poses pass Gate 0 **and** the same multi-component
pure-core failure recurs may a component operation be studied.

### Gate 2: one deterministic component-action A/B

Freeze fragments, poses, Etear, thresholds, search budgets, exact cover, and
the evaluation split. Compare:

- A: frozen v4.4.1 construction;
- B: bounded component merge/split/move proposals using existing geometric
  evidence only.

B may proceed only if it improves oracle candidate recall or exact yield by at
least `0.05`, produces no false-automatic assembly, does not reduce automatic
precision, and stays inside the preregistered compute budget. Otherwise close
the deterministic component branch.

### Gate 3: learned fine-tear descriptor only after a residual remains

If B leaves a retrieval failure rather than a search failure, train a narrow
boundary descriptor on physical seam strips/masks. Use same-position
different-note and near-straight false seams as hard negatives. Compare it with
frozen Etear under the same candidate graph and report rank metrics plus final
false-automatic outcomes. Classification accuracy alone cannot pass the gate.

ERL, PuzzleFlow, and GA are out of scope unless this narrower experiment first
shows that deterministic evidence and bounded component actions are
insufficient.

## Stage stop rule

The three papers are exhausted for the current evidence state when:

- rank and reciprocal-best-buddy diagnostics are implemented;
- reciprocal-best-buddy predictions are generated independently of truth
  eligibility, with unknown-truth predictions reported separately;
- simulator mask distributions are exportable for a future real/synthetic
  comparison;
- component-level coverage is reported without changing reconstruction;
- the physical-pilot A/B/C ladder is preregistered;
- no production threshold, candidate rule, or solver path has changed.

Those conditions mean the project should stop literature-driven algorithm work
and collect the physical pilot. More seed-7 simulation tuning, RL, flow
matching, or GA would add complexity without resolving the measured upstream
reality blocker.

## References

- X. Song et al., "ERL-MPP: Evolutionary Reinforcement Learning with
  Multi-head Puzzle Perception for Solving Large-scale Jigsaw Puzzles of Eroded
  Gaps," AAAI 2025.
- O. I. Shahar, G. Elkin, and O. Ben-Shahar, "The Missing GAP: From Solving
  Square Jigsaw Puzzles to Handling Real World Archaeological Fragments,"
  arXiv:2605.12077v1, accepted at CVPR 2026.
- D. Rikaa et al., "A Generic Hybrid Framework for 2D Visual Reconstruction,"
  arXiv:2501.19325v1, 2025.
