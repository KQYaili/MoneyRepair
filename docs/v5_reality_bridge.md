# v5 Reality Bridge Alpha 2

## Decision boundary

v4.4.1 is frozen at commit `af57a41`. The fixed-budget
`disjoint_round_robin` intervention passed its preregistered seed-7 gate, so the
project does not spend another correction on the same simulation discovery
case. v5 asks a different question:

> When raw acquisition error is introduced, which stage fails first: mask
> extraction, top-k pose recall, uncertainty routing, or the frozen
> reconstruction core?

The alpha-1 implementation at `cffbb10` established the measurement funnel.
Alpha 2 hardens only its measuring instruments: it does not change the locator
ranking, routing thresholds, Etear, candidate generation, or exact cover.

## Pipeline

```text
photo / scan
  -> local fragment crops and masks
  -> observable quality gate
  -> top-k canonical poses
  -> uncertainty route: automatic / review / insufficient-evidence
  -> placed handoff datasets
  -> v4.4.1 core (not run in this alpha)
```

Evaluation annotations are stored in a separate JSON file and never copied into
production `Fragment` objects. Each annotation contains a gold mask and a full
3x3 crop-to-canonical transform. `route_pose_candidates()` receives neither;
the production loader also strips legacy truth keys from embedded metadata.

## Physical tolerance contract

The default proxy contract declares:

```text
scan DPI                         300
segmentation tolerance          0.085 mm
registration tolerance          0.085 mm
minimum fragment span           20 mm
effectiveness ratio             1.0
```

This derives, rather than tunes, the pose audit gates:

```text
combined translation tolerance  2.008 px
angular tolerance               0.974 degrees
minimum effective feature       0.510 mm = 6.024 px
```

The minimum effective feature uses the same tolerance-accounting form used in
the physical-restoration reference:

```text
D_min = r_e * (2 * (T_seg + T_reg) + 0.25 * H) + T_seg + T_reg
```

Mask IoU remains descriptive. External boundaries are compared using continuous
Euclidean pixel-centre distances, so `1.004 px` is not rounded up to a 2-pixel
Chebyshev dilation. Internal missing area, internal extraneous area, disconnected
components, and holes are audited separately. The initial interior-area ceiling
is 1%; real runs must replace every proxy tolerance using the calibration split.

## Commands

Generate a deterministic annotated proxy capture:

```bash
moneyrepair simulate-capture \
  --output-dir runs/v5_proxy/cardinal \
  --pieces 8 \
  --seed 7 \
  --orientation-mode cardinal
```

Run the diagnostic:

```bash
moneyrepair reality-bridge \
  --manifest runs/v5_proxy/cardinal/manifest.json \
  --annotations runs/v5_proxy/cardinal/annotations.json \
  --output-dir runs/v5_proxy/cardinal/run
```

For real scans, first run `segment-scan`. Its manifest now records whether the
capture fixture constrains fragments to cardinal rotations or permits free
orientation. Add `references.front` and `references.back`, or pass both
reference paths to `reality-bridge`.

The report writes:

- per-fragment masks, candidate poses, uncertainty fields, and routes;
- annotated top-k/top-1 recall and complete transform error without using truth
  for routing;
- coarse-position count, the fixed internal coarse top-10, refined candidates,
  final filters, and transform-family coverage for each recall miss;
- boundary/interior/topology mask metrics, automatic pose precision, and an
  explicit false-automatic count;
- `handoff_front.npz` / `handoff_back.npz` for automatic placements only;
- `downstream_core_status: not_run`, so a placement artifact cannot be mistaken
  for a reconstruction result.

## Alpha proxy result

All rows use one synthetic note, eight fragments, seed 7, and the same default
physical tolerance contract. Timings are single local WSL runs and are
descriptive only.

| acquisition proxy | mask ready | top-k recall | top-1 | automatic / pose precision / release precision | mean mask IoU | localized miss |
|---|---:|---:|---:|---:|---:|---|
| cardinal, clean, K=3 | 8/8 | 4/8 | 4/8 | 4 / 1.000 / 1.000 | 1.000 | 4 coarse-shortlist misses |
| cardinal, RGB noise 5 + 8% isolated mask dropout, K=3 | 0/8 | n/a | n/a | 4 / 1.000 / 0.000 | 0.932 | segmentation first |
| cardinal, clean, K=10 | 8/8 | 4/8 | 4/8 | 4 / 1.000 / 1.000 | 1.000 | 4 coarse-shortlist misses |
| free angle, clean, K=3 | 8/8 | 0/8 | 0/8 | 0 / n/a / n/a | 1.000 | 8 transform-family misses |

The controlled result is a more specific negative finding. Increasing returned
K does not recover missing truth because all four clean cardinal misses occur
before the fixed internal coarse top-10 shortlist. The truth transforms are
inside the cardinal rigid model family, so continuous-angle search is not the
answer to that row. Free-angle capture instead produces eight measured
transform-family misses and no `sigma_theta`.

The degraded row corrects an alpha-1 measurement flaw. Its external boundaries
still pass, but mean interior missing area is `0.068`, above the 1% policy gate,
so segmentation correctly becomes the first bottleneck. The observable router
still emits four poses; this is why automatic pose precision and the dataset
mask gate are reported separately.

## Current conclusion and stop rule

For clean cardinal proxy input the first observed bottleneck remains **pose
recall**, now localized to the coarse shortlist. For the deliberately damaged
mask proxy it is segmentation, and for free angles it is the transform family.
The locator and router are frozen at these measurements. The next empirical
action is the preregistered [real-capture diagnostic pilot](v5_real_capture_pilot.md),
not another synthetic parameter search.

After real capture:

1. If masks exceed the physical tolerance band, change acquisition or
   segmentation only.
2. If masks pass but truth is absent from top-k, inspect its measured stage and
   model-family coverage. Add continuous angle and `sigma_theta` only for angle
   family misses; add scale/affine only when physical residuals require them.
3. If reliable true poses enter automatic handoff but reconstruction still
   shows the multi-component core wall, open a new component-bridge study.

Robotics, RL, and ODEWorld remain outside this core path.
