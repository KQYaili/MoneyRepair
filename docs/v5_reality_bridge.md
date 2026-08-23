# v5 Reality Bridge Alpha

## Decision boundary

v4.4.1 is frozen at commit `af57a41`. The fixed-budget
`disjoint_round_robin` intervention passed its preregistered seed-7 gate, so the
project does not spend another correction on the same simulation discovery
case. v5 asks a different question:

> When raw acquisition error is introduced, which stage fails first: mask
> extraction, top-k pose recall, uncertainty routing, or the frozen
> reconstruction core?

This alpha implements the measurement funnel. It does not claim real-banknote
reconstruction and it deliberately does not change Etear, candidate generation,
or exact cover.

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

Evaluation annotations may contain a ground-truth mask and pose. They are used
only to compute the funnel metrics. `route_pose_candidates()` does not receive
or inspect either annotation.

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

Mask IoU remains a descriptive metric. The annotated segmentation gate instead
checks that every disagreement lies inside the declared segmentation tolerance
band. Real runs must replace the proxy tolerances with measured scanner/camera
and annotation repeatability.

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
  --output-dir runs/v5_proxy/cardinal/run
```

For real scans, first run `segment-scan`. Its manifest now records whether the
capture fixture constrains fragments to cardinal rotations or permits free
orientation. Add `references.front` and `references.back`, or pass both
reference paths to `reality-bridge`.

The report writes:

- per-fragment masks, candidate poses, uncertainty fields, and routes;
- annotated top-k/top-1 recall without using truth for routing;
- `handoff_front.npz` / `handoff_back.npz` for automatic placements only;
- `downstream_core_status: not_run`, so a placement artifact cannot be mistaken
  for a reconstruction result.

## Alpha proxy result

All rows use one synthetic note, eight fragments, seed 7, and the same default
physical tolerance contract. Timings are single local WSL runs and are
descriptive only.

| acquisition proxy | mask ready | top-k recall | top-1 | automatic handoff | mean mask IoU | locator ms/fragment |
|---|---:|---:|---:|---:|---:|---:|
| cardinal, clean, K=3 | 8/8 | 4/8 | 4/8 | 4/8 | 1.000 | 67.4 |
| cardinal, RGB noise 5 + 8% isolated mask dropout, K=3 | 8/8 | 4/8 | 4/8 | 4/8 | 0.932 | 65.3 |
| cardinal, clean, K=10 | 8/8 | 4/8 | 4/8 | 4/8 | 1.000 | 114.3 |
| free angle, clean, K=3 | 8/8 | 0/8 | 0/8 | 0/8 | 1.000 | 92.1 |

The controlled result is already a useful negative finding. Increasing the
returned K does not recover any missing truth, so the problem is upstream of
output truncation. Even the friendly cardinal proxy loses half the true poses;
free-angle capture loses all of them because the current locator searches only
0/90/180/270 degrees and reports no `sigma_theta`.

## Current conclusion and stop rule

The first observed v5 proxy bottleneck is **pose recall**, before uncertainty
calibration and before v4.4.1. Therefore the next empirical action is a small,
annotated real capture set. No downstream reconstruction change is justified
while true poses fail to enter the candidate set.

After real capture:

1. If masks exceed the physical tolerance band, change acquisition or
   segmentation only.
2. If masks pass but truth is absent from top-k, change registration only,
   starting with continuous angle/scale/affine refinement and calibrated
   `sigma_theta`.
3. If reliable true poses enter automatic handoff but reconstruction still
   shows the multi-component core wall, open a new component-bridge study.

Robotics, RL, and ODEWorld remain outside this core path.
