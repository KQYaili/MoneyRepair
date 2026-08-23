# v5 Real-Capture Diagnostic Pilot

## Purpose

This is a diagnostic pilot, not a publication-scale real-banknote dataset and
not evidence that MoneyRepair reconstructs real torn notes. It measures which
upstream stage fails first under repeatable physical acquisition while the
v4.4.1 reconstruction core and the v5 alpha locator remain frozen.

The preregistered stage order is:

```text
acquisition / segmentation
  -> pose proposal recall
  -> pose ranking and automatic routing
  -> uncertainty coverage
  -> frozen v4.4.1 core, only if Gates 1-3 pass
```

## Material split

- Use 8 independently printed, double-sided paper proxies with one common
  template. Do not use currency in this pilot.
- Tear each proxy into 8 physical fragments: 64 fragments total.
- Reserve 2 proxies / 16 fragments for tolerance calibration only.
- Freeze every tolerance and decision threshold before evaluating the remaining
  6 proxies / 48 fragments.
- Stratify each proxy into 2 small, 4 medium, and 2 large fragments. Mark any
  fragment below the declared 20 mm span instead of applying the same angular
  tolerance silently.

Repeated observations of one fragment are correlated measurements. Report both
observation-level and physical-fragment-level results; never count repeats as
independent fragments.

## Ground truth

Before tearing:

- scan both sides at high resolution;
- fix immutable canonical front/back references and hashes;
- keep parent identity only in the evaluation annotation file.

After tearing:

- make a 600-1200 DPI gold scan of both sides of every fragment;
- annotate each mask independently at least twice;
- retain that high-resolution mask as the gold master, then register and rasterize
  a capture-specific evaluation mask at the observation crop resolution;
- store the complete capture-specific 3x3 crop-to-canonical transform;
- record annotator, repeat, disagreement, physical area, span, and effective
  radius.

Production observations and evaluation truth must be separate files. The
production loader strips evaluation-only keys even if an old manifest embeds
them.

## Paired acquisition tracks

Use the same physical fragments in every track and independently remove and
replace them for each repeat.

### Scanner-cardinal

- 300 DPI production input;
- cardinal orientation fixture;
- scale ruler in every scene;
- automatic enhancement, crop, sharpening, and colour correction disabled or
  fully recorded;
- 3 independent placements.

### Phone-cardinal

- fixed top-view mount, fiducials, height, exposure, and focus;
- lens calibration and local pixels/mm from the planar fiducials;
- 3 independent placements.

### Phone-free

- same device, calibration target, and background;
- arbitrary in-plane angle plus mild perspective variation;
- 3 independent placements.

Eight proxy sets across three repeats and three tracks produce approximately 72
multi-fragment scenes:

```text
8 * (3 scanner + 3 phone-cardinal + 3 phone-free) = 72 scenes
```

## Data contracts

The production manifest must contain only observable fields:

- observation, acquisition, session, and repeat IDs;
- source image, source mask, and scene bounding box;
- modality, device, resolution, and declared orientation mode;
- scanner DPI or phone fiducial calibration;
- lens/calibration version;
- lighting, exposure, focus, and capture height where applicable;
- canonical front/back reference hashes;
- segmentation algorithm, version, confidence, and capture timestamp.

The separate evaluation annotation file must contain:

- fragment ID, physical fragment ID, and parent ID;
- ground-truth side and gold mask;
- complete capture-specific crop-to-canonical affine or homography;
- physical area, span, and effective radius;
- annotator/repeat metadata and annotation uncertainty.

## Tolerance calibration

The current 300 DPI, 0.085 mm segmentation, and 0.085 mm registration values are
declared proxies. Replace them using only the 16 calibration fragments:

- effective scanner scale from a physical ruler;
- phone pixels/mm, lens distortion, and homography from fiducials;
- segmentation tolerance from repeated masks, such as a preregistered 95th
  percentile surface distance;
- registration tolerance from independent capture/replacement residuals;
- per-fragment angular tolerance from its effective radius;
- annotation uncertainty, capture repeatability, segmentation uncertainty,
  template mismatch, and locator error reported separately.

`r_e=1` is an acceptance policy, not a parameter estimated from the pilot.
`H=0` remains appropriate for the digital registration stage.

## Preregistered gates

### Gate 1: mask

For scanner-cardinal and phone-cardinal:

- at least 90% of evaluation fragments pass the physical boundary tolerance in
  at least 2 of 3 repeats;
- interior missing and extraneous area stay below the frozen calibration limit
  (the initial policy ceiling is 1% each);
- no parent proxy has a fragment-level pass rate below 75%;
- connectivity and extraneous-component metrics are reported separately from
  IoU.

Failure means acquisition/segmentation is the first wall. Do not modify the
locator.

### Gate 2: pose proposal recall

Compute only on mask-ready fragments:

- K=3 is primary and K=10 is diagnostic;
- fragment-level top-k recall must be at least 90%;
- decompose misses into side, translation, angle, scale/affine, coarse
  shortlist, fine refinement, and final filtering.

K=3 failure with K=10 success is a final candidate budget/ranking issue. Failure
at both K values requires the internal stage audit before changing the transform
family.

### Gate 3: ranking and route safety

- top-1 accuracy must be at least 85%;
- automatic pose precision must be at least 98%;
- preregister zero false-automatic placements for this small pilot;
- automatic recall is an efficiency metric and must not be raised by weakening
  the review threshold.

### Gate 4: uncertainty

- report translation interval coverage and interval width together;
- a wide interval is not evidence of useful calibration;
- free-angle observations without `sigma_theta` must route to review;
- add angle, scale, or affine uncertainty only when that transform family is
  introduced by a later, separately gated registration study.

Only after Gates 1-3 pass may the frozen v4.4.1 core run. A failure on the 48
evaluation fragments cannot be followed by tuning on those same fragments and
then relabelled as validation.

## Stop and rescue rules

```text
mask failure
  -> acquisition / segmentation only

mask pass + top-k failure
  -> inspect the recorded locator stage and transform-family coverage

top-k pass + top-1 or automatic-precision failure
  -> ranking / uncertainty only

reliable automatic handoff + core failure
  -> reopen reconstruction research on a new dataset/protocol
```

Continuous angle and `sigma_theta` are the next intervention only for measured
free-angle model-family misses. Add scale/affine only when phone residuals show
that rigid registration is inadequate. For scanner-cardinal misses inside the
current model family, inspect the coarse grid, fixed internal top-10 shortlist,
score function, and fine refinement first.

Robotics, RL, ODEWorld, and the unverified v6-v10 scaffold remain outside this
pilot.
