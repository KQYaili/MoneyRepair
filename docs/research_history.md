# Research History

This chronology explains how the current system was reached. It is not a list
of simultaneously supported algorithms. [STATUS.md](../STATUS.md) overrides
historical recommendations.

## v1-v1.5: Simulation And Search Baseline

The project began with synthetic banknote fragments, packed compatibility
matrices, depth-first coverage search, visualization, and reproducible seeds.
v1.5 added blur, noise, staining, missing pixels, strategy benchmarks, and
machine-readable reports. This established the engineering testbed but not a
realistic same-denomination identity model.

## v2-v2.5: Production Contracts And Reporting

v2 separated acquisition QA, manifest ingestion, packed matrix construction,
branch-and-bound search, and operator confirmation. v2.5 introduced a shared
scientific plotting style and editable process diagrams. Draw.io is now the
canonical editable diagram format; VSDX is an optional compatibility export.

## v3: Chimera Diagnosis

Multi-note simulation exposed cross-note chimeras hidden by single-note tests.
Appearance clustering could remove them in friendly global-gain simulations,
but failed when note count increased, appearance spread narrowed, or wear
became spatially non-uniform. Appearance therefore moved from identity evidence
to optional localization/tie-breaking support.

## v4-v4.2: Pose Search And Tear Geometry

The project added raw/placed pose machinery, packed/JIT paths, pressure tests,
per-note partitions, and placed-coordinate tear diagnostics. Contact-count,
boundary colour, and naive full-contour similarity were falsified as reliable
tear-mate discriminators. Per-note jagged geometry confirmed that the useful
signal is physical tear structure, not global tone.

## v4.3: Adaptive Evidence And Assembly Context

`Etear` replaced a fixed overlap-length gate with physically interpretable
evidence and uncertainty-aware routing. High-confidence cores were augmented
by whole-assembly gap recovery. In the canonical `N=20`, seeds `7/8/9`
ablation, routed v4.3 raised `p=24` yield/precision from `0.533/0.846` to
`0.917/0.981` while retaining perfect `p=8/16` results.

Mechanism decomposition showed that Etear mainly removes false edges, while
group-gap recovery supplies the fine-fragment recall gain.

## v4.3.2-v4.4.1: Wall Localization

Normalized-compute scaling preserved the v4.3 gain through `N=50`. A single
`N=100`, `p=24`, seed-7 diagnostic then localized missing yield before exact
cover.

Three preregistered interventions followed:

1. Oracle false-edge deletion improved recall only `0.840 -> 0.860`, below the
   `+0.050` rescue gate.
2. Residual-gap-first proposal returned a NULL result, `0.840 -> 0.840`, and
   localized misses upstream of the gap stage.
3. Fragment-disjoint base selection at fixed K improved oracle recall and exact
   yield `0.840 -> 0.900` with precision `1.000`.

The remaining ten misses lack a pure automatic component reaching the frozen
core threshold. v4.4.1 is therefore frozen at the multi-component core wall.

## v5: Reality Bridge

v5 stops tuning the simulator and builds the measurement path needed for raw
physical captures:

- production observations and evaluation truth are physically separated;
- mask boundary, interior, topology, and scale errors are audited separately;
- full crop-to-canonical transforms and pose-stage funnels are measured;
- tolerances derive from a physical alignment model;
- a 64-fragment/72-scene scanner/phone pilot is preregistered;
- a three-paper transfer audit adds rank, reciprocal-best-buddy, mask-shape,
  and component-coverage diagnostics without changing reconstruction.

The clean cardinal synthetic proxy reaches only `4/8` top-K pose recall. The
physical pilot remains uncollected, so no real reconstruction claim is made.

## Current Freeze

All prior development branches are represented in the linear history of
`main`. The repository keeps one canonical active branch and preserves release
tags for historical checkpoints.

The next allowed work is to collect and audit the physical pilot. Component
bridges or learned seam descriptors are conditional on reliable physical
handoff and recurrence of the same failure mechanism.
