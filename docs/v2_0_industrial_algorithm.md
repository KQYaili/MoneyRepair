# v2.0 Industrial Algorithm

v2.0 should stop expanding every possible research idea and make production
tradeoffs. The target is reliable reconstruction batches, not perfect academic
puzzle solving.

## Chosen production pipeline

1. Acquisition contract
   - Require high-resolution photos/scans with color card or stable lighting.
   - Store camera/scanner metadata and acquisition batch id.
   - Reject frames whose foreground segmentation quality is below threshold.

2. Manifest-first data model
   - Keep the manifest as the source of truth for id, label, side, affine
     placement, mask, and source image.
   - Optional OCR only fills `label`; it must not override manual labels without
     an explicit `--overwrite`.

3. Compatibility matrix as primary pruning
   - Use overlap and reference RGB checks to mark impossible pairs.
   - Import precomputed pair records when upstream matching already exists.
   - Store packed bits as the production format.

4. Search engine
   - Use branch-and-bound DFS with configurable ordering strategy.
   - Default to `area_degree` for production because it balances coverage and
     graph pruning.
   - Always emit multiple candidates plus an HTML report for manual acceptance.

5. Batch confirmation loop
   - Confirmed fragments leave the active set.
   - Rejected candidates are recorded by sorted fragment-id key.
   - Every accepted note gets a reproducible state entry.

## Explicit tradeoffs

- Prefer approximate affine placement plus RGB/overlap pruning over exact edge
  reconstruction for most fragments.
- Use contour/edge matching only for ambiguous residual fragments.
- Keep OCR optional and auditable rather than making it a hard dependency.
- Keep Visio-style editable schematics for methods and reports, but keep
  quantitative benchmark plots in Python for reproducibility.
- Do not optimize for a single unique automated answer; optimize for a small,
  inspectable candidate set.

## Production hardening still needed

- Real acquisition QA metrics: blur, glare, segmentation confidence, color drift.
- Parallel pairwise comparison for 20k fragments.
- Memory-mapped or chunked matrix import/export when `.npz` is not enough.
- Audit log for every accepted/rejected note.
- Operator UI for candidate review.
- Golden datasets with scanned and photographed fragments.
