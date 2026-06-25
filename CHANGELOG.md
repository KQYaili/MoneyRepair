# Changelog

## 2.0.0

Industrial production tradeoffs over open-ended research:

- acquisition quality contract (`quality.py`): focus, glare, segmentation
  confidence, and color-drift metrics with accept/reject thresholds and an
  `assess-quality` command;
- grid-accelerated compatibility build (`compute_compatibility_fast`) that
  writes packed bits directly for the ~20k fragment target, plus streaming
  incompatible-pair export (`build-matrix --engine fast --pairs-out`);
- packed-matrix helpers: `compatible_pair_count` and `restrict_packed_to_ids`;
- audit log on every confirm/reject with operator and reason;
- one auditable `run-pipeline` command that gates frames, prunes, searches,
  renders candidates, and writes a run manifest with input hashes, parameters,
  timings, and the QA summary.

## 1.5.0

Simulation realism and measurable algorithm comparison:

- realistic synthetic fragment degradation;
- solver ordering strategies: `area`, `degree`, and `area_degree`;
- strategy benchmark command;
- Python SVG/PDF/TIFF strategy report command;
- v1.5/v2.0/v2.5 planning documents.

## 1.0.0

Initial simulation-first release:

- synthetic banknote and fragment generation;
- scan/photo segmentation into fragment crops, masks, and manifests;
- manifest ingestion with affine placement into note coordinates;
- RGB reference scoring against front/back note images;
- contour tags, direction histograms, and curve similarity matching;
- packed compatibility matrix storage and precomputed pair import;
- DFS coverage search with candidate visualization reports;
- batch confirmation loop for manual reconstruction;
- benchmark and matrix footprint commands.
- GitHub CI, release checklist, and publishing guide.
