# v2.5 Scientific Reporting

> **Historical design note**. This document represents an earlier design round and is superseded by the authoritative [STATUS.md](../STATUS.md) in the repository root.

v2.5 focuses on polished reporting for papers, technical reports, and internal
reviews.

## Figure contract

Core conclusion:
MoneyRepair turns approximate fragment placement plus pairwise incompatibility
evidence into a small, inspectable set of high-coverage banknote candidates.

Evidence chain:

1. Dataset realism panel: clean synthetic vs realistic degraded synthetic vs
   segmented scan/photo crops.
2. Algorithm panel: ordering-strategy timing and coverage comparison.
3. Production panel: matrix pruning, DFS search, candidate report, and batch
   confirmation loop.
4. QA panel: coverage, active fragment count, rejected candidates, and residual
   ambiguous fragments.

Archetype:
Asymmetric mixed-modality figure: one schematic hero panel plus subordinate
benchmark plots and QA panels.

Export contract:
Use SVG/PDF/TIFF with editable text for Python plots. Keep Visio-style diagrams
editable as VSDX when manually produced.

## Tool split

Use Visio-style editable diagrams for:

- pipeline architecture;
- acquisition and manifest flow;
- production branch-and-bound logic;
- operator review loop.

The diagram style should follow the editable-spec approach used by the provided
Visio-oriented references:

- https://github.com/ywq177995212697-droid/visio-scientific-figures
- https://github.com/pengjunchi0/codex-visio-paper-figure-skill
- https://github.com/0Antique/Auto-Visio-Helper
- https://github.com/Rss3208/Visiomaster

Use Python plots for:

- benchmark bars and lines;
- matrix footprint comparisons;
- coverage distributions;
- ablation tables rendered as figures.

Current Python report command:

```bash
moneyrepair report-strategies --input runs/v1_5_strategy_benchmark.json --output-prefix runs/v1_5_strategy_report
```

## Shipped in 2.5

- `style.py` centralises the palette and publication rcParams; `reports.py` and
  `figures.py` both use it for a consistent look across panels.
- `figures.py` renders the multi-panel evidence figure
  (`render_report_figure`) and the standard QA / algorithm / footprint /
  coverage panels (`assemble_standard_panels`).
- Every figure ships a `<prefix>_data.csv` source table and a
  `<prefix>_manifest.json` with per-panel claims, export paths, the palette, and
  SHA-256 provenance for each source artifact.
- `validate_report` is the report-level QA gate: it flags missing panels, empty
  series, missing exports, or a missing palette.
- `diagrams.py` emits the editable Visio-style schematic as a JSON node/edge
  spec plus an editable-text SVG of the acquisition → manifest → pruning → DFS →
  report → operator-review loop.

Commands:

```bash
moneyrepair report-figures --output-prefix runs/report \
  --strategy-benchmark runs/v1_5_strategy_benchmark.json \
  --quality clean=runs/qa_clean.json --quality degraded=runs/qa_degraded.json \
  --claim "Approximate placement plus pairwise incompatibility yields a small inspectable candidate set."
moneyrepair export-diagram --name production-pipeline --output-prefix runs/pipeline
```

## Next refinement

- Convert the editable diagram spec to VSDX once a local Visio runtime is
  available (the JSON spec is already structured for it).
- Add a hero composite that embeds the diagram SVG beside the Python panels.
- Add text-overflow detection to the report QA gate.
