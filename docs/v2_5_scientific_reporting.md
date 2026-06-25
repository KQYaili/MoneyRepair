# v2.5 Scientific Reporting

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

## Next refinement

- Add a source-data CSV next to every figure.
- Add a figure manifest with panel titles, claim, export paths, and provenance.
- Add a methods schematic export path for editable VSDX once a local Visio
  runtime is available.
- Add report-level QA: text not clipped, no missing panels, source JSON hashes,
  and consistent palette across panels.
