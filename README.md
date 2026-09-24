# MoneyRepair

MoneyRepair is a geometry-first research system for reconstructing hand-torn,
near-identical paper fragments. It combines a physically defined acquisition
contract, canonical pose hypotheses, tear-boundary evidence, bounded candidate
construction, and exact-cover selection.

The deterministic simulation core is frozen at v4.4.1. The v5 code adds the
measurement infrastructure needed to test masks and poses on physical captures;
it does **not** establish real-banknote reconstruction performance. The next
valid experiment is the preregistered physical capture pilot.

> [!IMPORTANT]
> [STATUS.md](STATUS.md) is the single source of truth for measured results,
> limitations, dead ends, and the next allowed experiment. No claim elsewhere
> in the repository may exceed it.

![MoneyRepair evidence summary](docs/figures/research_evidence_summary.png)

The figure is generated only from committed simulation benchmark JSON. The
rightmost panel is deliberate: independent physical masks, pose handoff, and
real seam evidence are still pending.

## Current Position

| Layer | State | Evidence boundary |
|---|---|---|
| Deterministic assembly core | frozen | measured in per-note fractal-tear simulation |
| Acquisition coordinate contract | implemented | scanner/phone ledger, calibration, hash and truth-separation checks |
| Raw-crop mask and pose audit | implemented | measured only on annotated synthetic capture proxies |
| Physical pilot | ready to collect | no qualifying physical dataset is committed or claimed |
| Learned v6-v10 tools | experimental | untrained/unverified; never routed into production |

The operating stance is high-precision automatic confirmation for cases with
sufficient evidence, plus an explicit human review queue. MoneyRepair does not
claim full automation for a 2,000-note fine-fragment pool.

## Architecture

![MoneyRepair production pipeline](docs/pipeline_diagram.svg)

The pipeline keeps acquisition truth separate from production observations:

1. Freeze the physical coordinate and tolerance contract.
2. Ingest fragment images and masks in local crop coordinates.
3. Propose canonical poses and observable uncertainty.
4. Score placed tear evidence and build core/gap assembly candidates.
5. Select globally consistent candidates with exact cover.
6. Automatically release only candidates that pass the evidence policy;
   route the rest to review.

The editable source is [pipeline_diagram.drawio](docs/pipeline_diagram.drawio).
Every checked-in diagram also has a deterministic JSON graph and an SVG
preview. See the [diagram catalog](docs/README.md#editable-diagrams).

## Measured Results

### Fine-fragment mechanism ablation

The canonical v4.3 experiment uses `N=20`, seeds `7/8/9`, fixed state/node
budgets, and no serial labels.

| pieces per note | fixed overlap yield / precision | routed v4.3 yield / precision |
|---:|---:|---:|
| 8 | 1.000 / 1.000 | 1.000 / 1.000 |
| 16 | 1.000 / 1.000 | 1.000 / 1.000 |
| 24 | 0.533 / 0.846 | **0.917 / 0.981** |

At `p=24`, adaptive tear evidence mainly removes false edges; whole-assembly
gap recovery restores recall. The result is simulation-only and does not
validate real tear geometry.

### Fixed-budget base selection

In the single preregistered `N=100`, `p=24`, seed-7 diagnostic, replacing
global top-K base ranking with fragment-disjoint rounds changes only the base
selector:

| selector | oracle candidate recall | exact yield | exact precision |
|---|---:|---:|---:|
| global | 0.840 | 0.840 | 0.966 |
| disjoint rounds | **0.900** | **0.900** | **1.000** |

The remaining ten misses have no pure automatic component reaching the frozen
core threshold. This is one seed, not a cross-seed or physical-data headline.

## Install

Use the isolated WSL Anaconda/Miniconda environment:

```bash
conda env create -f environment.yml
conda activate moneyrepair
pip install -e ".[dev,reports]"
```

Python 3.10-3.13 are covered by CI. OCR and experimental ML dependencies are
optional:

```bash
pip install -e ".[ocr]"   # optional Tesseract wrapper
pip install -e ".[ml]"    # experimental research modules only
```

## Quick Smoke Test

```bash
moneyrepair smoke --output-dir runs/smoke --pieces 18 --coverage 0.98
```

This writes a synthetic dataset, packed compatibility matrix, candidate
solutions, visualizations, and an auditable `run_manifest.json`.

## Physical Pilot

Create and validate the frozen 64-fragment/72-scene ledger before collecting
data:

```bash
moneyrepair pilot-init \
  --output-dir runs/v5_physical_pilot \
  --generate-reference-master

moneyrepair pilot-freeze-coordinate-contract \
  --pilot-dir runs/v5_physical_pilot \
  --scanner-dpi 300 \
  --phone-pixels-per-mm 11.811023622047244

moneyrepair pilot-validate \
  --pilot-dir runs/v5_physical_pilot \
  --output preflight_report.json
```

The generated duplex master is explicitly labelled `NOT CURRENCY`. For a real
capture, keep gold masks/transforms in a separate annotation file and audit the
truth-blind handoff:

```bash
moneyrepair reality-bridge \
  --manifest runs/capture/manifest.json \
  --annotations runs/capture/annotations.json \
  --reference-front references/front.png \
  --reference-back references/back.png \
  --output-dir runs/capture/reality_audit
```

Read [the physical-pilot protocol](docs/v5_real_capture_pilot.md) before using
these commands. The pipeline must repair only the first failed stage: mask,
pose recall, uncertainty routing, or downstream assembly.

## Reproduce The Main Simulation Evidence

```bash
moneyrepair tearfit-v43-ablation \
  --notes 20 \
  --pieces-list 8,16,24 \
  --seeds 7,8,9 \
  --no-time-limits \
  --output runs/v4_3_geometry_ablation.json

moneyrepair tearfit-v44-base-selection \
  --notes 100 \
  --pieces-per-note 24 \
  --seed 7 \
  --output runs/v4_4_1_base_selection_n100_seed7.json
```

The committed source measurements live in [docs/benchmarks](docs/benchmarks).
Long runs should preserve raw JSON, seeds, budgets, and manifests.

## Diagrams And Figures

Generate an editable Draw.io source, JSON graph, and SVG preview:

```bash
moneyrepair export-diagram \
  --name production-pipeline \
  --output-prefix docs/pipeline_diagram
```

Available names are `production-pipeline`, `acquisition-flow`, `search-logic`,
`operator-loop`, and `research-gates`. VSDX is an optional compatibility export
via `--vsdx`; Draw.io is the checked-in editable format.

Regenerate the consolidated scientific figure from committed benchmark JSON:

```bash
python docs/figures/make_research_summary.py
```

## Repository Map

```text
src/moneyrepair/              supported deterministic package
src/moneyrepair/baselines/    falsified or superseded comparison paths
src/moneyrepair/experimental/ unverified v6-v10 research scaffolds
tests/                        deterministic unit and integration tests
docs/benchmarks/              committed source measurements
docs/figures/                 plotting scripts and rendered figures
docs/                         current guides and historical reports
```

Start with [the documentation index](docs/README.md), [pipeline guide](docs/pipeline.md),
and [research history](docs/research_history.md).

## Development Checks

```bash
python -m pytest -q
python -m compileall -q src
ruff check src tests docs/figures
mypy src/moneyrepair --ignore-missing-imports
```

`mypy` remains advisory for legacy modules; pytest, compileall, and Ruff are
blocking checks. Do not commit real banknote scans, private fragment images,
OCR caches, or generated `runs/` outputs.

MoneyRepair is released under the [MIT License](LICENSE).
