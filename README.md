# MoneyRepair

**MoneyRepair** is an industrial, geometry-first reconstruction toolkit for hand-torn near-identical banknotes and paper documents.

It registers torn fragments to a canonical banknote frame, extracts physical tear-boundary coincidence, evaluates adaptive evidence ($E_{\text{tear}}$) and whole-assembly gap fit ($G$), and solves globally consistent non-overlapping, serial-deduplicated assemblies using exact-cover branch-and-bound search.

---

> [!IMPORTANT]
> **Authoritative System Status & Boundaries:**
> All measured metrics, capability bounds ($N=20 \sim 200$, $p=8 \sim 24$), documented dead ends, and future resume-paths are governed by **[STATUS.md](STATUS.md)** in the repository root. No claim elsewhere in this repository may exceed what `STATUS.md` supports.

---

## 🏛️ System Architecture

MoneyRepair is structured around an end-to-end batch processing pipeline:

```
[ Scan / Raw Photo ] ──► [ 1. Quality QA Gate ] ──► [ 2. Auto-Locator ]
                                                           │
[ Operator Confirmation ] ◄── [ 4. Exact-Cover Solver ] ◄── [ 3. Compatibility Matrix ]
```

![MoneyRepair Production Pipeline Flow](docs/pipeline_diagram.svg)

- **1. Quality QA Gate**: Evaluates focus (Laplacian variance), glare (luminance clipping), and mask solidity.
- **2. Auto-Locator**: Estimates candidate poses (X, Y, 0°/90°/180°/270°, front/back) using Numba JIT coarse-to-fine template matching.
- **3. Compatibility Matrix**: Evaluates spatial overlap, interlock contact criteria, and DBSCAN appearance tone clustering to prevent cross-note chimeras.
- **4. Exact-Cover Solver**: Runs zero-allocation branch-and-bound search over packed bit-matrices to find optimal note assemblies.
- **5. Interactive Review Loop**: Presents candidate reports to operators for confirmation or rejection.

---

## 🚀 Quickstart & Setup

### Environment Setup

With WSL Anaconda or Miniconda:

```bash
conda create -n moneyrepair python=3.11 -y
conda activate moneyrepair
pip install -e ".[dev]"
```

Or from the checked-in environment file:

```bash
conda env create -f environment.yml
conda activate moneyrepair
```

### 1. Synthetic Pipeline Smoke Test

Run an end-to-end synthetic reconstruction:

```bash
moneyrepair smoke --output-dir runs/smoke --pieces 18 --coverage 0.98
```

### 2. v4.3 Mechanism Ablation ($p=8, 16, 24$)

Run same-seed A/B ablation comparing `baseline`, `effectiveness`, `effectiveness_gap`, and `v43_routed`:

```bash
moneyrepair tearfit-v43-ablation \
  --notes 5 \
  --pieces-list 8,16,24 \
  --seeds 7,13 \
  --output runs/v4_3_ablation.json
```

### 3. Production Batch Pipeline

Run an auditable pipeline batch with quality gating and run manifest generation:

```bash
moneyrepair run-pipeline \
  --dataset runs/smoke/demo_fragments.npz \
  --output-dir runs/production_run \
  --auto-locate \
  --coverage 0.97
```

---

## 📊 v4.3 Mechanism Ablation Performance

Measured performance across $p=8, 16, 24$ regimes under deterministic search budgets (N=20, seeds 7/8/9):

| 路线 / 算法 | $p=8$ Yield (Prec) | $p=16$ Yield (Prec) | $p=24$ Yield (Prec) | 伪边率 (False Edge) | 核心作用与能力界定 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`baseline`** (固定重叠) | 1.00 (1.00) | 1.00 (1.00) | 0.53 (0.85) | 0.018 ~ 0.094 | 粗碎片高速；缺少物理几何特征约束。 |
| **`effectiveness`** (Etear 边得分) | 0.67 (1.00) | 0.67 (0.90) | 0.67 (0.80) | **0.000 ~ 0.006** | **降噪保真 (Precision Guard)**：伪边率降低 70%。 |
| **`effectiveness_gap`** (Etear + Gap 填补) | 0.67 (1.00) | **1.00 (1.00)** | **0.92 (0.98)** | **0.000 ~ 0.006** | **召回修复 (Recall Recovery)**：借助组装全局缺口拉回落单碎片。 |
| **`v43_routed`** (自适应路由) | **1.00 (1.00)** | **1.00 (1.00)** | **0.92 (0.98)** | **0.000 ~ 0.006** | **动态分流 (Regime Triage)**：粗碎片高速放行，细碎片 Gap 深度拼合。 |

See **[docs/v4_3_tear_effectiveness.md](docs/v4_3_tear_effectiveness.md)** and **[docs/v4_3_ab_benchmark.md](docs/v4_3_ab_benchmark.md)** for full details.

---

## 📚 Documentation Directory

Explore the complete documentation in **[docs/README.md](docs/README.md)**:

- **[Pipeline Notes](docs/pipeline.md)**: Production pipeline, quality gates, DFS logic, and operator loop.
- **[v4.3 Adaptive Geometry](docs/v4_3_tear_effectiveness.md)**: Math formulas for $E_{\text{tear}}$ and whole-assembly gap fit $G$.
- **[v4.3 A/B Benchmarks](docs/v4_3_ab_benchmark.md)**: Detailed $p=8, 16, 24$ ablation breakdown tables.
- **[Auto-Locator Deduction](docs/v4_0_algorithm_deduction.md)**: Mathematical analysis and proofs for JIT template matching.
- **[Chimera Discrimination](docs/v3_0_chimera_discrimination.md)**: DBSCAN tone gain clustering and multi-note pool hardening.

---

## 🧪 Testing

Run unit and integration tests across Linux/WSL and Windows:

```bash
# Run core test suite (torch-free)
python -m pytest -m "not experimental"

# Run full test suite
python -m pytest
```

---

## 📜 License & Governance

MoneyRepair is licensed under the MIT License. See [LICENSE](LICENSE) for details.
Governance and contribution principles are detailed in [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
