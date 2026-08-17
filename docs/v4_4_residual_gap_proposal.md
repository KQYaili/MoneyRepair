# MoneyRepair v4.4 — Residual-Gap-First Candidate Proposal System

## Executive Summary

MoneyRepair v4.4 fundamentally shifts candidate generation from **Edge-First / Pair-First local expansion** to **Residual-Gap-First Candidate Construction**.

Inspired by physical restoration principles in cultural heritage preservation (*Alex Kachkine et al., Nature 642, 343-350, 2025*), v4.4 separates candidate proposal from global exact-cover selection. Instead of requiring fragments to possess strong pairwise accepted edge evidence prior to candidate generation, v4.4 computes the **explicit residual gap geometry** of partial assemblies and proposes candidate sets specifically tailored to close each gap.

---

## 🎯 Core Paradigm Shift

```
[ Traditional Edge-First Pipeline (v4.0 - v4.3) ]
Pair Evidence (Etear) ──► Pair Filter ──► Candidate Beam Expansion ──► Candidate Pool

[ v4.4 Residual-Gap-First Pipeline ]
High-Confidence Core
        │
        ▼
Residual Gap Map (ResidualGapRegion)
        │
        ▼
Gap Complexity Routing (Simple / Moderate / Complex)
        │
        ▼
Whole-Assembly Proposal Generation (E_proposal & Tolerance Test)
        │
        ▼
Context-Aware Candidate Pool ──► Exact-Cover Solver (DLX)
```

---

## 📐 Mathematical Formulation

### 1. Residual Gap Component Extraction
Given a set of fragments $F$ and a partial core assembly $C$ with union mask $M_C = \bigcup_{i \in C} M_i$, the residual gap domain $\Omega_{\text{gap}}$ is extracted as:

$$\Omega_{\text{gap}} = M_{\text{canonical}} \setminus M_C$$

Connected component labeling over $\Omega_{\text{gap}}$ identifies individual `ResidualGapRegion` objects $R_k$, each characterized by:
- Area $A(R_k)$ and perimeter $P(R_k)$
- Adjacent fragment index set $\mathcal{N}(R_k)$

### 2. Proposal Effectiveness ($E_{\text{proposal}}$)
Candidate proposals are evaluated on whole-assembly **before $\rightarrow$ after improvement** rather than isolated pairwise similarity:

$$E_{\text{proposal}} = \frac{L_{\text{closed}} + \alpha N_{\text{independent\_seams}} + \beta \Delta C_{\text{gap}}}{L_{\text{newly\_exposed}} + \gamma A_{\text{sliver}} + \delta U_{\text{tolerance}} + \epsilon}$$

Where:
- $L_{\text{closed}}$: Total seam perimeter of $\Omega_{\text{gap}}$ explained/closed by the candidate subset.
- $N_{\text{independent\_seams}}$: Count of independent boundary directions providing joint support.
- $A_{\text{sliver}}$: Residual gap area remaining below the minimum informative scale ($S_{\text{min}}$).
- $U_{\text{tolerance}}$: Pose uncertainty penalty.

### 3. Minimum Informative Gap Scale ($S_{\text{min}}$)
To prevent infinite search over sub-tolerance artifacts, gaps smaller than $S_{\text{min}}$ are classified as non-informative slivers:

$$S_{\text{min}} = \max\left(3, \text{tolerance} + \text{fray\_layers}\right)$$

Proposals that reduce a gap to an unclosable sub-tolerance sliver are rejected ($E_{\text{proposal}} = 0$).

---

## 🔀 Gap-wise Complexity Routing

Rather than applying a global algorithm selection across all fragments, v4.4 routes compute resources on a **per-gap basis**:

| Gap Complexity Class | Condition | Search Strategy |
| :--- | :--- | :--- |
| **`simple`** | Adjacent to 1 fragment, $A(R_k) \le 128\text{px}$ | Single-fragment near-neighbour fast lookup |
| **`moderate`** | Adjacent to $2 \sim 3$ fragments, $A(R_k) \le 256\text{px}$ | Local subset beam search ($k \le 4$ fragments) |
| **`complex`** | Adjacent to $\ge 4$ fragments or $A(R_k) > 256\text{px}$ | Local exact-cover search over candidate set |

---

## 🔍 Candidate Funnel & Diagnostic Suite

v4.4 introduces explicit candidate funnel diagnostics (`v44_candidate_funnel_diagnostic`) to isolate recall bottlenecks:

1. **`weak_pair_proposal_gate_limiter`**: Exact candidate failed to generate due to pairwise evidence gates.
2. **`proposal_recall_rescued_without_quality_rescue`**: Candidate created by gap proposal but failed final coverage/score threshold.
3. **`inconclusive_gap_budget_saturation`**: Search state budget exhausted during gap expansion.
4. **`deeper_candidate_construction_wall`**: Fundamental candidate representation limit.

---

## 🧪 Verification & Test Suite

The v4.4 implementation is covered by unit tests in `tests/test_tearfit.py`:
- `test_compute_residual_gap_components_finds_known_hole`
- `test_classify_gap_complexity_boundaries`
- `test_gap_proposal_rejects_sliver_and_accepts_closed_gap`
- `test_v44_gap_first_valid_cover_and_v43_routed_fingerprint_unchanged`

All 125 tests in the test suite pass cleanly (`108 core + 17 v4.4 tests`).
