"""MoneyRepair simulation toolkit."""

from moneyrepair.batch import AuditEvent, BatchState, ConfirmedNote
from moneyrepair.benchmark import MatrixFootprint, SyntheticBenchmark, compare_solver_strategies, estimate_matrix_footprint, run_synthetic_benchmark
from moneyrepair.compat import (
    CompatibilityMatrix,
    PackedCompatibilityMatrix,
    compatibility_from_pair_records,
    compute_compatibility_clustered,
    compute_compatibility_fast,
    write_incompatible_pairs,
)
from moneyrepair.diagnostics import diagnose_groups, diagnose_solutions, solution_purity
from moneyrepair.diagrams import DiagramSpec, production_pipeline_spec, write_diagram
from moneyrepair.figures import FigurePanel, assemble_standard_panels, render_report_figure, validate_report
from moneyrepair.fingerprint import (
    cluster_fragments_by_appearance,
    discriminative_compatibility,
    fragment_appearance,
)
from moneyrepair.ingest import fragments_from_manifest
from moneyrepair.interlock import (
    InterlockCompatibilityStats,
    TearInterlockScore,
    apply_interlock_constraints_with_stats,
    compute_interlock_compatibility,
    compute_interlock_compatibility_with_stats,
    iter_contact_candidate_pairs,
    tear_interlock_score,
)
from moneyrepair.labels import update_manifest_labels
from moneyrepair.pipeline import run_production_pipeline
from moneyrepair.policy_compare import POLICY_COMPARE_STRATEGIES, run_policy_controller_comparison
from moneyrepair.pressure import run_pressure_case, run_pressure_sweep
from moneyrepair.quality import FrameQuality, QualityThresholds, assess_fragments, summarize_quality
from moneyrepair.reference import ReferenceScore, score_best_reference_side, score_fragments_by_side
from moneyrepair.realism import RealismProfile, make_realistic_synthetic_fragments
from moneyrepair.reports import write_strategy_report
from moneyrepair.scan import connected_components, segment_scan_to_manifest
from moneyrepair.simulate import make_multi_note_fragments, make_synthetic_fragments
from moneyrepair.solver import CoverageSolution, solve_covering_sets
from moneyrepair.tearfit import (
    AssemblyCandidate,
    FractalTearConfig,
    TEARFIT_COVER_OBJECTIVES,
    TEARFIT_SEED_STRATEGIES,
    TearFitDiagnostics,
    TearFitComparisonCase,
    TearFitEdge,
    TearFitTrialResult,
    make_fractal_tear_fragments,
    run_tearfit_sweep,
    run_tearfit_strategy_comparison,
    run_tearfit_trial,
    score_absolute_tear_pairs,
    tearfit_comparison_cases,
)
from moneyrepair.types import Fragment

__version__ = "4.2.1"

__all__ = [
    "AuditEvent",
    "AssemblyCandidate",
    "BatchState",
    "CompatibilityMatrix",
    "ConfirmedNote",
    "CoverageSolution",
    "DiagramSpec",
    "FigurePanel",
    "Fragment",
    "FractalTearConfig",
    "FrameQuality",
    "InterlockCompatibilityStats",
    "MatrixFootprint",
    "PackedCompatibilityMatrix",
    "POLICY_COMPARE_STRATEGIES",
    "QualityThresholds",
    "ReferenceScore",
    "RealismProfile",
    "SyntheticBenchmark",
    "TEARFIT_COVER_OBJECTIVES",
    "TEARFIT_SEED_STRATEGIES",
    "TearFitComparisonCase",
    "TearInterlockScore",
    "TearFitDiagnostics",
    "TearFitEdge",
    "TearFitTrialResult",
    "apply_interlock_constraints_with_stats",
    "__version__",
    "assemble_standard_panels",
    "assess_fragments",
    "cluster_fragments_by_appearance",
    "compatibility_from_pair_records",
    "compare_solver_strategies",
    "compute_compatibility_clustered",
    "compute_compatibility_fast",
    "compute_interlock_compatibility",
    "compute_interlock_compatibility_with_stats",
    "iter_contact_candidate_pairs",
    "connected_components",
    "diagnose_groups",
    "diagnose_solutions",
    "discriminative_compatibility",
    "fragment_appearance",
    "fragments_from_manifest",
    "estimate_matrix_footprint",
    "make_multi_note_fragments",
    "make_fractal_tear_fragments",
    "make_synthetic_fragments",
    "make_realistic_synthetic_fragments",
    "production_pipeline_spec",
    "solution_purity",
    "tear_interlock_score",
    "render_report_figure",
    "run_pressure_case",
    "run_pressure_sweep",
    "run_policy_controller_comparison",
    "run_production_pipeline",
    "run_tearfit_sweep",
    "run_tearfit_strategy_comparison",
    "run_tearfit_trial",
    "run_synthetic_benchmark",
    "segment_scan_to_manifest",
    "score_best_reference_side",
    "score_absolute_tear_pairs",
    "score_fragments_by_side",
    "solve_covering_sets",
    "summarize_quality",
    "tearfit_comparison_cases",
    "update_manifest_labels",
    "validate_report",
    "write_diagram",
    "write_incompatible_pairs",
    "write_strategy_report",
]
