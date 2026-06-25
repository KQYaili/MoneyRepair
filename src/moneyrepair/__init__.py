"""MoneyRepair simulation toolkit."""

from moneyrepair.batch import AuditEvent, BatchState, ConfirmedNote
from moneyrepair.benchmark import MatrixFootprint, SyntheticBenchmark, compare_solver_strategies, estimate_matrix_footprint, run_synthetic_benchmark
from moneyrepair.compat import (
    CompatibilityMatrix,
    PackedCompatibilityMatrix,
    compatibility_from_pair_records,
    compute_compatibility_fast,
    write_incompatible_pairs,
)
from moneyrepair.ingest import fragments_from_manifest
from moneyrepair.labels import update_manifest_labels
from moneyrepair.pipeline import run_production_pipeline
from moneyrepair.quality import FrameQuality, QualityThresholds, assess_fragments, summarize_quality
from moneyrepair.reference import ReferenceScore, score_best_reference_side, score_fragments_by_side
from moneyrepair.realism import RealismProfile, make_realistic_synthetic_fragments
from moneyrepair.reports import write_strategy_report
from moneyrepair.scan import connected_components, segment_scan_to_manifest
from moneyrepair.simulate import make_synthetic_fragments
from moneyrepair.solver import CoverageSolution, solve_covering_sets
from moneyrepair.types import Fragment

__version__ = "2.0.0"

__all__ = [
    "AuditEvent",
    "BatchState",
    "CompatibilityMatrix",
    "ConfirmedNote",
    "CoverageSolution",
    "Fragment",
    "FrameQuality",
    "MatrixFootprint",
    "PackedCompatibilityMatrix",
    "QualityThresholds",
    "ReferenceScore",
    "RealismProfile",
    "SyntheticBenchmark",
    "__version__",
    "assess_fragments",
    "compatibility_from_pair_records",
    "compare_solver_strategies",
    "compute_compatibility_fast",
    "connected_components",
    "fragments_from_manifest",
    "estimate_matrix_footprint",
    "make_synthetic_fragments",
    "make_realistic_synthetic_fragments",
    "run_production_pipeline",
    "run_synthetic_benchmark",
    "segment_scan_to_manifest",
    "score_best_reference_side",
    "score_fragments_by_side",
    "solve_covering_sets",
    "summarize_quality",
    "update_manifest_labels",
    "write_incompatible_pairs",
    "write_strategy_report",
]
