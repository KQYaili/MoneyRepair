"""MoneyRepair simulation toolkit."""

from moneyrepair.batch import BatchState, ConfirmedNote
from moneyrepair.benchmark import MatrixFootprint, SyntheticBenchmark, estimate_matrix_footprint, run_synthetic_benchmark
from moneyrepair.compat import CompatibilityMatrix, PackedCompatibilityMatrix, compatibility_from_pair_records
from moneyrepair.ingest import fragments_from_manifest
from moneyrepair.labels import update_manifest_labels
from moneyrepair.reference import ReferenceScore, score_best_reference_side, score_fragments_by_side
from moneyrepair.scan import connected_components, segment_scan_to_manifest
from moneyrepair.simulate import make_synthetic_fragments
from moneyrepair.solver import CoverageSolution, solve_covering_sets
from moneyrepair.types import Fragment

__all__ = [
    "BatchState",
    "CompatibilityMatrix",
    "ConfirmedNote",
    "CoverageSolution",
    "Fragment",
    "MatrixFootprint",
    "PackedCompatibilityMatrix",
    "ReferenceScore",
    "SyntheticBenchmark",
    "compatibility_from_pair_records",
    "connected_components",
    "fragments_from_manifest",
    "estimate_matrix_footprint",
    "make_synthetic_fragments",
    "run_synthetic_benchmark",
    "segment_scan_to_manifest",
    "score_best_reference_side",
    "score_fragments_by_side",
    "solve_covering_sets",
    "update_manifest_labels",
]
