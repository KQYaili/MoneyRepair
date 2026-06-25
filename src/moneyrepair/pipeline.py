from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

from moneyrepair.compat import compute_compatibility_fast
from moneyrepair.quality import QualityThresholds, assess_fragments, summarize_quality
from moneyrepair.simulate import load_dataset
from moneyrepair.solver import solve_covering_sets
from moneyrepair.types import Fragment
from moneyrepair.visualize import render_solution_gallery, write_solution_report


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version() -> str:
    try:
        from moneyrepair import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def run_production_pipeline(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    target_coverage: float = 0.99,
    max_solutions: int = 10,
    order_strategy: str = "area_degree",
    time_limit_seconds: float | None = None,
    thresholds: QualityThresholds | None = None,
    drop_rejected_frames: bool = True,
    cell: int | None = None,
    max_overlap_pixels: int = 0,
    max_overlap_ratio: float = 0.0,
) -> dict:
    """Run one auditable production reconstruction batch.

    The pipeline gates frames on the acquisition contract, prunes with the
    grid-accelerated compatibility build, runs the branch-and-bound search, and
    writes a run manifest with input hashes, parameters, timings, the QA summary,
    and every output path so a batch can be reproduced and audited later.
    """

    thresholds = thresholds or QualityThresholds()
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}

    started = perf_counter()
    template, fragments = load_dataset(dataset_path)
    timings["load"] = perf_counter() - started

    started = perf_counter()
    quality_reports = assess_fragments(fragments, thresholds=thresholds, reference=template)
    quality_summary = summarize_quality(quality_reports, thresholds)
    timings["quality"] = perf_counter() - started

    passed_ids = {report.source for report in quality_reports if report.passed}
    if drop_rejected_frames:
        active: list[Fragment] = [fragment for fragment in fragments if fragment.id in passed_ids]
    else:
        active = list(fragments)
    dropped_ids = [fragment.id for fragment in fragments if fragment.id not in {item.id for item in active}]

    quality_path = output_dir / "quality_report.json"
    quality_path.write_text(
        json.dumps(
            {"summary": quality_summary, "frames": [report.to_dict() for report in quality_reports]},
            indent=2,
        ),
        encoding="utf-8",
    )

    started = perf_counter()
    matrix = compute_compatibility_fast(
        active,
        max_overlap_pixels=max_overlap_pixels,
        max_overlap_ratio=max_overlap_ratio,
        cell=cell,
    )
    matrix_path = output_dir / "matrix.npz"
    matrix.save(matrix_path)
    timings["build_matrix"] = perf_counter() - started

    started = perf_counter()
    solutions = solve_covering_sets(
        active,
        matrix,
        target_coverage=target_coverage,
        max_solutions=max_solutions,
        time_limit_seconds=time_limit_seconds,
        order_strategy=order_strategy,
    )
    timings["solve"] = perf_counter() - started
    timings["total"] = sum(value for key, value in timings.items() if key != "total")

    candidates_path = output_dir / "candidates.json"
    candidates_path.write_text(
        json.dumps(
            [
                {
                    "fragment_ids": list(solution.fragment_ids),
                    "coverage": solution.coverage,
                    "area": solution.area,
                }
                for solution in solutions
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    vis_dir = output_dir / "vis"
    report_path = output_dir / "report.html"
    image_paths = render_solution_gallery(template, active, solutions, vis_dir, limit=max_solutions)
    write_solution_report(solutions[:max_solutions], image_paths, report_path)

    manifest = {
        "tool": "moneyrepair",
        "version": _package_version(),
        "stage": "production_pipeline",
        "inputs": {
            "dataset": str(dataset_path),
            "dataset_sha256": _file_sha256(dataset_path),
            "fragments_total": len(fragments),
        },
        "parameters": {
            "target_coverage": target_coverage,
            "max_solutions": max_solutions,
            "order_strategy": order_strategy,
            "time_limit_seconds": time_limit_seconds,
            "drop_rejected_frames": drop_rejected_frames,
            "cell": cell,
            "max_overlap_pixels": max_overlap_pixels,
            "max_overlap_ratio": max_overlap_ratio,
            "thresholds": thresholds.to_dict(),
        },
        "quality": {
            "accepted": quality_summary["accepted"],
            "rejected": quality_summary["rejected"],
            "acceptance_rate": quality_summary["acceptance_rate"],
            "reason_counts": quality_summary["reason_counts"],
            "dropped_fragment_ids": dropped_ids,
        },
        "search": {
            "active_fragments": len(active),
            "solutions_found": len(solutions),
            "best_coverage": solutions[0].coverage if solutions else None,
        },
        "timings_seconds": timings,
        "outputs": {
            "matrix": str(matrix_path),
            "candidates": str(candidates_path),
            "quality_report": str(quality_path),
            "report": str(report_path),
            "visualizations": str(vis_dir),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["outputs"]["run_manifest"] = str(manifest_path)
    return manifest
