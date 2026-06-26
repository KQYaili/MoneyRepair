from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

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
    auto_locate: bool = False,
    reference_front: str | Path | np.ndarray | None = None,
    reference_back: str | Path | np.ndarray | None = None,
    precise_bound_threshold: int = 24,
    score_margin: float | None = None,
    min_score: float | None = None,
    max_boundary_diff: float = -1.0,
    discriminate_appearance: bool = False,
    discriminate_tolerance: float = 0.05,
    touch_priority: bool = True,
    include_interlock: bool = False,
    min_interlock_contact: int = 8,
    min_interlock_ratio: float = 0.03,
) -> dict:
    """Run one auditable production reconstruction batch.

    The pipeline gates frames on the acquisition contract, prunes with the
    grid-accelerated compatibility build, optionally applies interlock geometry
    constraints, runs the branch-and-bound search, and writes a run manifest
    with input hashes, parameters, timings, the QA summary, and every output
    path so a batch can be reproduced and audited later.
    """

    thresholds = thresholds or QualityThresholds()
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}

    started = perf_counter()
    template, fragments = load_dataset(dataset_path)
    timings["load"] = perf_counter() - started
    if auto_locate and discriminate_appearance:
        template_shape = template.shape[:2]
        if any(fragment.mask.shape != template_shape for fragment in fragments):
            raise ValueError(
                "--discriminate-appearance currently requires fragments already in template coordinates; "
                "disable it for raw auto-locate crops or compute appearance after pose placement."
            )

    # ====== JIT Warm-up ======
    started_warmup = perf_counter()
    from moneyrepair.solver import sum_candidate_areas
    sum_candidate_areas(np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64))
    if auto_locate:
        from moneyrepair.locator import locate_fragment_poses
        dummy_template = np.zeros((4, 4, 3), dtype=np.uint8)
        dummy_frag = Fragment(
            id="dummy",
            mask=np.ones((2, 2), dtype=bool),
            image=np.zeros((2, 2, 3), dtype=np.uint8),
        )
        locate_fragment_poses(dummy_frag, dummy_template, dummy_template, top_k=1, coarse_step=2)
    timings["jit_warmup"] = perf_counter() - started_warmup

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
    if auto_locate:
        from moneyrepair.locator import locate_fragment_poses, build_pose_compatibility_matrix, _crop_foreground, _rotate_image_and_mask
        
        ref_front = template
        ref_back = template
        if reference_front is not None:
            if isinstance(reference_front, (str, Path)):
                from moneyrepair.ingest import load_rgb
                ref_front = load_rgb(reference_front)
            else:
                ref_front = reference_front
        if reference_back is not None:
            if isinstance(reference_back, (str, Path)):
                from moneyrepair.ingest import load_rgb
                ref_back = load_rgb(reference_back)
            else:
                ref_back = reference_back

        placed_fragments: list[Fragment] = []
        h, w = template.shape[:2]
        pose_candidates_dict = {}
        for fragment in active:
            # Estimate candidate poses using real front/back references
            poses = locate_fragment_poses(
                fragment, ref_front, ref_back,
                top_k=3,
                score_margin=score_margin,
                min_score=min_score,
            )
            pose_candidates_dict[fragment.id] = [
                {
                    "pose_id": p.pose_id,
                    "side": p.side,
                    "tx": int(p.tx),
                    "ty": int(p.ty),
                    "angle": int(p.angle),
                    "score": round(float(p.score), 4)
                }
                for p in poses
            ]
            
            # Place fragment
            for p in poses:
                raw_crop_img, raw_crop_mask = _crop_foreground(fragment.image, fragment.mask)
                crop_img, crop_mask = _rotate_image_and_mask(raw_crop_img, raw_crop_mask, p.angle)
                
                placed_mask = np.zeros((h, w), dtype=bool)
                ch, cw = crop_mask.shape[:2]
                placed_mask[p.ty : p.ty + ch, p.tx : p.tx + cw] = crop_mask
                
                placed_img = np.zeros((h, w, 3), dtype=np.uint8)
                target_region = placed_img[p.ty : p.ty + ch, p.tx : p.tx + cw]
                # Filter background: only copy pixels where crop_mask is True
                target_region[crop_mask] = crop_img[crop_mask]
                
                placed_fragments.append(
                    Fragment(
                        id=p.pose_id,
                        mask=placed_mask,
                        label=fragment.label,
                        side=p.side,
                        image=placed_img,
                        meta={
                            "original_id": fragment.id,
                            "pose_id": p.pose_id,
                            "side": p.side,
                            "tx": p.tx,
                            "ty": p.ty,
                            "angle": p.angle,
                        }
                    )
                )
        
        active_search = placed_fragments
        pose_candidates_path = output_dir / "pose_candidates.json"
        pose_candidates_path.write_text(json.dumps(pose_candidates_dict, indent=2), encoding="utf-8")
        # Compute appearance clusters if requested
        groups = None
        if discriminate_appearance:
            from moneyrepair.fingerprint import cluster_fragments_by_appearance
            groups = cluster_fragments_by_appearance(active, template, tolerance=discriminate_tolerance)

        matrix = build_pose_compatibility_matrix(
            active_search,
            cell=cell,
            groups=groups,
            max_overlap_pixels=max_overlap_pixels,
            max_overlap_ratio=max_overlap_ratio,
            max_boundary_diff=max_boundary_diff,
        )
    else:
        active_search = active
        matrix = compute_compatibility_fast(
            active_search,
            max_overlap_pixels=max_overlap_pixels,
            max_overlap_ratio=max_overlap_ratio,
            cell=cell,
        )
    interlock_stats = None
    if include_interlock:
        from moneyrepair.compat import CompatibilityMatrix, PackedCompatibilityMatrix
        if isinstance(matrix, CompatibilityMatrix):
            matrix = PackedCompatibilityMatrix.from_dense(matrix)
        from moneyrepair.interlock import apply_interlock_constraints_with_stats
        matrix, interlock_stats = apply_interlock_constraints_with_stats(
            matrix,
            active_search,
            cell=cell,
            min_contact_edges=min_interlock_contact,
            min_contact_ratio=min_interlock_ratio,
        )

    matrix_path = output_dir / "matrix.npz"
    matrix.save(matrix_path)
    timings["build_matrix"] = perf_counter() - started

    started = perf_counter()
    solutions = solve_covering_sets(
        active_search,
        matrix,
        target_coverage=target_coverage,
        max_solutions=max_solutions,
        time_limit_seconds=time_limit_seconds,
        order_strategy=order_strategy,
        precise_bound_threshold=precise_bound_threshold,
        touch_priority=touch_priority,
    )
    timings["solve"] = perf_counter() - started
    timings["total"] = sum(value for key, value in timings.items() if key not in ("total", "total_without_jit_warmup"))
    timings["total_without_jit_warmup"] = timings["total"] - timings.get("jit_warmup", 0.0)

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
    image_paths = render_solution_gallery(template, active_search, solutions, vis_dir, limit=max_solutions)
    write_solution_report(solutions[:max_solutions], image_paths, report_path)

    ref_info = {}
    if reference_front is not None:
        if isinstance(reference_front, (str, Path)):
            ref_info["front"] = str(reference_front)
            ref_info["front_sha256"] = _file_sha256(reference_front)
        else:
            ref_info["front"] = "numpy_array"
            ref_info["front_sha256"] = hashlib.sha256(np.ascontiguousarray(reference_front).tobytes()).hexdigest()
    if reference_back is not None:
        if isinstance(reference_back, (str, Path)):
            ref_info["back"] = str(reference_back)
            ref_info["back_sha256"] = _file_sha256(reference_back)
        else:
            ref_info["back"] = "numpy_array"
            ref_info["back_sha256"] = hashlib.sha256(np.ascontiguousarray(reference_back).tobytes()).hexdigest()

    manifest_outputs = {
        "matrix": str(matrix_path),
        "candidates": str(candidates_path),
        "quality_report": str(quality_path),
        "report": str(report_path),
        "visualizations": str(vis_dir),
    }
    if auto_locate:
        manifest_outputs["pose_candidates"] = str(pose_candidates_path)

    manifest = {
        "tool": "moneyrepair",
        "version": _package_version(),
        "stage": "production_pipeline",
        "inputs": {
            "dataset": str(dataset_path),
            "dataset_sha256": _file_sha256(dataset_path),
            "fragments_total": len(fragments),
            "references": ref_info,
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
            "auto_locate": auto_locate,
            "thresholds": thresholds.to_dict(),
            "precise_bound_threshold": precise_bound_threshold,
            "score_margin": score_margin,
            "min_score": min_score,
            "max_boundary_diff": max_boundary_diff,
            "discriminate_appearance": discriminate_appearance,
            "discriminate_tolerance": discriminate_tolerance,
            "touch_priority": touch_priority,
            "include_interlock": include_interlock,
            "min_interlock_contact": min_interlock_contact,
            "min_interlock_ratio": min_interlock_ratio,
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
            "interlock_stats": {
                "bbox_candidate_pairs": interlock_stats.bbox_candidate_pairs,
                "scored_contact_pairs": interlock_stats.scored_contact_pairs,
                "rejected_pairs": interlock_stats.rejected_pairs,
            } if interlock_stats is not None else None,
        },
        "timings_seconds": timings,
        "outputs": manifest_outputs,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["outputs"]["run_manifest"] = str(manifest_path)
    return manifest
