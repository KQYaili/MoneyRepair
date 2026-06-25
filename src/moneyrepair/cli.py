from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moneyrepair.batch import load_batch_state, save_batch_state
from moneyrepair.benchmark import (
    compare_solver_strategies,
    estimate_matrix_footprint,
    run_synthetic_benchmark,
    write_matrix_footprint,
    write_synthetic_benchmark,
)
from moneyrepair.compat import (
    PackedCompatibilityMatrix,
    compatibility_from_pair_records,
    compute_compatibility,
    filter_compatibility_to_ids,
    load_pair_records,
)
from moneyrepair.features import describe_contours, match_similar_contours
from moneyrepair.ingest import fragments_from_manifest, load_rgb
from moneyrepair.labels import parse_roi, update_manifest_labels
from moneyrepair.reference import load_references, load_score_thresholds, score_best_reference_side, score_fragments_by_side, scores_to_jsonable
from moneyrepair.realism import RealismProfile, make_realistic_synthetic_fragments
from moneyrepair.reports import load_strategy_results, write_strategy_report
from moneyrepair.scan import segment_scan_to_manifest
from moneyrepair.simulate import load_dataset, make_synthetic_fragments, save_dataset
from moneyrepair.solver import CoverageSolution, solve_covering_sets
from moneyrepair.visualize import render_solution_gallery, write_solution_report


def _cmd_simulate(args: argparse.Namespace) -> None:
    template, fragments = make_synthetic_fragments(
        pieces=args.pieces,
        width=args.width,
        height=args.height,
        seed=args.seed,
        side=args.side,
    )
    save_dataset(args.output, template, fragments)
    print(f"wrote {len(fragments)} fragments to {args.output}")


def _cmd_simulate_realistic(args: argparse.Namespace) -> None:
    profile = RealismProfile(
        brightness_jitter=args.brightness_jitter,
        contrast_jitter=args.contrast_jitter,
        color_jitter=args.color_jitter,
        blur_radius_max=args.blur_radius_max,
        noise_sigma=args.noise_sigma,
        illumination_strength=args.illumination_strength,
        jpeg_quality_min=args.jpeg_quality_min,
        jpeg_quality_max=args.jpeg_quality_max,
    )
    template, fragments, profile = make_realistic_synthetic_fragments(
        pieces=args.pieces,
        width=args.width,
        height=args.height,
        seed=args.seed,
        side=args.side,
        profile=profile,
    )
    save_dataset(args.output, template, fragments)
    if args.profile_output:
        Path(args.profile_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.profile_output).write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    print(f"wrote {len(fragments)} realistic fragments to {args.output}")


def _cmd_build_matrix(args: argparse.Namespace) -> None:
    _, fragments = load_dataset(args.dataset)
    matrix = compute_compatibility(
        fragments,
        max_overlap_pixels=args.max_overlap_pixels,
        max_overlap_ratio=args.max_overlap_ratio,
    )
    if args.reference_scores and args.max_reference_rmse is not None:
        allowed_ids = load_score_thresholds(args.reference_scores, max_rmse=args.max_reference_rmse)
        matrix = filter_compatibility_to_ids(matrix, allowed_ids)
    matrix.save(args.output)
    incompatible = int((~matrix.compatible).sum() - len(matrix.ids))
    print(f"wrote matrix for {len(matrix.ids)} fragments to {args.output}; incompatible_pairs={incompatible // 2}")


def _solutions_to_json(solutions) -> list[dict]:
    return [
        {
            "fragment_ids": list(solution.fragment_ids),
            "coverage": solution.coverage,
            "area": solution.area,
        }
        for solution in solutions
    ]


def _solutions_from_json(raw: list[dict]) -> list[CoverageSolution]:
    return [
        CoverageSolution(
            fragment_ids=tuple(str(fragment_id) for fragment_id in item["fragment_ids"]),
            coverage=float(item["coverage"]),
            area=int(item["area"]),
        )
        for item in raw
    ]


def _cmd_solve(args: argparse.Namespace) -> None:
    _, fragments = load_dataset(args.dataset)
    matrix = PackedCompatibilityMatrix.load(args.matrix)
    allowed_ids = None
    if args.allowed_ids:
        allowed_ids = {line.strip() for line in Path(args.allowed_ids).read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    solutions = solve_covering_sets(
        fragments,
        matrix,
        target_coverage=args.coverage,
        max_solutions=args.max_solutions,
        start_id=args.start_id,
        time_limit_seconds=args.time_limit,
        allowed_ids=allowed_ids,
        order_strategy=args.order_strategy,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_solutions_to_json(solutions), indent=2), encoding="utf-8")
    print(f"wrote {len(solutions)} solutions to {output}")


def _cmd_visualize(args: argparse.Namespace) -> None:
    template, fragments = load_dataset(args.dataset)
    solutions = _solutions_from_json(json.loads(Path(args.solutions).read_text(encoding="utf-8")))
    output_dir = Path(args.output_dir)
    image_paths = render_solution_gallery(template, fragments, solutions, output_dir, limit=args.limit)
    if args.report:
        write_solution_report(solutions[: args.limit], image_paths, args.report)
    print(f"wrote {len(image_paths)} visualization images to {output_dir}")


def _cmd_describe_contours(args: argparse.Namespace) -> None:
    _, fragments = load_dataset(args.dataset)
    records = describe_contours(fragments, direction_bins=args.direction_bins)
    payload = [
        {
            "fragment_id": record.fragment_id,
            "tags": list(record.tags),
            "boundary_points": record.boundary_points,
            "bbox": list(record.bbox),
            "direction_histogram": list(record.direction_histogram),
        }
        for record in records
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote contour descriptions for {len(records)} fragments to {output}")


def _cmd_match_contours(args: argparse.Namespace) -> None:
    _, fragments = load_dataset(args.dataset)
    matches = match_similar_contours(
        fragments,
        max_distance=args.max_distance,
        limit=args.limit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(matches, indent=2), encoding="utf-8")
    print(f"wrote {len(matches)} contour matches to {output}")


def _cmd_import_pairs(args: argparse.Namespace) -> None:
    _, fragments = load_dataset(args.dataset)
    pairs = load_pair_records(args.pairs)
    matrix = compatibility_from_pair_records(
        (fragment.id for fragment in fragments),
        pairs,
        relation=args.relation,
    )
    matrix.save(args.output)
    print(f"wrote packed matrix for {len(matrix.ids)} fragments from {len(pairs)} pair records to {args.output}")


def _write_batch_candidates(path: Path, solutions: list[CoverageSolution], active_count: int) -> None:
    payload = {
        "active_fragment_count": active_count,
        "solutions": _solutions_to_json(solutions),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_batch_candidates(path: str | Path) -> list[CoverageSolution]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload["solutions"] if isinstance(payload, dict) and "solutions" in payload else payload
    return _solutions_from_json(raw)


def _cmd_batch_next(args: argparse.Namespace) -> None:
    template, fragments = load_dataset(args.dataset)
    matrix = PackedCompatibilityMatrix.load(args.matrix)
    state = load_batch_state(args.state)
    allowed_ids = state.active_fragment_ids(fragments)
    solutions = solve_covering_sets(
        fragments,
        matrix,
        target_coverage=args.coverage,
        max_solutions=args.max_solutions,
        start_id=args.start_id,
        time_limit_seconds=args.time_limit,
        allowed_ids=allowed_ids,
    )
    solutions = state.filter_rejected(solutions)
    output_dir = Path(args.output_dir)
    candidates_path = output_dir / "candidates.json"
    vis_dir = output_dir / "vis"
    report_path = output_dir / "report.html"
    _write_batch_candidates(candidates_path, solutions, active_count=len(allowed_ids))
    image_paths = render_solution_gallery(template, fragments, solutions, vis_dir, limit=args.max_solutions)
    write_solution_report(solutions[: args.max_solutions], image_paths, report_path)
    print(f"active_fragments={len(allowed_ids)}")
    print(f"confirmed_notes={len(state.confirmed_notes)}")
    print(f"candidates={candidates_path}")
    print(f"report={report_path}")


def _cmd_batch_confirm(args: argparse.Namespace) -> None:
    state = load_batch_state(args.state)
    solutions = _read_batch_candidates(args.candidates)
    if args.index < 0 or args.index >= len(solutions):
        raise ValueError(f"candidate index {args.index} is out of range")
    solution = solutions[args.index]
    note_id = args.note_id or state.next_note_id(prefix=args.note_prefix)
    state.add_confirmation(note_id, solution)
    save_batch_state(args.state, state)
    print(f"confirmed {note_id} with {len(solution.fragment_ids)} fragments at coverage={solution.coverage:.4%}")


def _cmd_batch_reject(args: argparse.Namespace) -> None:
    state = load_batch_state(args.state)
    solutions = _read_batch_candidates(args.candidates)
    if args.index < 0 or args.index >= len(solutions):
        raise ValueError(f"candidate index {args.index} is out of range")
    solution = solutions[args.index]
    state.reject_solution(solution)
    save_batch_state(args.state, state)
    print(f"rejected candidate {args.index} with {len(solution.fragment_ids)} fragments")


def _manifest_note_template(args: argparse.Namespace) -> np.ndarray:
    if args.reference_front:
        return load_rgb(args.reference_front)
    if args.reference_back:
        return load_rgb(args.reference_back)
    raw = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    note = raw.get("note") or {}
    if "height" not in note or "width" not in note:
        raise ValueError("manifest needs note.height/note.width when no reference image is provided")
    return np.zeros((int(note["height"]), int(note["width"]), 3), dtype=np.uint8)


def _cmd_ingest_manifest(args: argparse.Namespace) -> None:
    template = _manifest_note_template(args)
    fragments = fragments_from_manifest(args.manifest, reference=template)
    save_dataset(args.output, template, fragments)
    print(f"wrote {len(fragments)} manifest fragments to {args.output}")


def _cmd_score_reference(args: argparse.Namespace) -> None:
    _, fragments = load_dataset(args.dataset)
    references = load_references(front=args.reference_front, back=args.reference_back)
    if args.best_side:
        scores = score_best_reference_side(fragments, references)
    else:
        scores = score_fragments_by_side(fragments, references)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores_to_jsonable(scores), indent=2), encoding="utf-8")
    print(f"wrote {len(scores)} reference scores to {output}")


def _cmd_segment_scan(args: argparse.Namespace) -> None:
    manifest = segment_scan_to_manifest(
        image_path=args.image,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        threshold=args.threshold,
        min_area=args.min_area,
        padding=args.padding,
        side=args.side,
        id_prefix=args.id_prefix,
        label_prefix=args.label_prefix,
        labels_file=args.labels_file,
        note_width=args.note_width,
        note_height=args.note_height,
        preserve_scan_coordinates=not args.origin_affine,
    )
    manifest_path = Path(args.manifest) if args.manifest else Path(args.output_dir) / "manifest.json"
    print(f"wrote {len(manifest['fragments'])} segmented fragments to {args.output_dir}")
    print(f"manifest={manifest_path}")


def _cmd_label_manifest(args: argparse.Namespace) -> None:
    manifest = update_manifest_labels(
        manifest_path=args.manifest,
        output_path=args.output,
        method=args.method,
        labels_file=args.labels_file,
        overwrite=args.overwrite,
        roi=parse_roi(args.roi),
        tesseract_config=args.tesseract_config,
    )
    output = Path(args.output) if args.output else Path(args.manifest)
    labeled = sum(1 for item in manifest.get("fragments", []) if item.get("label"))
    print(f"wrote labels for {labeled} fragments to {output}")


def _cmd_estimate_matrix(args: argparse.Namespace) -> None:
    if args.output:
        footprint = write_matrix_footprint(args.output, fragments=args.fragments)
    else:
        footprint = estimate_matrix_footprint(args.fragments)
    print(f"fragments={footprint.fragments}")
    print(f"dense_bool_mb={footprint.dense_bool_mb:.2f}")
    print(f"packed_mb={footprint.packed_mb:.2f}")
    if args.output:
        print(f"output={args.output}")


def _cmd_benchmark_synthetic(args: argparse.Namespace) -> None:
    kwargs = {
        "pieces": args.pieces,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "target_coverage": args.coverage,
        "max_solutions": args.max_solutions,
        "time_limit_seconds": args.time_limit,
        "order_strategy": args.order_strategy,
    }
    if args.output:
        result = write_synthetic_benchmark(args.output, **kwargs)
    else:
        result = run_synthetic_benchmark(**kwargs)
    payload = result.to_dict()
    print(json.dumps(payload, indent=2))
    if args.output:
        print(f"output={args.output}")


def _cmd_benchmark_strategies(args: argparse.Namespace) -> None:
    results = compare_solver_strategies(
        pieces=args.pieces,
        width=args.width,
        height=args.height,
        seed=args.seed,
        target_coverage=args.coverage,
        max_solutions=args.max_solutions,
        time_limit_seconds=args.time_limit,
        strategies=tuple(args.strategies.split(",")),
    )
    payload = [result.to_dict() for result in results]
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if args.output:
        print(f"output={args.output}")


def _cmd_report_strategies(args: argparse.Namespace) -> None:
    outputs = write_strategy_report(
        load_strategy_results(args.input),
        output_prefix=args.output_prefix,
        title=args.title,
        dpi=args.dpi,
    )
    print(json.dumps(outputs, indent=2))


def _cmd_smoke(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    dataset_path = output_dir / "demo_fragments.npz"
    matrix_path = output_dir / "demo_matrix.npz"
    solutions_path = output_dir / "solutions.json"
    vis_dir = output_dir / "vis"

    template, fragments = make_synthetic_fragments(
        pieces=args.pieces,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    save_dataset(dataset_path, template, fragments)
    matrix = compute_compatibility(fragments)
    matrix.save(matrix_path)
    solutions = solve_covering_sets(
        fragments,
        matrix,
        target_coverage=args.coverage,
        max_solutions=args.max_solutions,
        time_limit_seconds=args.time_limit,
    )
    solutions_path.parent.mkdir(parents=True, exist_ok=True)
    solutions_path.write_text(json.dumps(_solutions_to_json(solutions), indent=2), encoding="utf-8")
    image_paths = render_solution_gallery(template, fragments, solutions, vis_dir)
    write_solution_report(solutions, image_paths, output_dir / "report.html")
    print(f"dataset={dataset_path}")
    print(f"matrix={matrix_path}")
    print(f"solutions={solutions_path}")
    print(f"visualizations={vis_dir}")
    print(f"report={output_dir / 'report.html'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moneyrepair")
    sub = parser.add_subparsers(dest="command", required=True)

    simulate = sub.add_parser("simulate", help="generate a synthetic fragment dataset")
    simulate.add_argument("--output", required=True)
    simulate.add_argument("--pieces", type=int, default=24)
    simulate.add_argument("--width", type=int, default=420)
    simulate.add_argument("--height", type=int, default=180)
    simulate.add_argument("--seed", type=int, default=7)
    simulate.add_argument("--side", default="front")
    simulate.set_defaults(func=_cmd_simulate)

    realistic = sub.add_parser("simulate-realistic", help="generate photometrically degraded synthetic fragments")
    realistic.add_argument("--output", required=True)
    realistic.add_argument("--pieces", type=int, default=24)
    realistic.add_argument("--width", type=int, default=420)
    realistic.add_argument("--height", type=int, default=180)
    realistic.add_argument("--seed", type=int, default=7)
    realistic.add_argument("--side", default="front")
    realistic.add_argument("--brightness-jitter", type=float, default=0.16)
    realistic.add_argument("--contrast-jitter", type=float, default=0.12)
    realistic.add_argument("--color-jitter", type=float, default=0.08)
    realistic.add_argument("--blur-radius-max", type=float, default=0.8)
    realistic.add_argument("--noise-sigma", type=float, default=7.0)
    realistic.add_argument("--illumination-strength", type=float, default=0.22)
    realistic.add_argument("--jpeg-quality-min", type=int, default=72)
    realistic.add_argument("--jpeg-quality-max", type=int, default=94)
    realistic.add_argument("--profile-output")
    realistic.set_defaults(func=_cmd_simulate_realistic)

    matrix = sub.add_parser("build-matrix", help="build packed compatibility matrix")
    matrix.add_argument("--dataset", required=True)
    matrix.add_argument("--output", required=True)
    matrix.add_argument("--max-overlap-pixels", type=int, default=0)
    matrix.add_argument("--max-overlap-ratio", type=float, default=0.0)
    matrix.add_argument("--reference-scores")
    matrix.add_argument("--max-reference-rmse", type=float)
    matrix.set_defaults(func=_cmd_build_matrix)

    solve = sub.add_parser("solve", help="search compatible high-coverage sets")
    solve.add_argument("--dataset", required=True)
    solve.add_argument("--matrix", required=True)
    solve.add_argument("--output", required=True)
    solve.add_argument("--coverage", type=float, default=0.99)
    solve.add_argument("--max-solutions", type=int, default=20)
    solve.add_argument("--start-id")
    solve.add_argument("--time-limit", type=float)
    solve.add_argument("--allowed-ids")
    solve.add_argument("--order-strategy", choices=("area", "degree", "area_degree"), default="area")
    solve.set_defaults(func=_cmd_solve)

    visualize = sub.add_parser("visualize", help="render solution overlays")
    visualize.add_argument("--dataset", required=True)
    visualize.add_argument("--solutions", required=True)
    visualize.add_argument("--output-dir", required=True)
    visualize.add_argument("--limit", type=int, default=20)
    visualize.add_argument("--report")
    visualize.set_defaults(func=_cmd_visualize)

    describe = sub.add_parser("describe-contours", help="extract contour tags and direction features")
    describe.add_argument("--dataset", required=True)
    describe.add_argument("--output", required=True)
    describe.add_argument("--direction-bins", type=int, default=8)
    describe.set_defaults(func=_cmd_describe_contours)

    match = sub.add_parser("match-contours", help="match similar contours after tag filtering")
    match.add_argument("--dataset", required=True)
    match.add_argument("--output", required=True)
    match.add_argument("--max-distance", type=float, default=0.25)
    match.add_argument("--limit", type=int, default=100)
    match.set_defaults(func=_cmd_match_contours)

    import_pairs = sub.add_parser("import-pairs", help="import precomputed compatible or incompatible pair records")
    import_pairs.add_argument("--dataset", required=True)
    import_pairs.add_argument("--pairs", required=True)
    import_pairs.add_argument("--output", required=True)
    import_pairs.add_argument("--relation", choices=("compatible", "incompatible"), default="incompatible")
    import_pairs.set_defaults(func=_cmd_import_pairs)

    batch_next = sub.add_parser("batch-next", help="search the next unconfirmed note and write an inspection report")
    batch_next.add_argument("--dataset", required=True)
    batch_next.add_argument("--matrix", required=True)
    batch_next.add_argument("--state", required=True)
    batch_next.add_argument("--output-dir", required=True)
    batch_next.add_argument("--coverage", type=float, default=0.99)
    batch_next.add_argument("--max-solutions", type=int, default=20)
    batch_next.add_argument("--start-id")
    batch_next.add_argument("--time-limit", type=float)
    batch_next.set_defaults(func=_cmd_batch_next)

    batch_confirm = sub.add_parser("batch-confirm", help="confirm a batch candidate and remove its fragments from future searches")
    batch_confirm.add_argument("--state", required=True)
    batch_confirm.add_argument("--candidates", required=True)
    batch_confirm.add_argument("--index", type=int, default=0)
    batch_confirm.add_argument("--note-id")
    batch_confirm.add_argument("--note-prefix", default="note")
    batch_confirm.set_defaults(func=_cmd_batch_confirm)

    batch_reject = sub.add_parser("batch-reject", help="remember a bad candidate so batch-next can skip it")
    batch_reject.add_argument("--state", required=True)
    batch_reject.add_argument("--candidates", required=True)
    batch_reject.add_argument("--index", type=int, default=0)
    batch_reject.set_defaults(func=_cmd_batch_reject)

    ingest = sub.add_parser("ingest-manifest", help="place real fragment images from a JSON manifest")
    ingest.add_argument("--manifest", required=True)
    ingest.add_argument("--output", required=True)
    ingest.add_argument("--reference-front")
    ingest.add_argument("--reference-back")
    ingest.set_defaults(func=_cmd_ingest_manifest)

    score = sub.add_parser("score-reference", help="compare placed fragment RGB pixels with reference note images")
    score.add_argument("--dataset", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--reference-front")
    score.add_argument("--reference-back")
    score.add_argument("--best-side", action="store_true")
    score.set_defaults(func=_cmd_score_reference)

    segment = sub.add_parser("segment-scan", help="split one scan/photo into fragment crops and a manifest")
    segment.add_argument("--image", required=True)
    segment.add_argument("--output-dir", required=True)
    segment.add_argument("--manifest")
    segment.add_argument("--threshold", type=float, default=22.0)
    segment.add_argument("--min-area", type=int, default=50)
    segment.add_argument("--padding", type=int, default=4)
    segment.add_argument("--side", default="front")
    segment.add_argument("--id-prefix", default="f")
    segment.add_argument("--label-prefix", default="")
    segment.add_argument("--labels-file")
    segment.add_argument("--note-width", type=int)
    segment.add_argument("--note-height", type=int)
    segment.add_argument("--origin-affine", action="store_true")
    segment.set_defaults(func=_cmd_segment_scan)

    label = sub.add_parser("label-manifest", help="fill manifest labels from CSV, filenames, ids, or optional OCR")
    label.add_argument("--manifest", required=True)
    label.add_argument("--output")
    label.add_argument("--method", choices=("filename", "id", "ocr"), default="filename")
    label.add_argument("--labels-file")
    label.add_argument("--overwrite", action="store_true")
    label.add_argument("--roi", help="OCR ROI as x0,y0,x1,y1; values <=1 are fractions of each crop")
    label.add_argument("--tesseract-config")
    label.set_defaults(func=_cmd_label_manifest)

    estimate = sub.add_parser("estimate-matrix", help="estimate dense vs packed compatibility matrix memory")
    estimate.add_argument("--fragments", type=int, required=True)
    estimate.add_argument("--output")
    estimate.set_defaults(func=_cmd_estimate_matrix)

    bench = sub.add_parser("benchmark-synthetic", help="time synthetic simulate/matrix/solve stages")
    bench.add_argument("--pieces", type=int, default=80)
    bench.add_argument("--width", type=int, default=480)
    bench.add_argument("--height", type=int, default=210)
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--coverage", type=float, default=0.98)
    bench.add_argument("--max-solutions", type=int, default=5)
    bench.add_argument("--time-limit", type=float, default=30.0)
    bench.add_argument("--order-strategy", choices=("area", "degree", "area_degree"), default="area")
    bench.add_argument("--output")
    bench.set_defaults(func=_cmd_benchmark_synthetic)

    bench_strategies = sub.add_parser("benchmark-strategies", help="compare DFS ordering strategies on one synthetic benchmark")
    bench_strategies.add_argument("--pieces", type=int, default=80)
    bench_strategies.add_argument("--width", type=int, default=480)
    bench_strategies.add_argument("--height", type=int, default=210)
    bench_strategies.add_argument("--seed", type=int, default=7)
    bench_strategies.add_argument("--coverage", type=float, default=0.98)
    bench_strategies.add_argument("--max-solutions", type=int, default=5)
    bench_strategies.add_argument("--time-limit", type=float, default=30.0)
    bench_strategies.add_argument("--strategies", default="area,degree,area_degree")
    bench_strategies.add_argument("--output")
    bench_strategies.set_defaults(func=_cmd_benchmark_strategies)

    report_strategies = sub.add_parser("report-strategies", help="render SVG/PDF/TIFF report from strategy benchmark JSON")
    report_strategies.add_argument("--input", required=True)
    report_strategies.add_argument("--output-prefix", required=True)
    report_strategies.add_argument("--title", default="MoneyRepair strategy benchmark")
    report_strategies.add_argument("--dpi", type=int, default=600)
    report_strategies.set_defaults(func=_cmd_report_strategies)

    smoke = sub.add_parser("smoke", help="run the full synthetic pipeline")
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--pieces", type=int, default=18)
    smoke.add_argument("--width", type=int, default=360)
    smoke.add_argument("--height", type=int, default=160)
    smoke.add_argument("--seed", type=int, default=7)
    smoke.add_argument("--coverage", type=float, default=0.98)
    smoke.add_argument("--max-solutions", type=int, default=5)
    smoke.add_argument("--time-limit", type=float, default=15.0)
    smoke.set_defaults(func=_cmd_smoke)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
