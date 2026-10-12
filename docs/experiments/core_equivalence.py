"""Item-by-item equivalence audit of two tear-fit core source trees.

Runs the same simulated v4.4.1 reconstruction with a reference tree (for
example the commit before a performance change) and a candidate tree, each in
its own process, then compares every decision the core makes: scored pairs and
their evidence, accepted core edges, core and gap candidates with their
sources, oracle candidate recall, selected assemblies and objective,
fingerprints, and search counters. Wall time and peak RSS are reported for both
sides and never compared. Measurement only; it changes no core behaviour.

Wall-clock limits stay off (the v4.4.1 preset already disables them), so the
search trajectory depends only on the fixed state and node budgets.
``--override KEY=VALUE`` changes a simulation or trial setting for every case.
``--direct-suite`` compares bit-level digests of the public core functions
under parameters that reconstruction never uses: serial labels and pose
uncertainty, another tolerance, pair-overlap and group-overlap limits, pair and
candidate constraints, seed restrictions, and node limits.

Examples:

    # Frozen N=100 benchmark, both arms
    python docs/experiments/core_equivalence.py --reference-ref da354f0

    # A labelled pool, then the direct-call suite
    python docs/experiments/core_equivalence.py --reference-ref da354f0 \\
        --case 8:16:9:global --override serial_ocr_rate=0.6 --override use_labels=true
    python docs/experiments/core_equivalence.py --reference-ref da354f0 --direct-suite
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, is_dataclass, replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import pickle
import platform
import resource
import subprocess
import sys
import tarfile
import tempfile
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = ("100:24:7:global", "100:24:7:disjoint_round_robin")
FLOAT_FIELDS = {
    "scored_pairs": (
        "overlap_ratio",
        "continuity_ratio",
        "bidirectional_balance",
        "normal_opposition",
        "curvature_entropy",
        "expected_accidental_hits",
        "pose_uncertainty",
        "effectiveness",
    ),
    "candidate": (
        "coverage",
        "raw_coverage",
        "score",
        "base_score",
        "constraint_bonus",
        "evidence_score",
    ),
}


def parse_case(text: str) -> dict:
    notes, pieces, seed, strategy = text.split(":")
    return {
        "notes": int(notes),
        "pieces": int(pieces),
        "seed": int(seed),
        "strategy": strategy,
    }


def parse_override(text: str) -> tuple[str, object]:
    """``KEY=VALUE`` with VALUE read as a bool, None, int, float or string."""

    key, separator, raw = text.partition("=")
    if not key or not separator:
        raise ValueError(f"expected KEY=VALUE, got {text!r}")
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return key, lowered == "true"
    if lowered in ("none", "null"):
        return key, None
    for cast in (int, float):
        try:
            return key, cast(raw)
        except ValueError:
            pass
    return key, raw


def case_name(case: dict) -> str:
    name = f"n{case['notes']}_p{case['pieces']}_s{case['seed']}_{case['strategy']}"
    overrides = case.get("overrides") or {}
    if overrides:
        name += "+" + "+".join(f"{key}={value}" for key, value in sorted(overrides.items()))
    return name


def peak_rss_mb() -> float:
    """Peak resident memory of this process image, in MiB.

    ``ru_maxrss`` survives ``execve``, so a worker started by a large parent
    would report the parent's size; Linux's VmHWM starts afresh with the image.
    """

    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def tree_digest(src: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(src.rglob("*.py")):
        digest.update(str(path.relative_to(src)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def export_ref(ref: str, destination: Path) -> Path:
    """Export ``src/`` of a git ref with ``git archive``; returns the src path."""

    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", "--format=tar", ref, "src"],
        check=True,
        capture_output=True,
    ).stdout
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as handle:
        handle.write(archive)
        handle.seek(0)
        with tarfile.open(fileobj=handle) as tar:
            try:
                tar.extractall(destination, filter="data")
            except TypeError:  # Python releases without extraction filters
                tar.extractall(destination)
    return destination / "src"


def _import_core(src: Path):
    sys.path.insert(0, str(src))
    import moneyrepair
    from moneyrepair import tearfit

    if not Path(moneyrepair.__file__).resolve().is_relative_to(src.resolve()):
        raise SystemExit(f"imported {moneyrepair.__file__}, expected a module under {src}")
    return tearfit


def _environment(src: Path) -> dict:
    import numpy

    return {
        "src": str(src),
        "tree": tree_digest(src),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "machine": platform.machine(),
        "numpy": numpy.__version__,
    }


def _candidate_item(candidate, source: str | None = None) -> dict:
    item = {
        "fragment_ids": tuple(candidate.fragment_ids),
        "labels": tuple(candidate.labels),
        "evidence_level": candidate.evidence_level,
        "gap_steps": candidate.gap_steps,
        "support_pixels": candidate.support_pixels,
    }
    if source is not None:
        item["source"] = source
    for name in FLOAT_FIELDS["candidate"]:
        item[name] = getattr(candidate, name)
    return item


def _flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            _flatten(f"{prefix}.{key}" if prefix else str(key), value[key], out)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _flatten(f"{prefix}.{field.name}", getattr(value, field.name), out)
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value


def run_worker(src: Path, case: dict, dump: Path) -> None:
    """Run one case in this process and pickle every compared item."""

    tearfit = _import_core(src)
    captured: list = []
    original = tearfit.reconstruct_placed_fragments

    def capture(*args, **kwargs):
        reconstruction = original(*args, **kwargs)
        captured.append((args[0], reconstruction))
        return reconstruction

    tearfit.reconstruct_placed_fragments = capture
    simulation_kwargs: dict = {
        "notes": case["notes"],
        "pieces_per_note": case["pieces"],
        "seed": case["seed"],
        "serial_ocr_rate": 0.0,
    }
    trial_kwargs = asdict(tearfit.TearFitCoreConfig.v4_4_1())
    del trial_kwargs["fray_layers"]
    trial_kwargs["base_selection_strategy"] = case["strategy"]
    simulation_fields = {field.name for field in fields(tearfit.FractalTearConfig)}
    trial_parameters = set(inspect.signature(tearfit.run_tearfit_trial).parameters) - {"config"}
    for key, value in (case.get("overrides") or {}).items():
        if key in simulation_fields:
            simulation_kwargs[key] = value
        elif key in trial_parameters:
            trial_kwargs[key] = value
        else:
            raise SystemExit(f"unknown override {key!r}")
    simulation = tearfit.FractalTearConfig(**simulation_kwargs)
    started = monotonic()
    result = tearfit.run_tearfit_trial(simulation, **trial_kwargs)
    wall = monotonic() - started
    peak_rss = peak_rss_mb()
    fragments, reconstruction = captured[-1]
    ids = [fragment.id for fragment in fragments]

    scored_pairs = []
    for edge in reconstruction.all_scores:
        item = {
            "pair": (ids[edge.left], ids[edge.right]),
            "evidence_level": edge.evidence_level,
            "overlap_pixels": edge.overlap_pixels,
            "left_hits": edge.left_hits,
            "right_hits": edge.right_hits,
            "contiguous_pixels": edge.contiguous_pixels,
            "mask_overlap_pixels": edge.mask_overlap_pixels,
        }
        for name in FLOAT_FIELDS["scored_pairs"]:
            item[name] = getattr(edge, name)
        scored_pairs.append(item)
    accepted_edges = [
        {
            "pair": (ids[edge.left], ids[edge.right]),
            "evidence_level": edge.evidence_level,
            "effectiveness": edge.effectiveness,
        }
        for edge in reconstruction.core_edges
    ]
    pool = [
        _candidate_item(candidate, reconstruction.candidate_source(candidate.fragment_ids))
        for candidate in reconstruction.candidates
    ]
    selected = [_candidate_item(candidate) for candidate in reconstruction.selected]
    objective_score = 0.0
    for candidate in reconstruction.selected:
        objective_score += candidate.score
    search_counters: dict = {}
    _flatten("", reconstruction.search_stats, search_counters)
    trial_metrics: dict = {}
    for name in (
        "fragments",
        "pair_scores",
        "accepted_edges",
        "true_possible_edges",
        "true_accepted_edges",
        "false_accepted_edges",
        "candidates",
        "core_candidates",
        "partial_core_candidates",
        "gap_candidates",
        "partial_gap_candidates",
        "selected_core_candidates",
        "selected_gap_candidates",
        "true_gap_candidates",
        "false_gap_candidates",
        "oracle_candidate_notes",
        "oracle_candidate_recall",
        "selected_score_total",
    ):
        trial_metrics[name] = getattr(result, name)
    _flatten("diagnostics", result.diagnostics, trial_metrics)
    payload = {
        "case": case,
        **_environment(src),
        "wall_seconds": wall,
        "peak_rss_mb": peak_rss,
        "stage_timings": dict(result.stage_timings),
        "items": {
            "scored_pairs": scored_pairs,
            "edge_decisions": dict(reconstruction.edge_decisions),
            "accepted_edges": accepted_edges,
            "core_candidates": [_candidate_item(c) for c in reconstruction.generated_candidates],
            "candidate_pool": pool,
            "selected": selected,
            "objective": {"count": len(selected), "score": objective_score},
            "fingerprints": {
                "selected_solution": reconstruction.selected_solution_fingerprint,
                "candidate_provenance": reconstruction.candidate_provenance_fingerprint,
            },
            "search_counters": search_counters,
            "trial_metrics": trial_metrics,
        },
    }
    dump.parent.mkdir(parents=True, exist_ok=True)
    with dump.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def canonical(value):
    """A comparable form that keeps every bit: floats as hex, dataclasses by field."""

    import numpy as np

    if is_dataclass(value) and not isinstance(value, type):
        return tuple((field.name, canonical(getattr(value, field.name))) for field in fields(value))
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), canonical(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(canonical(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(repr(canonical(item)) for item in value))
    return repr(value)


def direct_suite(tearfit) -> dict[str, object]:
    """Outputs of the public core functions under parameters reconstruction never uses."""

    results: dict[str, object] = {}

    def with_pose_uncertainty(fragments):
        noisy = []
        for index, fragment in enumerate(fragments):
            if index % 3 == 0:
                meta = dict(fragment.meta)
                meta.update(sigma_x=0.4 + 0.1 * (index % 5), sigma_y=0.3, sigma_theta=0.5)
                fragment = replace(fragment, meta=meta)
            noisy.append(fragment)
        return noisy

    score = tearfit.score_absolute_tear_pairs
    generate = tearfit.generate_assembly_candidates
    augment = tearfit.augment_candidates_with_group_gap
    cover = tearfit.select_exact_cover_candidates
    for notes, pieces, seed, ocr in ((4, 8, 1, 0.6), (6, 16, 2, 0.0), (8, 24, 3, 0.6), (10, 24, 7, 0.0)):
        _template, clean = tearfit.make_fractal_tear_fragments(
            tearfit.FractalTearConfig(notes=notes, pieces_per_note=pieces, seed=seed, serial_ocr_rate=ocr)
        )
        fragments = with_pose_uncertainty(clean)
        tag = f"n{notes}_p{pieces}_s{seed}_ocr{ocr}"
        for scoring in ("overlap", "effectiveness"):
            for use_labels in (False, True):
                results[f"pairs/{tag}/{scoring}/labels={use_labels}"] = score(
                    clean, scoring=scoring, use_labels=use_labels, min_overlap_pixels=4
                )
        results[f"pairs/{tag}/pose_uncertainty"] = score(fragments, scoring="effectiveness")
        results[f"pairs/{tag}/tolerance=3"] = score(clean, scoring="effectiveness", tolerance=3)
        results[f"pairs/{tag}/no_pair_overlap_limit"] = score(
            clean, scoring="effectiveness", max_pair_overlap_pixels=-1
        )
        results[f"pairs/{tag}/pair_overlap_limit=40"] = score(
            clean, scoring="effectiveness", max_pair_overlap_pixels=40
        )

        all_scores, edges = score(fragments, scoring="effectiveness", use_labels=True)
        automatic = [edge for edge in edges if edge.evidence_level == "automatic"]
        ids = [fragment.id for fragment in fragments]
        pairs = [(ids[edge.left], ids[edge.right]) for edge in automatic]

        def run_generate(name: str, **kwargs) -> list:
            stats: dict = {}
            output = generate(
                fragments,
                automatic,
                max_pieces=pieces + 2,
                time_limit_seconds=None,
                search_stats=stats,
                **kwargs,
            )
            results[f"core/{tag}/{name}"] = (output, stats)
            return output

        cores = run_generate(
            "beam=32", beam_width=32, max_expanded_states=4000, minimum_candidate_raw_coverage=0.78
        )
        run_generate("beam=8", beam_width=8, max_expanded_states=4000, minimum_candidate_raw_coverage=0.78)
        run_generate(
            "group_gap",
            gap_edges=all_scores,
            enable_group_gap=True,
            beam_width=16,
            max_expanded_states=2500,
        )
        run_generate("group_overlap_limit=40", beam_width=16, max_expanded_states=2500, max_group_overlap_pixels=40)
        run_generate(
            "pair_constraints",
            beam_width=16,
            max_expanded_states=2500,
            seed_strategy="all",
            forbidden_pairs=pairs[0:3],
            preferred_pairs=pairs[3:6],
            forced_pairs=pairs[6:8],
            forbidden_candidates=[candidate.fragment_ids for candidate in cores[:3]],
        )
        run_generate("seed_whitelist", beam_width=16, max_expanded_states=2500, seed_whitelist=ids[::2])
        if ocr:
            labels = sorted({fragment.label for fragment in fragments if fragment.label})
            run_generate("seed_labels", beam_width=16, max_expanded_states=2500, seed_labels=labels[::2])

        complete = [candidate for candidate in cores if candidate.coverage >= 0.93]
        partial = [candidate for candidate in cores if candidate.coverage < 0.93]
        for strategy in ("global", "disjoint_round_robin"):
            for pool_name in ("weak_pair", "boundary_contact"):
                for base_name, base in (("complete", complete), ("partial", partial)):
                    stats = {}
                    output = augment(
                        fragments,
                        base,
                        all_scores,
                        max_pieces=pieces + 2,
                        max_base_candidates=24,
                        base_selection_strategy=strategy,
                        proposal_pool=pool_name,
                        time_limit_seconds=None,
                        max_expanded_states=3000,
                        search_stats=stats,
                    )
                    results[f"gap/{tag}/{strategy}/{pool_name}/{base_name}"] = (output, stats)
        stats = {}
        output = augment(
            fragments,
            partial,
            all_scores,
            max_pieces=pieces + 2,
            max_base_candidates=24,
            max_group_overlap_pixels=40,
            time_limit_seconds=None,
            max_expanded_states=3000,
            search_stats=stats,
        )
        results[f"gap/{tag}/group_overlap_limit=40"] = (output, stats)

        evidence = tearfit.tear_boundary_evidence(fragments, tolerance=2)
        lookup = {(min(edge.left, edge.right), max(edge.left, edge.right)): edge for edge in all_scores}
        index_of = {fragment.id: index for index, fragment in enumerate(fragments)}
        scored = []
        for candidate in cores[:12]:
            state = frozenset(index_of[fragment_id] for fragment_id in candidate.fragment_ids)
            union = fragments[0].mask.copy()
            union[:] = False
            for index in state:
                union |= fragments[index].mask
            for other in range(0, len(fragments), 3):
                if other not in state:
                    scored.append(
                        tearfit.score_fragment_against_assembly(
                            fragments, state, union, other, lookup, evidence
                        )
                    )
        results[f"assembly_scores/{tag}"] = scored

        pool = sorted(
            {candidate.fragment_ids: candidate for candidate in cores}.values(),
            key=lambda candidate: (-candidate.score, candidate.fragment_ids),
        )
        for objective in ("score_then_count", "count_then_score"):
            for limit in (50, 5000, 60000):
                stats = {}
                results[f"cover/{tag}/{objective}/nodes={limit}"] = (
                    cover(
                        pool,
                        objective=objective,
                        time_limit_seconds=None,
                        max_search_nodes=limit,
                        search_stats=stats,
                    ),
                    stats,
                )
        if len(pool) >= 6:
            stats = {}
            results[f"cover/{tag}/candidate_constraints"] = (
                cover(
                    pool,
                    time_limit_seconds=None,
                    forbidden_candidates=[pool[1].fragment_ids],
                    locked_candidates=[pool[0].fragment_ids],
                    preferred_candidates=[pool[5].fragment_ids],
                    max_search_nodes=20000,
                    search_stats=stats,
                ),
                stats,
            )
    return results


def run_direct_worker(src: Path, dump: Path) -> None:
    tearfit = _import_core(src)
    started = monotonic()
    results = direct_suite(tearfit)
    wall = monotonic() - started
    suite = {}
    for name, value in results.items():
        output = value[0] if isinstance(value, tuple) else value
        suite[name] = {
            "digest": hashlib.sha256(repr(canonical(value)).encode()).hexdigest(),
            "items": len(output) if isinstance(output, (list, tuple)) else 1,
        }
    payload = {
        "suite": suite,
        **_environment(src),
        "wall_seconds": wall,
        "peak_rss_mb": peak_rss_mb(),
    }
    dump.parent.mkdir(parents=True, exist_ok=True)
    with dump.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _same_value(left, right, tolerance: float) -> tuple[bool, float]:
    """Exact (including the sign of zero) unless a tolerance is declared."""

    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right), 0.0
        difference = abs(left - right)
        if tolerance == 0.0:
            same = left == right and math.copysign(1.0, left) == math.copysign(1.0, right)
            return same, difference
        return difference <= tolerance, difference
    return left == right, 0.0


def _compare_records(left, right, tolerance: float) -> dict:
    """Compare two values: lists in order, dicts by key, scalars directly."""

    if isinstance(left, list) and isinstance(right, list):
        report = {"reference_count": len(left), "candidate_count": len(right), "differences": 0}
        max_float = 0.0
        first = None
        for index in range(max(len(left), len(right))):
            if index >= len(left) or index >= len(right):
                report["differences"] += 1
                if first is None:
                    first = {"index": index, "reference": left[index] if index < len(left) else None,
                             "candidate": right[index] if index < len(right) else None}
                continue
            item = _compare_records(left[index], right[index], tolerance)
            max_float = max(max_float, item["max_float_difference"])
            if item["differences"]:
                report["differences"] += 1
                if first is None:
                    first = {"index": index, "reference": left[index], "candidate": right[index]}
        report["max_float_difference"] = max_float
        report["first_difference"] = first
        return report
    if isinstance(left, dict) and isinstance(right, dict):
        differences = 0
        max_float = 0.0
        first = None
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                differences += 1
                first = first or {"key": key, "reference": left.get(key), "candidate": right.get(key)}
                continue
            item = _compare_records(left[key], right[key], tolerance)
            max_float = max(max_float, item["max_float_difference"])
            if item["differences"]:
                differences += 1
                first = first or {"key": key, "reference": left[key], "candidate": right[key]}
        return {"differences": differences, "max_float_difference": max_float, "first_difference": first}
    same, difference = _same_value(left, right, tolerance)
    return {"differences": 0 if same else 1, "max_float_difference": difference,
            "first_difference": None if same else {"reference": left, "candidate": right}}


def _performance(reference: dict, candidate: dict, *, stages: bool) -> dict:
    keys = ["tree", "machine", "numpy", "wall_seconds", "peak_rss_mb"]
    if stages:
        keys.append("stage_timings")
    return {
        side: {key: dump[key] for key in keys}
        for side, dump in (("reference", reference), ("candidate", candidate))
    }


def compare_dumps(reference: dict, candidate: dict, tolerance: float) -> dict:
    categories = {}
    for name in reference["items"]:
        report = _compare_records(reference["items"][name], candidate["items"].get(name), tolerance)
        report["identical"] = report["differences"] == 0
        categories[name] = report
    return {
        "case": reference["case"],
        "equivalent": all(report["identical"] for report in categories.values()),
        "float_tolerance": tolerance,
        "categories": categories,
        "performance": _performance(reference, candidate, stages=True),
    }


def compare_direct(reference: dict, candidate: dict) -> dict:
    names = sorted(set(reference["suite"]) | set(candidate["suite"]))
    different = [
        name
        for name in names
        if reference["suite"].get(name, {}).get("digest") != candidate["suite"].get(name, {}).get("digest")
    ]
    return {
        "suite": "direct",
        "entries": len(names),
        "equivalent": not different,
        "different": different,
        "performance": _performance(reference, candidate, stages=False),
    }


def _run_worker_process(arguments: list[str]) -> None:
    subprocess.run([sys.executable, str(Path(__file__).resolve()), *arguments], check=True)


def _dump_for(side: str, src: Path, case: dict, out_dir: Path, reuse: bool) -> dict:
    path = out_dir / f"{case_name(case)}.{side}.{tree_digest(src)}.pkl"
    if not (reuse and path.exists()):
        arguments = ["--worker", "--src", str(src),
                     "--case", "{notes}:{pieces}:{seed}:{strategy}".format(**case), "--dump", str(path)]
        for key, value in (case.get("overrides") or {}).items():
            arguments += ["--override", f"{key}={value}"]
        _run_worker_process(arguments)
    with path.open("rb") as handle:
        return pickle.load(handle)


def _direct_dump_for(side: str, src: Path, out_dir: Path, reuse: bool) -> dict:
    path = out_dir / f"direct_suite.{side}.{tree_digest(src)}.pkl"
    if not (reuse and path.exists()):
        _run_worker_process(["--worker-direct", "--src", str(src), "--dump", str(path)])
    with path.open("rb") as handle:
        return pickle.load(handle)


def _print_report(report: dict) -> None:
    status = "EQUIVALENT" if report["equivalent"] else "DIFFERENT"
    if report.get("suite") == "direct":
        print(f"\ndirect-call suite: {status} ({report['entries']} entries)")
        for name in report["different"]:
            print(f"  DIFFERENT {name}")
    else:
        print(f"\n{case_name(report['case'])}: {status}")
        for name, item in report["categories"].items():
            counts = ""
            if "reference_count" in item:
                counts = f" {item['reference_count']} vs {item['candidate_count']}"
            print(f"  {name:18s} {'identical' if item['identical'] else 'DIFFERENT'}{counts}")
    for side, perf in report["performance"].items():
        print(f"  {side:9s} {perf['wall_seconds']:9.1f} s  peak RSS {perf['peak_rss_mb']:8.1f} MB"
              f"  tree {perf['tree']} ({perf['machine']}, numpy {perf['numpy']})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-ref", help="git ref whose src/ is the reference tree")
    parser.add_argument("--reference-src", type=Path, help="reference src/ directory")
    parser.add_argument("--candidate-src", type=Path, default=ROOT / "src")
    parser.add_argument("--case", action="append", help="N:pieces:seed:strategy")
    parser.add_argument("--override", action="append", default=[], help="KEY=VALUE for every case")
    parser.add_argument("--direct-suite", action="store_true",
                        help="compare the direct-call suite (trials run only with --case)")
    parser.add_argument("--float-tolerance", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "core_equivalence")
    parser.add_argument("--report", type=Path, help="JSON report path (default: in --output-dir)")
    parser.add_argument("--reuse", action="store_true", help="reuse dumps of unchanged trees")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-direct", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--src", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dump", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        overrides = dict(parse_override(text) for text in args.override)
    except ValueError as error:
        parser.error(str(error))
    if args.worker:
        case = parse_case(args.case[0])
        if overrides:
            case["overrides"] = overrides
        run_worker(args.src, case, args.dump)
        return 0
    if args.worker_direct:
        run_direct_worker(args.src, args.dump)
        return 0
    if (args.reference_ref is None) == (args.reference_src is None):
        parser.error("give exactly one of --reference-ref or --reference-src")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.reference_ref is not None:
        reference_src = export_ref(args.reference_ref, out_dir / f"ref-{args.reference_ref}")
    else:
        reference_src = args.reference_src
    texts = args.case or ([] if args.direct_suite else list(DEFAULT_CASES))
    reports = []
    for text in texts:
        case = parse_case(text)
        if overrides:
            case["overrides"] = overrides
        reference = _dump_for("reference", reference_src, case, out_dir, args.reuse)
        candidate = _dump_for("candidate", args.candidate_src, case, out_dir, args.reuse)
        report = compare_dumps(reference, candidate, args.float_tolerance)
        _print_report(report)
        reports.append(report)
    if args.direct_suite:
        report = compare_direct(
            _direct_dump_for("reference", reference_src, out_dir, args.reuse),
            _direct_dump_for("candidate", args.candidate_src, out_dir, args.reuse),
        )
        _print_report(report)
        reports.append(report)
    summary = args.report or out_dir / "core_equivalence_report.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {summary}")
    return 0 if all(report["equivalent"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
