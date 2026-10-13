"""Runtime, memory and outcome growth of the simulated v4.4.1 core.

Measurement only. Runs the v4.4.1 preset over a grid of note counts, pieces
per note and seeds, one process per run so that peak RSS belongs to that run.
Budgets are the preset's preregistered workload-normalised rates (candidate
states per scored pair, gap states per fragment, cover nodes per note), never
fixed per run, and wall-clock limits stay off. Each run records stage
timings, peak RSS, pair and candidate counts, search sizes, per-stage
truncation flags, oracle candidate recall and exact yield/precision as one
JSON line. Existing lines are kept, so an interrupted grid resumes.

Example:

    python docs/experiments/core_scaling.py --notes 20,50,100,200 \\
        --pieces 16,24 --seeds 7,8,9 --jobs 3
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]


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


def environment() -> dict:
    """Interpreter, libraries, machine and a digest of the core sources."""

    import numba
    import numpy

    from moneyrepair import tearfit

    src = Path(tearfit.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(src.rglob("*.py")):
        digest.update(str(path.relative_to(src)).encode())
        digest.update(path.read_bytes())
    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "numba": numba.__version__,
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "core_tree": digest.hexdigest()[:16],
    }


def run_case(notes: int, pieces: int, seed: int, strategy: str) -> dict:
    from moneyrepair import tearfit

    simulation = tearfit.FractalTearConfig(
        notes=notes, pieces_per_note=pieces, seed=seed, serial_ocr_rate=0.0
    )
    trial_kwargs = asdict(tearfit.TearFitCoreConfig.v4_4_1())
    del trial_kwargs["fray_layers"]
    trial_kwargs["base_selection_strategy"] = strategy
    started = monotonic()
    result = tearfit.run_tearfit_trial(simulation, **trial_kwargs)
    wall = monotonic() - started
    stats = result.search_stats
    config = result.config
    return {
        "case": {"notes": notes, "pieces": pieces, "seed": seed, "strategy": strategy},
        "resolved_algorithm": config["resolved_algorithm"],
        "budgets": {
            name: config[name]
            for name in (
                "resolved_candidate_state_limit",
                "resolved_gap_state_limit",
                "resolved_partial_gap_state_limit",
                "resolved_cover_node_limit",
            )
        },
        "environment": environment(),
        "wall_seconds": wall,
        "peak_rss_mb": peak_rss_mb(),
        "stage_timings": dict(result.stage_timings),
        "counts": {
            "fragments": result.fragments,
            "pair_scores": result.pair_scores,
            "accepted_edges": result.accepted_edges,
            "core_candidates": result.core_candidates,
            "partial_core_candidates": result.partial_core_candidates,
            "gap_candidates": result.gap_candidates,
            "candidates": result.candidates,
            "selected": result.diagnostics.confirmed,
        },
        "search": {
            "core_expanded_states": stats["core"].get("expanded_states", 0),
            "complete_gap_expanded_states": stats["complete_gap"].get("expanded_states", 0),
            "partial_gap_expanded_states": stats["partial_gap"].get("expanded_states", 0),
            "gap_proposal_evaluations": stats["complete_gap"].get("proposal_evaluations", 0)
            + stats["partial_gap"].get("proposal_evaluations", 0),
            "cover_search_nodes": stats["exact_cover"].get("search_nodes", 0),
        },
        "truncated": {
            "core": bool(stats["core"].get("state_limit_reached", False)),
            "complete_gap": bool(stats["complete_gap"].get("state_limit_reached", False)),
            "partial_gap": bool(stats["partial_gap"].get("state_limit_reached", False)),
            "exact_cover": bool(stats["exact_cover"].get("node_limit_reached", False)),
            "any_time_limit": any(
                bool(stage.get("time_limit_reached", False)) for stage in stats.values()
            ),
        },
        "outcome": {
            "oracle_candidate_recall": result.oracle_candidate_recall,
            "exact_yield": result.diagnostics.exact_yield,
            "exact_precision": result.diagnostics.exact_precision,
            "chimeras": result.diagnostics.chimeras,
            "manual_notes_remaining": result.diagnostics.manual_notes_remaining,
        },
        "selected_solution_fingerprint": result.selected_solution_fingerprint,
    }


def _key(case: dict) -> tuple:
    return (case["notes"], case["pieces"], case["seed"], case["strategy"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notes", default="20,50,100,200")
    parser.add_argument("--pieces", default="16,24")
    parser.add_argument("--seeds", default="7,8,9")
    parser.add_argument("--strategy", default="disjoint_round_robin")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--cpus", default="", help="comma-separated CPUs to pin workers to")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "core_scaling.jsonl")
    parser.add_argument("--worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        notes, pieces, seed = (int(value) for value in args.worker.split(":"))
        print(json.dumps(run_case(notes, pieces, seed, args.strategy)))
        return 0

    done = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(_key(json.loads(line)["case"]))
    cases = [
        (int(n), int(p), int(s))
        for n in args.notes.split(",")
        for p in args.pieces.split(",")
        for s in args.seeds.split(",")
        if (int(n), int(p), int(s), args.strategy) not in done
    ]
    # Largest runs first so the pool finishes evenly.
    cases.sort(key=lambda case: -case[0] * case[1])
    cpus = [int(value) for value in args.cpus.split(",") if value]
    free = list(cpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def launch(case: tuple[int, int, int]) -> None:
        cpu = free.pop() if free else None
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "{}:{}:{}".format(*case), "--strategy", args.strategy]
        started = monotonic()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            preexec_fn=(lambda: os.sched_setaffinity(0, {cpu})) if cpu is not None else None,
        )
        if cpu is not None:
            free.append(cpu)
        if completed.returncode != 0:
            print(f"FAILED {case}: {completed.stderr[-2000:]}", file=sys.stderr)
            return
        line = completed.stdout.strip().splitlines()[-1]
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(f"done {case} in {monotonic() - started:.1f} s", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        list(pool.map(launch, cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
