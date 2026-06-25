from __future__ import annotations

from collections import Counter

from moneyrepair.solver import CoverageSolution
from moneyrepair.types import Fragment


def solution_purity(solution: CoverageSolution, lookup: dict[str, Fragment]) -> dict:
    """Score one solution against the secret true ``note_id`` of its fragments."""

    note_ids = [lookup[fid].meta.get("note_id") for fid in solution.fragment_ids if fid in lookup]
    known = [note_id for note_id in note_ids if note_id]
    distinct = sorted(set(known))
    dominant: str | None = None
    dominant_fraction = 0.0
    if known:
        dominant, count = Counter(known).most_common(1)[0]
        dominant_fraction = count / len(known)
    return {
        "fragments": len(solution.fragment_ids),
        "distinct_notes": len(distinct),
        "is_chimera": len(distinct) > 1,
        "dominant_note": dominant,
        "dominant_fraction": dominant_fraction,
        "coverage": solution.coverage,
    }


def diagnose_solutions(solutions: list[CoverageSolution], fragments: list[Fragment]) -> dict:
    """Aggregate chimera and recovery statistics over a solution set.

    A chimera is a solution mixing fragments from more than one true note. A note
    is "exactly recovered" when some solution's fragment set equals that note's
    full set of fragments.
    """

    lookup = {fragment.id: fragment for fragment in fragments}
    per_solution = [solution_purity(solution, lookup) for solution in solutions]
    chimeras = sum(1 for item in per_solution if item["is_chimera"])
    pure = sum(1 for item in per_solution if not item["is_chimera"] and item["distinct_notes"] == 1)

    true_notes: dict[str, set[str]] = {}
    for fragment in fragments:
        note_id = fragment.meta.get("note_id")
        if note_id:
            true_notes.setdefault(note_id, set()).add(fragment.id)

    solution_sets = [frozenset(solution.fragment_ids) for solution in solutions]
    exactly_recovered = sorted(
        note_id for note_id, frag_set in true_notes.items() if frozenset(frag_set) in solution_sets
    )
    pure_notes_found = sorted(
        {item["dominant_note"] for item in per_solution if not item["is_chimera"] and item["dominant_note"]}
    )

    return {
        "solutions": len(solutions),
        "chimeras": chimeras,
        "pure": pure,
        "chimera_rate": chimeras / len(solutions) if solutions else 0.0,
        "true_notes": len(true_notes),
        "pure_notes_found": pure_notes_found,
        "exactly_recovered_notes": exactly_recovered,
        "per_solution": per_solution,
    }
