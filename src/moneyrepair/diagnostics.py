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
    solution_set_counts = Counter(solution_sets)
    exactly_recovered = sorted(
        note_id for note_id, frag_set in true_notes.items() if frozenset(frag_set) in solution_sets
    )
    uniquely_exact_recovered = sorted(
        note_id
        for note_id, frag_set in true_notes.items()
        if solution_set_counts[frozenset(frag_set)] == 1
    )
    pure_notes_found = sorted(
        {item["dominant_note"] for item in per_solution if not item["is_chimera"] and item["dominant_note"]}
    )
    true_note_count = len(true_notes)

    return {
        "solutions": len(solutions),
        "chimeras": chimeras,
        "pure": pure,
        "chimera_rate": chimeras / len(solutions) if solutions else 0.0,
        "true_notes": true_note_count,
        "pure_notes_found": pure_notes_found,
        "pure_notes_found_count": len(pure_notes_found),
        "pure_notes_found_rate": len(pure_notes_found) / true_note_count if true_note_count else 0.0,
        "exactly_recovered_notes": exactly_recovered,
        "exactly_recovered_count": len(exactly_recovered),
        "exactly_recovered_rate": len(exactly_recovered) / true_note_count if true_note_count else 0.0,
        "uniquely_exact_recovered_notes": uniquely_exact_recovered,
        "uniquely_exact_recovered_count": len(uniquely_exact_recovered),
        "uniquely_exact_recovered_rate": len(uniquely_exact_recovered) / true_note_count if true_note_count else 0.0,
        "per_solution": per_solution,
    }


def diagnose_groups(fragments: list[Fragment], groups: dict[str, int]) -> dict:
    """Diagnose whether discrimination groups uniquely identify true notes.

    Unlike :func:`diagnose_solutions`, this does not depend on the DFS top-k
    output. It is the pressure-test metric for large pools where
    ``max_solutions`` can hide merged identities behind the first pure-looking
    candidates.
    """

    true_notes: dict[str, set[str]] = {}
    group_members: dict[int, set[str]] = {}
    group_notes: dict[int, set[str]] = {}
    note_groups: dict[str, set[int]] = {}
    for fragment in fragments:
        note_id = fragment.meta.get("note_id")
        group_id = groups.get(fragment.id)
        if note_id is not None:
            true_notes.setdefault(note_id, set()).add(fragment.id)
        if group_id is None:
            continue
        group_members.setdefault(group_id, set()).add(fragment.id)
        if note_id is not None:
            group_notes.setdefault(group_id, set()).add(note_id)
            note_groups.setdefault(note_id, set()).add(group_id)

    note_set_to_id = {frozenset(fragment_ids): note_id for note_id, fragment_ids in true_notes.items()}
    exact_recoverable_notes = sorted(
        note_set_to_id[frozenset(member_set)]
        for member_set in group_members.values()
        if frozenset(member_set) in note_set_to_id
    )
    mixed_groups = sorted(group_id for group_id, note_ids in group_notes.items() if len(note_ids) > 1)
    mixed_notes = sorted({note_id for group_id in mixed_groups for note_id in group_notes[group_id]})
    split_notes = sorted(note_id for note_id, group_ids in note_groups.items() if len(group_ids) > 1)
    true_note_count = len(true_notes)

    return {
        "groups": len(group_members),
        "true_notes": true_note_count,
        "cluster_deficit": true_note_count - len(group_members),
        "mixed_groups": mixed_groups,
        "mixed_group_count": len(mixed_groups),
        "mixed_notes": mixed_notes,
        "mixed_note_count": len(mixed_notes),
        "split_notes": split_notes,
        "split_note_count": len(split_notes),
        "exact_recoverable_notes": exact_recoverable_notes,
        "exact_recoverable_count": len(exact_recoverable_notes),
        "exact_recoverable_rate": len(exact_recoverable_notes) / true_note_count if true_note_count else 0.0,
    }
