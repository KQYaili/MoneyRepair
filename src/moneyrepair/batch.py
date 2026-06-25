from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from moneyrepair.solver import CoverageSolution
from moneyrepair.types import Fragment


@dataclass(frozen=True)
class ConfirmedNote:
    note_id: str
    fragment_ids: tuple[str, ...]
    coverage: float
    area: int
    accepted_at: str


@dataclass
class BatchState:
    confirmed_notes: list[ConfirmedNote] = field(default_factory=list)
    rejected_solution_keys: set[tuple[str, ...]] = field(default_factory=set)

    @property
    def used_fragment_ids(self) -> set[str]:
        used: set[str] = set()
        for note in self.confirmed_notes:
            used.update(note.fragment_ids)
        return used

    def active_fragment_ids(self, fragments: list[Fragment]) -> set[str]:
        used = self.used_fragment_ids
        return {fragment.id for fragment in fragments if fragment.id not in used}

    def next_note_id(self, prefix: str = "note") -> str:
        return f"{prefix}-{len(self.confirmed_notes) + 1:05d}"

    def add_confirmation(self, note_id: str, solution: CoverageSolution) -> None:
        self.confirmed_notes.append(
            ConfirmedNote(
                note_id=note_id,
                fragment_ids=solution.fragment_ids,
                coverage=solution.coverage,
                area=solution.area,
                accepted_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    def reject_solution(self, solution: CoverageSolution) -> None:
        self.rejected_solution_keys.add(tuple(sorted(solution.fragment_ids)))

    def filter_rejected(self, solutions: list[CoverageSolution]) -> list[CoverageSolution]:
        return [solution for solution in solutions if tuple(sorted(solution.fragment_ids)) not in self.rejected_solution_keys]

    def to_dict(self) -> dict:
        return {
            "confirmed_notes": [
                {
                    "note_id": note.note_id,
                    "fragment_ids": list(note.fragment_ids),
                    "coverage": note.coverage,
                    "area": note.area,
                    "accepted_at": note.accepted_at,
                }
                for note in self.confirmed_notes
            ],
            "rejected_solution_keys": [list(key) for key in sorted(self.rejected_solution_keys)],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BatchState":
        state = cls()
        state.confirmed_notes = [
            ConfirmedNote(
                note_id=str(item["note_id"]),
                fragment_ids=tuple(str(fragment_id) for fragment_id in item["fragment_ids"]),
                coverage=float(item["coverage"]),
                area=int(item["area"]),
                accepted_at=str(item.get("accepted_at", "")),
            )
            for item in payload.get("confirmed_notes", [])
        ]
        state.rejected_solution_keys = {
            tuple(sorted(str(fragment_id) for fragment_id in item))
            for item in payload.get("rejected_solution_keys", [])
        }
        return state


def load_batch_state(path: str | Path) -> BatchState:
    path = Path(path)
    if not path.exists():
        return BatchState()
    return BatchState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_batch_state(path: str | Path, state: BatchState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")


def solution_key(solution: CoverageSolution) -> tuple[str, ...]:
    return tuple(sorted(solution.fragment_ids))
