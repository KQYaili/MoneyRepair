from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from moneyrepair.ingest import load_rgb
from moneyrepair.types import Fragment


@dataclass(frozen=True)
class ReferenceScore:
    fragment_id: str
    side: str
    pixels: int
    mean_abs_error: float
    rmse: float


def load_references(front: str | Path | None = None, back: str | Path | None = None) -> dict[str, np.ndarray]:
    references: dict[str, np.ndarray] = {}
    if front is not None:
        references["front"] = load_rgb(front)
    if back is not None:
        references["back"] = load_rgb(back)
    if not references:
        raise ValueError("at least one reference image is required")
    return references


def score_fragment_against_reference(fragment: Fragment, reference: np.ndarray, side: str | None = None) -> ReferenceScore:
    if fragment.image is None:
        raise ValueError(f"fragment {fragment.id} has no RGB image")
    if reference.shape[:2] != fragment.mask.shape:
        raise ValueError("reference and fragment mask must share note coordinates")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("reference must be an RGB image")

    mask = fragment.mask
    pixels = int(mask.sum())
    if pixels == 0:
        return ReferenceScore(fragment.id, side or fragment.side, 0, float("inf"), float("inf"))

    diff = fragment.image[mask].astype(np.float32) - reference[mask].astype(np.float32)
    mean_abs = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return ReferenceScore(fragment.id, side or fragment.side, pixels, mean_abs, rmse)


def score_fragments_by_side(
    fragments: list[Fragment],
    references: Mapping[str, np.ndarray],
) -> list[ReferenceScore]:
    scores: list[ReferenceScore] = []
    for fragment in fragments:
        if fragment.side not in references:
            continue
        scores.append(score_fragment_against_reference(fragment, references[fragment.side], side=fragment.side))
    return scores


def score_best_reference_side(
    fragments: list[Fragment],
    references: Mapping[str, np.ndarray],
) -> list[ReferenceScore]:
    scores: list[ReferenceScore] = []
    for fragment in fragments:
        side_scores = [score_fragment_against_reference(fragment, image, side=side) for side, image in references.items()]
        scores.append(min(side_scores, key=lambda item: item.rmse))
    return scores


def scores_to_jsonable(scores: list[ReferenceScore]) -> list[dict]:
    return [
        {
            "fragment_id": score.fragment_id,
            "side": score.side,
            "pixels": score.pixels,
            "mean_abs_error": score.mean_abs_error,
            "rmse": score.rmse,
        }
        for score in scores
    ]


def load_score_thresholds(path: str | Path, max_rmse: float) -> set[str]:
    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(item["fragment_id"]) for item in raw if float(item["rmse"]) <= max_rmse}
