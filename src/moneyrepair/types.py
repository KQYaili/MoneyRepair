from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Fragment:
    """A note fragment placed in a common note coordinate frame."""

    id: str
    mask: np.ndarray
    label: str | None = None
    side: str = "front"
    image: np.ndarray | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mask.dtype != np.bool_:
            object.__setattr__(self, "mask", self.mask.astype(bool))
        if self.mask.ndim != 2:
            raise ValueError("fragment mask must be a 2D array")
        if self.image is not None:
            if self.image.shape[:2] != self.mask.shape:
                raise ValueError("fragment image must share the mask height/width")

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def fragment_areas(fragments: list[Fragment]) -> np.ndarray:
    return np.array([fragment.area for fragment in fragments], dtype=np.int64)


def stack_masks(fragments: list[Fragment]) -> np.ndarray:
    if not fragments:
        raise ValueError("at least one fragment is required")
    shape = fragments[0].mask.shape
    if any(fragment.mask.shape != shape for fragment in fragments):
        raise ValueError("all fragment masks must share one coordinate frame")
    return np.stack([fragment.mask for fragment in fragments], axis=0)
