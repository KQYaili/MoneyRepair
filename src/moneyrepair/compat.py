from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from moneyrepair.types import Fragment

_POPCOUNT_TABLE = np.array([bin(value).count("1") for value in range(256)], dtype=np.int64)


@dataclass(frozen=True)
class CompatibilityMatrix:
    """Pairwise same-note compatibility.

    ``compatible[i, j]`` means fragments i and j may belong to the same note.
    The diagonal is always false because a fragment is not paired with itself.
    """

    ids: tuple[str, ...]
    compatible: np.ndarray

    def __post_init__(self) -> None:
        if self.compatible.dtype != np.bool_:
            object.__setattr__(self, "compatible", self.compatible.astype(bool))
        if self.compatible.shape != (len(self.ids), len(self.ids)):
            raise ValueError("compatibility matrix shape does not match ids")

    def index(self, fragment_id: str) -> int:
        return self.ids.index(fragment_id)

    def is_compatible(self, left: int | str, right: int | str) -> bool:
        left_index = self.index(left) if isinstance(left, str) else left
        right_index = self.index(right) if isinstance(right, str) else right
        return bool(self.compatible[left_index, right_index])

    def compatible_indices(self, index: int, candidates: tuple[int, ...]) -> tuple[int, ...]:
        if not candidates:
            return ()
        candidate_array = np.fromiter(candidates, dtype=np.int64)
        return tuple(candidate_array[self.compatible[index, candidate_array]].tolist())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        packed = np.packbits(self.compatible, axis=1)
        np.savez_compressed(path, ids=np.array(self.ids), packed=packed, n=len(self.ids))

    @classmethod
    def load(cls, path: str | Path) -> "CompatibilityMatrix":
        data = np.load(Path(path), allow_pickle=False)
        ids = tuple(str(value) for value in data["ids"])
        n = int(data["n"])
        compatible = np.unpackbits(data["packed"], axis=1, count=n).astype(bool)
        return cls(ids=ids, compatible=compatible)


@dataclass
class PackedCompatibilityMatrix:
    """Packed pairwise compatibility for large fragment sets."""

    ids: tuple[str, ...]
    packed: np.ndarray
    n: int | None = None

    def __post_init__(self) -> None:
        self.n = len(self.ids) if self.n is None else int(self.n)
        expected_width = (self.n + 7) // 8
        if self.packed.dtype != np.uint8:
            self.packed = self.packed.astype(np.uint8)
        if self.packed.shape != (self.n, expected_width):
            raise ValueError("packed matrix shape does not match ids")

    @classmethod
    def from_dense(cls, matrix: CompatibilityMatrix) -> "PackedCompatibilityMatrix":
        return cls(ids=matrix.ids, packed=np.packbits(matrix.compatible, axis=1), n=len(matrix.ids))

    @classmethod
    def filled(cls, ids: Iterable[str], value: bool = True) -> "PackedCompatibilityMatrix":
        id_tuple = tuple(ids)
        n = len(id_tuple)
        width = (n + 7) // 8
        packed = np.full((n, width), 255 if value else 0, dtype=np.uint8)
        if value and n > 0:
            pad = width * 8 - n
            if pad:
                packed[:, -1] &= np.uint8((0xFF << pad) & 0xFF)
        matrix = cls(ids=id_tuple, packed=packed, n=n)
        for index in range(n):
            matrix.set_compatible(index, index, False)
        return matrix

    def compatible_pair_count(self) -> int:
        """Number of undirected compatible pairs, counted straight from bits."""

        if self.n in (0, None):
            return 0
        return int(_POPCOUNT_TABLE[self.packed].sum() // 2)

    def index(self, fragment_id: str) -> int:
        return self.ids.index(fragment_id)

    def _bit_position(self, index: int) -> tuple[int, int]:
        return index // 8, 7 - (index % 8)

    def set_compatible(self, left: int | str, right: int | str, value: bool) -> None:
        left_index = self.index(left) if isinstance(left, str) else left
        right_index = self.index(right) if isinstance(right, str) else right
        self._set_one(left_index, right_index, value)

    def set_pair_compatible(self, left: int | str, right: int | str, value: bool) -> None:
        left_index = self.index(left) if isinstance(left, str) else left
        right_index = self.index(right) if isinstance(right, str) else right
        self._set_one(left_index, right_index, value)
        self._set_one(right_index, left_index, value)

    def _set_one(self, row: int, col: int, value: bool) -> None:
        byte_index, bit_offset = self._bit_position(col)
        bit = np.uint8(1 << bit_offset)
        if value:
            self.packed[row, byte_index] = np.uint8(self.packed[row, byte_index] | bit)
        else:
            self.packed[row, byte_index] = np.uint8(self.packed[row, byte_index] & np.uint8(255 ^ int(bit)))

    def is_compatible(self, left: int | str, right: int | str) -> bool:
        left_index = self.index(left) if isinstance(left, str) else left
        right_index = self.index(right) if isinstance(right, str) else right
        byte_index, bit_offset = self._bit_position(right_index)
        return bool(self.packed[left_index, byte_index] & np.uint8(1 << bit_offset))

    def compatible_indices(self, index: int, candidates: tuple[int, ...]) -> tuple[int, ...]:
        if not candidates:
            return ()
        row = np.unpackbits(self.packed[index], count=self.n).astype(bool)
        candidate_array = np.fromiter(candidates, dtype=np.int64)
        return tuple(candidate_array[row[candidate_array]].tolist())

    def to_dense(self) -> CompatibilityMatrix:
        compatible = np.unpackbits(self.packed, axis=1, count=self.n).astype(bool)
        return CompatibilityMatrix(ids=self.ids, compatible=compatible)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=np.array(self.ids), packed=self.packed, n=self.n)

    @classmethod
    def load(cls, path: str | Path) -> "PackedCompatibilityMatrix":
        data = np.load(Path(path), allow_pickle=False)
        ids = tuple(str(value) for value in data["ids"])
        return cls(ids=ids, packed=data["packed"].astype(np.uint8), n=int(data["n"]))


def compute_compatibility(
    fragments: list[Fragment],
    max_overlap_pixels: int = 0,
    max_overlap_ratio: float = 0.0,
) -> CompatibilityMatrix:
    """Build pairwise compatibility from placed masks.

    The pair is incompatible when their pixel overlap is larger than both
    configured tolerances. Ratio is measured against the smaller fragment.
    """

    ids = tuple(fragment.id for fragment in fragments)
    n = len(fragments)
    compatible = np.ones((n, n), dtype=bool)
    np.fill_diagonal(compatible, False)
    bboxes = [fragment.bbox for fragment in fragments]
    areas = np.array([max(fragment.area, 1) for fragment in fragments], dtype=np.int64)

    for i in range(n):
        x0_i, y0_i, x1_i, y1_i = bboxes[i]
        for j in range(i + 1, n):
            x0_j, y0_j, x1_j, y1_j = bboxes[j]
            x0 = max(x0_i, x0_j)
            y0 = max(y0_i, y0_j)
            x1 = min(x1_i, x1_j)
            y1 = min(y1_i, y1_j)
            if x0 >= x1 or y0 >= y1:
                continue

            overlap = int(np.logical_and(fragments[i].mask[y0:y1, x0:x1], fragments[j].mask[y0:y1, x0:x1]).sum())
            ratio = overlap / float(min(areas[i], areas[j]))
            if overlap > max_overlap_pixels and ratio > max_overlap_ratio:
                compatible[i, j] = False
                compatible[j, i] = False
    return CompatibilityMatrix(ids=ids, compatible=compatible)


def _pair_overlap_pixels(
    left: Fragment,
    right: Fragment,
    left_bbox: tuple[int, int, int, int],
    right_bbox: tuple[int, int, int, int],
) -> int:
    x0 = max(left_bbox[0], right_bbox[0])
    y0 = max(left_bbox[1], right_bbox[1])
    x1 = min(left_bbox[2], right_bbox[2])
    y1 = min(left_bbox[3], right_bbox[3])
    if x0 >= x1 or y0 >= y1:
        return 0
    return int(np.logical_and(left.mask[y0:y1, x0:x1], right.mask[y0:y1, x0:x1]).sum())


def _auto_cell_size(bboxes: list[tuple[int, int, int, int]]) -> int:
    dims = sorted(max(x1 - x0, y1 - y0) for x0, y0, x1, y1 in bboxes if x1 > x0 and y1 > y0)
    if not dims:
        return 1
    return max(1, int(dims[len(dims) // 2]))


def _bbox_cells(bbox: tuple[int, int, int, int], cell: int) -> Iterable[tuple[int, int]]:
    x0, y0, x1, y1 = bbox
    for cy in range(y0 // cell, (y1 - 1) // cell + 1):
        for cx in range(x0 // cell, (x1 - 1) // cell + 1):
            yield cx, cy


def iter_incompatible_pairs(
    fragments: list[Fragment],
    max_overlap_pixels: int = 0,
    max_overlap_ratio: float = 0.0,
    cell: int | None = None,
) -> Iterable[tuple[int, int]]:
    """Yield index pairs ``(i, j)`` (i < j) whose masks overlap past tolerance.

    Fragments are hashed into a coarse spatial grid by bounding box, so only
    spatially close pairs are ever compared. For fragments that tile a note this
    keeps the work near the number of truly adjacent pairs instead of ``n^2``.
    """

    bboxes = [fragment.bbox for fragment in fragments]
    areas = [max(fragment.area, 1) for fragment in fragments]
    if cell is None:
        cell = _auto_cell_size(bboxes)
    cell = max(1, int(cell))

    buckets: dict[tuple[int, int], list[int]] = {}
    for index, bbox in enumerate(bboxes):
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        for key in _bbox_cells(bbox, cell):
            buckets.setdefault(key, []).append(index)

    seen: set[tuple[int, int]] = set()
    for members in buckets.values():
        count = len(members)
        for a in range(count):
            i = members[a]
            for b in range(a + 1, count):
                j = members[b]
                pair = (i, j) if i < j else (j, i)
                if pair in seen:
                    continue
                seen.add(pair)
                overlap = _pair_overlap_pixels(fragments[pair[0]], fragments[pair[1]], bboxes[pair[0]], bboxes[pair[1]])
                if overlap <= max_overlap_pixels:
                    continue
                ratio = overlap / float(min(areas[pair[0]], areas[pair[1]]))
                if ratio > max_overlap_ratio:
                    yield pair


def compute_compatibility_fast(
    fragments: list[Fragment],
    max_overlap_pixels: int = 0,
    max_overlap_ratio: float = 0.0,
    cell: int | None = None,
) -> PackedCompatibilityMatrix:
    """Grid-pruned compatibility build that writes packed bits directly.

    Equivalent result to :func:`compute_compatibility` but never materialises the
    dense ``n*n`` boolean array, so it scales to the ~20k fragment production
    target where the dense matrix would cost hundreds of megabytes.
    """

    ids = tuple(fragment.id for fragment in fragments)
    matrix = PackedCompatibilityMatrix.filled(ids, value=True)
    for i, j in iter_incompatible_pairs(fragments, max_overlap_pixels, max_overlap_ratio, cell):
        matrix.set_pair_compatible(i, j, False)
    return matrix


def write_incompatible_pairs(
    path: str | Path,
    fragments: list[Fragment],
    max_overlap_pixels: int = 0,
    max_overlap_ratio: float = 0.0,
    cell: int | None = None,
) -> int:
    """Stream incompatible pairs to a CSV without holding a dense matrix."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [fragment.id for fragment in fragments]
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fragment_a", "fragment_b"])
        for i, j in iter_incompatible_pairs(fragments, max_overlap_pixels, max_overlap_ratio, cell):
            writer.writerow([ids[i], ids[j]])
            count += 1
    return count


def incompatibility_matrix(compatibility: CompatibilityMatrix) -> np.ndarray:
    incompatible = ~compatibility.compatible.copy()
    np.fill_diagonal(incompatible, False)
    return incompatible


def filter_compatibility_to_ids(
    compatibility: CompatibilityMatrix,
    allowed_ids: set[str],
) -> CompatibilityMatrix:
    """Mark fragments outside ``allowed_ids`` incompatible with every partner."""

    filtered = compatibility.compatible.copy()
    for index, fragment_id in enumerate(compatibility.ids):
        if fragment_id not in allowed_ids:
            filtered[index, :] = False
            filtered[:, index] = False
    np.fill_diagonal(filtered, False)
    return CompatibilityMatrix(ids=compatibility.ids, compatible=filtered)


def restrict_packed_to_ids(
    matrix: PackedCompatibilityMatrix,
    allowed_ids: set[str],
) -> PackedCompatibilityMatrix:
    """Mark fragments outside ``allowed_ids`` incompatible with every partner."""

    packed = matrix.packed.copy()
    for index, fragment_id in enumerate(matrix.ids):
        if fragment_id not in allowed_ids:
            packed[index, :] = 0
            byte_index = index // 8
            bit = np.uint8(1 << (7 - (index % 8)))
            packed[:, byte_index] &= np.uint8(255) ^ bit
    return PackedCompatibilityMatrix(ids=matrix.ids, packed=packed, n=matrix.n)


def load_pair_records(path: str | Path) -> list[tuple[str, str]]:
    """Load two-column pair records from CSV, TSV, or whitespace text."""

    records: list[tuple[str, str]] = []
    for raw_line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            parts = next(csv.reader([line]))
        else:
            parts = line.split()
        if len(parts) < 2:
            continue
        left = parts[0].strip()
        right = parts[1].strip()
        if left.lower() in {"left", "fragment_a", "id_a"}:
            continue
        records.append((left, right))
    return records


def compatibility_from_pair_records(
    ids: Iterable[str],
    pairs: Iterable[tuple[str, str]],
    relation: str = "incompatible",
) -> PackedCompatibilityMatrix:
    """Build a packed matrix from precomputed pair records."""

    if relation not in {"compatible", "incompatible"}:
        raise ValueError("relation must be 'compatible' or 'incompatible'")
    default_value = relation == "incompatible"
    matrix = PackedCompatibilityMatrix.filled(ids, value=default_value)
    pair_value = relation == "compatible"
    id_to_index = {fragment_id: index for index, fragment_id in enumerate(matrix.ids)}

    for left, right in pairs:
        if left not in id_to_index or right not in id_to_index:
            raise ValueError(f"pair references unknown fragment id: {left!r}, {right!r}")
        matrix.set_pair_compatible(id_to_index[left], id_to_index[right], pair_value)
    for index in range(matrix.n or 0):
        matrix.set_compatible(index, index, False)
    return matrix
