from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from moneyrepair.types import Fragment


def synthetic_banknote(width: int = 420, height: int = 180, seed: int = 7) -> np.ndarray:
    """Create a deterministic note-like RGB template for software simulation."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.empty((height, width, 3), dtype=np.uint8)
    base[..., 0] = np.clip(182 + 28 * np.sin(xx / 31.0) + yy * 0.08, 0, 255)
    base[..., 1] = np.clip(52 + 18 * np.cos((xx + yy) / 29.0), 0, 255)
    base[..., 2] = np.clip(70 + 22 * np.sin(yy / 17.0), 0, 255)

    image = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(image, mode="RGBA")
    draw.rounded_rectangle((6, 6, width - 7, height - 7), radius=10, outline=(80, 15, 30, 230), width=3)
    draw.ellipse((width * 0.09, height * 0.16, width * 0.36, height * 0.84), outline=(250, 210, 160, 150), width=4)
    draw.rectangle((width * 0.54, height * 0.18, width * 0.88, height * 0.39), fill=(230, 220, 160, 55))
    draw.text((width * 0.67, height * 0.55), "100", fill=(245, 230, 190, 210))
    draw.text((width * 0.12, height * 0.10), "RMB", fill=(255, 235, 205, 170))

    arr = np.asarray(image).copy()
    noise = rng.normal(0, 3, arr.shape).astype(np.int16)
    return np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _voronoi_partition(height: int, width: int, pieces: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.column_stack(
        [
            rng.integers(0, height, size=pieces),
            rng.integers(0, width, size=pieces),
        ]
    ).astype(np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    pixels = np.stack([yy, xx], axis=-1).astype(np.float32)
    labels = np.empty((height, width), dtype=np.int32)

    best = np.full((height, width), np.inf, dtype=np.float32)
    for index, (cy, cx) in enumerate(centers):
        wobble = 10.0 * np.sin((xx + index * 19) / 37.0) + 8.0 * np.cos((yy - index * 11) / 23.0)
        dist = (pixels[..., 0] - cy) ** 2 + (pixels[..., 1] - cx) ** 2 + wobble
        update = dist < best
        labels[update] = index
        best[update] = dist[update]
    return labels


def make_synthetic_fragments(
    pieces: int = 24,
    width: int = 420,
    height: int = 180,
    seed: int = 7,
    side: str = "front",
) -> tuple[np.ndarray, list[Fragment]]:
    """Generate a synthetic note template and non-overlapping fragment masks."""

    template = synthetic_banknote(width=width, height=height, seed=seed)
    labels = _voronoi_partition(height=height, width=width, pieces=pieces, seed=seed + 1)
    fragments: list[Fragment] = []

    for index in range(pieces):
        mask = labels == index
        if not np.any(mask):
            continue
        image = np.where(mask[..., None], template, 0)
        fragments.append(
            Fragment(
                id=f"f{index:05d}",
                label=f"{index:05d}",
                side=side,
                mask=mask,
                image=image,
            )
        )
    return template, fragments


def _serial_roi(height: int, width: int) -> tuple[int, int, int, int]:
    """Bottom-left band where a banknote serial number lives."""

    return int(width * 0.06), int(height * 0.62), int(width * 0.42), int(height * 0.92)


def _smooth_field(height: int, width: int, rng: np.random.Generator, grid: int = 6) -> np.ndarray:
    """Low-frequency random field in [-1, 1]."""

    grid = max(2, int(grid))
    low = rng.normal(0, 1, (grid, grid)).astype(np.float32)
    low = (low - low.mean()) / (low.std() + 1e-6)
    image = Image.fromarray(np.clip((low + 3.0) / 6.0 * 255, 0, 255).astype(np.uint8), mode="L")
    resampling = getattr(Image, "Resampling", Image)
    field = np.asarray(image.resize((width, height), resample=resampling.BICUBIC), dtype=np.float32) / 255.0
    field = (field - field.mean()) / (field.std() + 1e-6)
    return np.clip(field / 3.0, -1.0, 1.0)


def _apply_spatial_wear(
    template: np.ndarray,
    rng: np.random.Generator,
    global_gain: np.ndarray,
    local_strength: float,
    gamma_spread: float,
    stain_count: int,
    stain_strength: float,
) -> np.ndarray:
    """Apply non-uniform note wear that a global gain fingerprint cannot invert."""

    height, width = template.shape[:2]
    image = template.astype(np.float32) / 255.0
    gamma = float(1.0 + rng.uniform(-gamma_spread, gamma_spread))
    image = np.power(np.clip(image, 0, 1), gamma)

    yy, xx = np.mgrid[0:height, 0:width]
    vignette_cx = rng.uniform(0.35, 0.65) * width
    vignette_cy = rng.uniform(0.35, 0.65) * height
    radius = np.sqrt(((xx - vignette_cx) / max(width, 1)) ** 2 + ((yy - vignette_cy) / max(height, 1)) ** 2)
    vignette = 1.0 - local_strength * 0.55 * radius / (radius.max() + 1e-6)

    field = _smooth_field(height, width, rng, grid=6)
    local = 1.0 + local_strength * field
    image = image * vignette[..., None] * local[..., None]

    for _ in range(max(0, int(stain_count))):
        cx = rng.uniform(0, width)
        cy = rng.uniform(0, height)
        sx = rng.uniform(width * 0.05, width * 0.22)
        sy = rng.uniform(height * 0.05, height * 0.22)
        blob = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2.0)
        tint = rng.uniform(-stain_strength, stain_strength, size=3)
        image = image * (1.0 + blob[..., None] * tint)

    image = image * global_gain.reshape(1, 1, 3)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def make_multi_note_fragments(
    notes: int = 3,
    pieces_per_note: int = 12,
    width: int = 420,
    height: int = 180,
    seed: int = 7,
    side: str = "front",
    appearance_spread: float = 0.18,
    noise_sigma: float = 4.0,
    wear_model: str = "global_gain",
    local_wear_strength: float = 0.0,
    gamma_spread: float = 0.0,
    stain_count: int = 0,
    stain_strength: float = 0.0,
    partition_model: str = "shared",
) -> tuple[np.ndarray, list[Fragment]]:
    """Generate fragments from ``notes`` distinct banknotes of one denomination.

    The default ``global_gain`` wear model is the friendly red/green testbed:
    every note is the same template with one per-channel appearance gain and a
    serial number. ``spatial`` adds local gamma, vignetting, stains, and
    low-frequency wear so the global-gain fingerprint no longer perfectly
    inverts the simulator. ``partition_model="shared"`` gives every note the
    same cut pattern; ``"per_note"`` gives each note an independent tear
    partition so geometry can carry identity signal.

    This is the honest testbed: pieces from different notes that cover disjoint
    regions do not overlap, so an overlap-only compatibility matrix will happily
    let the search stitch them into a high-coverage chimera. Single-note
    simulation could never surface that failure mode.
    """

    if notes < 1:
        raise ValueError("notes must be >= 1")
    if wear_model not in {"global_gain", "spatial"}:
        raise ValueError("wear_model must be 'global_gain' or 'spatial'")
    if partition_model not in {"shared", "per_note"}:
        raise ValueError("partition_model must be 'shared' or 'per_note'")
    template = synthetic_banknote(width=width, height=height, seed=seed)
    rng = np.random.default_rng(seed + 5_000)
    x0, y0, x1, y1 = _serial_roi(height, width)
    fragments: list[Fragment] = []

    # ``shared`` is the deliberately hardest chimera testbed: every physical
    # note is cut by the same partition, so shape cannot help. ``per_note`` is
    # closer to real hand-tearing, where each note has its own tear geometry and
    # torn-edge fit becomes a discriminative signal independent of appearance.
    shared_labels = (
        _voronoi_partition(height=height, width=width, pieces=pieces_per_note, seed=seed + 1)
        if partition_model == "shared"
        else None
    )

    for note_index in range(notes):
        labels = (
            shared_labels
            if shared_labels is not None
            else _voronoi_partition(height=height, width=width, pieces=pieces_per_note, seed=seed + 1 + note_index * 10_007)
        )
        gain = 1.0 + rng.uniform(-appearance_spread, appearance_spread, size=3)
        serial = f"SN{note_index:08d}"
        note_id = f"note-{note_index:03d}"
        if wear_model == "spatial":
            strength = local_wear_strength if local_wear_strength > 0 else 0.10
            gamma = gamma_spread if gamma_spread > 0 else 0.04
            stains = stain_count if stain_count > 0 else 3
            stain = stain_strength if stain_strength > 0 else 0.10
            toned = _apply_spatial_wear(template, rng, gain, strength, gamma, stains, stain)
        else:
            toned = np.clip(template.astype(np.float32) * gain, 0, 255)
        toned = np.clip(toned + rng.normal(0, noise_sigma, toned.shape), 0, 255).astype(np.uint8)

        for piece in range(pieces_per_note):
            mask = labels == piece
            if not np.any(mask):
                continue
            image = np.where(mask[..., None], toned, 0)
            carries_serial = bool(mask[y0:y1, x0:x1].sum() >= 12)
            fragments.append(
                Fragment(
                    id=f"n{note_index:03d}f{piece:03d}",
                    label=serial if carries_serial else None,
                    side=side,
                    mask=mask,
                    image=image,
                    meta={
                        "note_id": note_id,
                        "serial": serial,
                        "gain": [round(float(value), 4) for value in gain],
                        "wear_model": wear_model,
                        "partition_model": partition_model,
                    },
                )
            )
    return template, fragments


def save_dataset(path: str | Path, template: np.ndarray, fragments: list[Fragment]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    masks = np.stack([fragment.mask for fragment in fragments], axis=0)
    images = np.stack(
        [
            fragment.image if fragment.image is not None else np.zeros((*fragment.mask.shape, 3), dtype=np.uint8)
            for fragment in fragments
        ],
        axis=0,
    )
    has_images = np.array([fragment.image is not None for fragment in fragments], dtype=bool)
    ids = np.array([fragment.id for fragment in fragments])
    labels = np.array([fragment.label or "" for fragment in fragments])
    sides = np.array([fragment.side for fragment in fragments])
    metas = np.array([json.dumps(fragment.meta) for fragment in fragments])
    np.savez_compressed(
        path,
        template=template,
        masks=masks,
        images=images,
        has_images=has_images,
        ids=ids,
        labels=labels,
        sides=sides,
        metas=metas,
    )


def load_dataset(path: str | Path) -> tuple[np.ndarray, list[Fragment]]:
    data = np.load(Path(path), allow_pickle=False)
    template = data["template"]
    masks = data["masks"].astype(bool)
    images = data["images"] if "images" in data.files else None
    has_images = data["has_images"].astype(bool) if "has_images" in data.files else np.zeros(len(masks), dtype=bool)
    ids = [str(value) for value in data["ids"]]
    labels = [str(value) or None for value in data["labels"]]
    sides = [str(value) for value in data["sides"]]
    metas = [json.loads(str(value)) for value in data["metas"]] if "metas" in data.files else [{} for _ in ids]
    fragments = [
        Fragment(
            id=ids[index],
            label=labels[index],
            side=sides[index],
            mask=masks[index],
            image=images[index] if images is not None and has_images[index] else np.where(masks[index][..., None], template, 0),
            meta=metas[index],
        )
        for index in range(len(ids))
    ]
    return template, fragments
