from __future__ import annotations

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
    np.savez_compressed(
        path,
        template=template,
        masks=masks,
        images=images,
        has_images=has_images,
        ids=ids,
        labels=labels,
        sides=sides,
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
    fragments = [
        Fragment(
            id=ids[index],
            label=labels[index],
            side=sides[index],
            mask=masks[index],
            image=images[index] if images is not None and has_images[index] else np.where(masks[index][..., None], template, 0),
        )
        for index in range(len(ids))
    ]
    return template, fragments
