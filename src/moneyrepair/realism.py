from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from moneyrepair.simulate import make_synthetic_fragments
from moneyrepair.types import Fragment


@dataclass(frozen=True)
class RealismProfile:
    brightness_jitter: float = 0.16
    contrast_jitter: float = 0.12
    color_jitter: float = 0.08
    blur_radius_max: float = 0.8
    noise_sigma: float = 7.0
    illumination_strength: float = 0.22
    jpeg_quality_min: int = 72
    jpeg_quality_max: int = 94
    edge_dropout: float = 0.0

    def to_dict(self) -> dict:
        return {
            "brightness_jitter": self.brightness_jitter,
            "contrast_jitter": self.contrast_jitter,
            "color_jitter": self.color_jitter,
            "blur_radius_max": self.blur_radius_max,
            "noise_sigma": self.noise_sigma,
            "illumination_strength": self.illumination_strength,
            "jpeg_quality_min": self.jpeg_quality_min,
            "jpeg_quality_max": self.jpeg_quality_max,
            "edge_dropout": self.edge_dropout,
        }


def _factor(rng: np.random.Generator, jitter: float) -> float:
    return float(1.0 + rng.uniform(-jitter, jitter))


def _illumination(image: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    if strength <= 0:
        return image
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    angle = rng.uniform(0, 2 * np.pi)
    direction = np.cos(angle) * (xx / max(width - 1, 1) - 0.5) + np.sin(angle) * (yy / max(height - 1, 1) - 0.5)
    field = 1.0 + strength * direction
    return np.clip(image.astype(np.float32) * field[..., None], 0, 255).astype(np.uint8)


def _jpeg_roundtrip(image: Image.Image, rng: np.random.Generator, profile: RealismProfile) -> Image.Image:
    quality = int(rng.integers(profile.jpeg_quality_min, profile.jpeg_quality_max + 1))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def augment_fragment_image(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    profile: RealismProfile,
) -> np.ndarray:
    """Apply deterministic camera/scan-like degradation to one placed fragment."""

    rgb = image.astype(np.uint8)
    pil = Image.fromarray(rgb, mode="RGB")
    pil = ImageEnhance.Brightness(pil).enhance(_factor(rng, profile.brightness_jitter))
    pil = ImageEnhance.Contrast(pil).enhance(_factor(rng, profile.contrast_jitter))
    pil = ImageEnhance.Color(pil).enhance(_factor(rng, profile.color_jitter))
    if profile.blur_radius_max > 0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0, profile.blur_radius_max))))
    pil = _jpeg_roundtrip(pil, rng, profile)
    arr = np.asarray(pil, dtype=np.uint8)
    arr = _illumination(arr, rng, profile.illumination_strength)
    if profile.noise_sigma > 0:
        noise = rng.normal(0, profile.noise_sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return np.where(mask[..., None], arr, 0)


def make_realistic_synthetic_fragments(
    pieces: int = 24,
    width: int = 420,
    height: int = 180,
    seed: int = 7,
    side: str = "front",
    profile: RealismProfile | None = None,
) -> tuple[np.ndarray, list[Fragment], RealismProfile]:
    """Generate synthetic fragments with capture-like photometric degradation."""

    profile = profile or RealismProfile()
    template, fragments = make_synthetic_fragments(pieces=pieces, width=width, height=height, seed=seed, side=side)
    rng = np.random.default_rng(seed + 10_000)
    degraded: list[Fragment] = []
    for fragment in fragments:
        image = fragment.image if fragment.image is not None else np.where(fragment.mask[..., None], template, 0)
        degraded.append(
            Fragment(
                id=fragment.id,
                label=fragment.label,
                side=fragment.side,
                mask=fragment.mask,
                image=augment_fragment_image(image, fragment.mask, rng, profile),
                tags=fragment.tags,
                meta={**fragment.meta, "realism_profile": profile.to_dict()},
            )
        )
    return template, degraded, profile
