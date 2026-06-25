from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from PIL import Image

from moneyrepair.types import Fragment

@dataclass(frozen=True)
class CandidatePose:
    """A candidate placement pose for a fragment on a banknote template."""

    fragment_id: str
    pose_id: str
    side: str  # "front" | "back"
    tx: int  # translation X in note space
    ty: int  # translation Y in note space
    angle: int  # rotation angle in degrees (0, 90, 180, 270)
    score: float  # similarity score (higher is better)

    def to_dict(self) -> dict:
        return {
            "fragment_id": self.fragment_id,
            "pose_id": self.pose_id,
            "side": self.side,
            "tx": self.tx,
            "ty": self.ty,
            "angle": self.angle,
            "score": self.score,
        }


def _rotate_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    angle: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate image and mask by 0, 90, 180, or 270 degrees."""
    if angle == 0:
        return image, mask
    elif angle == 90:
        return np.rot90(image, 1, (0, 1)), np.rot90(mask, 1, (0, 1))
    elif angle == 180:
        return np.rot90(image, 2, (0, 1)), np.rot90(mask, 2, (0, 1))
    elif angle == 270:
        return np.rot90(image, 3, (0, 1)), np.rot90(mask, 3, (0, 1))
    else:
        raise ValueError("Angle must be 0, 90, 180, or 270")


def _crop_foreground(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop the bounding box of the foreground mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return image[:0, :0], mask[:0, :0]
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _match_score(
    crop_img: np.ndarray,
    crop_mask: np.ndarray,
    ref_window: np.ndarray,
) -> float:
    """Compute brightness-normalized MAE similarity score in [0, 1]."""
    # Cast to float64
    obs = crop_img.astype(np.float64)[crop_mask]
    ref = ref_window.astype(np.float64)[crop_mask]
    if len(obs) == 0:
        return 0.0

    # Solve observed = gain * ref for each channel
    gains = np.ones(3)
    for c in range(3):
        r = ref[:, c]
        denom = float(np.dot(r, r))
        if denom > 1e-6:
            gains[c] = float(np.dot(obs[:, c], r) / denom)

    # Normalize gain to prevent extreme scaling
    gains = np.clip(gains, 0.4, 2.5)
    normalized_ref = ref * gains[None, :]
    
    # Compute mean absolute error (MAE)
    mae = float(np.mean(np.abs(obs - normalized_ref)))
    
    # Map MAE to similarity score in [0, 1]
    return float(np.exp(-mae / 35.0))


def locate_fragment_poses(
    fragment: Fragment,
    ref_front: np.ndarray,
    ref_back: np.ndarray,
    *,
    top_k: int = 3,
    coarse_step: int = 8,
) -> list[CandidatePose]:
    """Find the best-fitting translations/rotations for a fragment on the template.

    Uses a coarse-to-fine grid search over X/Y positions, sides, and rotation angles.
    """
    if fragment.image is None:
        return []
    
    # Extract tight crop
    raw_crop_img, raw_crop_mask = _crop_foreground(fragment.image, fragment.mask)
    if raw_crop_mask.size == 0:
        return []

    ref_h, ref_w = ref_front.shape[:2]
    candidates: list[CandidatePose] = []

    # Coarse search
    for side_name, ref_img in (("front", ref_front), ("back", ref_back)):
        for angle in (0, 90, 180, 270):
            # Rotate crop
            crop_img, crop_mask = _rotate_image_and_mask(raw_crop_img, raw_crop_mask, angle)
            ch, cw = crop_mask.shape[:2]
            if ch > ref_h or cw > ref_w:
                continue

            # Sliding window coarse search
            for ty in range(0, ref_h - ch + 1, coarse_step):
                for tx in range(0, ref_w - cw + 1, coarse_step):
                    ref_window = ref_img[ty : ty + ch, tx : tx + cw]
                    score = _match_score(crop_img, crop_mask, ref_window)
                    candidates.append(
                        CandidatePose(
                            fragment_id=fragment.id,
                            pose_id="",
                            side=side_name,
                            tx=tx,
                            ty=ty,
                            angle=angle,
                            score=score,
                        )
                    )

    # Sort candidates and keep top matches for refinement
    candidates = sorted(candidates, key=lambda p: p.score, reverse=True)[:10]

    refined_candidates: list[CandidatePose] = []
    # Fine search (refining X, Y in a small neighborhood)
    for p in candidates:
        ref_img = ref_front if p.side == "front" else ref_back
        crop_img, crop_mask = _rotate_image_and_mask(raw_crop_img, raw_crop_mask, p.angle)
        ch, cw = crop_mask.shape[:2]
        
        best_score = p.score
        best_tx = p.tx
        best_ty = p.ty

        # Search local neighborhood
        for dy in range(-coarse_step // 2, coarse_step // 2 + 1):
            for dx in range(-coarse_step // 2, coarse_step // 2 + 1):
                tx = int(np.clip(p.tx + dx, 0, ref_w - cw))
                ty = int(np.clip(p.ty + dy, 0, ref_h - ch))
                ref_window = ref_img[ty : ty + ch, tx : tx + cw]
                score = _match_score(crop_img, crop_mask, ref_window)
                if score > best_score:
                    best_score = score
                    best_tx = tx
                    best_ty = ty

        refined_candidates.append(
            CandidatePose(
                fragment_id=p.fragment_id,
                pose_id="",  # filled later
                side=p.side,
                tx=best_tx,
                ty=best_ty,
                angle=p.angle,
                score=best_score,
            )
        )

    # De-duplicate close poses
    unique_poses: list[CandidatePose] = []
    for p in sorted(refined_candidates, key=lambda x: x.score, reverse=True):
        # Avoid duplicate overlap of near-identical poses
        is_dup = False
        for up in unique_poses:
            if up.side == p.side and up.angle == p.angle and abs(up.tx - p.tx) <= 2 and abs(up.ty - p.ty) <= 2:
                is_dup = True
                break
        if not is_dup:
            pose_idx = len(unique_poses)
            pose_id = f"{p.fragment_id}_pose{pose_idx}"
            unique_poses.append(
                CandidatePose(
                    fragment_id=p.fragment_id,
                    pose_id=pose_id,
                    side=p.side,
                    tx=p.tx,
                    ty=p.ty,
                    angle=p.angle,
                    score=p.score,
                )
            )
            if len(unique_poses) >= top_k:
                break

    return unique_poses
