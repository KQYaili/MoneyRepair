from __future__ import annotations

import json
import numba
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


def downsample_2x_image(img: np.ndarray) -> np.ndarray:
    """Downsample image by 2x using area average."""
    h, w = img.shape[:2]
    h_down = h // 2
    w_down = w // 2
    img_down = (
        img[0:2*h_down:2, 0:2*w_down:2].astype(np.float32) +
        img[1:2*h_down:2, 0:2*w_down:2].astype(np.float32) +
        img[0:2*h_down:2, 1:2*w_down:2].astype(np.float32) +
        img[1:2*h_down:2, 1:2*w_down:2].astype(np.float32)
    ) / 4.0
    return img_down.astype(np.uint8)


def downsample_2x_mask(mask: np.ndarray) -> np.ndarray:
    """Downsample binary mask by 2x using Max Pooling (region logical OR)."""
    h, w = mask.shape
    h_down = h // 2
    w_down = w // 2
    mask_down = (
        mask[0:2*h_down:2, 0:2*w_down:2] |
        mask[1:2*h_down:2, 0:2*w_down:2] |
        mask[0:2*h_down:2, 1:2*w_down:2] |
        mask[1:2*h_down:2, 1:2*w_down:2]
    )
    return mask_down


@numba.njit(fastmath=True, cache=True)
def numba_coarse_search_loop(
    ref_img,       # (ref_h, ref_w, 3) float64
    crop_img,      # (ch, cw, 3) float64
    mask_indices,  # (P, 2) int32
    ref_h, ref_w, ch, cw,
    coarse_step,
):
    num_y = (ref_h - ch) // coarse_step + 1
    num_x = (ref_w - cw) // coarse_step + 1
    total_positions = num_y * num_x
    
    results_tx = np.empty(total_positions, dtype=np.int32)
    results_ty = np.empty(total_positions, dtype=np.int32)
    results_score = np.empty(total_positions, dtype=np.float64)
    
    P = mask_indices.shape[0]
    obs = np.empty((P, 3), dtype=np.float64)
    for p in range(P):
        y = mask_indices[p, 0]
        x = mask_indices[p, 1]
        obs[p, 0] = crop_img[y, x, 0]
        obs[p, 1] = crop_img[y, x, 1]
        obs[p, 2] = crop_img[y, x, 2]
        
    idx = 0
    for ty in range(0, ref_h - ch + 1, coarse_step):
        for tx in range(0, ref_w - cw + 1, coarse_step):
            # Compute gain factor for each channel
            gains = np.ones(3, dtype=np.float64)
            for c in range(3):
                dot_obs_ref = 0.0
                dot_ref_ref = 0.0
                for p in range(P):
                    y = mask_indices[p, 0]
                    x = mask_indices[p, 1]
                    r_val = ref_img[ty + y, tx + x, c]
                    o_val = obs[p, c]
                    dot_obs_ref += o_val * r_val
                    dot_ref_ref += r_val * r_val
                if dot_ref_ref > 1e-6:
                    g = dot_obs_ref / dot_ref_ref
                    if g < 0.4:
                        gains[c] = 0.4
                    elif g > 2.5:
                        gains[c] = 2.5
                    else:
                        gains[c] = g
            
            # Compute MAE
            mae_sum = 0.0
            for c in range(3):
                g = gains[c]
                for p in range(P):
                    y = mask_indices[p, 0]
                    x = mask_indices[p, 1]
                    r_val = ref_img[ty + y, tx + x, c]
                    o_val = obs[p, c]
                    mae_sum += abs(o_val - g * r_val)
                    
            mae = mae_sum / (3.0 * P)
            score = np.exp(-mae / 35.0)
            
            results_tx[idx] = tx
            results_ty[idx] = ty
            results_score[idx] = score
            idx += 1
            
    return results_tx[:idx], results_ty[:idx], results_score[:idx]


def locate_fragment_poses(
    fragment: Fragment,
    ref_front: np.ndarray,
    ref_back: np.ndarray,
    *,
    top_k: int = 3,
    coarse_step: int = 8,
) -> list[CandidatePose]:
    """Find the best-fitting translations/rotations for a fragment on the template.

    Uses a coarse-to-fine hybrid search combining resolution pyramids, JIT compilation,
    and fine-res neighborhood refinement.
    """
    if fragment.image is None:
        return []
    
    # Extract tight crop
    raw_crop_img, raw_crop_mask = _crop_foreground(fragment.image, fragment.mask)
    if raw_crop_mask.size == 0:
        return []

    ref_h, ref_w = ref_front.shape[:2]
    
    # Pre-downsample templates for Level 1 (0.5x resolution)
    ref_front_down = downsample_2x_image(ref_front)
    ref_back_down = downsample_2x_image(ref_back)
    ref_front_down_64 = ref_front_down.astype(np.float64)
    ref_back_down_64 = ref_back_down.astype(np.float64)
    
    step_down = max(1, coarse_step // 2)
    candidates: list[CandidatePose] = []

    # 1. Coarse search on Level 1 (0.5x)
    for side_name, ref_img_down_64 in (("front", ref_front_down_64), ("back", ref_back_down_64)):
        for angle in (0, 90, 180, 270):
            # Rotate crop
            crop_img, crop_mask = _rotate_image_and_mask(raw_crop_img, raw_crop_mask, angle)
            
            # Downsample crop for Level 1
            crop_img_down = downsample_2x_image(crop_img)
            crop_mask_down = downsample_2x_mask(crop_mask)
            ch_down, cw_down = crop_mask_down.shape[:2]
            
            ref_h_down, ref_w_down = ref_img_down_64.shape[:2]
            if ch_down > ref_h_down or cw_down > ref_w_down:
                continue

            mask_indices = np.argwhere(crop_mask_down).astype(np.int32)
            if mask_indices.size == 0:
                continue

            crop_img_down_64 = crop_img_down.astype(np.float64)

            # Call JIT-compiled search loop
            tx_arr, ty_arr, score_arr = numba_coarse_search_loop(
                ref_img_down_64,
                crop_img_down_64,
                mask_indices,
                ref_h_down, ref_w_down,
                ch_down, cw_down,
                step_down
            )

            # Accumulate candidates mapping back to Level 0 (x2)
            for idx in range(len(score_arr)):
                candidates.append(
                    CandidatePose(
                        fragment_id=fragment.id,
                        pose_id="",
                        side=side_name,
                        tx=tx_arr[idx] * 2,
                        ty=ty_arr[idx] * 2,
                        angle=angle,
                        score=score_arr[idx],
                    )
                )

    # Sort candidates and keep top matches for refinement
    candidates = sorted(candidates, key=lambda p: p.score, reverse=True)[:10]

    refined_candidates: list[CandidatePose] = []
    # 2. Fine search at Level 0 (1x) in a small neighborhood
    for p in candidates:
        ref_img = ref_front if p.side == "front" else ref_back
        crop_img, crop_mask = _rotate_image_and_mask(raw_crop_img, raw_crop_mask, p.angle)
        ch, cw = crop_mask.shape[:2]
        
        best_score = -1.0
        best_tx = p.tx
        best_ty = p.ty

        search_range = max(2, coarse_step // 2)
        for dy in range(-search_range, search_range + 1):
            for dx in range(-search_range, search_range + 1):
                tx = p.tx + dx
                ty = p.ty + dy
                if 0 <= tx <= ref_w - cw and 0 <= ty <= ref_h - ch:
                    ref_window = ref_img[ty : ty + ch, tx : tx + cw]
                    score = _match_score(crop_img, crop_mask, ref_window)
                    if score > best_score:
                        best_score = score
                        best_tx = tx
                        best_ty = ty

        refined_candidates.append(
            CandidatePose(
                fragment_id=p.fragment_id,
                pose_id="",
                side=p.side,
                tx=best_tx,
                ty=best_ty,
                angle=p.angle,
                score=best_score,
            )
        )

    # 3. De-duplicate close poses
    unique_poses: list[CandidatePose] = []
    for p in sorted(refined_candidates, key=lambda x: x.score, reverse=True):
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


def build_pose_compatibility_matrix(
    placed_fragments: list[Fragment],
    cell: int | None = None,
) -> CompatibilityMatrix:
    """Build compatibility matrix for placed fragments, enforcing mutual exclusion of poses from the same fragment."""
    from moneyrepair.compat import compute_compatibility_fast, CompatibilityMatrix
    
    # 1. Compute physical overlap compatibility using the fast grid engine
    packed = compute_compatibility_fast(placed_fragments, cell=cell)
    matrix = packed.to_dense()
    
    # 2. Enforce mutual exclusion for poses of the same physical fragment
    n = len(placed_fragments)
    for i in range(n):
        orig_i = placed_fragments[i].meta.get("original_id")
        if orig_i is None:
            continue
        for j in range(i + 1, n):
            orig_j = placed_fragments[j].meta.get("original_id")
            if orig_i == orig_j:
                matrix.compatible[i, j] = False
                matrix.compatible[j, i] = False
                
    return matrix
