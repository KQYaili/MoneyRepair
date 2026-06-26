import time
import numpy as np
from moneyrepair.types import Fragment
from moneyrepair.simulate import synthetic_banknote
from moneyrepair.locator import locate_fragment_poses, _rotate_image_and_mask

def test_hybrid_locator_correctness_and_performance():
    # 1. Create templates
    width, height = 360, 160
    template_front = synthetic_banknote(width=width, height=height, seed=42)
    template_back = synthetic_banknote(width=width, height=height, seed=100)

    # 2. Extract a true fragment crop at a known position (side=front, rotation=0)
    tx, ty = 80, 40
    fw, fh = 40, 30
    
    mask = np.zeros((height, width), dtype=bool)
    mask[ty : ty + fh, tx : tx + fw] = True
    
    raw_img = np.where(mask[..., None], template_front, 0)
    
    frag = Fragment(
        id="hybrid_test_frag",
        mask=mask,
        image=raw_img,
    )

    # 3. Find candidate poses and time it (take minimum of 3 runs to isolate JIT compilation and OS noise)
    # Warm up to trigger JIT compilation
    locate_fragment_poses(frag, template_front, template_back, top_k=3, coarse_step=8)
    
    elapsed_times = []
    poses = []
    for _ in range(3):
        start_time = time.perf_counter()
        poses = locate_fragment_poses(frag, template_front, template_back, top_k=3, coarse_step=8)
        elapsed_times.append(time.perf_counter() - start_time)
        
    elapsed = min(elapsed_times)
    print(f"Hybrid locator solved in {elapsed * 1000:.2f} ms")
    
    # 4. Assert correctness
    assert len(poses) > 0
    top = poses[0]
    assert top.side == "front"
    # Ensure it locates the translation within the coarse resolution boundary (coarse_step = 8)
    assert abs(top.tx - tx) <= 1
    assert abs(top.ty - ty) <= 1
    assert top.angle == 0
    assert top.score > 0.8
    
    # 5. Assert performance (should be way under 800 ms to account for environment/compilation overhead)
    assert elapsed < 0.8


def test_multiple_poses_of_same_fragment_mutual_exclusion():
    from moneyrepair.locator import build_pose_compatibility_matrix
    from moneyrepair.solver import solve_covering_sets
    
    # 1. Create two virtual placed fragments belonging to the same original_id
    mask_a = np.zeros((4, 4), dtype=bool)
    mask_a[0, :] = True
    mask_b = np.zeros((4, 4), dtype=bool)
    mask_b[1, :] = True
    
    frag_a = Fragment(
        id="f0_p0",
        mask=mask_a,
        meta={"original_id": "f0", "pose_id": "f0_p0"}
    )
    frag_b = Fragment(
        id="f0_p1",
        mask=mask_b,
        meta={"original_id": "f0", "pose_id": "f0_p1"}
    )
    
    # Another independent fragment
    mask_c = np.zeros((4, 4), dtype=bool)
    mask_c[2, :] = True
    frag_c = Fragment(
        id="f1_p0",
        mask=mask_c,
        meta={"original_id": "f1", "pose_id": "f1_p0"}
    )
    
    placed_fragments = [frag_a, frag_b, frag_c]
    matrix = build_pose_compatibility_matrix(placed_fragments)
    
    # Assert that f0_p0 and f0_p1 are marked incompatible
    idx_a = matrix.index("f0_p0")
    idx_b = matrix.index("f0_p1")
    assert not matrix.compatible[idx_a, idx_b]
    assert not matrix.compatible[idx_b, idx_a]
    
    # Run solver
    solutions = solve_covering_sets(
        placed_fragments,
        matrix,
        target_coverage=0.4,
        max_solutions=5
    )
    
    # Assert that no solution contains both f0_p0 and f0_p1
    for sol in solutions:
        ids = set(sol.fragment_ids)
        assert not ("f0_p0" in ids and "f0_p1" in ids)


def test_hybrid_locator_robustness_to_noise_and_rotation():
    # 1. Create templates
    width, height = 360, 160
    template_front = synthetic_banknote(width=width, height=height, seed=42)
    template_back = synthetic_banknote(width=width, height=height, seed=100)

    # 2. Extract a true fragment crop at a known position on back template
    tx, ty = 120, 50
    fw, fh = 50, 40
    
    mask = np.zeros((height, width), dtype=bool)
    mask[ty : ty + fh, tx : tx + fw] = True
    
    # Crop the back template region
    raw_img = np.where(mask[..., None], template_back, 0)
    
    # Extract tight crop image and mask
    from moneyrepair.locator import _crop_foreground
    crop_img, crop_mask = _crop_foreground(raw_img, mask)
    
    # Rotate the cropped fragment by 270 degrees counter-clockwise (so best match requires 90 degrees rotation)
    rotated_img, rotated_mask = _rotate_image_and_mask(crop_img, crop_mask, 270)
    
    # Construct a fragment with the rotated crop
    frag_mask = np.zeros((100, 100), dtype=bool)
    rh, rw = rotated_mask.shape[:2]
    frag_mask[10 : 10 + rh, 10 : 10 + rw] = rotated_mask
    
    frag_img = np.zeros((100, 100, 3), dtype=np.uint8)
    frag_img[10 : 10 + rh, 10 : 10 + rw][rotated_mask] = rotated_img[rotated_mask]
    
    # Apply color cast / brightness gain to the fragment image to simulate real camera characteristics
    frag_img = frag_img.astype(np.float64)
    # Gain of 0.8 on red, 1.2 on green, 0.9 on blue
    frag_img[..., 0] *= 0.8
    frag_img[..., 1] *= 1.2
    frag_img[..., 2] *= 0.9
    # Add random noise
    rng = np.random.default_rng(12345)
    noise = rng.normal(0, 5, frag_img.shape)
    frag_img = np.clip(frag_img + noise, 0, 255).astype(np.uint8)
    
    frag = Fragment(
        id="robustness_test_frag",
        mask=frag_mask,
        image=frag_img,
    )
    
    # 3. Locate poses
    poses = locate_fragment_poses(frag, template_front, template_back, top_k=3, coarse_step=8)
    
    assert len(poses) > 0
    top = poses[0]
    
    # The correct side is back
    assert top.side == "back"
    
    # The correct rotation angle is 90 degrees
    assert top.angle == 90
    
    # The translation should match the original tx, ty within a tolerance of coarse_step / 2
    assert abs(top.tx - tx) <= 1
    assert abs(top.ty - ty) <= 1
    
    # The score should remain high (e.g. > 0.7) despite color shift and noise
    assert top.score > 0.7


def test_adaptive_top_k_filtering():
    width, height = 360, 160
    template_front = synthetic_banknote(width=width, height=height, seed=42)
    template_back = synthetic_banknote(width=width, height=height, seed=100)
    tx, ty = 80, 40
    fw, fh = 40, 30
    
    mask = np.zeros((height, width), dtype=bool)
    mask[ty : ty + fh, tx : tx + fw] = True
    raw_img = np.where(mask[..., None], template_front, 0)
    
    frag = Fragment(id="test_adaptive", mask=mask, image=raw_img)
    
    # Locate with min_score=1.05 (should return empty list since score is in (0, 1])
    poses_high = locate_fragment_poses(frag, template_front, template_back, min_score=1.05)
    assert len(poses_high) == 0
    
    # Locate with score_margin = 0.0001 (should keep only the best candidate)
    poses_tight = locate_fragment_poses(frag, template_front, template_back, score_margin=0.0001)
    assert len(poses_tight) <= 1


def test_numba_boundary_continuity():
    from moneyrepair.locator import numba_boundary_continuity
    # Create two adjacent masks of size 10x10
    mask_a = np.zeros((10, 10), dtype=bool)
    mask_a[:5, :] = True
    mask_b = np.zeros((10, 10), dtype=bool)
    mask_b[5:, :] = True
    
    # Case A: continuous colors (both all white)
    img_a = np.ones((10, 10, 3), dtype=np.uint8) * 200
    img_b = np.ones((10, 10, 3), dtype=np.uint8) * 200
    assert numba_boundary_continuity(img_a, mask_a, img_b, mask_b, max_boundary_diff=10.0)
    
    # Case B: discontinuous colors (one white, one black)
    img_b_dark = np.zeros((10, 10, 3), dtype=np.uint8)
    assert not numba_boundary_continuity(img_a, mask_a, img_b_dark, mask_b, max_boundary_diff=10.0)


def test_clustered_pose_compatibility_matrix():
    from moneyrepair.locator import build_pose_compatibility_matrix
    # Create placed fragments representing different physical fragments
    mask_a = np.zeros((4, 4), dtype=bool)
    mask_a[0, :] = True
    mask_b = np.zeros((4, 4), dtype=bool)
    mask_b[1, :] = True
    
    frag_a = Fragment(id="f0_p0", mask=mask_a, image=np.zeros((4, 4, 3), dtype=np.uint8), meta={"original_id": "f0"})
    frag_b = Fragment(id="f1_p0", mask=mask_b, image=np.zeros((4, 4, 3), dtype=np.uint8), meta={"original_id": "f1"})
    
    # With groups: f0 is in group 0, f1 is in group 1 (different notes)
    # They should be marked incompatible even though they don't overlap!
    groups = {"f0": 0, "f1": 1}
    matrix_clustered = build_pose_compatibility_matrix([frag_a, frag_b], groups=groups)
    idx_a = matrix_clustered.index("f0_p0")
    idx_b = matrix_clustered.index("f1_p0")
    assert not matrix_clustered.compatible[idx_a, idx_b]
    
    # Without groups (default), they don't overlap so they are compatible
    matrix_normal = build_pose_compatibility_matrix([frag_a, frag_b])
    assert matrix_normal.compatible[idx_a, idx_b]


def test_pose_compatibility_matrix_forwards_overlap_tolerance():
    from moneyrepair.locator import build_pose_compatibility_matrix

    mask_a = np.zeros((4, 4), dtype=bool)
    mask_a[0:2, 0:2] = True
    mask_b = np.zeros((4, 4), dtype=bool)
    mask_b[1:3, 1:3] = True
    frag_a = Fragment(id="f0_p0", mask=mask_a, image=np.zeros((4, 4, 3), dtype=np.uint8), meta={"original_id": "f0"})
    frag_b = Fragment(id="f1_p0", mask=mask_b, image=np.zeros((4, 4, 3), dtype=np.uint8), meta={"original_id": "f1"})

    strict = build_pose_compatibility_matrix([frag_a, frag_b])
    tolerant = build_pose_compatibility_matrix([frag_a, frag_b], max_overlap_pixels=1)
    idx_a = strict.index("f0_p0")
    idx_b = strict.index("f1_p0")

    assert not strict.compatible[idx_a, idx_b]
    assert tolerant.compatible[idx_a, idx_b]


def test_clustered_pose_compatibility_matrix_forwards_overlap_tolerance():
    from moneyrepair.locator import build_pose_compatibility_matrix

    mask_a = np.zeros((4, 4), dtype=bool)
    mask_a[0:2, 0:2] = True
    mask_b = np.zeros((4, 4), dtype=bool)
    mask_b[1:3, 1:3] = True
    frag_a = Fragment(id="f0_p0", mask=mask_a, image=np.zeros((4, 4, 3), dtype=np.uint8), meta={"original_id": "f0"})
    frag_b = Fragment(id="f1_p0", mask=mask_b, image=np.zeros((4, 4, 3), dtype=np.uint8), meta={"original_id": "f1"})
    groups = {"f0": 0, "f1": 0}

    strict = build_pose_compatibility_matrix([frag_a, frag_b], groups=groups)
    tolerant = build_pose_compatibility_matrix([frag_a, frag_b], groups=groups, max_overlap_pixels=1)
    idx_a = strict.index("f0_p0")
    idx_b = strict.index("f1_p0")

    assert not strict.compatible[idx_a, idx_b]
    assert tolerant.compatible[idx_a, idx_b]


