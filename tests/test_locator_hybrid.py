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

