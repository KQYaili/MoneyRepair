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
    
    # 5. Assert performance (should be way under 120 ms now due to Level 1 downsampling and JIT)
    assert elapsed < 0.12
