import numpy as np
import pytest
from PIL import Image

from moneyrepair.types import Fragment
from moneyrepair.simulate import synthetic_banknote
from moneyrepair.locator import locate_fragment_poses, CandidatePose, _rotate_image_and_mask


def test_locate_fragment_poses_finds_correct_pose():
    # 1. Create templates
    width, height = 120, 60
    template_front = synthetic_banknote(width=width, height=height, seed=42)
    template_back = synthetic_banknote(width=width, height=height, seed=100)

    # 2. Extract a true fragment crop at a known position
    tx, ty = 20, 10
    fw, fh = 40, 20
    
    # Crop template to form a fragment mask and image
    mask = np.zeros((height, width), dtype=bool)
    mask[ty : ty + fh, tx : tx + fw] = True
    
    # Simulate some rotation: e.g., 90 degrees
    # If the fragment was rotated, the crop we get is rotated
    raw_img = np.where(mask[..., None], template_front, 0)
    
    # We create a fragment that is placed (so it already occupies the coordinate space)
    frag = Fragment(
        id="test_frag",
        mask=mask,
        image=raw_img,
    )

    # Find candidate poses
    poses = locate_fragment_poses(frag, template_front, template_back, top_k=3, coarse_step=4)

    assert len(poses) > 0
    # The top candidate should represent the true side, translation, and 0 degree rotation
    top = poses[0]
    assert top.side == "front"
    assert abs(top.tx - tx) <= 1
    assert abs(top.ty - ty) <= 1
    assert top.angle == 0
    assert top.score > 0.8
