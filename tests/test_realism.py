import numpy as np

from moneyrepair.realism import RealismProfile, make_realistic_synthetic_fragments
from moneyrepair.simulate import make_synthetic_fragments


def test_realistic_synthetic_fragments_preserve_masks_and_degrade_rgb():
    template, clean = make_synthetic_fragments(pieces=8, width=80, height=40, seed=3)
    _, realistic, profile = make_realistic_synthetic_fragments(
        pieces=8,
        width=80,
        height=40,
        seed=3,
        profile=RealismProfile(noise_sigma=10.0, blur_radius_max=0.2),
    )

    assert profile.noise_sigma == 10.0
    assert len(realistic) == len(clean)
    np.testing.assert_array_equal(realistic[0].mask, clean[0].mask)
    assert not np.array_equal(realistic[0].image, clean[0].image)
    assert realistic[0].image.shape == template.shape
