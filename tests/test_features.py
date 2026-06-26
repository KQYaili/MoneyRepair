import numpy as np

from moneyrepair.features import describe_contours, match_similar_contours
from moneyrepair.types import Fragment


def test_describe_contours_reports_tags_and_histogram():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:3, :3] = True
    records = describe_contours([Fragment("a", mask)], direction_bins=4)

    assert records[0].fragment_id == "a"
    assert "corner" in records[0].tags
    assert len(records[0].direction_histogram) == 4


def test_match_similar_contours_uses_tag_compatible_pairs():
    mask_a = np.zeros((12, 12), dtype=bool)
    mask_b = np.zeros((12, 12), dtype=bool)
    mask_c = np.zeros((12, 12), dtype=bool)
    mask_a[1:5, 1:5] = True
    mask_b[1:5, 7:11] = True
    mask_c[7:10, 7:10] = True

    matches = match_similar_contours(
        [
            Fragment("a", mask_a),
            Fragment("b", mask_b),
            Fragment("c", mask_c),
        ],
        max_distance=0.05,
    )

    assert any({item["left"], item["right"]} == {"a", "b"} for item in matches)


def test_match_raw_crop_contours_finds_subsegment_match():
    from moneyrepair.features import match_raw_crop_contours

    mask_a = np.zeros((32, 32), dtype=bool)
    mask_a[5:15, 5:15] = True

    mask_b = np.zeros((32, 32), dtype=bool)
    mask_b[5:15, 14:24] = True

    matches = match_raw_crop_contours(
        [
            Fragment("a", mask_a),
            Fragment("b", mask_b),
        ],
        segment_length=8,
        max_distance=0.2,
    )

    assert len(matches) == 1
    assert matches[0]["left"] == "a"
    assert matches[0]["right"] == "b"
    assert matches[0]["distance"] < 0.2
    assert "left_start" in matches[0]
    assert "right_start" in matches[0]
    assert isinstance(matches[0]["reversed"], bool)
    assert isinstance(matches[0]["estimated_rotation"], float)
    assert isinstance(matches[0]["estimated_translation"], list)
    assert len(matches[0]["estimated_translation"]) == 2
