import numpy as np

from moneyrepair.interlock import (
    compute_interlock_compatibility,
    compute_interlock_compatibility_with_stats,
    contact_edge_count,
    iter_contact_candidate_pairs,
    tear_interlock_score,
)
from moneyrepair.types import Fragment


def test_contact_edge_count_scores_complementary_masks():
    left_mask = np.zeros((8, 8), dtype=bool)
    left_mask[:, :4] = True
    right_mask = np.zeros((8, 8), dtype=bool)
    right_mask[:, 4:] = True

    short_mask = np.zeros((8, 8), dtype=bool)
    short_mask[0:2, 4:] = True

    assert contact_edge_count(left_mask, right_mask) == 8
    assert contact_edge_count(left_mask, short_mask) == 2


def test_interlock_compatibility_rejects_weak_touch_when_thresholded():
    left_mask = np.zeros((8, 8), dtype=bool)
    left_mask[:, :4] = True
    right_mask = np.zeros((8, 8), dtype=bool)
    right_mask[0:2, 4:] = True
    left = Fragment(id="left", mask=left_mask)
    right = Fragment(id="right", mask=right_mask)

    score = tear_interlock_score(left, right)
    matrix = compute_interlock_compatibility([left, right], min_contact_edges=1, min_contact_ratio=0.2)

    assert score.contact_edges == 2
    assert score.contact_ratio < 0.2
    assert not matrix.to_dense().compatible[0, 1]


def test_interlock_compatibility_keeps_non_adjacent_pairs():
    left_mask = np.zeros((8, 8), dtype=bool)
    left_mask[:, :2] = True
    right_mask = np.zeros((8, 8), dtype=bool)
    right_mask[:, 5:] = True
    matrix = compute_interlock_compatibility(
        [Fragment(id="left", mask=left_mask), Fragment(id="right", mask=right_mask)],
        min_contact_edges=1,
        min_contact_ratio=0.9,
    )

    assert matrix.to_dense().compatible[0, 1]


def test_interlock_stats_and_sparse_candidates_do_not_scan_all_pairs():
    fragments = []
    for idx, x0 in enumerate((0, 5, 20, 30)):
        mask = np.zeros((8, 40), dtype=bool)
        mask[:, x0 : x0 + 4] = True
        fragments.append(Fragment(id=f"f{idx}", mask=mask))

    candidate_pairs = list(iter_contact_candidate_pairs(fragments, cell=4))
    matrix, stats = compute_interlock_compatibility_with_stats(
        fragments,
        cell=4,
        min_contact_edges=1,
        min_contact_ratio=0.9,
    )

    assert len(candidate_pairs) < len(fragments) * (len(fragments) - 1) // 2
    assert stats.bbox_candidate_pairs == len(candidate_pairs)
    assert stats.scored_contact_pairs <= stats.bbox_candidate_pairs
    assert matrix.compatible_pair_count() <= len(fragments) * (len(fragments) - 1) // 2
