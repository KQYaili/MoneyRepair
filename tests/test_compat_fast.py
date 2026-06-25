import numpy as np

from moneyrepair.compat import (
    compute_compatibility,
    compute_compatibility_fast,
    restrict_packed_to_ids,
    write_incompatible_pairs,
)
from moneyrepair.types import Fragment


def _band_fragment(frag_id: str, x0: int, x1: int, height: int = 10, width: int = 16) -> Fragment:
    mask = np.zeros((height, width), dtype=bool)
    mask[:, x0:x1] = True
    return Fragment(id=frag_id, mask=mask)


def _overlapping_fragments() -> list[Fragment]:
    # a/b overlap on columns 3..4; c/d/e are disjoint bands.
    return [
        _band_fragment("a", 0, 5),
        _band_fragment("b", 3, 8),
        _band_fragment("c", 9, 11),
        _band_fragment("d", 11, 13),
        _band_fragment("e", 13, 16),
    ]


def test_fast_matches_naive_compatibility():
    fragments = _overlapping_fragments()
    naive = compute_compatibility(fragments)
    fast = compute_compatibility_fast(fragments).to_dense()

    assert fast.ids == naive.ids
    np.testing.assert_array_equal(fast.compatible, naive.compatible)
    # a and b are the only incompatible pair.
    assert not naive.is_compatible("a", "b")
    assert naive.is_compatible("c", "d")


def test_compatible_pair_count_handles_padding():
    fragments = _overlapping_fragments()  # 5 fragments -> last byte has padding bits
    fast = compute_compatibility_fast(fragments)
    total_pairs = len(fragments) * (len(fragments) - 1) // 2
    naive = compute_compatibility(fragments)
    naive_incompatible = int((~naive.compatible).sum() - len(naive.ids)) // 2

    assert fast.compatible_pair_count() == total_pairs - naive_incompatible


def test_write_incompatible_pairs_streams_only_conflicts(tmp_path):
    fragments = _overlapping_fragments()
    output = tmp_path / "pairs.csv"
    count = write_incompatible_pairs(output, fragments)

    assert count == 1
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "fragment_a,fragment_b"
    assert lines[1].split(",") == ["a", "b"]


def test_restrict_packed_to_ids_isolates_removed_fragment():
    fragments = _overlapping_fragments()
    fast = compute_compatibility_fast(fragments)
    restricted = restrict_packed_to_ids(fast, {"c", "d", "e"}).to_dense()

    assert not restricted.is_compatible("a", "c")
    assert not restricted.is_compatible("a", "b")
    assert restricted.is_compatible("c", "d")
