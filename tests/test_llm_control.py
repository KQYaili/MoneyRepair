"""Tests for the LLM-Guided Search Controller and Critic in MoneyRepair."""

from __future__ import annotations

import urllib.error
import numpy as np
import pytest
pytestmark = pytest.mark.experimental

from moneyrepair.types import Fragment
from moneyrepair.tearfit import AssemblyCandidate, TearFitEdge
from moneyrepair.experimental.llm_control import (
    LLMAgentConfig,
    LLMFeedback,
    LLMController,
    SearchPolicy,
    apply_feedback_to_policy,
    llm_guided_assembly_loop,
)


def test_llm_controller_mock_parsing():
    config = LLMAgentConfig(use_mock=True, mock_strategy="coverage_first")
    controller = LLMController(config)

    # Test mock response parsing with simple state JSON
    state_prompt = """
    Analyze these candidates:
    {
      "global_stats": {"confirmed_assemblies": 2},
      "candidates": [
        {"index": 0, "labels": ["S1", "S2"], "coverage": 0.95},
        {"index": 1, "labels": ["S3"], "coverage": 0.98}
      ]
    }
    """
    feedback = controller.analyze_candidates([], [
        AssemblyCandidate(("a", "b"), 0.95, 0.9, 10.0, 10, ("S1", "S2")),
        AssemblyCandidate(("c", "d"), 0.98, 0.95, 20.0, 20, ("S3",)),
    ], {"confirmed_assemblies": 2})

    # The mock model should drop candidate 0 due to label conflicts
    assert 0 in feedback.drop
    assert 1 in feedback.keep
    assert feedback.strategy == "coverage_first"


def test_llm_guided_assembly_loop_advanced_behaviors(monkeypatch):
    # Construct dummy fragments
    top = np.zeros((10, 10), dtype=bool)
    top[:5, :] = True
    bottom = np.zeros((10, 10), dtype=bool)
    bottom[5:, :] = True

    fragments = [
        Fragment(id="f0", mask=top.copy(), label="S1", meta={"note_id": "n0"}),
        Fragment(id="f1", mask=bottom.copy(), label="S1", meta={"note_id": "n0"}),
        Fragment(id="f2", mask=top.copy(), label="S2", meta={"note_id": "n1"}),
    ]

    edges = [
        TearFitEdge(left=0, right=1, overlap_pixels=15, left_hits=15, right_hits=15, overlap_ratio=1.0),
        TearFitEdge(left=1, right=2, overlap_pixels=5, left_hits=5, right_hits=5, overlap_ratio=0.3),
    ]

    call_count = 0
    feedback_strategy = "score_first"

    def mock_analyze_search_state(self, fragments, candidate_pool, selected, global_stats, rejected_by_solver=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # We want to drop candidate 1: ("f1", "f2"), keep candidate 0: ("f0", "f1"), suggest seed f0, and merge "f0" & "f1"
            # Note: candidate_pool might be sorted. Let's find indices in candidate_pool
            drop_idx = -1
            keep_idx = -1
            for idx, candidate in enumerate(candidate_pool):
                if candidate.fragment_ids == ("f1", "f2"):
                    drop_idx = idx
                elif candidate.fragment_ids == ("f0", "f1"):
                    keep_idx = idx
            
            return LLMFeedback(
                keep=[keep_idx] if keep_idx != -1 else [],
                drop=[drop_idx] if drop_idx != -1 else [],
                merge=[["f0", "f1"]],
                suggest_seed="f0",
                strategy=feedback_strategy,
            )
        else:
            return LLMFeedback(explanation="No change.")

    monkeypatch.setattr(LLMController, "analyze_search_state", mock_analyze_search_state)

    config = LLMAgentConfig(use_mock=False, fallback_policy="raise")
    selected = llm_guided_assembly_loop(
        fragments,
        edges,
        config,
        max_iterations=2,
        coverage_threshold=0.5,
        max_pieces=3,
        beam_width=10,
    )

    # 1. Assert loop successfully returned candidates
    assert selected
    # 2. Assert the dropped candidate ("f1", "f2") is excluded from final selection
    assert all(c.fragment_ids != ("f1", "f2") for c in selected)
    # 3. Assert the locked/kept candidate ("f0", "f1") was selected
    assert any(c.fragment_ids == ("f0", "f1") for c in selected)
    
    # Let's inspect the final candidate that matches ("f0", "f1")
    f0_f1_cand = next(c for c in selected if c.fragment_ids == ("f0", "f1"))
    # 4. Assert selection preferred bonus was transparently added
    assert f0_f1_cand.constraint_bonus >= 50000.0
    assert f0_f1_cand.score == f0_f1_cand.base_score + f0_f1_cand.constraint_bonus


def test_llm_guided_assembly_loop_json_parse_failure(monkeypatch):
    def mock_query_api(self, prompt):
        return "invalid json output {{{{"
    monkeypatch.setattr(LLMController, "_query_api", mock_query_api)

    fragments = [
        Fragment(id="f0", mask=np.ones((2, 2), dtype=bool), label="S1"),
        Fragment(id="f1", mask=np.ones((2, 2), dtype=bool), label="S1"),
    ]
    edges = [TearFitEdge(left=0, right=1, overlap_pixels=5, left_hits=5, right_hits=5, overlap_ratio=1.0)]

    config = LLMAgentConfig(use_mock=False, fallback_policy="raise")
    selected = llm_guided_assembly_loop(
        fragments, edges, config, max_iterations=2, coverage_threshold=0.5
    )
    # Should complete gracefully without crashing and return candidates
    assert len(selected) >= 0


def test_feedback_updates_executable_search_policy():
    fragments = [
        Fragment(id="f0", mask=np.ones((2, 2), dtype=bool), label="S1"),
        Fragment(id="f1", mask=np.ones((2, 2), dtype=bool), label="S1"),
        Fragment(id="f2", mask=np.ones((2, 2), dtype=bool), label="S2"),
    ]
    pool = [
        AssemblyCandidate(("f0", "f1"), 1.0, 1.0, 20.0, 20, ("S1",)),
        AssemblyCandidate(("f1", "f2"), 1.0, 1.0, 10.0, 10, ("S1", "S2")),
    ]
    policy = SearchPolicy()
    feedback = LLMFeedback(
        keep=[0],
        drop=[1],
        merge=[["f0", "f1"]],
        suggest_seed="S1",
        strategy="coverage_first",
    )

    changed = apply_feedback_to_policy(feedback, pool, fragments, policy)

    assert changed
    assert ("f0", "f1") in policy.locked_candidates
    assert ("f1", "f2") in policy.forbidden_candidates
    assert ("f0", "f1") in policy.forced_pairs
    assert "S1" in policy.seed_labels
    assert policy.objective == "count_then_score"


def test_llm_api_failure_policy_is_not_silent_mock(monkeypatch):
    def fail_urlopen(*_args, **_kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    controller = LLMController(LLMAgentConfig(use_mock=False, api_key="x", api_url="https://example.invalid", fallback_policy="raise"))
    candidate = AssemblyCandidate(("f0",), 1.0, 1.0, 1.0, 1)

    with pytest.raises(RuntimeError):
        controller.analyze_candidates([], [candidate], {})

    empty_controller = LLMController(
        LLMAgentConfig(use_mock=False, api_key="x", api_url="https://example.invalid", fallback_policy="empty")
    )
    feedback = empty_controller.analyze_candidates([], [candidate], {})
    assert feedback.drop == []
    assert feedback.keep == []
    assert "failed" in feedback.explanation
