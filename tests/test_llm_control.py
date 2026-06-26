"""Tests for the LLM-Guided Search Controller and Critic in MoneyRepair."""

from __future__ import annotations

import urllib.error

import numpy as np
import pytest
from moneyrepair.types import Fragment
from moneyrepair.tearfit import AssemblyCandidate, TearFitEdge
from moneyrepair.llm_control import (
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


def test_llm_guided_assembly_loop_execution():
    config = LLMAgentConfig(use_mock=True)
    
    # Construct minimal dummy fragments and edges
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
    
    # Run the iterative loop
    selected = llm_guided_assembly_loop(
        fragments,
        edges,
        config,
        max_iterations=2,
        coverage_threshold=0.5,
        max_pieces=3,
        beam_width=10
    )
    
    assert selected
    assert selected[0].fragment_ids == ("f0", "f1")


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
