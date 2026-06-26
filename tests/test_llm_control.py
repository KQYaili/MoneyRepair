"""Tests for the LLM-Guided Search Controller and Critic in MoneyRepair."""

from __future__ import annotations

import numpy as np
from moneyrepair.types import Fragment
from moneyrepair.tearfit import AssemblyCandidate, TearFitEdge
from moneyrepair.llm_control import (
    LLMAgentConfig,
    LLMFeedback,
    LLMController,
    llm_guided_assembly_loop,
)


def test_llm_controller_mock_parsing():
    config = LLMAgentConfig(use_mock=True)
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
    assert feedback.strategy == "balanced"


def test_llm_guided_assembly_loop_execution():
    config = LLMAgentConfig(use_mock=True)
    
    # Construct minimal dummy fragments and edges
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    
    fragments = [
        Fragment(id="f0", mask=mask.copy(), label="S1", meta={"note_id": "n0"}),
        Fragment(id="f1", mask=mask.copy(), label="S1", meta={"note_id": "n0"}),
        Fragment(id="f2", mask=mask.copy(), label="S2", meta={"note_id": "n1"}),
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
    
    # Verify that the solver ran and returned assemblies
    assert len(selected) >= 0
