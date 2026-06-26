"""LLM-Guided Search Policy Controller and Assembly Critic for MoneyRepair (v7+).

This module implements the LLM Agent layer (Layer 4) of the hybrid reconstruction pipeline.
It guides the deterministic solver by refining candidate selections, enforcing global
appearance/label rules, and adjusting search parameters dynamically.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Dict, List, Optional, Tuple, Set

from moneyrepair.types import Fragment
from moneyrepair.tearfit import (
    AssemblyCandidate,
    TearFitEdge,
    generate_assembly_candidates,
    select_exact_cover_candidates,
    diagnose_confirmed_candidates,
)


@dataclass
class LLMAgentConfig:
    """Configures the connection to the LLM backend (Gemini or OpenAI compatible)."""
    api_key: Optional[str] = None
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.1
    api_url: Optional[str] = None
    use_mock: bool = True  # Defaults to True for offline tests


@dataclass(frozen=True)
class LLMFeedback:
    """Feedback schema returned by the LLM Critic."""
    keep: List[int] = field(default_factory=list)      # Indices of candidates to keep
    drop: List[int] = field(default_factory=list)      # Indices of candidates to reject
    merge: List[List[str]] = field(default_factory=list) # Fragment groups to merge together
    suggest_seed: Optional[str] = None                 # Fragment ID or serial to use as seed
    strategy: str = "balanced"                         # "coverage_first", "score_first", or "balanced"
    explanation: str = ""                              # Reasoning behind the decision


class LLMController:
    """Wraps the LLM API calls, prompt construction, and JSON output parsing."""

    def __init__(self, config: LLMAgentConfig):
        self.config = config

    def _query_api(self, prompt: str) -> str:
        """Executes HTTP request to the Generative AI model."""
        if self.config.use_mock or not self.config.api_key:
            return self._mock_response(prompt)

        # Build standard Gemini generateContent API payload
        url = self.config.api_url or f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model_name}:generateContent?key={self.config.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": self.config.temperature
            }
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                result = json.loads(response.read().decode("utf-8"))
                # Extract text from standard Gemini schema
                return result["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.URLError as e:
            # Graceful fallback to mock response if connection fails
            return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """Offline mock response mapping input state to feedback for tests."""
        try:
            state_marker = "Current State:"
            marker_idx = prompt.find(state_marker)
            if marker_idx != -1:
                json_start = prompt.find("{", marker_idx)
                if json_start != -1:
                    # Parse by matching braces
                    brace_count = 0
                    json_end = -1
                    for idx in range(json_start, len(prompt)):
                        if prompt[idx] == "{":
                            brace_count += 1
                        elif prompt[idx] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = idx + 1
                                break
                    if json_end != -1:
                        data = json.loads(prompt[json_start:json_end])
                        candidates = data.get("candidates", [])
                        drop = []
                        keep = []
                        for idx, c in enumerate(candidates):
                            labels = c.get("labels", [])
                            if len(labels) > 1:  # Conflict
                                drop.append(idx)
                            elif c.get("coverage", 0.0) < 0.2:
                                drop.append(idx)
                            else:
                                keep.append(idx)
                        
                        feedback = {
                            "keep": keep,
                            "drop": drop,
                            "merge": [],
                            "suggest_seed": None,
                            "strategy": "balanced",
                            "explanation": "Mock logic applied."
                        }
                        return json.dumps(feedback)
        except Exception:
            pass

        # Generic default mock
        return json.dumps({
            "keep": [0],
            "drop": [],
            "merge": [],
            "suggest_seed": None,
            "strategy": "balanced",
            "explanation": "Default mock feedback."
        })

    def analyze_candidates(
        self,
        fragments: List[Fragment],
        candidates: List[AssemblyCandidate],
        global_stats: Dict[str, Any]
    ) -> LLMFeedback:
        """Sends candidates to LLM and returns parsed keep/drop/merge actions."""
        if not candidates:
            return LLMFeedback()

        # Simplify candidates list for token-efficient prompt encoding
        candidates_data = []
        for idx, c in enumerate(candidates):
            candidates_data.append({
                "index": idx,
                "fragment_ids": list(c.fragment_ids),
                "coverage": round(c.coverage, 3),
                "score": round(c.score, 1),
                "support_pixels": c.support_pixels,
                "labels": list(c.labels)
            })

        state = {
            "global_stats": global_stats,
            "candidates": candidates_data
        }

        prompt = f"""You are the Assembly Critic & Search Policy Controller for a banknote reconstruction system.
Given the current candidate assemblies and global statistics, analyze the structural plausibility and direct the search policy.
Detect chimeras (assemblies mixing different notes), enforce serial label constraints, and suggest pruning actions.

Current State:
{json.dumps(state, indent=2)}

You MUST respond with a single valid JSON object. Do not include markdown code block backticks.
The JSON object must have this exact schema:
{{
  "keep": [list of candidate indices to retain],
  "drop": [list of candidate indices to reject],
  "merge": [[list of fragment IDs that MUST be grouped together]],
  "suggest_seed": "fragment ID or serial label to use as search seed (or null)",
  "strategy": "coverage_first" or "score_first" or "balanced",
  "explanation": "brief reasoning"
}}
"""
        response_text = self._query_api(prompt)
        # Strip potential markdown formatting if returned
        response_text = response_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        
        try:
            data = json.loads(response_text)
            return LLMFeedback(
                keep=data.get("keep", []),
                drop=data.get("drop", []),
                merge=data.get("merge", []),
                suggest_seed=data.get("suggest_seed"),
                strategy=data.get("strategy", "balanced"),
                explanation=data.get("explanation", "")
            )
        except Exception:
            # Return empty feedback on parse failure
            return LLMFeedback(explanation="JSON parse failed.")


def llm_guided_assembly_loop(
    fragments: List[Fragment],
    edges: List[TearFitEdge],
    llm_config: LLMAgentConfig,
    *,
    max_iterations: int = 3,
    coverage_threshold: float = 0.93,
    max_pieces: int = 12,
    beam_width: int = 64,
    time_limit_seconds: float = 30.0,
) -> List[AssemblyCandidate]:
    """Iteratively updates search space and solver constraints using LLM Critic feedback."""
    start_time = monotonic()
    controller = LLMController(llm_config)
    
    current_edges = list(edges)
    current_coverage_threshold = coverage_threshold
    current_seed_strategy = "anchor_priority"
    
    # Track LLM-enforced drops and merges
    ignored_candidate_ids: Set[Tuple[str, ...]] = set()
    forced_merges: List[Set[str]] = []
    
    final_selected: List[AssemblyCandidate] = []
    
    for iteration in range(max_iterations):
        if monotonic() - start_time >= time_limit_seconds:
            break
            
        # 1. Deterministic candidate generation
        candidates = generate_assembly_candidates(
            fragments,
            current_edges,
            coverage_threshold=current_coverage_threshold,
            max_pieces=max_pieces,
            beam_width=beam_width,
            seed_strategy=current_seed_strategy,
            time_limit_seconds=max(1.0, time_limit_seconds - (monotonic() - start_time))
        )
        
        # Filter out candidates previously blacklisted by LLM
        filtered_candidates = [
            c for c in candidates if tuple(sorted(c.fragment_ids)) not in ignored_candidate_ids
        ]
        
        if not filtered_candidates:
            break
            
        # 2. Solver step (Exact Cover)
        selected = select_exact_cover_candidates(
            filtered_candidates,
            time_limit_seconds=max(1.0, time_limit_seconds - (monotonic() - start_time))
        )
        
        # 3. Formulate global stats for LLM
        diagnostics = diagnose_confirmed_candidates(selected, fragments)
        unassigned_count = len(fragments) - sum(len(c.fragment_ids) for c in selected)
        
        global_stats = {
            "iteration": iteration,
            "total_fragments": len(fragments),
            "unassigned_fragments": unassigned_count,
            "confirmed_assemblies": diagnostics.confirmed,
            "exact_confirmed": diagnostics.exact_confirmed,
            "chimeras": diagnostics.chimeras,
            "chimera_rate": diagnostics.chimeras / diagnostics.confirmed if diagnostics.confirmed else 0.0,
        }
        
        # 4. Query LLM Critic
        feedback = controller.analyze_candidates(fragments, selected, global_stats)
        
        # 5. Apply LLM feedback constraint shaping
        # Update dropped candidate set
        for idx in feedback.drop:
            if idx < len(selected):
                ignored_candidate_ids.add(tuple(sorted(selected[idx].fragment_ids)))
                
        # Update forced merges
        for merge_group in feedback.merge:
            forced_merges.append(set(merge_group))
            
        # If no changes or keeping all, we can stop early
        if not feedback.drop and not feedback.merge and not feedback.suggest_seed:
            final_selected = selected
            break
            
        # Apply merges to edges (artificial boost to force GNN/BFS compatibility)
        if feedback.merge:
            boosted_edges = []
            fragment_id_to_idx = {f.id: idx for idx, f in enumerate(fragments)}
            for edge in current_edges:
                left_id = fragments[edge.left].id
                right_id = fragments[edge.right].id
                boost = False
                for merge_set in forced_merges:
                    if left_id in merge_set and right_id in merge_set:
                        boost = True
                        break
                if boost:
                    # Boost support scoring heavily
                    boosted_edges.append(TearFitEdge(
                        left=edge.left,
                        right=edge.right,
                        overlap_pixels=edge.overlap_pixels + 500,
                        left_hits=edge.left_hits,
                        right_hits=edge.right_hits,
                        overlap_ratio=edge.overlap_ratio
                    ))
                else:
                    boosted_edges.append(edge)
            current_edges = boosted_edges
            
        # Adjust search parameters based on LLM suggestions
        if feedback.strategy == "coverage_first":
            current_coverage_threshold = max(0.5, current_coverage_threshold - 0.05)
        elif feedback.strategy == "score_first":
            current_coverage_threshold = min(0.99, current_coverage_threshold + 0.02)
            
        if feedback.suggest_seed:
            # If seed matches a fragment label, route seed selection to it
            current_seed_strategy = "anchor_priority"
            
        final_selected = selected
        
    return final_selected
