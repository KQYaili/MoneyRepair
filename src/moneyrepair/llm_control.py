"""LLM-guided search policy controller and assembly critic for MoneyRepair.

The LLM layer is deliberately kept above the deterministic geometry and solver
layers. It reads candidate assemblies and search diagnostics, then updates an
explicit search policy. It does not predict pixel geometry, mutate tear scores,
or replace exact-cover search.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from itertools import combinations
from time import monotonic
from typing import Any, Literal

from moneyrepair.tearfit import (
    AssemblyCandidate,
    TearFitEdge,
    diagnose_confirmed_candidates,
    generate_assembly_candidates,
    select_exact_cover_candidates,
)
from moneyrepair.types import Fragment

FallbackPolicy = Literal["mock", "empty", "raise"]
MockStrategy = Literal["balanced", "coverage_first", "score_first", "broaden_search"]


@dataclass
class LLMAgentConfig:
    """Configures the LLM backend.

    ``use_mock`` is intended for offline tests. In production, API failures
    follow ``fallback_policy`` instead of silently pretending that the model ran.
    """

    api_key: str | None = None
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.1
    api_url: str | None = None
    use_mock: bool = False
    fallback_policy: FallbackPolicy = "raise"
    mock_strategy: MockStrategy = "balanced"


@dataclass(frozen=True)
class LLMFeedback:
    """Strict feedback schema returned by the LLM critic."""

    keep: list[int] = field(default_factory=list)
    drop: list[int] = field(default_factory=list)
    merge: list[list[str]] = field(default_factory=list)
    suggest_seed: str | None = None
    strategy: str = "balanced"
    explanation: str = ""


@dataclass
class SearchPolicy:
    """Executable constraints controlled by the LLM layer."""

    seed_whitelist: set[str] = field(default_factory=set)
    seed_labels: set[str] = field(default_factory=set)
    forbidden_candidates: set[tuple[str, ...]] = field(default_factory=set)
    locked_candidates: set[tuple[str, ...]] = field(default_factory=set)
    preferred_candidates: set[tuple[str, ...]] = field(default_factory=set)
    forced_pairs: set[tuple[str, str]] = field(default_factory=set)
    preferred_pairs: set[tuple[str, str]] = field(default_factory=set)
    forbidden_pairs: set[tuple[str, str]] = field(default_factory=set)
    coverage_delta: float = 0.0
    objective: str = "score_then_count"
    request_more_candidates: bool = False


def _candidate_key(fragment_ids: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    return tuple(sorted(fragment_ids))


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _empty_feedback(reason: str = "") -> str:
    return json.dumps(
        {
            "keep": [],
            "drop": [],
            "merge": [],
            "suggest_seed": None,
            "strategy": "balanced",
            "explanation": reason,
        }
    )


def _policy_signature(policy: SearchPolicy) -> tuple[Any, ...]:
    return (
        frozenset(policy.seed_whitelist),
        frozenset(policy.seed_labels),
        frozenset(policy.forbidden_candidates),
        frozenset(policy.locked_candidates),
        frozenset(policy.preferred_candidates),
        frozenset(policy.forced_pairs),
        frozenset(policy.preferred_pairs),
        frozenset(policy.forbidden_pairs),
        policy.coverage_delta,
        policy.objective,
        policy.request_more_candidates,
    )


def apply_feedback_to_policy(
    feedback: LLMFeedback,
    candidate_pool: list[AssemblyCandidate],
    fragments: list[Fragment],
    policy: SearchPolicy,
) -> bool:
    """Convert LLM JSON feedback into executable search constraints."""

    before = _policy_signature(policy)

    for index in feedback.drop:
        if 0 <= int(index) < len(candidate_pool):
            key = _candidate_key(candidate_pool[int(index)].fragment_ids)
            policy.forbidden_candidates.add(key)
            policy.locked_candidates.discard(key)
            policy.preferred_candidates.discard(key)
            print(f"[LLM Coordinator] Constraint: FORBID candidate {key}")

    for index in feedback.keep:
        if 0 <= int(index) < len(candidate_pool):
            key = _candidate_key(candidate_pool[int(index)].fragment_ids)
            if key not in policy.forbidden_candidates:
                policy.locked_candidates.add(key)
                policy.preferred_candidates.add(key)
                print(f"[LLM Coordinator] Constraint: LOCK candidate {key}")

    for group in feedback.merge:
        ids = [fragment_id for fragment_id in group if fragment_id]
        for left, right in combinations(sorted(set(ids)), 2):
            pair = _pair_key(left, right)
            policy.forced_pairs.add(pair)
            policy.preferred_pairs.add(pair)
            print(f"[LLM Coordinator] Constraint: MERGE pair {pair}")

    if feedback.suggest_seed:
        seed = str(feedback.suggest_seed)
        policy.seed_whitelist.clear()
        policy.seed_labels.clear()
        fragment_ids = {fragment.id for fragment in fragments}
        labels = {fragment.label for fragment in fragments if fragment.label}
        if seed in fragment_ids:
            policy.seed_whitelist.add(seed)
            print(f"[LLM Coordinator] Seed Whitelist: Added fragment seed {seed}")
        elif seed in labels:
            policy.seed_labels.add(seed)
            print(f"[LLM Coordinator] Seed Labels: Added label seed {seed}")
        else:
            policy.seed_labels.add(seed)
            print(f"[LLM Coordinator] Seed Labels: Added unknown label seed {seed}")
    else:
        if policy.seed_whitelist or policy.seed_labels:
            print("[LLM Coordinator] Seed Whitelist/Labels cleared.")
        policy.seed_whitelist.clear()
        policy.seed_labels.clear()

    if feedback.strategy == "coverage_first":
        policy.objective = "count_then_score"
        policy.coverage_delta = min(policy.coverage_delta, -0.05)
    elif feedback.strategy == "score_first":
        policy.objective = "score_then_count"
        policy.coverage_delta = max(policy.coverage_delta, 0.02)
    elif feedback.strategy == "balanced":
        policy.request_more_candidates = False
    elif feedback.strategy == "broaden_search":
        policy.request_more_candidates = True
        policy.coverage_delta = min(policy.coverage_delta, -0.05)

    if feedback.strategy != "balanced":
        print(f"[LLM Coordinator] Strategy: Set to '{feedback.strategy}' (objective={policy.objective}, coverage_delta={policy.coverage_delta:.2f})")

    return _policy_signature(policy) != before


class LLMController:
    """Wraps LLM API calls, prompt construction, and JSON output parsing."""

    def __init__(self, config: LLMAgentConfig):
        self.config = config

    def _handle_unavailable(self, prompt: str, reason: str) -> str:
        if self.config.fallback_policy == "mock":
            return self._mock_response(prompt)
        if self.config.fallback_policy == "empty":
            return _empty_feedback(reason)
        raise RuntimeError(reason)

    def _query_api(self, prompt: str) -> str:
        """Execute the HTTP request to an OpenAI-compatible or Gemini endpoint."""

        if self.config.use_mock:
            return self._mock_response(prompt)
        if not self.config.api_key:
            return self._handle_unavailable(prompt, "LLM API key is missing.")

        url = self.config.api_url or (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.model_name}:generateContent?key={self.config.api_key}"
        )
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": self.config.temperature,
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.URLError as exc:
            return self._handle_unavailable(prompt, f"LLM API request failed: {exc}")

    def _mock_response(self, prompt: str) -> str:
        """Offline mock response mapping input state to feedback for tests."""

        try:
            state_marker = "Current State:"
            marker_idx = prompt.find(state_marker)
            if marker_idx != -1:
                json_start = prompt.find("{", marker_idx)
                if json_start != -1:
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
                        drop: list[int] = []
                        keep: list[int] = []
                        for idx, candidate in enumerate(candidates):
                            labels = candidate.get("labels", [])
                            if len(labels) > 1:
                                drop.append(idx)
                            elif candidate.get("coverage", 0.0) < 0.2:
                                drop.append(idx)
                            elif candidate.get("solver_selected", True):
                                keep.append(idx)
                        return json.dumps(
                            {
                                "keep": keep,
                                "drop": drop,
                                "merge": [],
                                "suggest_seed": None,
                                "strategy": self.config.mock_strategy,
                                "explanation": "Mock logic applied.",
                            }
                        )
        except Exception:
            pass

        return _empty_feedback("Default mock feedback.")

    def analyze_search_state(
        self,
        fragments: list[Fragment],
        candidate_pool: list[AssemblyCandidate],
        selected_candidates: list[AssemblyCandidate],
        global_stats: dict[str, Any],
        rejected_by_solver: dict[str, Any] | None = None,
    ) -> LLMFeedback:
        """Send top-K pool, selected candidates, and rejected summary to the LLM."""

        if not candidate_pool:
            return LLMFeedback()

        selected_keys = {_candidate_key(candidate.fragment_ids) for candidate in selected_candidates}
        candidates_data = []
        for idx, candidate in enumerate(candidate_pool):
            key = _candidate_key(candidate.fragment_ids)
            candidates_data.append(
                {
                    "index": idx,
                    "fragment_ids": list(candidate.fragment_ids),
                    "coverage": round(candidate.coverage, 3),
                    "raw_coverage": round(candidate.raw_coverage, 3),
                    "score": round(candidate.score, 1),
                    "support_pixels": candidate.support_pixels,
                    "labels": list(candidate.labels),
                    "solver_selected": key in selected_keys,
                }
            )

        state = {
            "global_stats": global_stats,
            "fragments": {
                "count": len(fragments),
                "labelled": sum(1 for fragment in fragments if fragment.label),
            },
            "candidates": candidates_data,
            "selected_candidate_indices": [
                idx
                for idx, candidate in enumerate(candidate_pool)
                if _candidate_key(candidate.fragment_ids) in selected_keys
            ],
            "rejected_by_solver": rejected_by_solver or {},
        }

        prompt = f"""You are the Assembly Critic and Search Policy Controller for a banknote reconstruction system.
You must only control deterministic search policy. Do not invent pixel-level geometry.
Read the top-K candidate pool, the candidates selected by exact cover, and the solver-rejected summary.
Detect implausible chimeras, preserve strong candidates, inject hard merge constraints only when justified,
and suggest a concrete seed fragment ID or serial label when it would reshape the next candidate-generation pass.

Current State:
{json.dumps(state, indent=2)}

Respond with exactly one JSON object and no markdown fences:
{{
  "keep": [candidate indices to lock into the next exact-cover pass],
  "drop": [candidate indices to forbid in the next pass],
  "merge": [[fragment IDs that must be constrained together]],
  "suggest_seed": "fragment ID or serial label to seed candidate generation, or null",
  "strategy": "coverage_first" or "score_first" or "balanced" or "broaden_search",
  "explanation": "brief reasoning"
}}
"""
        response_text = self._query_api(prompt)
        response_text = response_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            data = json.loads(response_text)
            return LLMFeedback(
                keep=[int(item) for item in data.get("keep", [])],
                drop=[int(item) for item in data.get("drop", [])],
                merge=[list(map(str, group)) for group in data.get("merge", [])],
                suggest_seed=data.get("suggest_seed"),
                strategy=data.get("strategy", "balanced"),
                explanation=data.get("explanation", ""),
            )
        except Exception:
            return LLMFeedback(explanation="JSON parse failed.")

    def analyze_candidates(
        self,
        fragments: list[Fragment],
        candidates: list[AssemblyCandidate],
        global_stats: dict[str, Any],
    ) -> LLMFeedback:
        """Backward-compatible wrapper for simple candidate-only tests."""

        return self.analyze_search_state(
            fragments,
            candidates,
            candidates,
            global_stats,
            rejected_by_solver={"count": 0, "top_rejected": []},
        )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _pool_for_llm(
    candidates: list[AssemblyCandidate],
    selected: list[AssemblyCandidate],
    limit: int,
) -> list[AssemblyCandidate]:
    pool = list(candidates[: max(0, limit)])
    seen = {_candidate_key(candidate.fragment_ids) for candidate in pool}
    for candidate in selected:
        key = _candidate_key(candidate.fragment_ids)
        if key not in seen:
            pool.append(candidate)
            seen.add(key)
    return pool


def _rejected_summary(candidates: list[AssemblyCandidate], selected: list[AssemblyCandidate]) -> dict[str, Any]:
    selected_keys = {_candidate_key(candidate.fragment_ids) for candidate in selected}
    rejected = [candidate for candidate in candidates if _candidate_key(candidate.fragment_ids) not in selected_keys]
    label_conflicts = sum(1 for candidate in rejected if len(candidate.labels) > 1)
    return {
        "count": len(rejected),
        "label_conflict_count": label_conflicts,
        "top_rejected": [
            {
                "fragment_ids": list(candidate.fragment_ids),
                "coverage": round(candidate.coverage, 3),
                "score": round(candidate.score, 1),
                "labels": list(candidate.labels),
            }
            for candidate in rejected[:8]
        ],
    }


def llm_guided_assembly_loop(
    fragments: list[Fragment],
    edges: list[TearFitEdge],
    llm_config: LLMAgentConfig,
    *,
    max_iterations: int = 3,
    coverage_threshold: float = 0.93,
    max_pieces: int = 12,
    beam_width: int = 64,
    candidate_pool_limit: int = 40,
    time_limit_seconds: float = 30.0,
) -> list[AssemblyCandidate]:
    """Iteratively update candidate generation and exact-cover constraints."""

    start_time = monotonic()
    controller = LLMController(llm_config)
    policy = SearchPolicy()
    final_selected: list[AssemblyCandidate] = []
    
    # Registry to accumulate all generated candidates across iterations
    all_generated_candidates: dict[tuple[str, ...], AssemblyCandidate] = {}

    for iteration in range(max_iterations):
        elapsed = monotonic() - start_time
        if elapsed >= time_limit_seconds:
            break

        remaining_time = max(1.0, time_limit_seconds - elapsed)
        current_threshold = _clamp(coverage_threshold + policy.coverage_delta, 0.5, 0.99)
        current_beam_width = beam_width * (2 if policy.request_more_candidates else 1)

        candidates = generate_assembly_candidates(
            fragments,
            edges,
            coverage_threshold=current_threshold,
            max_pieces=max_pieces,
            beam_width=current_beam_width,
            seed_strategy="anchor_priority",
            seed_whitelist=policy.seed_whitelist or None,
            seed_labels=policy.seed_labels or None,
            forced_pairs=policy.forced_pairs or None,
            preferred_pairs=policy.preferred_pairs or None,
            forbidden_pairs=policy.forbidden_pairs or None,
            forbidden_candidates=policy.forbidden_candidates or None,
            time_limit_seconds=remaining_time,
        )
        
        # Accumulate newly generated candidates
        for candidate in candidates:
            key = _candidate_key(candidate.fragment_ids)
            if key not in all_generated_candidates or candidate.score > all_generated_candidates[key].score:
                all_generated_candidates[key] = candidate

        # Prepare candidates for the solver by filtering registry against forbidden policy list
        solver_candidates = [
            c for key, c in all_generated_candidates.items()
            if key not in policy.forbidden_candidates
        ]
        
        if not solver_candidates:
            break

        # Sort solver candidates to match the standard select_exact_cover_candidates input ordering
        solver_candidates.sort(key=lambda item: (-item.score, item.fragment_ids))

        try:
            selected = select_exact_cover_candidates(
                solver_candidates,
                time_limit_seconds=max(1.0, time_limit_seconds - (monotonic() - start_time)),
                objective=policy.objective,
                forbidden_candidates=policy.forbidden_candidates or None,
                locked_candidates=policy.locked_candidates or None,
                preferred_candidates=policy.preferred_candidates or None,
            )
        except ValueError:
            # A bad LLM lock should not poison the deterministic solver forever.
            policy.preferred_candidates.update(policy.locked_candidates)
            policy.locked_candidates.clear()
            selected = select_exact_cover_candidates(
                solver_candidates,
                time_limit_seconds=max(1.0, time_limit_seconds - (monotonic() - start_time)),
                objective=policy.objective,
                forbidden_candidates=policy.forbidden_candidates or None,
                preferred_candidates=policy.preferred_candidates or None,
            )

        final_selected = selected
        diagnostics = diagnose_confirmed_candidates(selected, fragments)
        assigned_ids = {fragment_id for candidate in selected for fragment_id in candidate.fragment_ids}
        
        candidate_pool = _pool_for_llm(solver_candidates, selected, candidate_pool_limit)

        # Compute conflict graph stats for the LLM
        conflict_pairs = 0
        total_pairs = len(candidate_pool) * (len(candidate_pool) - 1) // 2
        if total_pairs > 0:
            for c1, c2 in combinations(candidate_pool, 2):
                if (set(c1.fragment_ids) & set(c2.fragment_ids)) or (set(c1.labels) & set(c2.labels)):
                    conflict_pairs += 1
            conflict_density = conflict_pairs / total_pairs
        else:
            conflict_density = 0.0

        global_stats = {
            "iteration": iteration,
            "total_fragments": len(fragments),
            "candidate_pool_size": len(solver_candidates),
            "unassigned_fragments": len(fragments) - len(assigned_ids),
            "confirmed_assemblies": diagnostics.confirmed,
            "exact_confirmed": diagnostics.exact_confirmed,
            "chimeras": diagnostics.chimeras,
            "chimera_rate": diagnostics.chimeras / diagnostics.confirmed if diagnostics.confirmed else 0.0,
            "conflict_pairs": conflict_pairs,
            "conflict_density": round(conflict_density, 3),
            "policy": {
                "seed_whitelist": sorted(policy.seed_whitelist),
                "seed_labels": sorted(policy.seed_labels),
                "forbidden_candidates": len(policy.forbidden_candidates),
                "locked_candidates": len(policy.locked_candidates),
                "forced_pairs": len(policy.forced_pairs),
                "objective": policy.objective,
                "coverage_delta": policy.coverage_delta,
            },
        }

        feedback = controller.analyze_search_state(
            fragments,
            candidate_pool,
            selected,
            global_stats,
            rejected_by_solver=_rejected_summary(solver_candidates, selected),
        )
        changed = apply_feedback_to_policy(feedback, candidate_pool, fragments, policy)
        if not changed:
            break

    return final_selected
