from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

from .adaptive_generation import (
    Action,
    AdaptiveGenerationConfig,
    AdaptiveGenerationEngine,
    DependencyGraphCache,
    GenerationUnit,
    UnitValidation,
)
from .io_utils import extract_json_object
from .llm import ChatModel, completion_history
from .longgenbench import (
    LongGenTask,
    SegmentContract,
    _compiled_document_summary,
    _prior_context,
    apply_compiled_contracts,
    assemble_document,
    chunked,
    compile_prompt_contracts,
    parse_generated_segments,
    parse_review_results,
    structural_issues,
    task_word_range,
    task_record,
    word_count,
)


ADAPTIVE_PIPELINE_VERSION = "generic_adaptive_generation_v2_1"


class AdaptiveHarnessIncompleteError(RuntimeError):
    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        metadata = output.get("metadata", {})
        structural = metadata.get("final_structural_issues", {})
        semantic = metadata.get("unresolved_semantic_issues", {})
        super().__init__(
            "adaptive harness produced an incomplete output: "
            f"structural={len(structural)}, semantic={len(semantic)}"
        )


def infer_target_word_range(task: LongGenTask) -> tuple[int, int]:
    return task_word_range(task)


def compact_source_brief(source: str, *, max_characters: int = 900) -> str:
    normalized = re.sub(r"\s+", " ", source).strip()
    if len(normalized) <= max_characters:
        return normalized
    candidate = normalized[:max_characters].rsplit(" ", 1)[0].rstrip(" ,;:")
    return candidate + "."


class LongGenBenchAdapter:
    def __init__(
        self,
        task: LongGenTask,
        document_summary: str,
        contracts: list[SegmentContract],
    ) -> None:
        self.task = task
        self.document_summary = document_summary
        self.contracts = {str(contract.index): contract for contract in contracts}
        self.minimum_words, self.maximum_words = infer_target_word_range(task)

    def action_for(
        self,
        unit: GenerationUnit,
        validation: UnitValidation,
    ) -> Action:
        return "extend" if validation.status == "short" else "generate"

    def build_prompt(
        self,
        action: Action,
        units: list[GenerationUnit],
        current_texts: dict[str, str],
        completed_texts: dict[str, str],
    ) -> str:
        if action == "extend":
            return self._build_extension_prompt(units, current_texts, completed_texts)
        return self._build_generation_prompt(units, completed_texts)

    def parse_response(
        self,
        action: Action,
        units: list[GenerationUnit],
        raw: str,
    ) -> dict[str, str]:
        expected = [self.contracts[unit.unit_id] for unit in units]
        return {
            str(index): text
            for index, text in parse_generated_segments(expected, raw).items()
        }

    def validate(self, unit: GenerationUnit, text: str) -> UnitValidation:
        count = word_count(text)
        if not text.strip():
            return UnitValidation(
                status="missing",
                measured_length=0,
                required_length=unit.minimum_length,
                issues=["segment is missing"],
            )
        if count < unit.minimum_length:
            return UnitValidation(
                status="short",
                measured_length=count,
                required_length=unit.minimum_length,
                issues=[f"word count {count} is below {unit.minimum_length}"],
            )
        return UnitValidation(
            status="valid",
            measured_length=count,
            required_length=unit.minimum_length,
        )

    def estimate_output_tokens(
        self,
        unit: GenerationUnit,
        action: Action,
        current_text: str,
    ) -> int:
        if action == "extend":
            missing_words = max(1, unit.minimum_length - word_count(current_text))
            return math.ceil(missing_words * 1.45) + 48
        return math.ceil(unit.target_length * 1.45) + 64

    def system_prompt(self, action: Action) -> str:
        if action == "extend":
            return "Continue only the requested segments without repeating their existing prose."
        return "Write only the explicitly requested long-form segments using the exact tags."

    def _build_generation_prompt(
        self,
        units: list[GenerationUnit],
        completed_texts: dict[str, str],
    ) -> str:
        indices = [int(unit.unit_id) for unit in units]
        contract_payload = [
            {
                "index": contract.index,
                "purpose": contract.purpose,
                "required_events": contract.required_events,
                "continuity": contract.continuity,
            }
            for contract in (self.contracts[unit.unit_id] for unit in units)
        ]
        return f"""LONGGEN_ADAPTIVE_GENERATE

Write only segment indices {indices}. Return every requested index exactly once.
Each segment must contain {self.minimum_words} to {self.maximum_words} English words.
Include every required event naturally at its assigned index. Keep ordinary content
concrete, coherent, varied, and consistent with the source brief.

Use exact tags for every requested segment:
<<<SEGMENT {indices[0]}>>>
reader-facing prose
<<<END SEGMENT {indices[0]}>>>

SOURCE_BRIEF:
{compact_source_brief(self.task.prompt)}

DOCUMENT_SUMMARY:
{self.document_summary}

PRIOR_CONTEXT:
{_completed_context(completed_texts)}

SEGMENT_CONTRACTS_JSON:
{json.dumps(contract_payload, ensure_ascii=False)}
"""

    def _build_extension_prompt(
        self,
        units: list[GenerationUnit],
        current_texts: dict[str, str],
        completed_texts: dict[str, str],
    ) -> str:
        items = []
        for unit in units:
            contract = self.contracts[unit.unit_id]
            current = current_texts.get(unit.unit_id, "")
            items.append(
                {
                    "index": contract.index,
                    "required_events": contract.required_events,
                    "current_word_count": word_count(current),
                    "minimum_words": unit.minimum_length,
                    "words_still_needed": max(1, unit.minimum_length - word_count(current)),
                    "current_text": current,
                }
            )
        indices = [int(unit.unit_id) for unit in units]
        return f"""LONGGEN_ADAPTIVE_EXTEND

Write only new continuation prose for segment indices {indices}.
Do not repeat, summarize, replace, or restart CURRENT_SEGMENTS.
Add enough concrete prose for each combined segment to reach its minimum length.
Preserve continuity and add any required event that is not yet clear.

Use exact tags for each continuation:
<<<SEGMENT {indices[0]}>>>
new continuation only
<<<END SEGMENT {indices[0]}>>>

SOURCE_BRIEF:
{compact_source_brief(self.task.prompt)}

PRIOR_CONTEXT:
{_completed_context(completed_texts)}

CURRENT_SEGMENTS_JSON:
{json.dumps(items, ensure_ascii=False)}
"""


def _completed_context(completed_texts: dict[str, str]) -> str:
    indexed = {
        int(unit_id): text
        for unit_id, text in completed_texts.items()
        if unit_id.isdigit() and text.strip()
    }
    return _prior_context(indexed, max_characters=500) or "(start of document)"


def build_adaptive_review_prompt(
    task: LongGenTask,
    contracts: list[SegmentContract],
    segments: dict[int, str],
) -> str:
    items = [
        {
            "index": contract.index,
            "required_events": contract.required_events,
            "text": segments.get(contract.index, ""),
        }
        for contract in contracts
    ]
    return f"""LONGGEN_ADAPTIVE_REVIEW_JSON

Check only whether every required event is clearly present at the assigned index.
Accept clear paraphrases. Ignore style and harmless extra detail.

Return valid JSON only:
{{
  "results": [
    {{"index": 1, "passed": true, "missing_events": [], "issues": []}}
  ]
}}

SOURCE_BRIEF:
{compact_source_brief(task.prompt)}

SEGMENTS_JSON:
{json.dumps(items, ensure_ascii=False)}
"""


def build_adaptive_repair_prompt(
    task: LongGenTask,
    contracts: list[SegmentContract],
    segments: dict[int, str],
    issues: dict[int, dict[str, Any]],
) -> str:
    minimum_words, maximum_words = infer_target_word_range(task)
    indices = [contract.index for contract in contracts]
    items = [
        {
            "index": contract.index,
            "required_events": contract.required_events,
            "current_text": segments.get(contract.index, ""),
            "issues": issues.get(contract.index, {}),
        }
        for contract in contracts
    ]
    return f"""LONGGEN_ADAPTIVE_REPAIR

Return complete replacements only for segment indices {indices}.
Resolve every listed semantic issue while preserving correct content.
Each replacement must contain {minimum_words} to {maximum_words} English words.

Use exact tags:
<<<SEGMENT {indices[0]}>>>
complete replacement prose
<<<END SEGMENT {indices[0]}>>>

SOURCE_BRIEF:
{compact_source_brief(task.prompt)}

FAILING_SEGMENTS_JSON:
{json.dumps(items, ensure_ascii=False)}
"""


def run_harness_adaptive(
    task: LongGenTask,
    llm: ChatModel,
    *,
    initial_batch_size: int = 8,
    maximum_batch_size: int = 8,
    generation_max_output_tokens: int = 4_096,
    extension_max_output_tokens: int = 2_048,
    review_max_output_tokens: int = 1_024,
    repair_max_output_tokens: int = 4_096,
    max_generation_attempts: int = 3,
    max_extension_attempts: int = 3,
    max_structural_recovery_rounds: int = 1,
    total_token_budget: int | None = None,
    deadline_seconds: float | None = None,
    fail_on_incomplete: bool = True,
    checkpoint_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    graph_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    document_summary = _compiled_document_summary(task)
    contracts = [
        SegmentContract(
            index=index,
            purpose=f"Write a distinct, coherent {task.segment_type.lower()} entry.",
        )
        for index in range(1, task.segment_count + 1)
    ]
    compiled_overrides = apply_compiled_contracts(contracts, compile_prompt_contracts(task))
    minimum_words, maximum_words = infer_target_word_range(task)
    units = [
        GenerationUnit(
            unit_id=str(contract.index),
            order=contract.index,
            minimum_length=minimum_words,
            target_length=maximum_words,
            mandatory=True,
            priority=1 if contract.required_events else 0,
            payload={
                "purpose": contract.purpose,
                "required_events": contract.required_events,
                "continuity": contract.continuity,
            },
        )
        for contract in contracts
    ]
    config = AdaptiveGenerationConfig(
        initial_batch_size=initial_batch_size,
        maximum_batch_size=maximum_batch_size,
        generation_max_output_tokens=generation_max_output_tokens,
        extension_max_output_tokens=extension_max_output_tokens,
        max_generation_attempts=max_generation_attempts,
        max_extension_attempts=max_extension_attempts,
        total_token_budget=total_token_budget,
        deadline_seconds=deadline_seconds,
    )
    adapter = LongGenBenchAdapter(task, document_summary, contracts)
    engine = AdaptiveGenerationEngine(
        llm,
        adapter,
        config=config,
        graph_cache=DependencyGraphCache(graph_cache_dir),
    )
    generation = engine.run(
        units,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )
    segments = {
        int(unit_id): text
        for unit_id, text in generation.texts.items()
        if unit_id.isdigit() and text.strip()
    }

    hard_issues = structural_issues(task, contracts, segments)
    structural_recovery_records: list[dict[str, Any]] = []
    for round_index in range(max_structural_recovery_rounds):
        if not hard_issues:
            break
        round_records = _recover_structural_issues(
            llm,
            adapter,
            units,
            segments,
            hard_issues,
            batch_size=maximum_batch_size,
            generation_max_output_tokens=generation_max_output_tokens,
            extension_max_output_tokens=extension_max_output_tokens,
            round_index=round_index,
        )
        structural_recovery_records.extend(round_records)
        hard_issues = structural_issues(task, contracts, segments)

    review_records: list[dict[str, Any]] = []
    semantic_issues: dict[int, dict[str, Any]] = {}
    review_targets = [
        contract
        for contract in contracts
        if contract.required_events
        and contract.index in segments
        and contract.index not in hard_issues
    ]
    for batch in chunked(review_targets, maximum_batch_size):
        payload = _complete_json(
            llm,
            build_adaptive_review_prompt(task, batch, segments),
            system="Review explicit segment requirements. Return valid JSON only.",
            max_output_tokens=review_max_output_tokens,
        )
        batch_results = parse_review_results(payload)
        for contract in batch:
            record = batch_results.get(
                contract.index,
                {
                    "passed": False,
                    "missing_events": list(contract.required_events),
                    "issues": ["review result missing"],
                },
            )
            review_records.append({"stage": "initial", "index": contract.index, **record})
            if not record["passed"]:
                semantic_issues[contract.index] = record

    repair_records: list[dict[str, Any]] = []
    unresolved_semantic_issues = dict(semantic_issues)
    failing_contracts = [
        contract for contract in contracts if contract.index in semantic_issues
    ]
    for batch in chunked(failing_contracts, max(1, maximum_batch_size // 2)):
        raw = llm.complete(
            build_adaptive_repair_prompt(task, batch, segments, semantic_issues),
            system="Repair only the requested segments and return exact tags.",
            temperature=0.1,
            max_output_tokens=repair_max_output_tokens,
        )
        completion = llm.completion_metadata()
        repaired = parse_generated_segments(batch, raw)
        candidate_segments = dict(segments)
        structurally_valid_candidates: list[SegmentContract] = []
        for contract in batch:
            candidate = repaired.get(contract.index, "").strip()
            if candidate and word_count(candidate) >= task.minimum_words:
                candidate_segments[contract.index] = candidate
                structurally_valid_candidates.append(contract)
        post_repair_results: dict[int, dict[str, Any]] = {}
        if structurally_valid_candidates:
            payload = _complete_json(
                llm,
                build_adaptive_review_prompt(
                    task,
                    structurally_valid_candidates,
                    candidate_segments,
                ),
                system="Validate repaired segment requirements. Return valid JSON only.",
                max_output_tokens=review_max_output_tokens,
            )
            post_repair_results = parse_review_results(payload)
        for contract in batch:
            candidate = repaired.get(contract.index, "").strip()
            post_review = post_repair_results.get(
                contract.index,
                {
                    "passed": False,
                    "missing_events": list(contract.required_events),
                    "issues": ["post-repair review result missing"],
                },
            )
            structurally_valid = (
                bool(candidate)
                and word_count(candidate) >= task.minimum_words
            )
            accepted = structurally_valid and bool(post_review["passed"])
            if accepted:
                segments[contract.index] = candidate
                unresolved_semantic_issues.pop(contract.index, None)
            else:
                unresolved_semantic_issues[contract.index] = post_review
            repair_records.append(
                {
                    "index": contract.index,
                    "accepted": accepted,
                    "candidate_words": word_count(candidate),
                    "post_review": post_review,
                    "completion": completion,
                }
            )
            review_records.append(
                {
                    "stage": "post_repair",
                    "index": contract.index,
                    **post_review,
                }
            )

    final_structural_issues = structural_issues(task, contracts, segments)
    final_text = assemble_document(task, segments)
    status = (
        "complete"
        if not final_structural_issues and not unresolved_semantic_issues
        else "incomplete"
    )
    output = {
        "task": task_record(task),
        "system": "harness_adaptive",
        "status": status,
        "final_text": final_text,
        "segments": [
            {"index": index, "text": text}
            for index, text in sorted(segments.items())
        ],
        "contracts": [
            {
                "index": contract.index,
                "purpose": contract.purpose,
                "required_events": contract.required_events,
                "continuity": contract.continuity,
            }
            for contract in contracts
        ],
        "metadata": {
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "model": llm.model,
            "pipeline_version": ADAPTIVE_PIPELINE_VERSION,
            "target_word_range": [minimum_words, maximum_words],
            "document_summary": document_summary,
            "compiled_contract_overrides": compiled_overrides,
            "generation": {
                "failed_units": generation.failed_units,
                "request_records": generation.request_records,
                "metrics": generation.metrics,
                "dependency_graph": generation.dependency_graph,
                "dependency_cache_hit": generation.dependency_cache_hit,
                "resumed_from_checkpoint": generation.resumed_from_checkpoint,
            },
            "review_results": review_records,
            "semantic_issue_indices": sorted(semantic_issues),
            "unresolved_semantic_issues": unresolved_semantic_issues,
            "structural_recovery_records": structural_recovery_records,
            "repair_records": repair_records,
            "final_structural_issues": final_structural_issues,
            "completion_history": completion_history(llm),
        },
    }
    if status == "incomplete" and fail_on_incomplete:
        raise AdaptiveHarnessIncompleteError(output)
    return output


def _recover_structural_issues(
    llm: ChatModel,
    adapter: LongGenBenchAdapter,
    units: list[GenerationUnit],
    segments: dict[int, str],
    issues: dict[int, dict[str, Any]],
    *,
    batch_size: int,
    generation_max_output_tokens: int,
    extension_max_output_tokens: int,
    round_index: int,
) -> list[dict[str, Any]]:
    by_index = {int(unit.unit_id): unit for unit in units}
    records: list[dict[str, Any]] = []
    completed_texts = {str(index): text for index, text in segments.items()}
    for action in ("extend", "generate"):
        targets = [
            by_index[index]
            for index in sorted(issues)
            if index in by_index
            and ((index in segments) == (action == "extend"))
        ]
        for batch in chunked(targets, batch_size):
            current_texts = {
                unit.unit_id: segments.get(int(unit.unit_id), "")
                for unit in batch
            }
            raw = llm.complete(
                adapter.build_prompt(
                    action,
                    batch,
                    current_texts,
                    completed_texts,
                ),
                system=adapter.system_prompt(action),
                temperature=0.1,
                max_output_tokens=(
                    extension_max_output_tokens
                    if action == "extend"
                    else generation_max_output_tokens
                ),
            )
            completion = llm.completion_metadata()
            parsed = adapter.parse_response(action, batch, raw)
            for unit in batch:
                index = int(unit.unit_id)
                candidate = parsed.get(unit.unit_id, "").strip()
                if candidate:
                    segments[index] = (
                        _append_continuation(segments.get(index, ""), candidate)
                        if action == "extend"
                        else candidate
                    )
                    completed_texts[unit.unit_id] = segments[index]
                validation = adapter.validate(unit, segments.get(index, ""))
                records.append(
                    {
                        "round": round_index + 1,
                        "action": action,
                        "index": index,
                        "parsed": bool(candidate),
                        "validation": {
                            "status": validation.status,
                            "measured_length": validation.measured_length,
                            "required_length": validation.required_length,
                            "issues": validation.issues,
                        },
                        "completion": completion,
                    }
                )
    return records


def _append_continuation(existing: str, continuation: str) -> str:
    existing = existing.strip()
    continuation = continuation.strip()
    if not existing:
        return continuation
    if continuation.startswith(existing):
        return continuation
    return f"{existing}\n\n{continuation}"


def _complete_json(
    llm: ChatModel,
    prompt: str,
    *,
    system: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    raw = llm.complete(
        prompt,
        system=system,
        temperature=0.0,
        response_format="json_object",
        max_output_tokens=max_output_tokens,
    )
    payload = extract_json_object(raw)
    if not isinstance(payload, dict):
        raise ValueError("adaptive review JSON must be an object")
    return payload
