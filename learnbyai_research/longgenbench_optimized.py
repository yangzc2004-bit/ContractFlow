from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .llm import ChatModel, completion_history
from .longgenbench import (
    LongGenTask,
    SegmentContract,
    _compiled_document_summary,
    _load_harness_checkpoint,
    _prior_context,
    _string_list,
    _write_harness_checkpoint,
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


OPTIMIZED_PIPELINE_VERSION = "adaptive_generation_recovery_v1"


def target_word_range(task: LongGenTask) -> tuple[int, int]:
    return task_word_range(task)


def compact_task_brief(task: LongGenTask) -> str:
    normalized = re.sub(r"\s+", " ", task.prompt).strip()
    first_sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0]
    if task.segment_type == "Week":
        return (
            f"{first_sentence} Write a coherent weekly diary with weather, work, family, "
            "and ordinary life updates."
        )
    if task.segment_type == "Menu Week":
        return (
            f"{first_sentence} Write a varied weekly restaurant menu with concrete dishes "
            "and concise descriptions."
        )
    if task.segment_type == "Floor":
        return (
            f"{first_sentence} Describe each floor's facilities, architectural features, "
            "and distinctive design."
        )
    if task.segment_type == "Block":
        return (
            f"{first_sentence} Describe each city block's facilities, architecture, "
            "and contribution to a diverse urban plan."
        )
    return first_sentence


def build_optimized_generation_prompt(
    task: LongGenTask,
    document_summary: str,
    contracts: list[SegmentContract],
    prior_context: str,
) -> str:
    minimum_words, maximum_words = target_word_range(task)
    indices = [contract.index for contract in contracts]
    contract_payload = [
        {
            "index": contract.index,
            "required_events": contract.required_events,
        }
        for contract in contracts
    ]
    example_index = indices[0]
    return f"""LONGGEN_WRITE_SEGMENTS_OPTIMIZED

Write only the requested segment indices: {indices}.
Do not write any earlier or later segment, and do not restart from segment 1 unless 1 is requested.

Requirements:
- Return every requested index exactly once.
- Each segment must contain {minimum_words} to {maximum_words} English words.
- Include every required event naturally in its assigned segment.
- Keep ordinary content concrete, varied, and consistent with TASK_BRIEF.
- Do not include "#*#" headings inside segment prose; the assembler adds them.
- Use these exact tags, replacing {example_index} with each requested index:

<<<SEGMENT {example_index}>>>
reader-facing prose
<<<END SEGMENT {example_index}>>>

TASK_BRIEF:
{compact_task_brief(task)}

DOCUMENT_SUMMARY:
{document_summary}

PRIOR_CONTEXT:
{prior_context or "(start of document)"}

REQUESTED_INDICES_JSON:
{json.dumps(indices)}

SEGMENT_CONTRACTS_JSON:
{json.dumps(contract_payload, ensure_ascii=False)}
"""


def build_optimized_review_prompt(
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
    return f"""LONGGEN_REVIEW_SEGMENTS_OPTIMIZED_JSON

Check whether each segment clearly contains every required event at the correct index.
Accept clear paraphrases. Do not criticize style or harmless extra detail.

Return valid JSON only:
{{
  "results": [
    {{"index": 1, "passed": true, "missing_events": [], "issues": []}}
  ]
}}

TASK_BRIEF:
{compact_task_brief(task)}

SEGMENTS_JSON:
{json.dumps(items, ensure_ascii=False)}
"""


def build_optimized_repair_prompt(
    task: LongGenTask,
    contracts: list[SegmentContract],
    segments: dict[int, str],
    issues: dict[int, dict[str, Any]],
) -> str:
    minimum_words, maximum_words = target_word_range(task)
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
    example_index = indices[0]
    return f"""LONGGEN_REPAIR_SEGMENTS_OPTIMIZED

Return complete replacements only for indices {indices}.
Each replacement must contain {minimum_words} to {maximum_words} English words.
Resolve every listed issue, include every required event at the correct index, and preserve
correct content. Do not emit any unrequested segment.

Use exact tags:
<<<SEGMENT {example_index}>>>
complete replacement prose
<<<END SEGMENT {example_index}>>>

TASK_BRIEF:
{compact_task_brief(task)}

FAILING_SEGMENTS_JSON:
{json.dumps(items, ensure_ascii=False)}
"""


def run_harness_optimized(
    task: LongGenTask,
    llm: ChatModel,
    *,
    batch_size: int = 8,
    generation_max_output_tokens: int = 4_096,
    review_max_output_tokens: int = 1_024,
    repair_max_output_tokens: int = 4_096,
    max_singleton_retries: int = 2,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint = _load_harness_checkpoint(checkpoint_path, task)
    document_summary = _compiled_document_summary(task)
    contracts = [
        SegmentContract(
            index=index,
            purpose=f"Write a distinct, coherent {task.segment_type.lower()} entry.",
        )
        for index in range(1, task.segment_count + 1)
    ]
    compiled_overrides = apply_compiled_contracts(contracts, compile_prompt_contracts(task))
    plan_payload = {
        "source": "deterministic_public_prompt_compiler",
        "pipeline_version": OPTIMIZED_PIPELINE_VERSION,
        "document_summary": document_summary,
        "segments": [{"index": contract.index} for contract in contracts],
    }

    if checkpoint:
        if checkpoint.get("plan_payload", {}).get("pipeline_version") != OPTIMIZED_PIPELINE_VERSION:
            raise ValueError("optimized checkpoint belongs to an incompatible pipeline version")
        segments = {
            int(index): str(text)
            for index, text in checkpoint.get("segments", {}).items()
            if str(text).strip()
        }
        generation_records = [
            dict(item) for item in checkpoint.get("generation_batches", []) if isinstance(item, dict)
        ]
        review_records = [
            dict(item) for item in checkpoint.get("review_results", []) if isinstance(item, dict)
        ]
        issues = {
            int(index): dict(record)
            for index, record in checkpoint.get("issues", {}).items()
            if isinstance(record, dict)
        }
        repair_records = [
            dict(item) for item in checkpoint.get("repair_records", []) if isinstance(item, dict)
        ]
    else:
        segments = {}
        generation_records = []
        review_records = []
        issues = {}
        repair_records = []
        _write_optimized_checkpoint(
            checkpoint_path,
            task,
            plan_payload,
            document_summary,
            contracts,
            compiled_overrides,
            segments,
            generation_records,
            review_records,
            issues,
            repair_records,
            stage="planned",
        )

    pending_batches = [
        batch
        for batch in chunked(
            [contract for contract in contracts if contract.index not in segments],
            batch_size,
        )
    ]
    singleton_attempts: dict[int, int] = {}
    while pending_batches:
        batch = pending_batches.pop(0)
        raw = llm.complete(
            build_optimized_generation_prompt(
                task,
                document_summary,
                batch,
                _prior_context(segments, max_characters=500),
            ),
            system="Write only the explicitly requested long-form segments and exact tags.",
            temperature=0.2,
            max_output_tokens=generation_max_output_tokens,
        )
        completion = llm.completion_metadata()
        parsed = parse_generated_segments(batch, raw)
        accepted: dict[int, str] = {}
        short_indices: list[int] = []
        for contract in batch:
            text = parsed.get(contract.index, "").strip()
            if text and word_count(text) >= task.minimum_words:
                accepted[contract.index] = text
            elif text:
                short_indices.append(contract.index)
        segments.update(accepted)
        missing = [
            contract
            for contract in batch
            if contract.index not in accepted
        ]
        generation_records.append(
            {
                "stage": "generation" if len(batch) == batch_size else "generation_recovery",
                "indices": [contract.index for contract in batch],
                "parsed_indices": sorted(parsed),
                "accepted_indices": sorted(accepted),
                "short_indices": short_indices,
                "missing_indices": [contract.index for contract in missing],
                "finish_reason": completion.get("finish_reason"),
                "completion": completion,
            }
        )
        if missing:
            if len(missing) > 1:
                split_at = max(1, len(missing) // 2)
                pending_batches.insert(0, missing[split_at:])
                pending_batches.insert(0, missing[:split_at])
            else:
                index = missing[0].index
                singleton_attempts[index] = singleton_attempts.get(index, 0) + 1
                if singleton_attempts[index] <= max_singleton_retries:
                    pending_batches.insert(0, missing)
        _write_optimized_checkpoint(
            checkpoint_path,
            task,
            plan_payload,
            document_summary,
            contracts,
            compiled_overrides,
            segments,
            generation_records,
            review_records,
            issues,
            repair_records,
            stage=f"generated_{len(segments)}",
        )

    hard_issues = structural_issues(task, contracts, segments)
    issues = dict(hard_issues)
    review_targets = [
        contract
        for contract in contracts
        if contract.required_events and contract.index in segments
    ]
    reviewed_indices = {
        int(record["index"])
        for record in review_records
        if "index" in record
    }
    for batch in chunked(review_targets, batch_size):
        outstanding = [contract for contract in batch if contract.index not in reviewed_indices]
        if not outstanding:
            continue
        payload = _complete_json(
            llm,
            build_optimized_review_prompt(task, outstanding, segments),
            system="Review explicit segment contracts. Return valid JSON only.",
            max_output_tokens=review_max_output_tokens,
        )
        batch_results = parse_review_results(payload)
        for contract in outstanding:
            record = batch_results.get(contract.index)
            if record is None:
                record = {
                    "passed": False,
                    "missing_events": list(contract.required_events),
                    "issues": ["review result missing"],
                }
            review_records.append({"index": contract.index, **record})
            if not record["passed"]:
                issues[contract.index] = record
            elif contract.index not in hard_issues:
                issues.pop(contract.index, None)
            reviewed_indices.add(contract.index)
        _write_optimized_checkpoint(
            checkpoint_path,
            task,
            plan_payload,
            document_summary,
            contracts,
            compiled_overrides,
            segments,
            generation_records,
            review_records,
            issues,
            repair_records,
            stage=f"reviewed_{len(reviewed_indices)}",
        )

    failing_contracts = [contract for contract in contracts if contract.index in issues]
    for batch in chunked(failing_contracts, max(1, batch_size // 2)):
        before_words = {
            contract.index: word_count(segments.get(contract.index, ""))
            for contract in batch
        }
        raw = llm.complete(
            build_optimized_repair_prompt(task, batch, segments, issues),
            system="Repair only the requested segments and return exact tags.",
            temperature=0.1,
            max_output_tokens=repair_max_output_tokens,
        )
        completion = llm.completion_metadata()
        repaired = parse_generated_segments(batch, raw)
        for contract in batch:
            text = repaired.get(contract.index, "").strip()
            accepted = bool(text) and word_count(text) >= task.minimum_words
            if accepted:
                segments[contract.index] = text
            repair_records.append(
                {
                    "index": contract.index,
                    "accepted": accepted,
                    "before_words": before_words[contract.index],
                    "after_words": word_count(text),
                    "completion": completion,
                }
            )
        _write_optimized_checkpoint(
            checkpoint_path,
            task,
            plan_payload,
            document_summary,
            contracts,
            compiled_overrides,
            segments,
            generation_records,
            review_records,
            issues,
            repair_records,
            stage=f"repaired_{len(repair_records)}",
        )

    final_structural_issues = structural_issues(task, contracts, segments)
    final_text = assemble_document(task, segments)
    return {
        "task": task_record(task),
        "system": "harness_optimized",
        "final_text": final_text,
        "segments": [{"index": index, "text": text} for index, text in sorted(segments.items())],
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
            "pipeline_version": OPTIMIZED_PIPELINE_VERSION,
            "batch_size": batch_size,
            "target_word_range": target_word_range(task),
            "document_summary": document_summary,
            "compiled_contract_overrides": compiled_overrides,
            "generation_batches": generation_records,
            "review_results": review_records,
            "repair_records": repair_records,
            "pre_repair_issue_indices": sorted(issues),
            "final_structural_issues": final_structural_issues,
            "completion_history": completion_history(llm),
            "resumed_from_checkpoint": checkpoint is not None,
        },
    }


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
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"optimized review returned invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("optimized review JSON must be an object")
    return payload


def _write_optimized_checkpoint(
    checkpoint_path: str | Path | None,
    task: LongGenTask,
    plan_payload: dict[str, Any],
    document_summary: str,
    contracts: list[SegmentContract],
    compiled_overrides: list[dict[str, Any]],
    segments: dict[int, str],
    generation_records: list[dict[str, Any]],
    review_records: list[dict[str, Any]],
    issues: dict[int, dict[str, Any]],
    repair_records: list[dict[str, Any]],
    *,
    stage: str,
) -> None:
    _write_harness_checkpoint(
        checkpoint_path,
        task,
        plan_payload,
        document_summary,
        contracts,
        [],
        compiled_overrides,
        segments,
        generation_records,
        review_records,
        issues,
        repair_records,
        stage=stage,
    )
