from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from .io_utils import extract_json_object
from .llm import ChatModel, completion_history


LONGGEN_TYPES = ("Week", "Floor", "Menu Week", "Block")
HARNESS_PIPELINE_VERSION = "contractflow_longgenbench_v0"


@dataclass
class SegmentContract:
    index: int
    purpose: str = ""
    required_events: list[str] = field(default_factory=list)
    continuity: str = ""


@dataclass
class LongGenTask:
    dataset_index: int
    prompt: str
    segment_type: str
    segment_count: int
    prefix: str
    minimum_words: int


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError("LongGenBench dataset must be a JSON list")
    return [dict(item) for item in value if isinstance(item, dict)]


def select_balanced_indices(dataset: list[dict[str, Any]], per_type: int) -> list[int]:
    selected: list[int] = []
    counts = {name: 0 for name in LONGGEN_TYPES}
    for index, item in enumerate(dataset):
        segment_type = str(item.get("type", ""))
        if segment_type not in counts or counts[segment_type] >= per_type:
            continue
        selected.append(index)
        counts[segment_type] += 1
        if all(count == per_type for count in counts.values()):
            break
    missing = {name: per_type - count for name, count in counts.items() if count < per_type}
    if missing:
        raise ValueError(f"dataset does not contain the requested balanced sample: {missing}")
    return selected


def task_from_dataset_item(dataset_index: int, item: dict[str, Any]) -> LongGenTask:
    prompt = str(item["prompt"])
    minimum_words = parse_minimum_words(prompt)
    return LongGenTask(
        dataset_index=dataset_index,
        prompt=prompt,
        segment_type=str(item["type"]),
        segment_count=int(item["number"]),
        prefix=str(item.get("prefix", "")),
        minimum_words=minimum_words,
    )


def parse_minimum_words(prompt: str) -> int:
    patterns = (
        r"each[^.\n]{0,100}?at least\s+(\d+)\s+words",
        r"each containing at least\s+(\d+)\s+words",
        r"each entry should be at least\s+(\d+)\s+words",
        r"between\s+(\d+)\s+and\s+\d+\s+words",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 100


def task_word_range(
    task: LongGenTask,
    *,
    fallback_growth: float = 1.1,
    fallback_extra_words: int = 12,
) -> tuple[int, int]:
    maximum_patterns = (
        r"between\s+\d+\s+and\s+(\d+)\s+words",
        r"at least\s+\d+\s+but no more than\s+(\d+)\s+words",
        r"maximum of\s+(\d+)\s+words",
    )
    for pattern in maximum_patterns:
        match = re.search(pattern, task.prompt, flags=re.IGNORECASE)
        if match:
            return task.minimum_words, max(task.minimum_words, int(match.group(1)))
    fallback = max(
        task.minimum_words + fallback_extra_words,
        math.ceil(task.minimum_words * fallback_growth),
    )
    return task.minimum_words, fallback


def task_id(task: LongGenTask) -> str:
    slug = task.segment_type.lower().replace(" ", "_")
    return f"{task.dataset_index:04d}_{slug}"


def dataset_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_selection_manifest(
    dataset_path: str | Path,
    dataset: list[dict[str, Any]],
    selected_indices: Iterable[int],
) -> dict[str, Any]:
    tasks = []
    for index in selected_indices:
        task = task_from_dataset_item(index, dataset[index])
        tasks.append(
            {
                "dataset_index": index,
                "task_id": task_id(task),
                "type": task.segment_type,
                "segment_count": task.segment_count,
                "minimum_words": task.minimum_words,
                "prompt_sha256": hashlib.sha256(task.prompt.encode("utf-8")).hexdigest(),
            }
        )
    return {
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": dataset_sha256(dataset_path),
        "selection_policy": "first_n_per_type_in_dataset_order",
        "tasks": tasks,
    }


def build_plan_prompt(task: LongGenTask) -> str:
    return f"""LONGGEN_CONTRACT_PLAN_JSON

Create an explicit segment-by-segment writing plan for the long-form generation task below.
The plan is an execution contract, not prose for the final answer.

Requirements:
- Produce exactly {task.segment_count} segment records, indexed 1 through {task.segment_count}.
- The segment type is {task.segment_type}.
- Read dates, ranges, intervals, and periodic rules carefully.
- Expand every single-instance, range, and periodic requirement into every affected segment.
- Put concrete required content in required_events for the exact affected segment.
- Put ordinary filler or continuity guidance in purpose and continuity.
- Do not invent additional special events.
- Do not use information outside USER_PROMPT.

Return valid JSON only:
{{
  "document_summary": "short global continuity plan",
  "segments": [
    {{
      "index": 1,
      "purpose": "ordinary content planned for this segment",
      "required_events": ["exact required event for this segment"],
      "continuity": "brief link from the prior segment"
    }}
  ]
}}

USER_PROMPT:
{task.prompt}
"""


def build_plan_audit_prompt(task: LongGenTask, plan: dict[str, Any]) -> str:
    compact_plan = {
        "document_summary": plan.get("document_summary", ""),
        "segments": [
            {
                "index": item.get("index"),
                "required_events": item.get("required_events", []),
            }
            for item in plan.get("segments", [])
            if isinstance(item, dict)
        ],
    }
    return f"""LONGGEN_CONTRACT_AUDIT_JSON

Audit the extracted long-form contracts against USER_PROMPT.
Focus only on incorrect or missing single-instance, range, and periodic requirements.
Do not rewrite the whole plan and do not add ordinary filler.

Return valid JSON only:
{{
  "corrections": [
    {{
      "index": 1,
      "replace_required_events": ["complete corrected event list for this segment"]
    }}
  ]
}}
Return an empty corrections list when the contracts are accurate.

USER_PROMPT:
{task.prompt}

CURRENT_CONTRACTS:
{json.dumps(compact_plan, ensure_ascii=False)}
"""


def parse_contract_plan(task: LongGenTask, payload: dict[str, Any]) -> tuple[str, list[SegmentContract]]:
    raw_segments = payload.get("segments", [])
    if not isinstance(raw_segments, list):
        raise ValueError("contract plan segments must be a list")
    by_index: dict[int, SegmentContract] = {}
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("index"))
        except (TypeError, ValueError):
            continue
        if not 1 <= index <= task.segment_count or index in by_index:
            continue
        by_index[index] = SegmentContract(
            index=index,
            purpose=str(raw.get("purpose", "")),
            required_events=_string_list(raw.get("required_events")),
            continuity=str(raw.get("continuity", "")),
        )
    missing = [index for index in range(1, task.segment_count + 1) if index not in by_index]
    if missing:
        raise ValueError(f"contract plan is missing segment indices: {missing[:10]}")
    return str(payload.get("document_summary", "")), [by_index[index] for index in sorted(by_index)]


def apply_plan_corrections(
    contracts: list[SegmentContract],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    by_index = {contract.index: contract for contract in contracts}
    applied: list[dict[str, Any]] = []
    raw_corrections = payload.get("corrections", [])
    if not isinstance(raw_corrections, list):
        return applied
    for raw in raw_corrections:
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("index"))
        except (TypeError, ValueError):
            continue
        contract = by_index.get(index)
        if contract is None or "replace_required_events" not in raw:
            continue
        replacement = _string_list(raw.get("replace_required_events"))
        before = list(contract.required_events)
        contract.required_events = replacement
        applied.append({"index": index, "before": before, "after": replacement})
    return applied


def compile_prompt_contracts(task: LongGenTask) -> dict[int, list[str]]:
    if task.segment_type == "Week":
        return _compile_week_contracts(task.prompt, task.segment_count)
    if task.segment_type == "Floor":
        return _compile_floor_contracts(task.prompt, task.segment_count)
    if task.segment_type == "Menu Week":
        return _compile_menu_contracts(task.prompt, task.segment_count)
    if task.segment_type == "Day":
        return _compile_day_contracts(task.prompt, task.segment_count)
    if task.segment_type == "Menu Day":
        return _compile_menu_day_contracts(task.prompt, task.segment_count)
    if task.segment_type == "Block":
        return _compile_block_contracts(task.prompt, task.segment_count)
    return {}


def apply_compiled_contracts(
    contracts: list[SegmentContract],
    compiled: dict[int, list[str]],
) -> list[dict[str, Any]]:
    overrides: list[dict[str, Any]] = []
    for contract in contracts:
        replacement = list(compiled.get(contract.index, []))
        before = list(contract.required_events)
        contract.required_events = replacement
        if before != replacement:
            overrides.append({"index": contract.index, "before": before, "after": replacement})
    return overrides


def build_generation_prompt(
    task: LongGenTask,
    document_summary: str,
    contracts: list[SegmentContract],
    prior_context: str,
) -> str:
    contract_payload = [
        {
            "index": contract.index,
            "purpose": contract.purpose,
            "required_events": contract.required_events,
            "continuity": contract.continuity,
        }
        for contract in contracts
    ]
    return f"""LONGGEN_WRITE_SEGMENTS

Write the requested long-form document segments using the execution contracts.

Requirements:
- Return exactly the requested segment indices.
- Each segment must contain at least {task.minimum_words} English words.
- Include every required_event naturally and unambiguously in its segment.
- Do not move a required event to another index.
- Maintain continuity with PRIOR_CONTEXT and DOCUMENT_SUMMARY.
- Return final reader-facing prose, not an outline or analysis.
- Do not put the prose in JSON. Use the exact tags below for every segment:

<<<SEGMENT 1>>>
complete segment prose
<<<END SEGMENT 1>>>

DOCUMENT_SUMMARY:
{document_summary}

PRIOR_CONTEXT:
{prior_context or "(start of document)"}

SEGMENT_CONTRACTS:
{json.dumps(contract_payload, ensure_ascii=False)}

USER_PROMPT:
{task.prompt}
"""


def build_segmented_generation_prompt(
    task: LongGenTask,
    indices: list[int],
    prior_context: str,
) -> str:
    return f"""LONGGEN_WRITE_SEGMENTS

Write the requested long-form document segments directly from USER_PROMPT.

Requirements:
- Return exactly these segment indices: {indices}.
- Each segment must contain at least {task.minimum_words} English words.
- Follow all requirements in USER_PROMPT.
- Maintain continuity with PRIOR_CONTEXT.
- Return final reader-facing prose, not an outline or analysis.
- Do not use JSON. Use the exact tags below for every segment:

<<<SEGMENT 1>>>
complete segment prose
<<<END SEGMENT 1>>>

PRIOR_CONTEXT:
{prior_context or "(start of document)"}

USER_PROMPT:
{task.prompt}
"""


def parse_generated_segments(
    expected: list[SegmentContract],
    payload: dict[str, Any] | str,
) -> dict[int, str]:
    expected_indices = {contract.index for contract in expected}
    result: dict[int, str] = {}
    if isinstance(payload, str):
        pattern = re.compile(
            r"<<<SEGMENT\s+(\d+)>>>\s*(.*?)\s*<<<END SEGMENT\s+\1>>>",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(payload):
            index = int(match.group(1))
            text = match.group(2).strip()
            if index in expected_indices and text and index not in result:
                result[index] = text
        return result
    raw_segments = payload.get("segments", [])
    if not isinstance(raw_segments, list):
        return result
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("index"))
        except (TypeError, ValueError):
            continue
        text = str(raw.get("text", "")).strip()
        if index in expected_indices and text and index not in result:
            result[index] = text
    return result


def build_review_prompt(
    task: LongGenTask,
    contracts: list[SegmentContract],
    segments: dict[int, str],
) -> str:
    items = [
        {
            "index": contract.index,
            "required_events": contract.required_events,
            "minimum_words": task.minimum_words,
            "text": segments.get(contract.index, ""),
        }
        for contract in contracts
    ]
    return f"""LONGGEN_REVIEW_SEGMENTS_JSON

Check each segment against its execution contract.
Mark passed=false when a required event is absent, materially incorrect, assigned to the
wrong index, or the segment is incomplete. Treat clear paraphrases as present.

Return valid JSON only:
{{
  "results": [
    {{
      "index": 1,
      "passed": true,
      "missing_events": [],
      "issues": []
    }}
  ]
}}

USER_PROMPT:
{task.prompt}

SEGMENTS_TO_REVIEW:
{json.dumps(items, ensure_ascii=False)}
"""


def parse_review_results(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        return result
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("index"))
        except (TypeError, ValueError):
            continue
        result[index] = {
            "passed": bool(raw.get("passed")),
            "missing_events": _string_list(raw.get("missing_events")),
            "issues": _string_list(raw.get("issues")),
        }
    return result


def build_repair_prompt(
    task: LongGenTask,
    contracts: list[SegmentContract],
    segments: dict[int, str],
    issues: dict[int, dict[str, Any]],
) -> str:
    items = [
        {
            "index": contract.index,
            "required_events": contract.required_events,
            "minimum_words": task.minimum_words,
            "current_text": segments.get(contract.index, ""),
            "review": issues.get(contract.index, {}),
        }
        for contract in contracts
    ]
    return f"""LONGGEN_REPAIR_SEGMENTS

Replace each failing segment with complete reader-facing prose.
Preserve correct content, resolve every review issue, include every required event at the
correct index, and meet the minimum word count. Return the complete replacement segment.
Do not use JSON. Use the exact tags below for every replacement:

<<<SEGMENT 1>>>
complete repaired segment prose
<<<END SEGMENT 1>>>

USER_PROMPT:
{task.prompt}

FAILING_SEGMENTS:
{json.dumps(items, ensure_ascii=False)}
"""


def build_structural_repair_prompt(
    task: LongGenTask,
    contracts: list[SegmentContract],
    segments: dict[int, str],
    issues: dict[int, dict[str, Any]],
) -> str:
    items = [
        {
            "index": contract.index,
            "minimum_words": task.minimum_words,
            "current_text": segments.get(contract.index, ""),
            "issues": issues.get(contract.index, {}).get("issues", []),
        }
        for contract in contracts
    ]
    return f"""LONGGEN_REPAIR_SEGMENTS

Replace each missing or structurally incomplete segment with complete reader-facing
prose. Follow USER_PROMPT, preserve correct content, and meet the minimum word count.
Return the complete replacement segment using the exact tags below:

<<<SEGMENT 1>>>
complete repaired segment prose
<<<END SEGMENT 1>>>

USER_PROMPT:
{task.prompt}

FAILING_SEGMENTS:
{json.dumps(items, ensure_ascii=False)}
"""


def build_direct_prompt(task: LongGenTask) -> str:
    return task.prompt


def assemble_document(task: LongGenTask, segments: dict[int, str]) -> str:
    blocks = []
    for index in range(1, task.segment_count + 1):
        text = segments.get(index, "").strip()
        if not text:
            continue
        existing_marker = re.match(
            rf"^(?:#\*#|###)\s*{re.escape(task.segment_type)}\s+{index}\b",
            text,
            flags=re.IGNORECASE,
        )
        blocks.append(
            text
            if existing_marker
            else f"#*# {task.segment_type} {index}:\n{text}"
        )
    return "\n\n".join(blocks) + "\n\n*** finished ***"


def parse_document_blocks(text: str, segment_type: str) -> dict[int, str]:
    escaped = re.escape(segment_type)
    marker = rf"(?:#\*#|###)\s*{escaped}\s+(\d+)[^:\n]*:\s*"
    matches = list(re.finditer(marker, text, flags=re.IGNORECASE))
    if not matches:
        fallback = rf"(?:^|\n)\s*{escaped}\s+(\d+)[^:\n]*:\s*"
        matches = list(re.finditer(fallback, text, flags=re.IGNORECASE))
    result: dict[int, str] = {}
    for position, match in enumerate(matches):
        index = int(match.group(1))
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        block = text[match.end() : end]
        block = re.split(r"\*\*\*\s*finished", block, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if index not in result and block:
            result[index] = block
    return result


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def structural_issues(
    task: LongGenTask,
    contracts: list[SegmentContract],
    segments: dict[int, str],
) -> dict[int, dict[str, Any]]:
    issues: dict[int, dict[str, Any]] = {}
    for contract in contracts:
        text = segments.get(contract.index, "")
        count = word_count(text)
        messages = []
        if not text:
            messages.append("segment is missing")
        if count < task.minimum_words:
            messages.append(f"word count {count} is below {task.minimum_words}")
        if messages:
            issues[contract.index] = {
                "passed": False,
                "missing_events": list(contract.required_events),
                "issues": messages,
            }
    return issues


def chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_direct(
    task: LongGenTask,
    llm: ChatModel,
    *,
    max_output_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    text = llm.complete(
        build_direct_prompt(task),
        system="Follow the user's long-form generation request exactly. Return only the requested document.",
        temperature=0.2,
        max_output_tokens=max_output_tokens,
    ).strip()
    blocks = parse_document_blocks(text, task.segment_type)
    return {
        "task": task_record(task),
        "system": "direct",
        "final_text": text,
        "segments": [{"index": index, "text": block} for index, block in sorted(blocks.items())],
        "metadata": {
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "model": llm.model,
            "completion_history": completion_history(llm),
        },
    }


def run_harness(
    task: LongGenTask,
    llm: ChatModel,
    *,
    batch_size: int,
    plan_max_output_tokens: int,
    generation_max_output_tokens: int,
    review_max_output_tokens: int,
    repair_max_output_tokens: int,
    checkpoint_path: str | Path | None = None,
    use_llm_plan: bool = True,
    use_contracts: bool = True,
    use_review_repair: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint = _load_harness_checkpoint(checkpoint_path, task)
    if checkpoint:
        plan_payload = dict(checkpoint["plan_payload"])
        document_summary = str(checkpoint.get("document_summary", ""))
        contracts = [
            SegmentContract(
                index=int(raw["index"]),
                purpose=str(raw.get("purpose", "")),
                required_events=_string_list(raw.get("required_events")),
                continuity=str(raw.get("continuity", "")),
            )
            for raw in checkpoint.get("contracts", [])
            if isinstance(raw, dict)
        ]
        if len(contracts) != task.segment_count:
            raise ValueError("harness checkpoint has an incompatible contract count")
        applied_corrections = [
            dict(item) for item in checkpoint.get("plan_corrections", []) if isinstance(item, dict)
        ]
        compiled_overrides = [
            dict(item)
            for item in checkpoint.get("compiled_contract_overrides", [])
            if isinstance(item, dict)
        ]
        segments = {
            int(index): str(text)
            for index, text in checkpoint.get("segments", {}).items()
            if str(text).strip()
        }
        generation_batches = [
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
        if use_contracts and use_llm_plan:
            plan_payload = complete_json(
                llm,
                build_plan_prompt(task),
                system="You are a long-form contract planner. Return valid JSON only.",
                temperature=0.1,
                max_output_tokens=plan_max_output_tokens,
            )
            document_summary, contracts = parse_contract_plan(task, plan_payload)
            audit_payload = complete_json(
                llm,
                build_plan_audit_prompt(task, plan_payload),
                system="You audit long-form execution contracts. Return valid JSON only.",
                temperature=0.0,
                max_output_tokens=plan_max_output_tokens,
            )
            applied_corrections = apply_plan_corrections(contracts, audit_payload)
        elif use_contracts:
            document_summary = _compiled_document_summary(task)
            contracts = [
                SegmentContract(
                    index=index,
                    purpose=f"Write a distinct, coherent {task.segment_type.lower()} entry.",
                )
                for index in range(1, task.segment_count + 1)
            ]
            plan_payload = {
                "source": "deterministic_public_prompt_compiler",
                "document_summary": document_summary,
                "segments": [{"index": contract.index} for contract in contracts],
            }
            applied_corrections = []
        else:
            document_summary = (
                f"Generate {task.segment_count} ordered {task.segment_type} segments "
                "from the public task prompt."
            )
            contracts = [
                SegmentContract(
                    index=index,
                    purpose=f"Write segment {index} of the requested document.",
                )
                for index in range(1, task.segment_count + 1)
            ]
            plan_payload = {
                "source": "segmented_generation_without_contracts",
                "document_summary": document_summary,
                "segments": [{"index": contract.index} for contract in contracts],
            }
            applied_corrections = []
        compiled_overrides = (
            apply_compiled_contracts(contracts, compile_prompt_contracts(task))
            if use_contracts
            else []
        )
        segments = {}
        generation_batches = []
        review_records = []
        issues = {}
        repair_records = []
        _write_harness_checkpoint(
            checkpoint_path,
            task,
            plan_payload,
            document_summary,
            contracts,
            applied_corrections,
            compiled_overrides,
            segments,
            generation_batches,
            review_records,
            issues,
            repair_records,
            stage="planned",
        )

    for batch in chunked(contracts, batch_size):
        if all(contract.index in segments for contract in batch):
            continue
        generation_prompt = (
            build_generation_prompt(
                task,
                document_summary,
                batch,
                _prior_context(segments),
            )
            if use_contracts
            else build_segmented_generation_prompt(
                task,
                [contract.index for contract in batch],
                _prior_context(segments),
            )
        )
        raw = llm.complete(
            generation_prompt,
            system="You write final long-form segments using the required segment tags.",
            temperature=0.25,
            max_output_tokens=generation_max_output_tokens,
        )
        generated = parse_generated_segments(batch, raw)
        segments.update(generated)
        generation_batches.append(
            {
                "indices": [contract.index for contract in batch],
                "returned_indices": sorted(generated),
            }
        )
        _write_harness_checkpoint(
            checkpoint_path,
            task,
            plan_payload,
            document_summary,
            contracts,
            applied_corrections,
            compiled_overrides,
            segments,
            generation_batches,
            review_records,
            issues,
            repair_records,
            stage=f"generated_through_{max(segments) if segments else 0}",
        )

    hard_issues = structural_issues(task, contracts, segments)
    issues.update(hard_issues)
    review_targets = (
        [contract for contract in contracts if contract.required_events]
        if use_review_repair
        else []
    )
    reviewed_indices = {
        int(record["index"])
        for record in review_records
        if "index" in record
    }
    for batch in chunked(review_targets, batch_size):
        if all(contract.index in reviewed_indices for contract in batch):
            continue
        payload = complete_json(
            llm,
            build_review_prompt(task, batch, segments),
            system="You review long-form segments against explicit contracts. Return valid JSON only.",
            temperature=0.0,
            max_output_tokens=review_max_output_tokens,
        )
        batch_results = parse_review_results(payload)
        review_records.extend(
            {"index": index, **record}
            for index, record in sorted(batch_results.items())
        )
        for contract in batch:
            record = batch_results.get(contract.index)
            if record is None:
                issues.setdefault(
                    contract.index,
                    {
                        "passed": False,
                        "missing_events": list(contract.required_events),
                        "issues": ["review result missing"],
                    },
                )
            elif not record["passed"]:
                issues[contract.index] = record
            elif contract.index not in hard_issues:
                issues.pop(contract.index, None)
        reviewed_indices.update(batch_results)
        _write_harness_checkpoint(
            checkpoint_path,
            task,
            plan_payload,
            document_summary,
            contracts,
            applied_corrections,
            compiled_overrides,
            segments,
            generation_batches,
            review_records,
            issues,
            repair_records,
            stage=f"reviewed_{len(reviewed_indices)}",
        )

    failing_contracts = (
        [contract for contract in contracts if contract.index in issues]
        if use_review_repair
        else []
    )
    repaired_indices = {
        int(record["index"])
        for record in repair_records
        if "index" in record
    }
    for batch in chunked(failing_contracts, max(1, batch_size // 2)):
        batch = [contract for contract in batch if contract.index not in repaired_indices]
        if not batch:
            continue
        repair_prompt = (
            build_repair_prompt(task, batch, segments, issues)
            if use_contracts
            else build_structural_repair_prompt(task, batch, segments, issues)
        )
        raw = llm.complete(
            repair_prompt,
            system="You repair complete long-form segments using the required segment tags.",
            temperature=0.15,
            max_output_tokens=repair_max_output_tokens,
        )
        repaired = parse_generated_segments(batch, raw)
        for index, text in repaired.items():
            before_words = word_count(segments.get(index, ""))
            segments[index] = text
            repair_records.append(
                {
                    "index": index,
                    "before_words": before_words,
                    "after_words": word_count(text),
                }
            )
        repaired_indices.update(repaired)
        _write_harness_checkpoint(
            checkpoint_path,
            task,
            plan_payload,
            document_summary,
            contracts,
            applied_corrections,
            compiled_overrides,
            segments,
            generation_batches,
            review_records,
            issues,
            repair_records,
            stage=f"repaired_{len(repaired_indices)}",
        )

    final_structural_issues = structural_issues(task, contracts, segments)
    final_text = assemble_document(task, segments)
    return {
        "task": task_record(task),
        "system": harness_system_name(use_contracts, use_review_repair),
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
            "pipeline_version": HARNESS_PIPELINE_VERSION,
            "batch_size": batch_size,
            "document_summary": document_summary,
            "plan_corrections": applied_corrections,
            "compiled_contract_overrides": compiled_overrides,
            "generation_batches": generation_batches,
            "review_results": review_records,
            "repair_records": repair_records,
            "pre_repair_issue_indices": sorted(issues),
            "final_structural_issues": final_structural_issues,
            "completion_history": completion_history(llm),
            "resumed_from_checkpoint": checkpoint is not None,
            "use_llm_plan": use_llm_plan and use_contracts,
            "use_contracts": use_contracts,
            "use_review_repair": use_review_repair,
        },
    }


def harness_system_name(use_contracts: bool, use_review_repair: bool) -> str:
    if use_contracts and use_review_repair:
        return "contractflow"
    if not use_contracts and use_review_repair:
        return "no_contract"
    if use_contracts and not use_review_repair:
        return "no_review_repair"
    return "neither"


def complete_json(
    llm: ChatModel,
    prompt: str,
    *,
    system: str,
    temperature: float,
    max_output_tokens: int,
    max_schema_retries: int = 1,
) -> dict[str, Any]:
    raw = ""
    error: Exception | None = None
    for attempt in range(max_schema_retries + 1):
        request_prompt = prompt
        if attempt:
            request_prompt = f"""LONGGEN_JSON_SCHEMA_CORRECTION

Repair the malformed response below into valid JSON without changing its substantive
content, omitting records, or adding commentary. Escape quotation marks inside strings.
Return corrected JSON only.

MALFORMED_RESPONSE:
{raw}

PARSER_ERROR:
{error}
"""
        raw = llm.complete(
            request_prompt,
            system=system,
            temperature=0.0 if attempt else temperature,
            response_format="json_object",
            max_output_tokens=max_output_tokens,
        )
        try:
            return extract_json_object(raw)
        except ValueError as caught:
            error = caught
    raise ValueError(f"JSON schema correction failed: {error}")


def task_record(task: LongGenTask) -> dict[str, Any]:
    return {
        "dataset_index": task.dataset_index,
        "task_id": task_id(task),
        "type": task.segment_type,
        "segment_count": task.segment_count,
        "minimum_words": task.minimum_words,
        "prefix": task.prefix,
        "prompt_sha256": hashlib.sha256(task.prompt.encode("utf-8")).hexdigest(),
    }


def output_segments(output: dict[str, Any]) -> dict[int, str]:
    segments: dict[int, str] = {}
    for raw in output.get("segments", []):
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("index"))
        except (TypeError, ValueError):
            continue
        text = str(raw.get("text", "")).strip()
        if text:
            segments[index] = text
    return segments


def _prior_context(segments: dict[int, str], max_characters: int = 1_200) -> str:
    if not segments:
        return ""
    latest = segments[max(segments)]
    return latest[-max_characters:]


def _compiled_document_summary(task: LongGenTask) -> str:
    first_sentence = re.split(r"(?<=[.!?])\s+", task.prompt.strip(), maxsplit=1)[0]
    return (
        f"{first_sentence} Produce exactly {task.segment_count} coherent "
        f"{task.segment_type} segments while following the compiled hard constraints."
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _compile_week_contracts(prompt: str, segment_count: int) -> dict[int, list[str]]:
    events: dict[int, list[str]] = {}
    birthday_line = re.search(
        r"Family member birthday:\s*(.+?)(?:\n|,\s*\n)",
        prompt,
        flags=re.IGNORECASE,
    )
    if birthday_line:
        for match in re.finditer(
            r"([^,]+?)\s*\(birthday on\s+([A-Za-z]+)\s+(\d{1,2})\)",
            birthday_line.group(1),
            flags=re.IGNORECASE,
        ):
            person = match.group(1).strip()
            week = _week_for_month_day(match.group(2), int(match.group(3)))
            _add_event(events, week, f"{person} birthday")

    range_match = re.search(
        r"\n2\)\s*(.+?)\s+in week\s+(\d+)\s*-\s*(\d+)",
        prompt,
        flags=re.IGNORECASE,
    )
    if range_match:
        description = range_match.group(1).strip().rstrip(".")
        for index in range(int(range_match.group(2)), int(range_match.group(3)) + 1):
            _add_event(events, index, description)

    periodic_match = re.search(
        r"\n3\)\s*(.+?)\s+every\s+(\d+)\s+weeks.*?starting from week\s+(\d+)",
        prompt,
        flags=re.IGNORECASE,
    )
    if periodic_match:
        description = periodic_match.group(1).strip().rstrip(".")
        interval = int(periodic_match.group(2))
        start = int(periodic_match.group(3))
        for index in range(start, segment_count + 1, interval):
            _add_event(events, index, description)
    return events


def _compile_floor_contracts(prompt: str, segment_count: int) -> dict[int, list[str]]:
    events: dict[int, list[str]] = {}
    for match in re.finditer(
        r"Designate Floor\s+(\d+)\s+for\s+(.+?)\s+use\.",
        prompt,
        flags=re.IGNORECASE,
    ):
        _add_event(events, int(match.group(1)), match.group(2).strip())

    range_match = re.search(
        r"Allocate Floors\s+(\d+)\s+to\s+(\d+)\s+for\s+(?:an?\s+)?(.+?)\.",
        prompt,
        flags=re.IGNORECASE,
    )
    if range_match:
        for index in range(int(range_match.group(1)), int(range_match.group(2)) + 1):
            _add_event(events, index, range_match.group(3).strip())

    periodic_match = re.search(
        r"Include\s+(?:an?\s+)?(.+?)\s+every\s+(\d+)\s+floors,\s+starting from Floor\s+(\d+)",
        prompt,
        flags=re.IGNORECASE,
    )
    if periodic_match:
        description = periodic_match.group(1).strip()
        interval = int(periodic_match.group(2))
        start = int(periodic_match.group(3))
        for index in range(start, segment_count + 1, interval):
            _add_event(events, index, description)
    return events


def _compile_menu_contracts(prompt: str, segment_count: int) -> dict[int, list[str]]:
    events: dict[int, list[str]] = {}
    for match in re.finditer(
        r"Celebrate\s+'(.+?)'\s+on\s+(\d{2})-(\d{2})\s+with a special dish:\s*(.+?)\.",
        prompt,
        flags=re.IGNORECASE,
    ):
        week = _week_for_numeric_date(int(match.group(2)), int(match.group(3)))
        _add_event(events, week, f"{match.group(1)} featuring {match.group(4).strip()}")

    range_match = re.search(
        r"Feature\s+'(.+?)'\s+from Week\s+(\d+)\s+to Week\s+(\d+)\s+with dishes like:\s*(.+?)\.",
        prompt,
        flags=re.IGNORECASE,
    )
    if range_match:
        description = f"{range_match.group(1)} featuring {range_match.group(4).strip()}"
        for index in range(int(range_match.group(2)), int(range_match.group(3)) + 1):
            _add_event(events, index, description)

    periodic_match = re.search(
        r"Include\s+'(.+?)'\s+every\s+(\d+)\s+weeks\s+starting from Week\s+(\d+),\s+serving:\s*(.+?)\.",
        prompt,
        flags=re.IGNORECASE,
    )
    if periodic_match:
        description = f"{periodic_match.group(1)} featuring {periodic_match.group(4).strip()}"
        interval = int(periodic_match.group(2))
        start = int(periodic_match.group(3))
        for index in range(start, segment_count + 1, interval):
            _add_event(events, index, description)
    return events


def _compile_day_contracts(prompt: str, segment_count: int) -> dict[int, list[str]]:
    events: dict[int, list[str]] = {}
    birthday_section = re.search(
        r"1\)\s*Family birthdays:\s*(.*?)(?=\n\s*2\))",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if birthday_section:
        for match in re.finditer(
            r"^\s*-\s*(.+?)\s+\(birthday on\s+([A-Za-z]+)\s+(\d{1,2})\)\s*$",
            birthday_section.group(1),
            flags=re.IGNORECASE | re.MULTILINE,
        ):
            person = match.group(1).strip()
            index = _day_for_month_day(match.group(2), int(match.group(3)))
            _add_event(events, index, f"{person} birthday")

    once_match = re.search(
        r"\n\s*2\)\s*(.+?)\s+on\s+([A-Za-z]+)\s+(\d{1,2})\s*$",
        prompt,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if once_match:
        index = _day_for_month_day(once_match.group(2), int(once_match.group(3)))
        _add_event(events, index, once_match.group(1).strip().rstrip("."))

    range_match = re.search(
        r"\n\s*3\)\s*(.+?)\s+from\s+([A-Za-z]+)\s+(\d{1,2})\s+to\s+"
        r"([A-Za-z]+)\s+(\d{1,2}),\s*2018",
        prompt,
        flags=re.IGNORECASE,
    )
    if range_match:
        description = range_match.group(1).strip().rstrip(".")
        start = _day_for_month_day(range_match.group(2), int(range_match.group(3)))
        end = _day_for_month_day(range_match.group(4), int(range_match.group(5)))
        for index in range(start, end + 1):
            _add_event(events, index, description)

    periodic_match = re.search(
        r"\n\s*4\)\s*(.+?)\s+starting from day\s+(\d+)\s+and repeating every\s+"
        r"(\d+)\s+days",
        prompt,
        flags=re.IGNORECASE,
    )
    if periodic_match:
        description = periodic_match.group(1).strip().rstrip(".")
        start = int(periodic_match.group(2))
        interval = int(periodic_match.group(3))
        for index in range(start, segment_count + 1, interval):
            _add_event(events, index, description)
    return events


def _compile_menu_day_contracts(
    prompt: str,
    segment_count: int,
) -> dict[int, list[str]]:
    events: dict[int, list[str]] = {}
    for match in re.finditer(
        r"Celebrate\s+'(.+?)'\s+on\s+([A-Za-z]+)\s+(\d{1,2})\s+"
        r"with a special dish:\s*(.+?)\.",
        prompt,
        flags=re.IGNORECASE,
    ):
        index = _day_for_month_day(match.group(2), int(match.group(3)))
        _add_event(
            events,
            index,
            f"{match.group(1)} featuring {match.group(4).strip()}",
        )

    range_match = re.search(
        r"Organize a festival,\s+'(.+?)',\s+from\s+([A-Za-z]+)\s+(\d{1,2})\s+"
        r"to\s+([A-Za-z]+)\s+(\d{1,2})\s+featuring dishes like:\s*(.+?)\.",
        prompt,
        flags=re.IGNORECASE,
    )
    if range_match:
        description = f"{range_match.group(1)} featuring {range_match.group(6).strip()}"
        start = _day_for_month_day(range_match.group(2), int(range_match.group(3)))
        end = _day_for_month_day(range_match.group(4), int(range_match.group(5)))
        for index in range(start, end + 1):
            _add_event(events, index, description)

    periodic_match = re.search(
        r"Schedule\s+'(.+?)'\s+every\s+(\d+)\s+days\s+starting from day\s+"
        r"([A-Za-z]+)\s+(\d{1,2}),\s+serving:\s*(.+?)\.",
        prompt,
        flags=re.IGNORECASE,
    )
    if periodic_match:
        description = (
            f"{periodic_match.group(1)} featuring {periodic_match.group(5).strip()}"
        )
        interval = int(periodic_match.group(2))
        start = _day_for_month_day(
            periodic_match.group(3),
            int(periodic_match.group(4)),
        )
        for index in range(start, segment_count + 1, interval):
            _add_event(events, index, description)
    return events


def _compile_block_contracts(prompt: str, segment_count: int) -> dict[int, list[str]]:
    events: dict[int, list[str]] = {}
    grid_match = re.search(
        r"with a\s+(\d+)x(\d+)\s+block grid",
        prompt,
        flags=re.IGNORECASE,
    )
    grid_width = int(grid_match.group(1)) if grid_match else 10
    grid_height = int(grid_match.group(2)) if grid_match else 10
    for match in re.finditer(
        r"Designate Block at\s+\((\d+),\s*(\d+)\)\s+for\s+(.+?)\s+use\.",
        prompt,
        flags=re.IGNORECASE,
    ):
        x, y = int(match.group(1)), int(match.group(2))
        _add_event(events, _block_index(x, y, grid_width), match.group(3).strip())

    range_match = re.search(
        r"Allocate\s+(?:an?\s+)?(.+?)\s+along the (?:row|column) from\s+"
        r"\((\d+),\s*(\d+)\)\s+to\s+\((\d+),\s*(\d+)\)",
        prompt,
        flags=re.IGNORECASE,
    )
    if range_match:
        description = range_match.group(1).strip()
        x1, y1, x2, y2 = map(int, range_match.groups()[1:])
        for x, y in _coordinate_path(x1, y1, x2, y2, step=1):
            _add_event(events, _block_index(x, y, grid_width), description)

    periodic_match = re.search(
        r"Include\s+(?:an?\s+)?(.+?)\s+starting from Block at\s+"
        r"\((\d+),\s*(\d+)\)\s+with an interval of every\s+(\d+)\s+blocks along the (row|column)",
        prompt,
        flags=re.IGNORECASE,
    )
    if periodic_match:
        description = periodic_match.group(1).strip()
        x, y = int(periodic_match.group(2)), int(periodic_match.group(3))
        interval = int(periodic_match.group(4))
        axis = periodic_match.group(5).lower()
        if axis == "column":
            for current_x in range(x, grid_width, interval):
                _add_event(
                    events,
                    _block_index(current_x, y, grid_width),
                    description,
                )
        else:
            for current_y in range(y, grid_height, interval):
                _add_event(
                    events,
                    _block_index(x, current_y, grid_width),
                    description,
                )
    return {
        index: descriptions
        for index, descriptions in events.items()
        if 1 <= index <= segment_count
    }


def _week_for_month_day(month_name: str, day: int) -> int:
    return ((_day_for_month_day(month_name, day) - 1) // 7) + 1


def _day_for_month_day(month_name: str, day: int) -> int:
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    month = months[month_name.strip().lower()]
    return (date(2018, month, day) - date(2018, 1, 1)).days + 1


def _week_for_numeric_date(month: int, day: int) -> int:
    return ((date(2018, month, day) - date(2018, 1, 1)).days // 7) + 1


def _block_index(x: int, y: int, width: int = 10) -> int:
    return y * width + x + 1


def _coordinate_path(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    step: int,
) -> Iterable[tuple[int, int]]:
    if y1 == y2:
        direction = 1 if x2 >= x1 else -1
        for x in range(x1, x2 + direction, direction * step):
            yield x, y1
    elif x1 == x2:
        direction = 1 if y2 >= y1 else -1
        for y in range(y1, y2 + direction, direction * step):
            yield x1, y


def _add_event(events: dict[int, list[str]], index: int, description: str) -> None:
    cleaned = re.sub(r"\s+", " ", description).strip(" .")
    if not cleaned:
        return
    values = events.setdefault(index, [])
    if cleaned not in values:
        values.append(cleaned)


def _load_harness_checkpoint(
    checkpoint_path: str | Path | None,
    task: LongGenTask,
) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    if checkpoint.get("task_id") != task_id(task):
        raise ValueError("harness checkpoint belongs to another task")
    checkpoint_version = checkpoint.get("pipeline_version")
    if checkpoint_version not in (None, HARNESS_PIPELINE_VERSION):
        raise ValueError(
            "harness checkpoint belongs to an incompatible pipeline version: "
            f"{checkpoint_version}"
        )
    return checkpoint


def _write_harness_checkpoint(
    checkpoint_path: str | Path | None,
    task: LongGenTask,
    plan_payload: dict[str, Any],
    document_summary: str,
    contracts: list[SegmentContract],
    applied_corrections: list[dict[str, Any]],
    compiled_overrides: list[dict[str, Any]],
    segments: dict[int, str],
    generation_batches: list[dict[str, Any]],
    review_records: list[dict[str, Any]],
    issues: dict[int, dict[str, Any]],
    repair_records: list[dict[str, Any]],
    *,
    stage: str,
) -> None:
    if checkpoint_path is None:
        return
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "task_id": task_id(task),
        "pipeline_version": HARNESS_PIPELINE_VERSION,
        "stage": stage,
        "plan_payload": plan_payload,
        "document_summary": document_summary,
        "contracts": [
            {
                "index": contract.index,
                "purpose": contract.purpose,
                "required_events": contract.required_events,
                "continuity": contract.continuity,
            }
            for contract in contracts
        ],
        "plan_corrections": applied_corrections,
        "compiled_contract_overrides": compiled_overrides,
        "segments": {str(index): text for index, text in sorted(segments.items())},
        "generation_batches": generation_batches,
        "review_results": review_records,
        "issues": {str(index): record for index, record in sorted(issues.items())},
        "repair_records": repair_records,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
