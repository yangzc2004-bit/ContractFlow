from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .io_utils import extract_json_object
from .llm import ChatModel, completion_history
from .longgenbench import LongGenTask, output_segments, task_from_dataset_item, word_count


@dataclass
class EvaluationCheck:
    check_id: str
    category: str
    index: int
    description: str
    context: str


def build_checks(item: dict[str, Any], output: dict[str, Any]) -> list[EvaluationCheck]:
    segments = output_segments(output)
    checks: list[EvaluationCheck] = []
    for category, field in (
        ("once", "checks_once"),
        ("range", "checks_range"),
        ("periodic", "checks_periodic"),
    ):
        raw_checks = item.get(field, {})
        if not isinstance(raw_checks, dict):
            continue
        for raw_index, description in raw_checks.items():
            index = int(raw_index)
            checks.append(
                EvaluationCheck(
                    check_id=f"{category}:{index}",
                    category=category,
                    index=index,
                    description=str(description),
                    context=segments.get(index, ""),
                )
            )
    return checks


def build_judge_prompt(checks: list[EvaluationCheck]) -> str:
    items = [
        {
            "id": check.check_id,
            "required_content": check.description,
            "context": check.context,
        }
        for check in checks
    ]
    return f"""LONGGEN_OFFICIAL_STYLE_CHECK_JSON

For each item, determine whether CONTEXT clearly includes the REQUIRED_CONTENT.
Accept a clear paraphrase. Do not infer an event that is not stated.

Return valid JSON only:
{{
  "results": [
    {{"id": "once:1", "present": true}}
  ]
}}

ITEMS:
{json.dumps(items, ensure_ascii=False)}
"""


def evaluate_output(
    dataset_item: dict[str, Any],
    output: dict[str, Any],
    llm: ChatModel,
    *,
    batch_size: int = 20,
    max_output_tokens: int = 4_096,
) -> dict[str, Any]:
    task = task_from_dataset_item(int(output["task"]["dataset_index"]), dataset_item)
    segments = output_segments(output)
    checks = build_checks(dataset_item, output)
    judged: dict[str, bool] = {}
    for start in range(0, len(checks), batch_size):
        batch = checks[start : start + batch_size]
        raw = llm.complete(
            build_judge_prompt(batch),
            system="You are a precise binary evaluator. Return valid JSON only.",
            temperature=0.0,
            response_format="json_object",
            max_output_tokens=max_output_tokens,
        )
        payload = extract_json_object(raw)
        for result in payload.get("results", []):
            if not isinstance(result, dict):
                continue
            check_id = str(result.get("id", ""))
            if check_id:
                judged[check_id] = bool(result.get("present"))

    category_values: dict[str, list[bool]] = {"once": [], "range": [], "periodic": []}
    check_records = []
    for check in checks:
        present = judged.get(check.check_id, False)
        category_values[check.category].append(present)
        check_records.append(
            {
                "id": check.check_id,
                "category": check.category,
                "index": check.index,
                "description": check.description,
                "present": present,
                "context_words": word_count(check.context),
            }
        )

    category_accuracy = {
        category: _mean(values)
        for category, values in category_values.items()
    }
    completion_rate = len(segments) / task.segment_count if task.segment_count else 0.0
    length_pass_rate = (
        sum(word_count(text) >= task.minimum_words for text in segments.values()) / task.segment_count
        if task.segment_count
        else 0.0
    )
    return {
        "task_id": output["task"]["task_id"],
        "dataset_index": task.dataset_index,
        "type": task.segment_type,
        "system": output["system"],
        "model": output.get("metadata", {}).get("model"),
        "segment_count": task.segment_count,
        "returned_segments": len(segments),
        "completion_rate": completion_rate,
        "length_pass_rate": length_pass_rate,
        "accuracy_once": category_accuracy["once"],
        "accuracy_range": category_accuracy["range"],
        "accuracy_periodic": category_accuracy["periodic"],
        "average_accuracy": _mean(list(category_accuracy.values())),
        "checks": check_records,
        "judge_completion_history": completion_history(llm),
        "status": "complete",
    }


def failed_evaluation(
    dataset_index: int,
    dataset_item: dict[str, Any],
    system: str,
    error: dict[str, Any],
) -> dict[str, Any]:
    task = task_from_dataset_item(dataset_index, dataset_item)
    checks = []
    for category, field in (
        ("once", "checks_once"),
        ("range", "checks_range"),
        ("periodic", "checks_periodic"),
    ):
        for raw_index, description in dataset_item.get(field, {}).items():
            checks.append(
                {
                    "id": f"{category}:{raw_index}",
                    "category": category,
                    "index": int(raw_index),
                    "description": str(description),
                    "present": False,
                    "context_words": 0,
                }
            )
    return {
        "task_id": f"{dataset_index:04d}_{task.segment_type.lower().replace(' ', '_')}",
        "dataset_index": dataset_index,
        "type": task.segment_type,
        "system": system,
        "model": error.get("completion_history", [{}])[-1].get("model")
        if error.get("completion_history")
        else None,
        "segment_count": task.segment_count,
        "returned_segments": 0,
        "completion_rate": 0.0,
        "length_pass_rate": 0.0,
        "accuracy_once": 0.0,
        "accuracy_range": 0.0,
        "accuracy_periodic": 0.0,
        "average_accuracy": 0.0,
        "checks": checks,
        "judge_completion_history": [],
        "status": "failed_technical",
        "generation_error_type": error.get("error_type"),
        "generation_error_message": error.get("error_message"),
    }


def summarize_evaluations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems = sorted({str(row["system"]) for row in rows})
    summary: dict[str, Any] = {"systems": {}}
    metric_names = (
        "completion_rate",
        "length_pass_rate",
        "accuracy_once",
        "accuracy_range",
        "accuracy_periodic",
        "average_accuracy",
    )
    for system in systems:
        system_rows = [row for row in rows if row["system"] == system]
        summary["systems"][system] = {
            "task_count": len(system_rows),
            **{
                metric: _mean([float(row[metric]) for row in system_rows])
                for metric in metric_names
            },
            "by_type": {
                segment_type: {
                    "task_count": len(type_rows),
                    **{
                        metric: _mean([float(row[metric]) for row in type_rows])
                        for metric in metric_names
                    },
                }
                for segment_type in sorted({str(row["type"]) for row in system_rows})
                if (type_rows := [row for row in system_rows if row["type"] == segment_type])
            },
        }
    return summary


def _mean(values: list[float | bool]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0
