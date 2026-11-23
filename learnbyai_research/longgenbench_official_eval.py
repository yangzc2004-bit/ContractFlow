from __future__ import annotations

import re
from typing import Any

from .llm import ChatModel, completion_history
from .longgenbench import task_from_dataset_item, task_id


OFFICIAL_EXAMPLES = [
    "Example 1: Context: The district's new residential area will feature two "
    "10-story apartment buildings, a shopping mall, and a park. The construction "
    "is set to begin at 9 AM, and the park will include a playground and jogging "
    "tracks. The plan also includes space for a small medical clinic, which will "
    "open from 10 AM to 6 PM. ### Instruction: Does this context include a medical "
    "clinic? Please answer with 'yes' or 'no' only. Answer: yes",
    "Example 2: Context: The menu for today's lunch at the office includes grilled "
    "turkey, mashed potatoes with gravy, roasted vegetables, and pumpkin pie for "
    "dessert. The meal will be served from 12 PM to 2 PM, and there will be a "
    "vegetarian option available. The meal is planned to accommodate 50 people, "
    "and the turkey will be served with cranberry sauce. ### Instruction: Does "
    "this context include mashed potatoes? Please answer with 'yes' or 'no' only. "
    "Answer: yes",
    "Example 3: Context: On April 15th, the weather was sunny with a high of 75\u00b0F. "
    "In the morning, I volunteered for a community cleanup from 9 AM to 12 PM. We "
    "collected trash and planted 20 new trees along the riverbank. After lunch, I "
    "helped organize the donation of clothes and food for a local shelter, where "
    "we served sandwiches and drinks. The day ended at 4 PM. ### Instruction: "
    "Does this context include long-distance running? Please answer with 'yes' or "
    "'no' only. Answer: no",
]


def official_output_blocks(final_text: str) -> list[str]:
    return final_text.split("#*#")


def official_word_count(final_text: str) -> int:
    return len(final_text.split())


def parse_blocks(output_blocks: list[str], segment_type: str) -> dict[int, str]:
    type_to_block: dict[int, str] = {}
    pattern = rf"{segment_type} (\d+)"
    for block in output_blocks:
        match = re.search(pattern, block)
        if match:
            identifier = int(match.group(1))
            if identifier not in type_to_block or type_to_block[identifier] is None:
                type_to_block[identifier] = block
    return type_to_block


def create_prompt(context: str, event_description: str) -> str:
    return (
        "\n".join(OFFICIAL_EXAMPLES)
        + " \n ### Refer to the examples above for how to answer. \nContext: "
        + context
        + "\n\n### Instruction: Now, for the following context, does it include the "
        + event_description
        + "? Please answer with 'yes' or 'no' only. Answer: "
    )


def classify_response(response: str) -> str:
    return "yes" if "yes" in response.strip().lower() else "no"


def completion_rate(type_to_block: dict[int, str], total_number: int) -> float:
    expected = set(range(1, total_number + 1))
    missing = expected - set(type_to_block)
    return (len(expected) - len(missing)) / len(expected) * 100 if expected else 0.0


def build_official_export_item(
    dataset_item: dict[str, Any],
    output: dict[str, Any] | None,
) -> dict[str, Any]:
    final_text = str(output.get("final_text", "")) if isinstance(output, dict) else ""
    return {
        "input": dataset_item["prompt"],
        "checks_once": dataset_item.get("checks_once", {}),
        "checks_range": dataset_item.get("checks_range", {}),
        "checks_periodic": dataset_item.get("checks_periodic", {}),
        "type": dataset_item["type"],
        "number": dataset_item["number"],
        "output_blocks": official_output_blocks(final_text),
        "word_count": official_word_count(final_text),
    }


def evaluate_official_output(
    dataset_item: dict[str, Any],
    output: dict[str, Any],
    llm: ChatModel,
    *,
    temperature: float = 0.0,
    max_output_tokens: int = 50,
) -> dict[str, Any]:
    dataset_index = int(output["task"]["dataset_index"])
    task = task_from_dataset_item(dataset_index, dataset_item)
    final_text = str(output.get("final_text", ""))
    blocks = official_output_blocks(final_text)
    type_to_block = parse_blocks(blocks, task.segment_type)
    cr = completion_rate(type_to_block, task.segment_count)

    results_by_category: dict[str, dict[str, str]] = {}
    total_checks_by_category: dict[str, int] = {}
    for category in ("once", "range", "periodic"):
        raw_checks = dataset_item.get(f"checks_{category}", {})
        if not isinstance(raw_checks, dict):
            raw_checks = {}
        total_checks_by_category[category] = len(raw_checks)
        category_results: dict[str, str] = {}
        for raw_index, event_description in raw_checks.items():
            identifier = int(raw_index)
            if identifier not in type_to_block:
                continue
            response = llm.complete(
                create_prompt(type_to_block[identifier], str(event_description)),
                system=None,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            category_results[str(identifier)] = classify_response(response)
        results_by_category[category] = category_results

    return {
        "task_id": output["task"].get("task_id") or task_id(task),
        "dataset_index": dataset_index,
        "type": task.segment_type,
        "number": task.segment_count,
        "system": output["system"],
        "model": output.get("metadata", {}).get("model"),
        "completion_rate": cr,
        "word_count": official_word_count(final_text),
        "output_block_count": len(blocks),
        "results_once": results_by_category["once"],
        "results_range": results_by_category["range"],
        "results_periodic": results_by_category["periodic"],
        "total_checks_once": total_checks_by_category["once"],
        "total_checks_range": total_checks_by_category["range"],
        "total_checks_periodic": total_checks_by_category["periodic"],
        "judge_completion_history": completion_history(llm),
        "status": "complete",
    }


def failed_official_output(
    dataset_index: int,
    dataset_item: dict[str, Any],
    system: str,
    error: dict[str, Any],
) -> dict[str, Any]:
    task = task_from_dataset_item(dataset_index, dataset_item)
    return {
        "task_id": task_id(task),
        "dataset_index": dataset_index,
        "type": task.segment_type,
        "number": task.segment_count,
        "system": system,
        "model": error.get("completion_history", [{}])[-1].get("model")
        if error.get("completion_history")
        else None,
        "completion_rate": 0.0,
        "word_count": 0,
        "output_block_count": 0,
        "results_once": {},
        "results_range": {},
        "results_periodic": {},
        "total_checks_once": len(dataset_item.get("checks_once", {}) or {}),
        "total_checks_range": len(dataset_item.get("checks_range", {}) or {}),
        "total_checks_periodic": len(dataset_item.get("checks_periodic", {}) or {}),
        "judge_completion_history": [],
        "status": "failed_technical",
        "generation_error_type": error.get("error_type"),
        "generation_error_message": error.get("error_message"),
    }


def _summarize_official_flat(
    records: list[dict[str, Any]],
    *,
    model_name: str,
) -> dict[str, Any]:
    categories = ("once", "range", "periodic")
    accuracies: dict[str, float] = {}
    strict_accuracies: dict[str, float] = {}
    check_counts: dict[str, int] = {}
    total_check_counts: dict[str, int] = {}
    yes_counts: dict[str, int] = {}
    for category in categories:
        values = [
            result
            for record in records
            for result in record.get(f"results_{category}", {}).values()
        ]
        check_counts[category] = len(values)
        yes_counts[category] = sum(value == "yes" for value in values)
        accuracies[category] = yes_counts[category] / len(values) if values else 0.0

        total_checks = sum(
            int(
                record.get(
                    f"total_checks_{category}",
                    len(record.get(f"results_{category}", {})),
                )
            )
            for record in records
        )
        total_check_counts[category] = total_checks
        strict_accuracies[category] = (
            yes_counts[category] / total_checks if total_checks else 0.0
        )

    average_accuracy = sum(accuracies.values()) / len(categories)
    strict_average_accuracy = sum(strict_accuracies.values()) / len(categories)
    total_evaluated = sum(check_counts.values())
    total_yes = sum(yes_counts.values())
    total_all_checks = sum(total_check_counts.values())

    qualified_words = [
        int(record.get("word_count", 0))
        for record in records
        if int(record.get("output_block_count", 0)) > 40
        and int(record.get("word_count", 0)) > 2000
    ]
    all_words = [int(record.get("word_count", 0)) for record in records]

    return {
        "Model": model_name,
        "task_count": len(records),
        "Completion Rate": (
            sum(float(record["completion_rate"]) for record in records) / len(records)
            if records
            else 0.0
        ),
        "Accuracy Once": accuracies["once"],
        "Accuracy Range": accuracies["range"],
        "Accuracy Periodic": accuracies["periodic"],
        "Average Accuracy": average_accuracy,
        "Strict Accuracy Once": strict_accuracies["once"],
        "Strict Accuracy Range": strict_accuracies["range"],
        "Strict Accuracy Periodic": strict_accuracies["periodic"],
        "Strict Average Accuracy": strict_average_accuracy,
        "Ratio Results/Checks": (
            total_evaluated / total_all_checks if total_all_checks else 0.0
        ),
        "Ratio Yes/Results": total_yes / total_evaluated if total_evaluated else 0.0,
        "Ratio Yes/Checks": total_yes / total_all_checks if total_all_checks else 0.0,
        "Average Word Count": (
            sum(qualified_words) / len(qualified_words) if qualified_words else None
        ),
        "Mean Word Count": sum(all_words) / len(all_words) if all_words else 0.0,
        "check_counts": check_counts,
        "total_check_counts": total_check_counts,
        "yes_counts": yes_counts,
    }


def summarize_official(
    records: list[dict[str, Any]],
    *,
    model_name: str,
) -> dict[str, Any]:
    summary = _summarize_official_flat(records, model_name=model_name)
    types = sorted({str(record["type"]) for record in records if record.get("type")})
    if types:
        summary["by_type"] = {
            segment_type: _summarize_official_flat(
                [record for record in records if record.get("type") == segment_type],
                model_name=model_name,
            )
            for segment_type in types
        }
    return summary
