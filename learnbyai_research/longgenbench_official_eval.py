from __future__ import annotations

import re
from typing import Any


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


def summarize_official(
    records: list[dict[str, Any]],
    *,
    model_name: str,
) -> dict[str, Any]:
    categories = ("once", "range", "periodic")
    accuracies: dict[str, float] = {}
    check_counts: dict[str, int] = {}
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

    average_accuracy = sum(accuracies.values()) / len(categories)
    return {
        "Model": model_name,
        "Completion Rate": (
            sum(float(record["completion_rate"]) for record in records) / len(records)
            if records
            else 0.0
        ),
        "Accuracy Once": accuracies["once"],
        "Accuracy Range": accuracies["range"],
        "Accuracy Periodic": accuracies["periodic"],
        "Average Accuracy": average_accuracy,
        "check_counts": check_counts,
        "yes_counts": yes_counts,
    }
