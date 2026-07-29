from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from ..contracts.evidence import formula_present, item_present, normalize
from ..schemas import CourseTask


RUNAWAY_CHAPTER_CHARACTERS = 30_000

_NUMBERED_CHAPTER_HEADING = re.compile(
    r"^#{1,6}[ \t]+(?:第[ \t]*(?P<zh>[0-9一二三四五六七八九十百零〇两]+)[ \t]*章|chapter[ \t]+(?P<en>[0-9]+))(?=\s|[:：.、-]|$).*$",
    re.IGNORECASE | re.MULTILINE,
)


def evaluate_task_spec(task: CourseTask, output: dict[str, Any]) -> dict[str, Any]:
    """Evaluate visible output against frozen task metadata, not its contracts."""
    chapters = output.get("chapters", [])
    chapter_texts = [str(item.get("content_markdown", "")) for item in chapters if isinstance(item, dict)]
    if str(output.get("pipeline", "")) == "direct_prompt" and len(chapter_texts) == 1:
        extracted = split_numbered_chapters(chapter_texts[0], task.chapter_count)
        if extracted is not None:
            chapter_texts = extracted
    lengths = [len(text) for text in chapter_texts]
    full_text = normalize("\n".join(chapter_texts))
    metadata = task.metadata
    concepts = [str(item) for item in metadata.get("must_cover_concepts", [])]
    aliases = metadata.get("concept_aliases", {}) if isinstance(metadata.get("concept_aliases", {}), dict) else {}
    formulas = [str(item) for item in metadata.get("must_include_formulas", [])]
    examples = [str(item) for item in metadata.get("must_include_examples", [])]
    early_text = normalize("\n".join(chapter_texts[: max(1, task.chapter_count // 2)]))
    forbidden = [str(item) for item in metadata.get("forbidden_early_topics", [])]

    missing_concepts = [
        concept
        for concept in concepts
        if not any(item_present(candidate, full_text) for candidate in [concept, *as_strings(aliases.get(concept, []))])
    ]
    missing_formulas = [item for item in formulas if not formula_present(item, full_text)]
    missing_examples = [item for item in examples if not item_present(item, full_text)]
    forbidden_leakage = [item for item in forbidden if item_present(item, early_text)]
    markdown_failures = markdown_failures_for(chapter_texts)
    finish_reasons = author_finish_reasons(output)
    truncated = any(reason == "length" for reason in finish_reasons)
    under_minimum = [index + 1 for index, length in enumerate(lengths) if length < task.chapter_min_characters]
    runaway = [index + 1 for index, length in enumerate(lengths) if length > RUNAWAY_CHAPTER_CHARACTERS]
    chapter_count_ok = len(chapter_texts) == task.chapter_count

    integrity_status = "complete"
    if not chapter_count_ok or under_minimum or runaway or truncated or markdown_failures:
        integrity_status = "incomplete"
    if not chapter_texts:
        integrity_status = "failed"
    return {
        "task_id": task.id,
        "pipeline": str(output.get("pipeline", "unknown")),
        "chapter_count": len(chapter_texts),
        "expected_chapter_count": task.chapter_count,
        "min_chapter_characters": min(lengths) if lengths else 0,
        "max_chapter_characters": max(lengths) if lengths else 0,
        "under_minimum_chapters": under_minimum,
        "runaway_chapters": runaway,
        "finish_reasons": finish_reasons,
        "truncated": truncated,
        "markdown_failures": markdown_failures,
        "integrity_status": integrity_status,
        "concept_total": len(concepts),
        "concept_coverage_rate": coverage_rate(len(concepts) - len(missing_concepts), len(concepts)),
        "missing_concepts": missing_concepts,
        "formula_total": len(formulas),
        "formula_coverage_rate": coverage_rate(len(formulas) - len(missing_formulas), len(formulas)),
        "missing_formulas": missing_formulas,
        "example_total": len(examples),
        "example_coverage_rate": coverage_rate(len(examples) - len(missing_examples), len(examples)),
        "missing_examples": missing_examples,
        "forbidden_early_leakage": forbidden_leakage,
    }


def split_numbered_chapters(markdown: str, expected_count: int) -> list[str] | None:
    """Extract an exact numbered chapter sequence from one-shot book Markdown.

    Direct prompting intentionally returns the whole book in one model response.
    This helper only recognizes chapter headings already present in that response;
    it never invents missing chapters or rewrites their content.
    """
    if expected_count <= 0:
        return None
    first_heading_by_number: dict[int, re.Match[str]] = {}
    for match in _NUMBERED_CHAPTER_HEADING.finditer(markdown):
        number = int(match.group("en")) if match.group("en") else _parse_chinese_number(match.group("zh") or "")
        if number is not None and 1 <= number <= expected_count and number not in first_heading_by_number:
            first_heading_by_number[number] = match
    if set(first_heading_by_number) != set(range(1, expected_count + 1)):
        return None
    ordered = [first_heading_by_number[number] for number in range(1, expected_count + 1)]
    if any(current.start() >= following.start() for current, following in zip(ordered, ordered[1:])):
        return None
    sections = [
        markdown[match.start() : (ordered[index + 1].start() if index + 1 < len(ordered) else len(markdown))].strip()
        for index, match in enumerate(ordered)
    ]
    return sections if all(sections) else None


def _parse_chinese_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    normalized = value.replace("〇", "零").replace("两", "二")
    digits = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if normalized == "十":
        return 10
    if "十" in normalized:
        tens, ones = normalized.split("十", 1)
        if tens not in ("", *digits) or ones not in ("", *digits):
            return None
        return (digits[tens] if tens else 1) * 10 + (digits[ones] if ones else 0)
    if len(normalized) == 1 and normalized in digits:
        return digits[normalized]
    return None


def evaluate_task_spec_outputs(task_by_id: dict[str, CourseTask], outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output in outputs:
        raw_task = output.get("task", {})
        task_id = str(raw_task.get("id", "")) if isinstance(raw_task, dict) else ""
        task = task_by_id.get(task_id)
        if task is None:
            raise ValueError(f"no frozen task specification for output task_id={task_id!r}")
        rows.append(evaluate_task_spec(task, output))
    return rows


def write_task_spec_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id", "pipeline", "chapter_count", "expected_chapter_count", "min_chapter_characters",
        "max_chapter_characters", "under_minimum_chapters", "runaway_chapters", "finish_reasons",
        "truncated", "markdown_failures", "integrity_status", "concept_total", "concept_coverage_rate",
        "missing_concepts", "formula_total", "formula_coverage_rate", "missing_formulas", "example_total",
        "example_coverage_rate", "missing_examples", "forbidden_early_leakage",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = {key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value for key, value in row.items()}
            writer.writerow(encoded)


def as_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)] if value else []


def coverage_rate(found: int, total: int) -> float:
    return 1.0 if total == 0 else round(found / total, 4)


def markdown_failures_for(chapters: list[str]) -> list[str]:
    failures: list[str] = []
    for index, text in enumerate(chapters, start=1):
        stripped = text.strip()
        if not stripped.startswith("#"):
            failures.append(f"chapter_{index}:missing_heading")
        if stripped.count("```") % 2:
            failures.append(f"chapter_{index}:unclosed_code_fence")
    return failures


def author_finish_reasons(output: dict[str, Any]) -> list[str]:
    metadata = output.get("metadata", {})
    if not isinstance(metadata, dict):
        return []
    history = metadata.get("author_completion_history", [])
    if not isinstance(history, list):
        return []
    return [str(item.get("finish_reason")) for item in history if isinstance(item, dict) and item.get("finish_reason")]
