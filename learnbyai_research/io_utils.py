from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .schemas import CourseOutput, CourseTask, PairwiseJudgment, task_from_dict, to_plain


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(to_plain(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_task(path: str | Path) -> CourseTask:
    return task_from_dict(read_json(path))


def load_output(path: str | Path) -> dict[str, Any]:
    return read_json(path)


def append_jsonl(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_plain(value), ensure_ascii=False))
        handle.write("\n")


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = _json_candidate(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as first_error:
        repaired = _repair_common_json(candidate)
        if repaired != candidate:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as repaired_error:
                raise ValueError(_json_decode_message(repaired, repaired_error)) from repaired_error
        raise ValueError(_json_decode_message(candidate, first_error)) from first_error


def _json_candidate(text: str) -> str:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", candidate, re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    return candidate


def _repair_common_json(candidate: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", candidate)


def _json_decode_message(candidate: str, error: json.JSONDecodeError) -> str:
    context_radius = 180
    start = max(0, error.pos - context_radius)
    end = min(len(candidate), error.pos + context_radius)
    snippet = candidate[start:end]
    return (
        f"Failed to parse JSON at line {error.lineno}, column {error.colno}: {error.msg}\n"
        f"Context around error:\n{snippet}"
    )


def output_from_json(data: dict[str, Any]) -> dict[str, Any]:
    return data


def course_markdown(output: CourseOutput | dict[str, Any]) -> str:
    data = to_plain(output)
    lines: list[str] = []
    task = data.get("task", {})
    lines.append(f"# {task.get('topic', 'Textbook')}")
    lines.append("")
    lines.append(f"Goal: {task.get('goal', '')}")
    lines.append("")
    for chapter in data.get("chapters", []):
        content = chapter.get("content_markdown", "").strip()
        if content:
            lines.append(content)
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def export_chapter_markdown_files(output: dict[str, Any], output_dir: str | Path) -> list[Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, chapter in enumerate(output.get("chapters", []), start=1):
        content = str(chapter.get("content_markdown") or "").strip()
        if not content:
            continue
        title = chapter_title(chapter, index)
        path = target_dir / f"{index:02d}_{slugify_filename(title)}.md"
        path.write_text(content + "\n", encoding="utf-8")
        written.append(path)
    return written


def chapter_title(chapter: dict[str, Any], index: int) -> str:
    plan = chapter.get("plan")
    if isinstance(plan, dict) and plan.get("title"):
        return str(plan["title"])
    return f"chapter_{index}"


def slugify_filename(value: str) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value.strip(), flags=re.UNICODE)
    slug = re.sub(r"_+", "_", slug).strip("_.")
    return slug or "chapter"
