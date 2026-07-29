from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(to_plain(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def append_jsonl(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_plain(value), ensure_ascii=False))
        handle.write("\n")


def to_plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain(item) for item in value]
    return value


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
