from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
import time
from typing import Any

from ..io_utils import course_markdown, extract_json_object
from ..llm import ChatModel, completion_history
from ..prompts import JUDGE_DIMENSIONS, build_pairwise_judge_prompt
from ..schemas import CourseTask, PairwiseJudgment


MAX_EVALUATION_ATTEMPTS = 3


@dataclass
class PairwiseJudge:
    llm: ChatModel

    def judge(
        self,
        task: CourseTask,
        output_a: dict[str, Any],
        output_b: dict[str, Any],
        *,
        swap: bool | None = None,
    ) -> PairwiseJudgment:
        started = time.perf_counter()
        history_start = len(completion_history(self.llm))
        presented_b_first = _presentation_should_swap(task, output_a, output_b) if swap is None else swap
        presented_a, presented_b = (output_b, output_a) if presented_b_first else (output_a, output_b)
        prompt = build_pairwise_judge_prompt(task, course_markdown(presented_a), course_markdown(presented_b))
        winner, dimensions, evidence, rationale = self._evaluate(prompt)
        if presented_b_first:
            winner = _swap_label(winner)
            dimensions = {name: _swap_label(value) for name, value in dimensions.items()}
            evidence = {
                name: {"A": _swap_textbook_labels(values["B"]), "B": _swap_textbook_labels(values["A"])}
                for name, values in evidence.items()
            }
            rationale = _swap_textbook_labels(rationale)
        return PairwiseJudgment(
            task_id=task.id,
            judge_model=self.llm.model,
            system_a=str(output_a.get("pipeline", "A")),
            system_b=str(output_b.get("pipeline", "B")),
            winner=winner,
            dimensions=dimensions,
            dimension_evidence=evidence,
            rationale=rationale,
            presentation_order="B_then_A" if presented_b_first else "A_then_B",
            completion_history=completion_history(self.llm)[history_start:],
            elapsed_seconds=round(time.perf_counter() - started, 4),
        )

    def _request_text(self, prompt: str) -> str:
        return self.llm.complete(
            prompt,
            system="You are a blind pairwise evaluator. Return valid JSON only.",
            temperature=0.0,
            response_format="json_object",
        )

    def _evaluate(self, prompt: str) -> tuple[str, dict[str, str], dict[str, dict[str, str]], str]:
        last_error: ValueError | None = None
        for _ in range(MAX_EVALUATION_ATTEMPTS):
            raw = self._request_text(prompt)
            try:
                return _validate_payload(extract_json_object(raw))
            except ValueError as error:
                try:
                    corrected = self._request_text(_build_correction_prompt(raw, str(error)))
                    return _validate_payload(extract_json_object(corrected))
                except ValueError as correction_error:
                    last_error = correction_error
        if last_error is not None:
            raise last_error
        raise RuntimeError("Pairwise judge completed no evaluation attempts.")


def _presentation_should_swap(task: CourseTask, output_a: dict[str, Any], output_b: dict[str, Any]) -> bool:
    identity = "\x1f".join(
        [
            task.id,
            str(output_a.get("pipeline", "A")),
            str(output_b.get("pipeline", "B")),
        ]
    )
    return bool(sha256(identity.encode("utf-8")).digest()[0] & 1)


def _validate_payload(payload: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, dict[str, str]], str]:
    winner = _clean_label(payload.get("winner"))
    dimensions = _clean_dimensions(payload.get("dimensions"))
    _validate_winner_consistency(winner, dimensions)
    evidence = _clean_evidence(payload.get("dimension_evidence"))
    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        raise ValueError("rationale must be a non-empty string")
    return winner, dimensions, evidence, rationale


def _clean_label(value: Any) -> str:
    if value not in {"A", "B", "Tie"}:
        raise ValueError("winner must be A, B, or Tie")
    return str(value)


def _clean_dimensions(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError("dimensions must be an object")
    if set(raw) != set(JUDGE_DIMENSIONS):
        raise ValueError("dimensions must contain exactly the configured evaluation dimensions")
    result: dict[str, str] = {}
    for key in JUDGE_DIMENSIONS:
        result[key] = _clean_label(raw[key])
    return result


def _clean_evidence(raw: Any) -> dict[str, dict[str, str]]:
    if not isinstance(raw, dict):
        raise ValueError("dimension_evidence must be an object")
    if set(raw) != set(JUDGE_DIMENSIONS):
        raise ValueError("dimension_evidence must cover every evaluation dimension")
    result: dict[str, dict[str, str]] = {}
    for key in JUDGE_DIMENSIONS:
        item = raw[key]
        if not isinstance(item, dict) or set(item) != {"A", "B"}:
            raise ValueError(f"dimension_evidence.{key} must contain exactly A and B")
        a_text = str(item["A"]).strip()
        b_text = str(item["B"]).strip()
        if not a_text or not b_text:
            raise ValueError(f"dimension_evidence.{key} must contain non-empty evidence for A and B")
        result[key] = {"A": a_text, "B": b_text}
    return result


def _validate_winner_consistency(winner: str, dimensions: dict[str, str]) -> None:
    if winner == "Tie":
        return
    other = _swap_label(winner)
    if sum(value == winner for value in dimensions.values()) < sum(value == other for value in dimensions.values()):
        raise ValueError("winner cannot lose more evaluation dimensions than the opposing output")


def _swap_label(value: str) -> str:
    return {"A": "B", "B": "A", "Tie": "Tie"}[value]


def _swap_textbook_labels(text: str) -> str:
    """Keep free-text evidence aligned with the restored A/B presentation order."""
    first = "[[TEXTBOOK_ONE]]"
    second = "[[TEXTBOOK_TWO]]"
    swapped = re.sub(r"\bTextbook A\b", first, text, flags=re.IGNORECASE)
    swapped = re.sub(r"\bTextbook B\b", second, swapped, flags=re.IGNORECASE)
    swapped = re.sub(r"\bA\b", first, swapped)
    swapped = re.sub(r"\bB\b", second, swapped)
    return swapped.replace(first, "Textbook B").replace(second, "Textbook A")


def _build_correction_prompt(payload: Any, error: str) -> str:
    return (
        "PAIRWISE_JUDGE_JSON_CORRECTION\n\n"
        "Repair the following evaluation JSON without changing its substantive judgment. "
        "Return JSON only. It must contain winner, rationale, every required dimension, and evidence for A and B in every dimension.\n\n"
        f"REQUIRED_DIMENSIONS: {JUDGE_DIMENSIONS}\n"
        f"VALIDATION_ERROR: {error}\n"
        f"INVALID_JSON_OBJECT:\n{payload}"
    )
