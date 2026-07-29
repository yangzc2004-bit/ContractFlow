from __future__ import annotations

import json
import re

from learnbyai_research.longgenbench import SegmentContract, task_from_dataset_item
from learnbyai_research.longgenbench_optimized import (
    build_optimized_generation_prompt,
    run_harness_optimized,
    target_word_range,
)


def dataset_item(number: int = 4) -> dict:
    return {
        "prompt": (
            f"Write {number} weekly entries. Each entry should be at least 20 words. "
            "Include a workshop every 2 weeks starting from week 2."
        ),
        "type": "Week",
        "number": number,
        "prefix": "#*# Week 1:",
        "checks_once": {},
        "checks_range": {},
        "checks_periodic": {},
    }


class AdaptiveFakeLLM:
    model = "adaptive-fake"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._metadata: dict = {}
        self._history: list[dict] = []
        self.failed_multi_batch = False

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        response_format: str | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        self.calls.append(prompt)
        self._metadata = {
            "request_max_output_tokens": max_output_tokens,
            "finish_reason": "stop",
            "elapsed_seconds": 0.0,
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        if "LONGGEN_REVIEW_SEGMENTS_OPTIMIZED_JSON" in prompt:
            indices = [
                int(value)
                for value in re.findall(r'"index":\s*(\d+)', prompt)
            ]
            result = json.dumps(
                {
                    "results": [
                        {"index": index, "passed": True, "missing_events": [], "issues": []}
                        for index in sorted(set(indices))
                    ]
                }
            )
        else:
            match = re.search(r"REQUESTED_INDICES_JSON:\s*(\[[^\]]*\])", prompt)
            indices = json.loads(match.group(1)) if match else [1]
            if len(indices) > 1 and not self.failed_multi_batch:
                indices = indices[:1]
                self.failed_multi_batch = True
                self._metadata["finish_reason"] = "length"
            words = " ".join(f"word{index}" for index in range(24))
            result = "\n".join(
                f"<<<SEGMENT {index}>>>\n{words}\n<<<END SEGMENT {index}>>>"
                for index in indices
            )
        self._history.append(dict(self._metadata))
        return result

    def completion_metadata(self) -> dict:
        return dict(self._metadata)

    def completion_history(self) -> list[dict]:
        return [dict(item) for item in self._history]


def test_target_word_range_adds_a_small_guard_band() -> None:
    task = task_from_dataset_item(0, dataset_item())
    assert target_word_range(task) == (20, 32)


def test_optimized_prompt_requests_only_real_batch_indices() -> None:
    task = task_from_dataset_item(0, dataset_item())
    prompt = build_optimized_generation_prompt(
        task,
        "summary",
        [SegmentContract(index=3), SegmentContract(index=4)],
        "",
    )
    assert "Write only the requested segment indices: [3, 4]" in prompt
    assert "<<<SEGMENT 3>>>" in prompt
    assert "USER_PROMPT:" not in prompt


def test_optimized_pipeline_splits_and_recovers_missing_segments() -> None:
    task = task_from_dataset_item(0, dataset_item())
    llm = AdaptiveFakeLLM()
    output = run_harness_optimized(task, llm, batch_size=4)
    assert len(output["segments"]) == 4
    assert output["metadata"]["final_structural_issues"] == {}
    assert any(
        record["stage"] == "generation_recovery"
        for record in output["metadata"]["generation_batches"]
    )
