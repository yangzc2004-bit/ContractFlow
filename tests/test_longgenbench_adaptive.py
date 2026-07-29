from __future__ import annotations

import json
import re

import pytest

from learnbyai_research.longgenbench import task_from_dataset_item, word_count
from learnbyai_research.longgenbench_adaptive import (
    AdaptiveHarnessIncompleteError,
    LongGenBenchAdapter,
    compact_source_brief,
    run_harness_adaptive,
)


def dataset_item(number: int = 2) -> dict:
    return {
        "prompt": (
            f"Write {number} weekly entries. Each entry should be at least 20 words. "
            "Keep the entries concrete and coherent."
        ),
        "type": "Week",
        "number": number,
        "prefix": "#*# Week 1:",
        "checks_once": {},
        "checks_range": {},
        "checks_periodic": {},
    }


class ContinuationFakeLLM:
    model = "continuation-fake"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._metadata: dict = {}
        self._history: list[dict] = []

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
            "finish_reason": "stop",
            "elapsed_seconds": 0.1,
            "request_max_output_tokens": max_output_tokens,
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 40,
                "total_tokens": 70,
            },
        }
        if "LONGGEN_ADAPTIVE_EXTEND" in prompt:
            indices = self._indices(prompt)
            result = "\n".join(
                self._tag(index, " ".join(f"extra{word}" for word in range(12)))
                for index in indices
            )
        elif "LONGGEN_ADAPTIVE_REVIEW_JSON" in prompt:
            indices = sorted(set(int(value) for value in re.findall(r'"index":\s*(\d+)', prompt)))
            result = json.dumps(
                {
                    "results": [
                        {"index": index, "passed": True, "missing_events": [], "issues": []}
                        for index in indices
                    ]
                }
            )
        else:
            indices = self._indices(prompt)
            blocks = []
            for index in indices:
                count = 10 if index == 1 else 22
                blocks.append(
                    self._tag(index, " ".join(f"word{index}_{word}" for word in range(count)))
                )
            result = "\n".join(blocks)
        self._history.append(dict(self._metadata))
        return result

    def completion_metadata(self) -> dict:
        return dict(self._metadata)

    def completion_history(self) -> list[dict]:
        return [dict(item) for item in self._history]

    @staticmethod
    def _indices(prompt: str) -> list[int]:
        match = re.search(r"(?:indices|segment indices)\s+(\[[^\]]+\])", prompt)
        assert match
        return [int(value) for value in json.loads(match.group(1))]

    @staticmethod
    def _tag(index: int, text: str) -> str:
        return f"<<<SEGMENT {index}>>>\n{text}\n<<<END SEGMENT {index}>>>"


class StructuralRecoveryFakeLLM(ContinuationFakeLLM):
    def __init__(self, *, recover: bool) -> None:
        super().__init__()
        self.recover = recover
        self.generation_calls = 0

    def complete(self, prompt: str, **kwargs) -> str:
        if "LONGGEN_ADAPTIVE_GENERATE" not in prompt:
            return super().complete(prompt, **kwargs)
        self.calls.append(prompt)
        self.generation_calls += 1
        self._metadata = {
            "finish_reason": "stop",
            "elapsed_seconds": 0.1,
            "request_max_output_tokens": kwargs.get("max_output_tokens"),
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 40,
                "total_tokens": 70,
            },
        }
        indices = self._indices(prompt)
        if self.generation_calls == 1:
            indices = [index for index in indices if index != 1]
        elif not self.recover:
            indices = [index for index in indices if index != 1]
        words = " ".join(f"recovered{word}" for word in range(22))
        result = "\n".join(self._tag(index, words) for index in indices)
        self._history.append(dict(self._metadata))
        return result


class FailedSemanticRepairFakeLLM(ContinuationFakeLLM):
    def complete(self, prompt: str, **kwargs) -> str:
        if "LONGGEN_ADAPTIVE_REVIEW_JSON" in prompt:
            self.calls.append(prompt)
            self._metadata = {
                "finish_reason": "stop",
                "elapsed_seconds": 0.1,
                "request_max_output_tokens": kwargs.get("max_output_tokens"),
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 20,
                    "total_tokens": 50,
                },
            }
            indices = sorted(
                set(int(value) for value in re.findall(r'"index":\s*(\d+)', prompt))
            )
            result = json.dumps(
                {
                    "results": [
                        {
                            "index": index,
                            "passed": False,
                            "missing_events": ["Attend a workshop"],
                            "issues": ["required event missing"],
                        }
                        for index in indices
                    ]
                }
            )
            self._history.append(dict(self._metadata))
            return result
        if "LONGGEN_ADAPTIVE_REPAIR" in prompt:
            self.calls.append(prompt)
            self._metadata = {
                "finish_reason": "stop",
                "elapsed_seconds": 0.1,
                "request_max_output_tokens": kwargs.get("max_output_tokens"),
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 40,
                    "total_tokens": 70,
                },
            }
            indices = sorted(
                set(int(value) for value in re.findall(r'"index":\s*(\d+)', prompt))
            )
            words = " ".join(f"ordinary{word}" for word in range(22))
            result = "\n".join(self._tag(index, words) for index in indices)
            self._history.append(dict(self._metadata))
            return result
        return super().complete(prompt, **kwargs)


def test_compact_source_brief_is_type_agnostic() -> None:
    source = "Write a document. " + ("detail " * 300)
    brief = compact_source_brief(source, max_characters=80)
    assert len(brief) <= 81
    assert brief.startswith("Write a document.")


def test_longgen_adapter_estimates_only_missing_extension_words() -> None:
    task = task_from_dataset_item(0, dataset_item(number=1))
    adapter = LongGenBenchAdapter(task, "summary", [])
    from learnbyai_research.adaptive_generation import GenerationUnit

    unit = GenerationUnit("1", 1, minimum_length=20, target_length=24)
    full_estimate = adapter.estimate_output_tokens(unit, "generate", "")
    extension_estimate = adapter.estimate_output_tokens(
        unit,
        "extend",
        " ".join("word" for _ in range(18)),
    )
    assert extension_estimate < full_estimate


def test_adaptive_pipeline_extends_short_segments_and_writes_metrics(tmp_path) -> None:
    task = task_from_dataset_item(0, dataset_item())
    llm = ContinuationFakeLLM()
    metrics_path = tmp_path / "metrics.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"

    output = run_harness_adaptive(
        task,
        llm,
        initial_batch_size=2,
        maximum_batch_size=2,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        graph_cache_dir=tmp_path / "graphs",
    )

    segments = {item["index"]: item["text"] for item in output["segments"]}
    assert word_count(segments[1]) == 22
    assert word_count(segments[2]) == 22
    assert "word1_0" in segments[1]
    assert "extra0" in segments[1]
    assert output["metadata"]["final_structural_issues"] == {}
    records = output["metadata"]["generation"]["request_records"]
    assert [record["action"] for record in records] == ["generate", "extend"]
    assert output["metadata"]["generation"]["metrics"]["extension_requests"] == 1
    assert checkpoint_path.exists()
    metric_lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(metric_lines) == 2
    assert [json.loads(line)["action"] for line in metric_lines] == ["generate", "extend"]


def test_adaptive_pipeline_resumes_completed_generation_without_new_calls(tmp_path) -> None:
    task = task_from_dataset_item(0, dataset_item())
    checkpoint_path = tmp_path / "checkpoint.json"
    graph_cache_dir = tmp_path / "graphs"
    run_harness_adaptive(
        task,
        ContinuationFakeLLM(),
        initial_batch_size=2,
        maximum_batch_size=2,
        checkpoint_path=checkpoint_path,
        graph_cache_dir=graph_cache_dir,
    )

    resumed_llm = ContinuationFakeLLM()
    output = run_harness_adaptive(
        task,
        resumed_llm,
        initial_batch_size=2,
        maximum_batch_size=2,
        checkpoint_path=checkpoint_path,
        graph_cache_dir=graph_cache_dir,
    )

    assert resumed_llm.calls == []
    assert output["metadata"]["generation"]["resumed_from_checkpoint"] is True
    assert output["metadata"]["generation"]["dependency_cache_hit"] is True


def test_adaptive_pipeline_recovers_hard_missing_issue_before_review() -> None:
    task = task_from_dataset_item(0, dataset_item())
    output = run_harness_adaptive(
        task,
        StructuralRecoveryFakeLLM(recover=True),
        initial_batch_size=2,
        maximum_batch_size=2,
        max_generation_attempts=1,
    )

    assert output["status"] == "complete"
    assert output["metadata"]["final_structural_issues"] == {}
    assert output["metadata"]["structural_recovery_records"][0]["index"] == 1


def test_adaptive_pipeline_fails_closed_when_hard_issue_remains() -> None:
    task = task_from_dataset_item(0, dataset_item())
    with pytest.raises(AdaptiveHarnessIncompleteError) as caught:
        run_harness_adaptive(
            task,
            StructuralRecoveryFakeLLM(recover=False),
            initial_batch_size=2,
            maximum_batch_size=2,
            max_generation_attempts=1,
        )

    assert caught.value.output["status"] == "incomplete"
    assert 1 in caught.value.output["metadata"]["final_structural_issues"]


def test_semantic_repair_must_pass_post_repair_review() -> None:
    item = dataset_item(number=1)
    item["prompt"] = (
        "Write 1 weekly entry. Each entry should be at least 20 words.\n"
        "3) Attend a workshop every 1 weeks, starting from week 1."
    )
    task = task_from_dataset_item(0, item)
    with pytest.raises(AdaptiveHarnessIncompleteError) as caught:
        run_harness_adaptive(
            task,
            FailedSemanticRepairFakeLLM(),
            initial_batch_size=1,
            maximum_batch_size=1,
        )

    output = caught.value.output
    assert output["metadata"]["repair_records"][0]["accepted"] is False
    assert output["metadata"]["unresolved_semantic_issues"]
    assert [record["stage"] for record in output["metadata"]["review_results"]] == [
        "initial",
        "post_repair",
    ]
