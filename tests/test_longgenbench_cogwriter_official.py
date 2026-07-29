from __future__ import annotations

import asyncio
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from learnbyai_research.longgenbench_cogwriter_official import (
    OfficialCogWriterCallAdapter,
    bound_refinement_loops,
    classify_official_prompt,
    configure_utf8_stdio,
    load_official_cogwriter,
    official_method_hashes,
    official_plan_entry_count,
    official_plan_key,
)

UPSTREAM_COGWRITER = Path("third_party/CogWriter")


def test_classify_official_prompt_only_uses_planning_budget_for_plans() -> None:
    assert (
        classify_official_prompt("Return your analysis and plan in ONLY this exact json format:")
        == "planning"
    )
    assert (
        classify_official_prompt(
            "Return your analysis and the final revised plan in ONLY this exact JSON format:"
        )
        == "planning"
    )
    assert classify_official_prompt("Write a 200-word weekly diary entry.") == "segment"
    assert classify_official_prompt("You are an expert editor.") == "segment"


def test_configure_utf8_stdio_is_safe_to_call_repeatedly() -> None:
    configure_utf8_stdio()
    configure_utf8_stdio()
    assert sys.stdout.encoding.lower().replace("-", "") == "utf8"


@pytest.mark.skipif(
    not UPSTREAM_COGWRITER.exists(),
    reason="optional upstream CogWriter checkout is not bundled",
)
def test_load_official_cogwriter_injects_llm_without_editing_upstream() -> None:
    async def fake_call(model: str, prompt: str) -> str:
        return "{}"

    cogwriter = load_official_cogwriter(UPSTREAM_COGWRITER, fake_call)

    cogwriter_module = inspect.getmodule(cogwriter.async_generate)
    assert cogwriter_module is not None
    from CogWriter_model.Agents import GenerationAgent, PlanningAgent

    assert PlanningAgent.async_call_llm is fake_call
    assert GenerationAgent.async_call_llm is fake_call
    assert asyncio.iscoroutinefunction(cogwriter.async_generate)
    assert len(official_method_hashes(UPSTREAM_COGWRITER)) == 4


@pytest.mark.skipif(
    not UPSTREAM_COGWRITER.exists(),
    reason="optional upstream CogWriter checkout is not bundled",
)
def test_loaded_cogwriter_class_comes_from_upstream_checkout() -> None:
    async def fake_call(model: str, prompt: str) -> str:
        return "{}"

    cogwriter = load_official_cogwriter(UPSTREAM_COGWRITER, fake_call)
    loaded_path = inspect.getfile(cogwriter).replace("\\", "/")
    assert loaded_path.endswith("third_party/CogWriter/CogWriter_model/CogWriter.py")


@pytest.mark.skipif(
    not UPSTREAM_COGWRITER.exists(),
    reason="optional upstream CogWriter checkout is not bundled",
)
def test_bounded_recovery_caps_all_four_length_refinement_loops() -> None:
    upstream = (
        UPSTREAM_COGWRITER
        / "CogWriter_model"
        / "Agents"
        / "GenerationAgent.py"
    ).read_text(encoding="utf-8")

    transformed = bound_refinement_loops(upstream, max_refinements=3)

    assert transformed.count("for _bounded_refinement_attempt_") == 4
    assert "range(3)" in transformed
    assert "while word_diff >" not in transformed


def test_official_plan_cardinality_uses_the_adapted_task_type() -> None:
    assert official_plan_key("Day") == "weekly_plan"
    assert official_plan_key("Menu Day") == "weekly_plan"
    assert official_plan_entry_count(
        {"block_plan": [{"block_id": "Block 1"}, {"block_id": "Block 2"}]},
        "Block",
    ) == 2
    assert official_plan_entry_count({"floor_plan": "not-a-list"}, "Floor") == 0


def test_call_adapter_preserves_upstream_empty_string_failure_semantics() -> None:
    @dataclass
    class FailingClient:
        model: str = "fake"

        def complete(self, prompt: str, **kwargs) -> str:
            raise RuntimeError("provider failed")

        def completion_history(self) -> list[dict]:
            return [{"success": False, "usage": {}}]

    history: list[dict] = []
    adapter = OfficialCogWriterCallAdapter(
        client_factory=FailingClient,
        planning_max_output_tokens=16_384,
        segment_max_output_tokens=2_048,
        temperature=0.2,
        history=history,
    )

    assert asyncio.run(adapter.complete("deepseek-v4-flash", "Write a segment.")) == ""
    assert history[0]["success"] is False
    assert history[0]["official_cogwriter_phase"] == "segment"
