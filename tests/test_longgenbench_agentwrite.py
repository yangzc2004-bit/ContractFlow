from __future__ import annotations

from learnbyai_research.longgenbench import task_from_dataset_item
from learnbyai_research.longgenbench_agentwrite import (
    parse_agentwrite_plan,
    run_agentwrite,
)


def dataset_item() -> dict:
    return {
        "prompt": "Write two weekly entries, each containing at least 100 words.",
        "type": "Week",
        "number": 2,
        "prefix": "#*# Week 1:",
        "checks_once": {},
        "checks_range": {},
        "checks_periodic": {},
    }


class FakeClient:
    model = "fake"

    def __init__(self, response: str, prompts: list[str]) -> None:
        self.response = response
        self.prompts = prompts

    def complete(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.response

    def completion_history(self) -> list[dict]:
        return []


def test_parse_agentwrite_plan_requires_all_consecutive_steps() -> None:
    plan = "\n".join(
        [
            "Segment 1 - Main Point: Alpha - Word Count: 100 words",
            "Segment 2 - Main Point: Beta - Word Count: 100 words",
        ]
    )

    steps = parse_agentwrite_plan(plan, 2)

    assert [step.index for step in steps] == [1, 2]
    assert steps[1].main_point == "Beta"


def test_run_agentwrite_generates_sequential_checkpointed_segments(tmp_path) -> None:
    task = task_from_dataset_item(0, dataset_item())
    planning_prompts: list[str] = []
    writing_prompts: list[str] = []
    plan = "\n".join(
        [
            "Segment 1 - Main Point: Alpha - Word Count: 100 words",
            "Segment 2 - Main Point: Beta - Word Count: 100 words",
        ]
    )
    writing_responses = iter(["Alpha body.", "Beta body."])

    def planning_factory() -> FakeClient:
        return FakeClient(plan, planning_prompts)

    def writing_factory() -> FakeClient:
        return FakeClient(next(writing_responses), writing_prompts)

    output = run_agentwrite(
        task,
        planning_factory,
        writing_factory,
        checkpoint_dir=tmp_path,
    )

    assert output["final_text"].startswith("#*# Week 1:\nAlpha body.")
    assert "#*# Week 2:\nBeta body." in output["final_text"]
    assert "Alpha body." in writing_prompts[1]
    assert "*** finished ***" not in writing_prompts[1]
    assert (tmp_path / "plan.json").exists()
    assert (tmp_path / "segments" / "0002.json").exists()
