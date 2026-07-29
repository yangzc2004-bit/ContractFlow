import json

from learnbyai_research.eval.judge import PairwiseJudge
from learnbyai_research.prompts import JUDGE_DIMENSIONS, build_pairwise_judge_prompt
from learnbyai_research.schemas import CourseTask, Material


def _payload(label: str) -> str:
    return json.dumps(
        {
            "winner": label,
            "dimensions": {item: label for item in JUDGE_DIMENSIONS},
            "dimension_evidence": {
                item: {"A": f"A evidence for {item}", "B": f"B evidence for {item}"}
                for item in JUDGE_DIMENSIONS
            },
            "rationale": "Textbook A has the stronger overall evidence than Textbook B.",
        }
    )


class CapturingJudgeLLM:
    model = "test-judge"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return _payload("A")


class CorrectingJudgeLLM(CapturingJudgeLLM):
    def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        if len(self.calls) == 1:
            return json.dumps({"winner": "A", "dimensions": {"goal_alignment": "A"}})
        return _payload("A")

class ParseCorrectingJudgeLLM(CapturingJudgeLLM):
    def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return "not valid JSON" if len(self.calls) == 1 else _payload("A")


class RetryingJudgeLLM(CapturingJudgeLLM):
    def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return "not valid JSON" if len(self.calls) <= 2 else _payload("A")


def _task() -> CourseTask:
    return CourseTask(
        id="judge-task",
        topic="Reinforcement learning",
        goal="Teach the foundations.",
        background="Knows probability.",
        chapter_count=8,
        chapter_min_characters=8000,
        chapter_target_characters=10000,
        materials=[Material(name="requirements", role="requirements", text="Include Bellman equations.")],
    )


def test_pairwise_prompt_includes_materials_length_and_complete_schema() -> None:
    prompt = build_pairwise_judge_prompt(_task(), "# A", "# B")

    assert "CHAPTER_MIN_CHARACTERS: 8000" in prompt
    assert "CHAPTER_TARGET_CHARACTERS: 10000" in prompt
    assert "Include Bellman equations." in prompt
    assert '"nominal_minimum_total_characters": 64000' in prompt
    assert "Do not infer additional requirements from a conventional syllabus" in prompt
    assert "Do not award positive credit merely for an unrequested named theorem or topic" in prompt
    for dimension in JUDGE_DIMENSIONS:
        assert f'"{dimension}": "A|B|Tie"' in prompt


def test_judge_uses_json_mode_and_restores_labels_after_swapped_presentation() -> None:
    llm = CapturingJudgeLLM()
    result = PairwiseJudge(llm).judge(
        _task(),
        {"pipeline": "direct_prompt", "task": {"topic": "T", "goal": "G"}, "chapters": [{"content_markdown": "# Direct"}]},
        {"pipeline": "full", "task": {"topic": "T", "goal": "G"}, "chapters": [{"content_markdown": "# Full"}]},
        swap=True,
    )

    assert llm.calls[0]["response_format"] == "json_object"
    assert "# Full" in llm.calls[0]["prompt"].split("OUTPUT_A:", 1)[1].split("OUTPUT_B:", 1)[0]
    assert result.presentation_order == "B_then_A"
    assert result.winner == "B"
    assert set(result.dimensions) == set(JUDGE_DIMENSIONS)
    assert result.dimension_evidence["goal_alignment"]["A"] == "Textbook A evidence for goal_alignment"
    assert result.rationale == "Textbook B has the stronger overall evidence than Textbook A."
    assert result.completion_history == []
    assert result.elapsed_seconds is not None


def test_judge_repairs_missing_dimensions_once() -> None:
    llm = CorrectingJudgeLLM()
    result = PairwiseJudge(llm).judge(
        _task(),
        {"pipeline": "direct_prompt", "task": {"topic": "T", "goal": "G"}, "chapters": [{"content_markdown": "# Direct"}]},
        {"pipeline": "full", "task": {"topic": "T", "goal": "G"}, "chapters": [{"content_markdown": "# Full"}]},
        swap=False,
    )

    assert len(llm.calls) == 2
    assert "PAIRWISE_JUDGE_JSON_CORRECTION" in llm.calls[1]["prompt"]
    assert result.winner == "A"


def test_judge_repairs_unparseable_json_once() -> None:
    llm = ParseCorrectingJudgeLLM()

    result = PairwiseJudge(llm).judge(
        _task(),
        {"pipeline": "direct_prompt", "task": {"topic": "T", "goal": "G"}, "chapters": [{"content_markdown": "# Direct"}]},
        {"pipeline": "full", "task": {"topic": "T", "goal": "G"}, "chapters": [{"content_markdown": "# Full"}]},
        swap=False,
    )

    assert len(llm.calls) == 2
    assert "PAIRWISE_JUDGE_JSON_CORRECTION" in llm.calls[1]["prompt"]
    assert "not valid JSON" in llm.calls[1]["prompt"]
    assert result.winner == "A"


def test_judge_retries_after_an_unparseable_correction() -> None:
    llm = RetryingJudgeLLM()

    result = PairwiseJudge(llm).judge(
        _task(),
        {"pipeline": "direct_prompt", "task": {"topic": "T", "goal": "G"}, "chapters": [{"content_markdown": "# Direct"}]},
        {"pipeline": "full", "task": {"topic": "T", "goal": "G"}, "chapters": [{"content_markdown": "# Full"}]},
        swap=False,
    )

    assert len(llm.calls) == 3
    assert result.winner == "A"
