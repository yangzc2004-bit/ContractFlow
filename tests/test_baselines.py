import json
from pathlib import Path

from learnbyai_research.llm import MockLLM
from learnbyai_research.pipelines.baselines import DirectPromptPipeline, SingleAgentPipeline
from learnbyai_research.schemas import CourseTask


class SingleAuthorLLM:
    model = "test-author"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        if "SINGLE_AGENT_PLAN_JSON" in prompt:
            return json.dumps(
                {
                    "course_overview": "A two-chapter course.",
                    "chapters": [
                        {"title": "Foundations", "description": "First concepts.", "purpose": "Build vocabulary."},
                        {"title": "Application", "description": "Use the concepts.", "purpose": "Practice."},
                    ],
                }
            )
        if "SINGLE_AGENT_WRITE_CHAPTER_MARKDOWN" in prompt:
            return "# Chapter\n\nA complete chapter draft."
        if "SINGLE_AGENT_REWRITE_SHORT_CHAPTER" in prompt:
            return "# Chapter\n\n" + "An expanded chapter draft. " * 4
        raise AssertionError("unexpected prompt")

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


def test_single_agent_stages_outline_then_writes_each_chapter() -> None:
    llm = SingleAuthorLLM()
    task = CourseTask(
        id="single",
        topic="Topic",
        goal="Goal",
        background="Background",
        chapter_count=2,
        chapter_min_characters=0,
        chapter_target_characters=0,
    )

    output = SingleAgentPipeline(llm).run(task)

    assert len(llm.calls) == 3
    assert "SINGLE_AGENT_PLAN_JSON" in llm.calls[0]["prompt"]
    assert all("SINGLE_AGENT_WRITE_CHAPTER_MARKDOWN" in call["prompt"] for call in llm.calls[1:])
    assert all(call["system"] == "You are a single textbook author. Return Markdown only." for call in llm.calls[1:])
    assert len(output.chapters) == 2
    assert output.metadata["generation_mode"] == "staged_single_author"
    assert output.metadata["chapter_min_characters"] == 0
    assert output.metadata["chapter_target_characters"] == 0
    assert output.metadata["author_completion_history"] == []


def test_single_agent_rewrites_a_short_chapter_once() -> None:
    llm = SingleAuthorLLM()
    task = CourseTask(
        id="single-rewrite",
        topic="Topic",
        goal="Goal",
        background="Background",
        chapter_count=2,
        chapter_min_characters=40,
        chapter_target_characters=50,
    )

    output = SingleAgentPipeline(llm).run(task)

    assert len(llm.calls) == 5
    assert sum("SINGLE_AGENT_REWRITE_SHORT_CHAPTER" in call["prompt"] for call in llm.calls) == 2
    assert all(len(chapter.content_markdown) >= 40 for chapter in output.chapters)
    assert len(output.metadata["length_rewrites"]) == 2
    assert output.metadata["author_completion_history"] == []


class PreReviewThresholdAuthor(SingleAuthorLLM):
    def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        if "SINGLE_AGENT_PLAN_JSON" in prompt:
            return json.dumps(
                {
                    "course_overview": "One chapter.",
                    "chapters": [{"title": "Foundations", "description": "", "purpose": ""}],
                }
            )
        if "SINGLE_AGENT_WRITE_CHAPTER_MARKDOWN" in prompt:
            return "# Foundations\n\n" + "a" * 40
        if "SINGLE_AGENT_REWRITE_SHORT_CHAPTER" in prompt:
            return "# Foundations\n\n" + "b" * 65
        raise AssertionError("unexpected prompt")


def test_single_agent_enriches_between_minimum_and_pre_review_target() -> None:
    llm = PreReviewThresholdAuthor()
    task = CourseTask(
        id="single-pre-review-target",
        topic="Topic",
        goal="Goal",
        background="Background",
        chapter_count=1,
        chapter_min_characters=40,
        chapter_target_characters=80,
        metadata={"pre_review_target_characters": 60},
    )

    output = SingleAgentPipeline(llm).run(task)

    assert sum("SINGLE_AGENT_REWRITE_SHORT_CHAPTER" in call["prompt"] for call in llm.calls) == 1
    assert len(output.chapters[0].content_markdown) >= 60
    assert output.chapters[0].length_enrichment_history[0]["accepted"] is True
    assert output.chapters[0].length_enrichment_history[0]["meets_pre_review_target"] is True


def test_direct_prompt_persists_mock_completion_history() -> None:
    task = CourseTask(
        id="direct",
        topic="Topic",
        goal="Goal",
        background="Background",
        chapter_count=2,
        chapter_min_characters=0,
        chapter_target_characters=0,
    )

    output = DirectPromptPipeline(MockLLM()).run(task)

    history = output.metadata["author_completion_history"]
    assert len(history) == 1
    assert history[0]["model"] == "mock-llm"
    assert history[0]["success"] is True


def test_single_agent_resumes_from_a_saved_chapter(tmp_path: Path) -> None:
    task = CourseTask(
        id="single-resume",
        topic="Topic",
        goal="Goal",
        background="Background",
        chapter_count=2,
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    first_llm = SingleAuthorLLM()
    first = SingleAgentPipeline(first_llm, checkpoint_dir=str(tmp_path)).run(task)
    assert len(first.chapters) == 2

    resumed_llm = SingleAuthorLLM()
    resumed = SingleAgentPipeline(resumed_llm, checkpoint_dir=str(tmp_path), resume_from_checkpoints=True).run(task)

    assert len(resumed_llm.calls) == 0
    assert len(resumed.chapters) == 2
    assert resumed.metadata["resumed_from_checkpoint"] is True
