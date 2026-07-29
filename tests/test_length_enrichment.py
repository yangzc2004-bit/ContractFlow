import json

import pytest

from learnbyai_research.pipelines.full import ContractGuidedPipeline
from learnbyai_research.schemas import CourseTask, chapter_pre_review_target


class FullLengthAuthor:
    model = "full-length-author"

    def __init__(self, enriched_body: str) -> None:
        self.enriched_body = enriched_body
        self.calls: list[str] = []

    def complete(self, prompt: str, **_kwargs) -> str:
        self.calls.append(prompt)
        if "PLAN_WITH_CONTRACTS_JSON" in prompt:
            return json.dumps(
                {
                    "course_bible": {},
                    "chapters": [
                        {"title": "Foundations", "description": "", "purpose": "", "contract": {"chapter_title": "Foundations"}}
                    ],
                }
            )
        if "WRITE_CHAPTER_MARKDOWN" in prompt:
            return "# Foundations\n\n" + "a" * 40
        if "ENRICH_CHAPTER_FOR_INSTRUCTIONAL_DEPTH" in prompt:
            return "# Foundations\n\n" + self.enriched_body
        raise AssertionError("unexpected prompt")

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


def _task() -> CourseTask:
    return CourseTask(
        id="full-pre-review-target",
        topic="Topic",
        goal="Goal",
        background="Background",
        chapter_count=1,
        chapter_min_characters=40,
        chapter_target_characters=80,
        metadata={"pre_review_target_characters": 60},
    )


def test_pre_review_target_defaults_to_minimum_and_rejects_an_invalid_floor() -> None:
    task = _task()
    assert chapter_pre_review_target(CourseTask(id="default", topic="", goal="", background="", chapter_min_characters=40)) == 40
    assert chapter_pre_review_target(task) == 60
    task.metadata["pre_review_target_characters"] = 39
    with pytest.raises(ValueError, match="at least"):
        chapter_pre_review_target(task)


def test_full_enriches_once_before_review_when_draft_is_between_minimum_and_target() -> None:
    author = FullLengthAuthor("b" * 65)
    output = ContractGuidedPipeline(llm=author, use_review_repair=False).run(_task())

    chapter = output.chapters[0]
    assert sum("ENRICH_CHAPTER_FOR_INSTRUCTIONAL_DEPTH" in prompt for prompt in author.calls) == 1
    assert len(chapter.content_markdown) >= 60
    assert chapter.length_enrichment_history == [
        {
            "before_chars": 55,
            "after_chars": 80,
            "pre_review_target_characters": 60,
            "accepted": True,
            "completion": {"finish_reason": "stop"},
        }
    ]


def test_full_keeps_the_original_draft_when_enrichment_is_shorter() -> None:
    author = FullLengthAuthor("b" * 10)
    output = ContractGuidedPipeline(llm=author, use_review_repair=False).run(_task())

    chapter = output.chapters[0]
    assert len(chapter.content_markdown) == 55
    assert chapter.length_enrichment_history[0]["accepted"] is False
