import json

from learnbyai_research.io_utils import write_json
from learnbyai_research.materials import route_materials
from learnbyai_research.pipelines.common import bible_from_payload, contract_from_payload, plan_from_payload
from learnbyai_research.pipelines.full import ContractGuidedPipeline
from learnbyai_research.schemas import ChapterOutput, CourseOutput, CourseTask, to_plain


class ResumeAuthor:
    model = "resume-author"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        if "WRITE_CHAPTER_MARKDOWN" in prompt:
            return "# Second\n\nA complete second chapter."
        raise AssertionError("resume should reuse the saved plan and first chapter")

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


class PassingReviewer:
    model = "resume-reviewer"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        return json.dumps({"passed": True, "issues": [], "summary": "No changes needed."})

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


def test_full_resume_reuses_written_chapter_and_plan(tmp_path) -> None:
    task = CourseTask(
        id="resume-task",
        topic="Resume topic",
        goal="Resume a checkpoint.",
        background="None.",
        chapter_count=2,
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    plan_payload = {
        "course_bible": {"target_learner": "tester"},
        "chapters": [
            {"title": "First", "description": "First chapter.", "purpose": "Introduce the topic.", "contract": {"chapter_title": "First"}},
            {"title": "Second", "description": "Second chapter.", "purpose": "Apply the topic.", "contract": {"chapter_title": "Second"}},
        ],
    }
    first_raw = plan_payload["chapters"][0]
    first_plan = plan_from_payload(first_raw)
    first_contract = contract_from_payload(first_raw["contract"], first_plan.title)
    bible = bible_from_payload(plan_payload)
    bible.chapter_contracts.append(first_contract)
    first_chapter = ChapterOutput(
        plan=first_plan,
        contract=first_contract,
        content_markdown="# First\n\nSaved chapter.",
        length_enrichment_history=[{"before_chars": 12, "after_chars": 20, "accepted": True}],
    )
    partial = CourseOutput(
        task=task,
        pipeline="full",
        course_bible=bible,
        chapters=[first_chapter],
        routed_materials=route_materials(task),
        metadata={"stage": "chapter_1_written"},
    )
    write_json(tmp_path / "plan.json", {"task_id": task.id, "pipeline": "full", "raw_chapter_count": 2, "plan": plan_payload})
    write_json(tmp_path / "latest.json", to_plain(partial))

    author = ResumeAuthor()
    reviewer = PassingReviewer()
    output = ContractGuidedPipeline(
        llm=author,
        reviewer_llm=reviewer,
        checkpoint_dir=str(tmp_path),
        resume_from_checkpoints=True,
    ).run(task)

    assert len(output.chapters) == 2
    assert output.chapters[0].content_markdown == "# First\n\nSaved chapter."
    assert output.chapters[0].length_enrichment_history == [{"before_chars": 12, "after_chars": 20, "accepted": True}]
    assert len(author.calls) == 1
    assert "CHAPTER_TITLE: Second" in author.calls[0]
    assert reviewer.calls == 2
    assert output.metadata["resumed_from_checkpoint"] is True
    assert output.metadata["resume_stage"] == "chapter_1_written"
