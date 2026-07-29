from learnbyai_research.eval.task_spec import RUNAWAY_CHAPTER_CHARACTERS, evaluate_task_spec
from learnbyai_research.schemas import CourseTask


def task() -> CourseTask:
    return CourseTask(
        id="spec-task", topic="Topic", goal="Goal", background="Background", chapter_count=2,
        chapter_min_characters=10, chapter_target_characters=20,
        metadata={"must_cover_concepts": ["alpha"], "concept_aliases": {"alpha": ["阿尔法"]},
                  "must_include_formulas": ["x=y"], "must_include_examples": ["worked example"],
                  "forbidden_early_topics": ["advanced topic"]},
    )


def test_task_spec_metrics_uses_task_metadata_and_detects_integrity_failures() -> None:
    output = {"pipeline": "single_agent", "task": {"id": "spec-task"}, "metadata": {"author_completion_history": [{"finish_reason": "stop"}]}, "chapters": [
        {"content_markdown": "# One\n\nalpha x=y worked example advanced topic"},
        {"content_markdown": "# Two\n\nalpha x=y worked example"},
    ]}
    result = evaluate_task_spec(task(), output)
    assert result["integrity_status"] == "complete"
    assert result["concept_coverage_rate"] == 1.0
    assert result["formula_coverage_rate"] == 1.0
    assert result["example_coverage_rate"] == 1.0
    assert result["forbidden_early_leakage"] == ["advanced topic"]


def test_task_spec_metrics_marks_short_runaway_and_truncated_outputs_incomplete() -> None:
    output = {"pipeline": "full", "task": {"id": "spec-task"}, "metadata": {"author_completion_history": [{"finish_reason": "length"}]}, "chapters": [
        {"content_markdown": "#\nx"},
        {"content_markdown": "# Two\n" + "x" * (RUNAWAY_CHAPTER_CHARACTERS + 1)},
    ]}
    result = evaluate_task_spec(task(), output)
    assert result["integrity_status"] == "incomplete"
    assert result["under_minimum_chapters"] == [1]
    assert result["runaway_chapters"] == [2]
    assert result["truncated"]


def test_task_spec_metrics_extracts_numbered_chapters_from_one_shot_direct_output() -> None:
    output = {
        "pipeline": "direct_prompt",
        "task": {"id": "spec-task"},
        "metadata": {"author_completion_history": [{"finish_reason": "stop"}]},
        "chapters": [{
            "content_markdown": (
                "# 教材标题\n\n"
                "# 第一章 基础\n\nalpha x=y worked example\n\n"
                "## Chapter 2 Application\n\nalpha x=y worked example"
            )
        }],
    }

    result = evaluate_task_spec(task(), output)

    assert result["chapter_count"] == 2
    assert result["integrity_status"] == "complete"


def test_task_spec_metrics_does_not_invent_missing_direct_chapters() -> None:
    output = {
        "pipeline": "direct_prompt",
        "task": {"id": "spec-task"},
        "chapters": [{"content_markdown": "# 第一章 Only\n\nalpha x=y worked example"}],
    }

    result = evaluate_task_spec(task(), output)

    assert result["chapter_count"] == 1
    assert result["integrity_status"] == "incomplete"
