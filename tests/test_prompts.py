from learnbyai_research.prompts import (
    build_chapter_writer_prompt,
    build_contract_plan_prompt,
    build_direct_prompt,
    build_repair_validation_prompt,
    build_review_prompt,
)
from learnbyai_research.schemas import ChapterContract, ChapterPlan, CourseBible, CourseTask, RoutedMaterials


def test_direct_prompt_includes_chapter_count_and_length_budget() -> None:
    task = CourseTask(
        id="budgeted",
        topic="Test topic",
        goal="Teach the topic.",
        background="Learner background.",
        chapter_count=8,
        chapter_min_characters=8000,
        chapter_target_characters=10000,
    )

    prompt = build_direct_prompt(task, RoutedMaterials())

    assert "Produce exactly 8 chapters" in prompt
    assert "at least 8000 characters" in prompt
    assert "about 10000 characters" in prompt
    assert "each with a distinct" not in prompt
    assert "Do not merge multiple planned chapters" not in prompt


def test_repair_validation_prompt_requires_issue_resolution_and_new_issue_reporting() -> None:
    task = CourseTask(id="repair-validation", topic="Topic", goal="Goal", background="Background")
    prompt = build_repair_validation_prompt(
        task,
        CourseBible(),
        ChapterPlan(title="Foundations", description="", purpose=""),
        ChapterContract(chapter_title="Foundations"),
        RoutedMaterials(reference="Source material."),
        [{"id": "R1", "severity": "medium", "category": "math", "message": "Fix it.", "suggestion": "Correct it."}],
        "# Foundations\n\nIncorrect content.",
        "# Foundations\n\nCorrected content.",
    )

    assert "VALIDATE_REPAIR_JSON" in prompt
    assert '"issue_id": "R1"' in prompt
    assert "resolved|partial|unresolved" in prompt
    assert "new_issues" in prompt
    assert "ORIGINAL_CHAPTER_MARKDOWN" in prompt
    assert "Source material." in prompt
    assert '"character_count_scope": "complete_markdown_string"' in prompt
    assert '"length_passed": false' in prompt
    assert "program-owned hard metrics" in prompt


def test_full_prompts_enforce_task_traceable_scope_control() -> None:
    task = CourseTask(id="scope", topic="Core topic", goal="Teach only the core.", background="Background")
    plan = ChapterPlan(title="Core", description="", purpose="")
    contract = ChapterContract(chapter_title="Core", required_topics=["core definition"])
    materials = RoutedMaterials(requirements="Cover the core definition.")

    plan_prompt = build_contract_plan_prompt(task, materials)
    writer_prompt = build_chapter_writer_prompt(task, CourseBible(), plan, contract, materials, 0, 1)
    review_prompt = build_review_prompt(task, CourseBible(), plan, contract, materials, "# Core\n\nText")
    validation_prompt = build_repair_validation_prompt(
        task,
        CourseBible(),
        plan,
        contract,
        materials,
        [{"id": "R1", "severity": "medium", "category": "scope_creep", "message": "Extra topic."}],
        "# Core\n\nOriginal",
        "# Core\n\nCandidate",
    )

    assert "3 to 6 core required topics" in plan_prompt
    assert "compact optional subsection" in plan_prompt
    assert "Do not add uncontracted major topics" in writer_prompt
    assert "scope_creep" in review_prompt
    assert "proof whose gap fails to establish its stated theorem as high severity" in review_prompt
    assert "scope_creep new issue" in validation_prompt
    assert "proof with a quantifier error" in validation_prompt
