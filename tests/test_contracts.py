from learnbyai_research.contracts import check_contract_sequence, repair_improved, verify_chapter_contract
from learnbyai_research.pipelines.full import (
    MAX_REPAIR_CHAPTER_CHARACTERS,
    _accept_local_repair,
    _course_integrity_failures,
    _finalize_chapter_integrity,
    _program_hard_metric_issues,
    _repair_validation_passes_gate,
    _review_result_after_repair,
    _semantic_repair_improved,
)
from learnbyai_research.schemas import (
    ChapterOutput,
    ChapterPlan,
    CourseTask,
    RepairIssueResolution,
    RepairValidationResult,
    ReviewIssue,
)
from learnbyai_research.schemas import ChapterContract


def test_verify_chapter_contract_accepts_satisfied_chapter() -> None:
    contract = ChapterContract(
        chapter_title="Markov Decision Processes",
        required_topics=["state", "action"],
        prerequisite_concepts=["probability"],
        forbidden_early_topics=["Q-learning"],
        required_examples=["grid-world"],
        required_formulas=["V(s)"],
    )
    content = (
        "# Markov Decision Processes\n\n"
        "We briefly review probability, then define state and action. "
        "A grid-world example shows how choices affect outcomes. "
        "The value of a state is written as $V(s)$."
    )

    result = verify_chapter_contract(content, contract, title="Markov Decision Processes")

    assert result.satisfied
    assert result.violation_score == 0
    assert result.evidence.covered_topics == ["state", "action"]


def test_verify_chapter_contract_reports_actionable_violations() -> None:
    contract = ChapterContract(
        chapter_title="Markov Decision Processes",
        required_topics=["state", "action"],
        forbidden_early_topics=["Q-learning"],
        required_examples=["grid-world"],
        required_formulas=["V(s)"],
    )
    content = "# Markov Decision Processes\n\nQ-learning updates are explained in detail."

    result = verify_chapter_contract(content, contract, title="Markov Decision Processes")
    violation_types = {item.type for item in result.violations}

    assert not result.satisfied
    assert "missing_required_topic" in violation_types
    assert "missing_required_example" in violation_types
    assert "missing_required_formula" in violation_types
    assert "premature_topic" in violation_types


def test_forbidden_multiword_topic_does_not_match_a_single_shared_word() -> None:
    contract = ChapterContract(
        chapter_title="Policy Evaluation",
        required_topics=["policy evaluation"],
        forbidden_early_topics=["policy gradient", "offline RL"],
    )
    content = "# Policy Evaluation\n\nThis chapter explains policy evaluation for a fixed policy in a finite MDP."

    result = verify_chapter_contract(content, contract, title="Policy Evaluation")

    assert result.satisfied
    assert result.evidence.premature_topics_found == []


def test_forbidden_topic_detects_real_expansion_but_allows_a_forward_pointer() -> None:
    contract = ChapterContract(chapter_title="Foundations", forbidden_early_topics=["policy gradient"])

    expanded = verify_chapter_contract(
        "# Foundations\n\nPolicy gradient estimates a parameter update from sampled returns.",
        contract,
        title="Foundations",
    )
    forward = verify_chapter_contract(
        "# Foundations\n\nPolicy gradient will be covered later in the course.",
        contract,
        title="Foundations",
    )

    assert expanded.evidence.premature_topics_found == ["policy gradient"]
    assert forward.evidence.premature_topics_found == []


def test_repair_improved_accepts_lower_violation_score() -> None:
    contract = ChapterContract(chapter_title="Foundations", required_topics=["problem framing"])
    before = verify_chapter_contract("# Foundations\n\nA short introduction.", contract, title="Foundations")
    after = verify_chapter_contract(
        "# Foundations\n\nThis section explains problem framing with a concrete learner goal.",
        contract,
        title="Foundations",
    )

    assert repair_improved(before, after)


def test_local_repair_accepts_a_shorter_complete_chapter_when_contract_score_improves() -> None:
    task = CourseTask(
        id="repair-length",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=20,
        chapter_target_characters=30,
    )
    contract = ChapterContract(chapter_title="Foundations", required_topics=["problem framing"])
    before_text = "# Foundations\n\n" + "A short introduction. " * 8
    candidate_text = "# Foundations\n\nproblem framing"
    before = verify_chapter_contract(before_text, contract, title="Foundations")
    after = verify_chapter_contract(candidate_text, contract, title="Foundations")

    accepted, reason = _accept_local_repair(
        task,
        before_text,
        candidate_text,
        before,
        after,
        validation_passed=True,
    )

    assert accepted
    assert reason is None


def test_local_repair_rejects_a_candidate_below_the_task_minimum() -> None:
    task = CourseTask(
        id="repair-minimum",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=50,
        chapter_target_characters=60,
    )
    contract = ChapterContract(chapter_title="Foundations", required_topics=["problem framing"])
    before_text = "# Foundations\n\n" + "problem framing explanation. " * 4
    candidate_text = "# Foundations\n\nproblem framing"
    before = verify_chapter_contract(before_text, contract, title="Foundations")
    after = verify_chapter_contract(candidate_text, contract, title="Foundations")

    accepted, reason = _accept_local_repair(
        task,
        before_text,
        candidate_text,
        before,
        after,
        validation_passed=True,
    )

    assert not accepted
    assert reason == "candidate_below_chapter_minimum"


def test_local_repair_rejects_a_candidate_above_the_runaway_limit() -> None:
    task = CourseTask(
        id="repair-runaway",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=20,
        chapter_target_characters=30,
    )
    contract = ChapterContract(chapter_title="Foundations", required_topics=["problem framing"])
    before_text = "# Foundations\n\n" + "problem framing explanation. " * 4
    candidate_text = "# Foundations\n\nproblem framing\n" + "x" * MAX_REPAIR_CHAPTER_CHARACTERS
    before = verify_chapter_contract(before_text, contract, title="Foundations")
    after = verify_chapter_contract(candidate_text, contract, title="Foundations")

    accepted, reason = _accept_local_repair(task, before_text, candidate_text, before, after)

    assert not accepted
    assert reason == "candidate_above_runaway_limit"


def test_local_repair_accepts_reviewer_confirmed_semantic_improvement_without_contract_change() -> None:
    task = CourseTask(
        id="semantic-repair",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    contract = ChapterContract(
        chapter_title="Foundations",
        required_topics=["problem framing"],
        required_formulas=["zeta_formula"],
    )
    before_text = "# Foundations\n\nThis chapter explains problem framing in a concrete setting."
    candidate_text = "# Foundations\n\nThis chapter corrects the mathematical explanation of problem framing in a concrete setting."
    before = verify_chapter_contract(before_text, contract, title="Foundations")
    after = verify_chapter_contract(candidate_text, contract, title="Foundations")

    accepted, reason = _accept_local_repair(
        task,
        before_text,
        candidate_text,
        before,
        after,
        validation_passed=True,
    )

    assert before.violation_score == after.violation_score == 2
    assert accepted
    assert reason is None


def test_local_repair_leaves_deterministic_contract_changes_to_reviewer_two() -> None:
    task = CourseTask(
        id="new-high",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    contract = ChapterContract(
        chapter_title="Foundations",
        required_topics=["foundation"],
        forbidden_early_topics=["Q-learning"],
        assessment_targets=["goal one", "goal two", "goal three"],
    )
    before_text = "# Foundations\n\nThis foundation introduces the basic setting."
    candidate_text = "# Foundations\n\nThis foundation covers goal one, goal two, and goal three. Q-learning updates are derived in detail."
    before = verify_chapter_contract(before_text, contract, title="Foundations")
    after = verify_chapter_contract(candidate_text, contract, title="Foundations")

    accepted, reason = _accept_local_repair(
        task,
        before_text,
        candidate_text,
        before,
        after,
        validation_passed=True,
    )

    assert before.violation_score == after.violation_score == 3
    assert accepted
    assert reason is None


def test_local_repair_rejects_invalid_markdown_structure() -> None:
    task = CourseTask(
        id="invalid-markdown",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    contract = ChapterContract(chapter_title="Foundations")
    before_text = "# Foundations\n\nValid chapter."
    before = verify_chapter_contract(before_text, contract, title="Foundations")

    missing_heading = "Foundations\n\nLong prose without a Markdown heading."
    accepted, reason = _accept_local_repair(
        task,
        before_text,
        missing_heading,
        before,
        verify_chapter_contract(missing_heading, contract, title="Foundations"),
        validation_passed=True,
    )
    assert not accepted
    assert reason == "candidate_missing_markdown_heading"

    unclosed_fence = "# Foundations\n\n```python\nprint('unfinished')"
    accepted, reason = _accept_local_repair(
        task,
        before_text,
        unclosed_fence,
        before,
        verify_chapter_contract(unclosed_fence, contract, title="Foundations"),
        validation_passed=True,
    )
    assert not accepted
    assert reason == "candidate_unclosed_code_fence"


def test_local_repair_rejects_provider_truncation() -> None:
    task = CourseTask(
        id="truncated-repair",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    contract = ChapterContract(chapter_title="Foundations")
    before_text = "# Foundations\n\nA complete-looking chapter."
    candidate_text = before_text + " More text."
    before = verify_chapter_contract(before_text, contract, title="Foundations")
    after = verify_chapter_contract(candidate_text, contract, title="Foundations")

    accepted, reason = _accept_local_repair(
        task,
        before_text,
        candidate_text,
        before,
        after,
        validation_passed=True,
        completion_metadata={"finish_reason": "length"},
    )

    assert not accepted
    assert reason == "candidate_truncated"


def test_program_hard_metric_issues_do_not_include_contract_semantics() -> None:
    task = CourseTask(
        id="hard-metrics-only",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )

    assert _program_hard_metric_issues(task, "# Foundations\n\nReadable Markdown.") == []


def test_contract_improvement_cannot_bypass_failed_reviewer_validation() -> None:
    task = CourseTask(
        id="validation-required",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    contract = ChapterContract(chapter_title="Foundations", required_topics=["problem framing"])
    before_text = "# Foundations\n\nA short introduction."
    candidate_text = "# Foundations\n\nThis chapter now teaches problem framing."
    before = verify_chapter_contract(before_text, contract, title="Foundations")
    after = verify_chapter_contract(candidate_text, contract, title="Foundations")

    accepted, reason = _accept_local_repair(
        task,
        before_text,
        candidate_text,
        before,
        after,
        validation_passed=False,
    )

    assert repair_improved(before, after)
    assert not accepted
    assert reason == "repair_validation_failed"


def test_semantic_repair_requires_high_issues_to_be_resolved_and_no_new_high_issue() -> None:
    original = [
        ReviewIssue(id="R1", severity="high", category="math", message="Incorrect derivation."),
        ReviewIssue(id="R2", severity="medium", category="code", message="Broken update."),
    ]
    accepted_validation = RepairValidationResult(
        passed=True,
        resolutions=[
            RepairIssueResolution(issue_id="R1", status="resolved"),
            RepairIssueResolution(issue_id="R2", status="resolved"),
        ],
    )
    unresolved_high = RepairValidationResult(
        passed=True,
        resolutions=[
            RepairIssueResolution(issue_id="R1", status="partial"),
            RepairIssueResolution(issue_id="R2", status="resolved"),
        ],
    )
    new_high = RepairValidationResult(
        passed=True,
        resolutions=[
            RepairIssueResolution(issue_id="R1", status="resolved"),
            RepairIssueResolution(issue_id="R2", status="resolved"),
        ],
        new_issues=[ReviewIssue(severity="high", category="continuity", message="New contradiction.")],
    )

    assert _semantic_repair_improved(original, accepted_validation)
    assert not _semantic_repair_improved(original, unresolved_high)
    assert not _semantic_repair_improved(original, new_high)


def test_repair_gate_ignores_summary_flag_when_only_medium_warning_remains() -> None:
    original = [
        ReviewIssue(id="R1", severity="high", category="math", message="Incorrect derivation."),
        ReviewIssue(id="R2", severity="medium", category="rigor", message="Imprecise estimate."),
    ]
    validation = RepairValidationResult(
        passed=False,
        resolutions=[
            RepairIssueResolution(issue_id="R1", status="resolved"),
            RepairIssueResolution(issue_id="R2", status="partial"),
        ],
    )

    assert _repair_validation_passes_gate(original, validation)

    review = _review_result_after_repair(original, validation)
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        review=review,
    )
    _finalize_chapter_integrity(chapter)

    assert not review.passed
    assert [issue.id for issue in review.issues] == ["R2"]
    assert chapter.integrity_status == "complete_with_warnings"
    assert chapter.integrity_failures == []


def test_repair_gate_rejects_unresolved_original_high_and_new_high() -> None:
    original = [ReviewIssue(id="R1", severity="high", category="math", message="Incorrect derivation.")]
    unresolved = RepairValidationResult(
        passed=True,
        resolutions=[RepairIssueResolution(issue_id="R1", status="partial")],
    )
    introduced_high = RepairValidationResult(
        passed=True,
        resolutions=[RepairIssueResolution(issue_id="R1", status="resolved")],
        new_issues=[ReviewIssue(id="N1", severity="high", category="code", message="New bug.")],
    )

    assert not _repair_validation_passes_gate(original, unresolved)
    assert not _repair_validation_passes_gate(original, introduced_high)


def test_course_integrity_does_not_treat_medium_or_low_warnings_as_failures() -> None:
    task = CourseTask(
        id="warnings-allowed",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_count=1,
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    validation = RepairValidationResult(
        passed=False,
        resolutions=[RepairIssueResolution(issue_id="R1", status="partial")],
    )
    original = [ReviewIssue(id="R1", severity="medium", category="rigor", message="Imprecise estimate.")]
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        content_markdown="# Foundations\n\nContent.",
        review=_review_result_after_repair(original, validation),
        integrity_status="complete_with_warnings",
    )

    assert _course_integrity_failures(task, [chapter], require_review=True) == []


def test_verify_chapter_contract_checks_bridge_assessment_and_preparation() -> None:
    contract = ChapterContract(
        chapter_title="Value Functions",
        required_topics=["state value"],
        bridge_from_previous="review probability foundations",
        bridge_to_next="prepare Bellman equations",
        prepares_for=["Bellman equations"],
        assessment_targets=["explain state values"],
        summary_for_next="sets up value iteration",
    )
    content = (
        "# Value Functions\n\n"
        "Recall probability foundations from the previous chapter. "
        "A state value describes expected future return.\n\n"
        "## Exercises\n\n"
        "1. Explain state values in a small grid-world.\n\n"
        "## Summary\n\n"
        "This prepares Bellman equations in the next chapter and sets up value iteration."
    )

    result = verify_chapter_contract(content, contract, title="Value Functions")

    assert result.satisfied
    assert result.evidence.assessment_targets_found == ["explain state values"]
    assert result.evidence.prepares_for_found == ["Bellman equations"]
    assert sorted(result.evidence.bridge_evidence_found) == [
        "bridge_from_previous",
        "bridge_to_next",
        "summary_for_next",
    ]


def test_contract_sequence_checker_reports_missing_targets_and_cycles() -> None:
    contracts = [
        ChapterContract(chapter_title="Foundations", required_topics=["state"], introduced_concepts=["state"]),
        ChapterContract(chapter_title="Policies", required_topics=["policy"], prerequisite_concepts=["state"]),
    ]
    good = check_contract_sequence(
        contracts,
        target_topics=["state", "policy"],
        chapter_dependencies=[
            {"chapter_title": "Foundations", "depends_on": []},
            {"chapter_title": "Policies", "depends_on": ["Foundations"]},
        ],
    )

    assert good.satisfied
    assert good.target_coverage_rate == 1.0

    bad = check_contract_sequence(
        contracts,
        target_topics=["state", "policy", "Bellman equation"],
        chapter_dependencies=[
            {"chapter_title": "Foundations", "depends_on": ["Policies"]},
            {"chapter_title": "Policies", "depends_on": ["Foundations"]},
        ],
    )
    violation_types = {item.type for item in bad.violations}

    assert not bad.satisfied
    assert "missing_target_topic" in violation_types
    assert "dependency_cycle" in violation_types


def test_contract_sequence_checker_accepts_target_aliases() -> None:
    contracts = [
        ChapterContract(
            chapter_title="有限马尔科夫决策过程",
            required_topics=["有限马尔科夫决策过程", "状态价值函数", "策略评估"],
        )
    ]

    result = check_contract_sequence(
        contracts,
        target_topics=["finite MDP", "state-value function", "policy evaluation"],
        target_aliases={
            "finite MDP": ["有限马尔科夫决策过程", "有限 MDP"],
            "state-value function": ["状态价值函数", "v_pi"],
            "policy evaluation": ["策略评估"],
        },
    )

    assert result.satisfied
    assert result.target_coverage_rate == 1.0
