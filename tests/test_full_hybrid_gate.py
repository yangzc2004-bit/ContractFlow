import json

import pytest

from learnbyai_research.io_utils import read_json
from learnbyai_research.pipelines.full import (
    ChapterIntegrityError,
    ContractGuidedPipeline,
    _normalize_contract_transitions,
)
from learnbyai_research.schemas import ChapterContract, ChapterOutput, ChapterPlan, CourseBible, CourseTask, RoutedMaterials


class AuthorRepairLLM:
    model = "author-repair-test"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, prompt: str, **_kwargs) -> str:
        self.calls.append(prompt)
        if "REPAIR_CHAPTER_MARKDOWN" in prompt:
            return "# Foundations\n\nThis chapter gives a corrected mathematical derivation for the foundation."
        if "RESPOND_TO_REPAIR_DISPUTE_JSON" in prompt:
            return json.dumps(
                {
                    "responses": [
                        {"issue_id": "N1", "position": "accept", "evidence": "The allegation is valid."}
                    ],
                    "summary": "The author accepts the new issue.",
                }
            )
        raise AssertionError("author should only receive repair prompts")

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


class ReviewerLLM:
    model = "reviewer-test"

    def __init__(self, *, validation_passed: bool = True) -> None:
        self.calls: list[str] = []
        self.validation_passed = validation_passed

    def complete(self, prompt: str, **_kwargs) -> str:
        self.calls.append(prompt)
        if "REVIEW_CHAPTER_JSON" in prompt:
            return json.dumps(
                {
                    "passed": True,
                    "issues": [
                        {
                            "severity": "medium",
                            "category": "math",
                            "message": "The derivation is inaccurate.",
                            "suggestion": "Correct the derivation.",
                        }
                    ],
                    "summary": "Needs a mathematical correction.",
                }
            )
        if "VALIDATE_REPAIR_JSON" in prompt:
            return json.dumps(
                {
                    "passed": self.validation_passed,
                    "resolutions": [
                        {
                            "issue_id": "R1",
                            "status": "resolved" if self.validation_passed else "unresolved",
                            "evidence": "The derivation is corrected." if self.validation_passed else "The issue remains.",
                        }
                    ],
                    "new_issues": [],
                    "summary": "The mathematical issue is resolved." if self.validation_passed else "The issue remains.",
                }
            )
        raise AssertionError("reviewer should only receive review or validation prompts")

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


class MediumCleanupAuthor:
    model = "medium-cleanup-author"

    def __init__(self, *, fail_second_round: bool = False) -> None:
        self.calls: list[str] = []
        self.fail_second_round = fail_second_round

    def complete(self, prompt: str, **_kwargs) -> str:
        self.calls.append(prompt)
        if len(self.calls) == 1:
            return "# Foundations\n\nPartially corrected derivation retained as a safe fallback."
        if self.fail_second_round:
            raise RuntimeError("temporary cleanup failure")
        return "# Foundations\n\nFully corrected derivation after optional medium cleanup."

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


class MediumCleanupReviewer:
    model = "medium-cleanup-reviewer"

    def __init__(self) -> None:
        self.validation_calls = 0

    def complete(self, prompt: str, **_kwargs) -> str:
        if "REVIEW_CHAPTER_JSON" in prompt:
            return json.dumps(
                {
                    "passed": False,
                    "issues": [
                        {
                            "severity": "medium",
                            "category": "rigor",
                            "message": "The derivation needs a more precise quantifier argument.",
                            "suggestion": "Correct the quantifier order.",
                        }
                    ],
                    "summary": "A medium rigor issue remains.",
                }
            )
        if "VALIDATE_REPAIR_JSON" in prompt:
            self.validation_calls += 1
            resolved = self.validation_calls == 2
            return json.dumps(
                {
                    "passed": resolved,
                    "resolutions": [
                        {
                            "issue_id": "R1",
                            "status": "resolved" if resolved else "partial",
                            "evidence": "Fully corrected." if resolved else "Improved but still imprecise.",
                        }
                    ],
                    "new_issues": [],
                    "summary": "Clean." if resolved else "Continue medium cleanup.",
                }
            )
        raise AssertionError("unexpected reviewer prompt")

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


class EventuallyValidJsonLLM:
    model = "eventually-valid-json"

    def __init__(self) -> None:
        self.responses = iter(["not json", "still not json", '{"passed": true}'])
        self.calls = 0

    def complete(self, _prompt: str, **_kwargs) -> str:
        self.calls += 1
        return next(self.responses)

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


def test_complete_json_allows_two_format_only_repairs() -> None:
    llm = EventuallyValidJsonLLM()
    pipeline = ContractGuidedPipeline(llm=llm)

    payload = pipeline._complete_json(llm, "return JSON", system="return JSON", temperature=0.0)

    assert payload == {"passed": True}
    assert llm.calls == 3


class TwoRoundAuthor:
    model = "two-round-author"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, prompt: str, **_kwargs) -> str:
        self.calls.append(prompt)
        if len(self.calls) == 1:
            return "# Foundations\n\nCandidate one still has the error."
        return "# Foundations\n\nCandidate two contains the verified correction."

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


class TwoRoundReviewer:
    model = "two-round-reviewer"

    def __init__(self) -> None:
        self.validation_calls = 0

    def complete(self, prompt: str, **_kwargs) -> str:
        if "REVIEW_CHAPTER_JSON" in prompt:
            return json.dumps(
                {
                    "passed": False,
                    "issues": [
                        {
                            "severity": "high",
                            "category": "factual_accuracy",
                            "message": "The worked example is wrong.",
                            "suggestion": "Recompute it.",
                        }
                    ],
                    "summary": "A factual repair is required.",
                }
            )
        if "VALIDATE_REPAIR_JSON" in prompt:
            self.validation_calls += 1
            passed = self.validation_calls == 2
            return json.dumps(
                {
                    "passed": passed,
                    "resolutions": [
                        {
                            "issue_id": "R1",
                            "status": "resolved" if passed else "unresolved",
                            "evidence": "Verified." if passed else "Still incorrect.",
                        }
                    ],
                    "new_issues": [],
                    "summary": "Resolved." if passed else "Repair again.",
                }
            )
        raise AssertionError("unexpected reviewer prompt")

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


class NewHighReviewer(TwoRoundReviewer):
    model = "new-high-reviewer"

    def complete(self, prompt: str, **_kwargs) -> str:
        if "REVIEW_CHAPTER_JSON" in prompt:
            return super().complete(prompt, **_kwargs)
        if "VALIDATE_REPAIR_JSON" in prompt:
            return json.dumps(
                {
                    "passed": True,
                    "resolutions": [
                        {"issue_id": "R1", "status": "resolved", "evidence": "Original issue fixed."}
                    ],
                    "new_issues": [
                        {
                            "severity": "high",
                            "category": "factual_accuracy",
                            "message": "The repair introduced a new false claim.",
                            "suggestion": "Remove the false claim.",
                        }
                    ],
                    "summary": "Original issue fixed, but a new high issue was introduced.",
                }
            )
        raise AssertionError("unexpected reviewer prompt")


def test_hybrid_gate_repairs_a_medium_issue_even_when_reviewer_marks_the_draft_passed() -> None:
    author = AuthorRepairLLM()
    reviewer = ReviewerLLM()
    pipeline = ContractGuidedPipeline(llm=author, reviewer_llm=reviewer, max_repair_rounds=1)
    task = CourseTask(
        id="hybrid-gate",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    contract = ChapterContract(chapter_title="Foundations")
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        contract=contract,
        content_markdown="# Foundations\n\nThis chapter explains the foundation.",
    )

    pipeline._review_and_repair(task, CourseBible(), RoutedMaterials(), chapter, set())

    assert "corrected mathematical derivation" in chapter.content_markdown
    assert chapter.review is not None and chapter.review.passed
    assert chapter.repair_history[0]["accepted"]
    assert chapter.repair_history[0]["rejection_reason"] is None
    assert chapter.repair_history[0]["acceptance_basis"] == "reviewer_validation_and_contract_improvement"
    assert chapter.repair_history[0]["repair_model"] == author.model
    assert chapter.repair_history[0]["review_model"] == reviewer.model
    assert sum("REPAIR_CHAPTER_MARKDOWN" in prompt for prompt in author.calls) == 1
    assert sum("REVIEW_CHAPTER_JSON" in prompt for prompt in reviewer.calls) == 1
    assert sum("VALIDATE_REPAIR_JSON" in prompt for prompt in reviewer.calls) == 1


def test_hybrid_gate_routes_repairs_to_dedicated_repair_model() -> None:
    author = AuthorRepairLLM()
    author.model = "draft-author"
    repairer = AuthorRepairLLM()
    repairer.model = "dedicated-repair"
    reviewer = ReviewerLLM()
    pipeline = ContractGuidedPipeline(
        llm=author,
        reviewer_llm=reviewer,
        repair_llm=repairer,
        max_repair_rounds=1,
    )
    task = CourseTask(
        id="dedicated-repair",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        contract=ChapterContract(chapter_title="Foundations"),
        content_markdown="# Foundations\n\nThis chapter explains the foundation.",
    )

    pipeline._review_and_repair(task, CourseBible(), RoutedMaterials(), chapter, set())

    assert author.calls == []
    assert sum("REPAIR_CHAPTER_MARKDOWN" in prompt for prompt in repairer.calls) == 1
    assert chapter.repair_history[0]["repair_model"] == "dedicated-repair"


def test_unresolved_medium_validation_is_accepted_with_a_warning() -> None:
    author = AuthorRepairLLM()
    reviewer = ReviewerLLM(validation_passed=False)
    pipeline = ContractGuidedPipeline(llm=author, reviewer_llm=reviewer, max_repair_rounds=1)
    task = CourseTask(
        id="failed-validation",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        contract=ChapterContract(chapter_title="Foundations"),
        content_markdown="# Foundations\n\nAn incomplete draft.",
    )

    pipeline._review_and_repair(task, CourseBible(), RoutedMaterials(), chapter, set())

    assert "corrected mathematical derivation" in chapter.content_markdown
    assert chapter.review is not None and not chapter.review.passed
    assert [issue.id for issue in chapter.review.issues] == ["R1"]
    assert chapter.repair_history[0]["accepted"]
    assert chapter.repair_history[0]["rejection_reason"] is None


def test_residual_medium_issue_uses_remaining_round_before_final_acceptance() -> None:
    author = MediumCleanupAuthor()
    reviewer = MediumCleanupReviewer()
    pipeline = ContractGuidedPipeline(
        llm=author,
        reviewer_llm=reviewer,
        repair_llm=author,
        max_repair_rounds=2,
    )
    task = CourseTask(
        id="medium-cleanup",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        contract=ChapterContract(chapter_title="Foundations"),
        content_markdown="# Foundations\n\nAn imprecise draft.",
    )

    pipeline._review_and_repair(task, CourseBible(), RoutedMaterials(), chapter, set())

    assert "Fully corrected derivation" in chapter.content_markdown
    assert chapter.review is not None and chapter.review.passed
    assert [item["accepted"] for item in chapter.repair_history] == [True, True]
    assert [item["cleanup_continued"] for item in chapter.repair_history] == [True, False]
    assert "cleanup_reason" in author.calls[1]


def test_optional_medium_cleanup_failure_retains_verified_fallback() -> None:
    author = MediumCleanupAuthor(fail_second_round=True)
    reviewer = MediumCleanupReviewer()
    pipeline = ContractGuidedPipeline(
        llm=author,
        reviewer_llm=reviewer,
        repair_llm=author,
        max_repair_rounds=2,
    )
    task = CourseTask(
        id="medium-cleanup-fallback",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        contract=ChapterContract(chapter_title="Foundations"),
        content_markdown="# Foundations\n\nAn imprecise draft.",
    )

    pipeline._review_and_repair(task, CourseBible(), RoutedMaterials(), chapter, set())

    assert "safe fallback" in chapter.content_markdown
    assert chapter.review is not None and not chapter.review.passed
    assert chapter.repair_history[0]["cleanup_continued"] is True
    assert chapter.repair_history[1]["error"]["fallback_retained"] is True


def test_reviewer_two_feedback_is_sent_to_the_author_for_a_second_repair_round() -> None:
    author = TwoRoundAuthor()
    reviewer = TwoRoundReviewer()
    pipeline = ContractGuidedPipeline(llm=author, reviewer_llm=reviewer, max_repair_rounds=2)
    task = CourseTask(
        id="two-round-repair",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        contract=ChapterContract(chapter_title="Foundations"),
        content_markdown="# Foundations\n\nOriginal incorrect chapter.",
    )

    pipeline._review_and_repair(task, CourseBible(), RoutedMaterials(), chapter, set())

    assert "verified correction" in chapter.content_markdown
    assert [item["accepted"] for item in chapter.repair_history] == [False, True]
    assert "previous_validation" in author.calls[1]
    assert "Candidate one still has the error" in author.calls[1]


def test_new_high_issue_blocks_acceptance_even_when_reviewer_two_sets_passed_true() -> None:
    author = AuthorRepairLLM()
    reviewer = NewHighReviewer()
    pipeline = ContractGuidedPipeline(llm=author, reviewer_llm=reviewer, max_repair_rounds=1)
    task = CourseTask(
        id="new-high-block",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    original = "# Foundations\n\nOriginal incorrect chapter."
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        contract=ChapterContract(chapter_title="Foundations"),
        content_markdown=original,
    )

    pipeline._review_and_repair(task, CourseBible(), RoutedMaterials(), chapter, set())

    assert chapter.content_markdown == original
    assert not chapter.repair_history[0]["accepted"]
    assert chapter.repair_history[0]["rejection_reason"] == "repair_validation_failed"
    assert chapter.repair_history[0]["needs_adjudication"] is False


class RebuttingAuthor(AuthorRepairLLM):
    def complete(self, prompt: str, **_kwargs) -> str:
        if "RESPOND_TO_REPAIR_DISPUTE_JSON" in prompt:
            self.calls.append(prompt)
            return json.dumps(
                {
                    "responses": [
                        {"issue_id": "N1", "position": "rebut", "evidence": "The displayed equality is exact."}
                    ],
                    "summary": "The alleged mathematical error is false.",
                }
            )
        return super().complete(prompt, **_kwargs)


class DisputeReviewer(NewHighReviewer):
    def __init__(self, *, withdraw: bool) -> None:
        super().__init__()
        self.withdraw = withdraw

    def complete(self, prompt: str, **kwargs) -> str:
        if "RECONSIDER_REPAIR_VALIDATION_JSON" in prompt:
            new_issues = [] if self.withdraw else [
                {
                    "id": "N1",
                    "severity": "high",
                    "category": "factual_accuracy",
                    "message": "The repair introduced a new false claim.",
                    "suggestion": "Remove the false claim.",
                }
            ]
            return json.dumps(
                {
                    "passed": self.withdraw,
                    "resolutions": [
                        {"issue_id": "R1", "status": "resolved", "evidence": "Original issue fixed."}
                    ],
                    "new_issues": new_issues,
                    "summary": "Withdrawn after rebuttal." if self.withdraw else "The allegation remains.",
                }
            )
        return super().complete(prompt, **kwargs)


def test_rebutted_new_high_is_reconsidered_and_can_be_withdrawn(tmp_path) -> None:
    author = RebuttingAuthor()
    reviewer = DisputeReviewer(withdraw=True)
    pipeline = ContractGuidedPipeline(
        llm=author,
        reviewer_llm=reviewer,
        max_repair_rounds=1,
        checkpoint_dir=str(tmp_path),
    )
    task = CourseTask(
        id="dispute-withdrawn",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        contract=ChapterContract(chapter_title="Foundations"),
        content_markdown="# Foundations\n\nOriginal incorrect chapter.",
    )

    pipeline._review_and_repair(
        task,
        CourseBible(),
        RoutedMaterials(),
        chapter,
        set(),
        chapter_number=1,
    )

    assert chapter.repair_history[0]["accepted"] is True
    assert chapter.repair_history[0]["needs_adjudication"] is False
    assert (tmp_path / "chapter_1_repair_round1.json").exists()
    assert (tmp_path / "chapter_1_validation_round1.json").exists()
    assert (tmp_path / "chapter_1_dispute_author_response_round1.json").exists()
    assert (tmp_path / "chapter_1_dispute_validation_round1.json").exists()
    assert "candidate_markdown" in read_json(tmp_path / "chapter_1_repair_round1.json")


def test_confirmed_disputed_high_marks_needs_adjudication() -> None:
    author = RebuttingAuthor()
    reviewer = DisputeReviewer(withdraw=False)
    pipeline = ContractGuidedPipeline(llm=author, reviewer_llm=reviewer, max_repair_rounds=1)
    task = CourseTask(
        id="dispute-confirmed",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    original = "# Foundations\n\nOriginal incorrect chapter."
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        contract=ChapterContract(chapter_title="Foundations"),
        content_markdown=original,
    )

    pipeline._review_and_repair(task, CourseBible(), RoutedMaterials(), chapter, set())

    assert chapter.content_markdown == original
    assert chapter.integrity_status == "needs_adjudication"
    assert chapter.repair_history[0]["rejection_reason"] == "needs_adjudication"


class SchemaRetryReviewer(ReviewerLLM):
    def __init__(self) -> None:
        super().__init__()
        self.validation_attempts = 0

    def complete(self, prompt: str, **kwargs) -> str:
        if "VALIDATE_REPAIR_JSON" in prompt:
            self.calls.append(prompt)
            self.validation_attempts += 1
            return json.dumps({"passed": True, "resolutions": [], "new_issues": [], "summary": "Malformed mapping."})
        if "RETRY_REPAIR_VALIDATION_SCHEMA_JSON" in prompt:
            self.calls.append(prompt)
            return json.dumps(
                {
                    "passed": True,
                    "resolutions": [
                        {"issue_id": "R1", "status": "resolved", "evidence": "Corrected after schema retry."}
                    ],
                    "new_issues": [],
                    "summary": "Valid mapping.",
                }
            )
        return super().complete(prompt, **kwargs)


def test_invalid_reviewer_issue_mapping_is_retried_once(tmp_path) -> None:
    author = AuthorRepairLLM()
    reviewer = SchemaRetryReviewer()
    pipeline = ContractGuidedPipeline(
        llm=author,
        reviewer_llm=reviewer,
        max_repair_rounds=1,
        checkpoint_dir=str(tmp_path),
    )
    task = CourseTask(
        id="schema-retry",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        contract=ChapterContract(chapter_title="Foundations"),
        content_markdown="# Foundations\n\nOriginal chapter.",
    )

    pipeline._review_and_repair(
        task,
        CourseBible(),
        RoutedMaterials(),
        chapter,
        set(),
        chapter_number=1,
    )

    assert chapter.repair_history[0]["accepted"] is True
    assert (tmp_path / "chapter_1_validation_round1_attempt1.json").exists()
    assert (tmp_path / "chapter_1_validation_round1_attempt2.json").exists()
    assert read_json(tmp_path / "chapter_1_validation_round1_attempt1.json")["schema_valid"] is False
    assert read_json(tmp_path / "chapter_1_validation_round1_attempt2.json")["schema_valid"] is True


class FailClosedAuthor:
    model = "fail-closed-author"

    def __init__(self) -> None:
        self.written_chapters = 0

    def complete(self, prompt: str, **_kwargs) -> str:
        if "PLAN_WITH_CONTRACTS_JSON" in prompt:
            return json.dumps(
                {
                    "course_bible": {},
                    "chapters": [
                        {"title": "First", "description": "", "purpose": "", "contract": {"chapter_title": "First"}},
                        {"title": "Second", "description": "", "purpose": "", "contract": {"chapter_title": "Second"}},
                    ],
                }
            )
        if "WRITE_CHAPTER_MARKDOWN" in prompt:
            self.written_chapters += 1
            return "# First\n\nIncorrect worked example."
        if "REPAIR_CHAPTER_MARKDOWN" in prompt:
            return "# First\n\nStill incorrect worked example."
        raise AssertionError("unexpected author prompt")

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


class FailClosedReviewer:
    model = "fail-closed-reviewer"

    def complete(self, prompt: str, **_kwargs) -> str:
        if "REVIEW_CHAPTER_JSON" in prompt:
            return json.dumps(
                {
                    "passed": False,
                    "issues": [
                        {"severity": "high", "category": "math", "message": "Wrong result.", "suggestion": "Fix it."}
                    ],
                    "summary": "Blocking error.",
                }
            )
        if "VALIDATE_REPAIR_JSON" in prompt:
            return json.dumps(
                {
                    "passed": False,
                    "resolutions": [
                        {"issue_id": "R1", "status": "unresolved", "evidence": "Still wrong."}
                    ],
                    "new_issues": [],
                    "summary": "Not fixed.",
                }
            )
        raise AssertionError("unexpected reviewer prompt")

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


def test_run_stops_after_checkpointing_a_chapter_with_unresolved_high_issue(tmp_path) -> None:
    author = FailClosedAuthor()
    pipeline = ContractGuidedPipeline(
        llm=author,
        reviewer_llm=FailClosedReviewer(),
        max_repair_rounds=1,
        checkpoint_dir=str(tmp_path),
    )
    task = CourseTask(
        id="fail-closed-run",
        topic="Topic",
        goal="Goal",
        background="None.",
        chapter_count=2,
        chapter_min_characters=0,
        chapter_target_characters=0,
    )

    with pytest.raises(ChapterIntegrityError):
        pipeline.run(task)

    assert author.written_chapters == 1
    assert read_json(tmp_path / "latest.json")["metadata"]["stage"] == "chapter_1_reviewed"
    assert read_json(tmp_path / "latest.json")["chapters"][0]["integrity_status"] == "incomplete"


def test_contract_transition_normalization_uses_actual_plan_order() -> None:
    raw = [
        {
            "title": "Shortest Paths",
            "description": "",
            "purpose": "",
            "setup_for_next": "Next, study minimum spanning trees.",
        }
    ]
    contract = ChapterContract(
        chapter_title="Shortest Paths",
        bridge_to_next="Prepare greedy proofs in chapter six.",
        summary_for_next="Prepare greedy proofs.",
    )

    _normalize_contract_transitions(raw, [contract])

    assert contract.bridge_to_next == "Next, study minimum spanning trees."
    assert contract.summary_for_next == "Next, study minimum spanning trees."
    assert "Prepare greedy proofs in chapter six." in contract.prepares_for


class ReviewFailureLLM:
    model = "review-failure-test"

    def complete(self, prompt: str, **_kwargs) -> str:
        if "REVIEW_CHAPTER_JSON" in prompt:
            raise RuntimeError("temporary network failure")
        raise AssertionError("unexpected prompt")

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


def test_review_network_failure_stops_the_pipeline_by_default() -> None:
    llm = ReviewFailureLLM()
    pipeline = ContractGuidedPipeline(llm=llm, reviewer_llm=llm, max_repair_rounds=1)
    task = CourseTask(
        id="review-network-failure",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        contract=ChapterContract(chapter_title="Foundations", required_topics=["foundation"]),
        content_markdown="# Foundations\n\nThis chapter explains the foundation.",
    )

    try:
        pipeline._review_and_repair(task, CourseBible(), RoutedMaterials(), chapter, set())
    except RuntimeError as error:
        assert "temporary network failure" in str(error)
    else:
        raise AssertionError("review network failures must stop the pipeline")

    assert chapter.review_error["stage"] == "review"
    assert chapter.review_error["allowed"] is False


class LengthOnlyReviewer:
    model = "length-only-reviewer"

    def complete(self, prompt: str, **_kwargs) -> str:
        if "REVIEW_CHAPTER_JSON" in prompt:
            return json.dumps(
                {
                    "passed": False,
                    "issues": [
                        {
                            "severity": "high",
                            "category": "contract coverage",
                            "message": "Chapter has 20 characters, below the required minimum.",
                            "suggestion": "Expand it.",
                        }
                    ],
                    "summary": "Estimated as too short.",
                }
            )
        raise AssertionError("no repair should be requested for an overridden length estimate")

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


def test_program_character_count_overrides_reviewer_length_estimate() -> None:
    reviewer = LengthOnlyReviewer()
    pipeline = ContractGuidedPipeline(llm=reviewer, reviewer_llm=reviewer, max_repair_rounds=1)
    task = CourseTask(
        id="length-authority",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=40,
        chapter_target_characters=50,
    )
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        contract=ChapterContract(chapter_title="Foundations"),
        content_markdown="# Foundations\n\n" + "Complete Markdown content. " * 4,
    )

    pipeline._review_and_repair(task, CourseBible(), RoutedMaterials(), chapter, set())

    assert chapter.review is not None and chapter.review.passed
    assert chapter.review.issues == []
    assert chapter.review_reconciliation[0]["action"] == "review_issue_removed"
    assert chapter.review_reconciliation[0]["length_passed"] is True


class FalseNewLengthReviewer(ReviewerLLM):
    def complete(self, prompt: str, **kwargs) -> str:
        if "VALIDATE_REPAIR_JSON" in prompt:
            self.calls.append(prompt)
            return json.dumps(
                {
                    "passed": False,
                    "resolutions": [
                        {"issue_id": "R1", "status": "resolved", "evidence": "The derivation is fixed."}
                    ],
                    "new_issues": [
                        {
                            "id": "N1",
                            "severity": "high",
                            "category": "contract coverage",
                            "message": "Candidate is below the minimum character requirement.",
                            "suggestion": "Expand it.",
                        }
                    ],
                    "summary": "Estimated as too short.",
                }
            )
        return super().complete(prompt, **kwargs)


def test_program_character_count_removes_false_new_length_issue_before_dispute() -> None:
    author = AuthorRepairLLM()
    reviewer = FalseNewLengthReviewer()
    pipeline = ContractGuidedPipeline(llm=author, reviewer_llm=reviewer, max_repair_rounds=1)
    task = CourseTask(
        id="validation-length-authority",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=40,
        chapter_target_characters=50,
    )
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        contract=ChapterContract(chapter_title="Foundations"),
        content_markdown="# Foundations\n\nAn incomplete draft with enough raw characters for the configured minimum.",
    )

    pipeline._review_and_repair(task, CourseBible(), RoutedMaterials(), chapter, set())

    assert chapter.repair_history[0]["accepted"] is True
    assert chapter.repair_history[0]["needs_adjudication"] is False
    assert not any("RESPOND_TO_REPAIR_DISPUTE_JSON" in prompt for prompt in author.calls)
    assert any(item["action"] == "new_issue_removed" for item in chapter.review_reconciliation)


class IdenticalThenFixedAuthor:
    model = "identical-then-fixed-author"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, prompt: str, **_kwargs) -> str:
        self.calls.append(prompt)
        if "IDENTICAL_REPAIR_RETRY" in prompt:
            return "# Foundations\n\nThe corrected derivation is now explicit and verifiable."
        marker = "CURRENT_CHAPTER_MARKDOWN:"
        return prompt.split(marker, 1)[-1].strip()

    def completion_metadata(self) -> dict:
        return {"finish_reason": "stop"}


def test_identical_repair_gets_one_internal_retry_without_consuming_round(tmp_path) -> None:
    author = IdenticalThenFixedAuthor()
    reviewer = ReviewerLLM()
    pipeline = ContractGuidedPipeline(
        llm=author,
        reviewer_llm=reviewer,
        max_repair_rounds=1,
        max_identical_repair_retries=1,
        checkpoint_dir=str(tmp_path),
    )
    task = CourseTask(
        id="identical-retry",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        contract=ChapterContract(chapter_title="Foundations"),
        content_markdown="# Foundations\n\nThe original inaccurate derivation.",
    )

    pipeline._review_and_repair(
        task,
        CourseBible(),
        RoutedMaterials(),
        chapter,
        set(),
        chapter_number=1,
    )

    assert len(chapter.repair_history) == 1
    assert chapter.repair_history[0]["round"] == 1
    assert chapter.repair_history[0]["accepted"] is True
    assert chapter.repair_history[0]["identical_retry_count"] == 1
    assert len(author.calls) == 2
    assert read_json(tmp_path / "chapter_1_repair_round1_attempt1.json")["identical_to_source"] is True
    assert read_json(tmp_path / "chapter_1_repair_round1_attempt2.json")["identical_to_source"] is False
    assert (tmp_path / "chapter_1_repair_round1.json").exists()


class AlwaysIdenticalAuthor(IdenticalThenFixedAuthor):
    model = "always-identical-author"

    def complete(self, prompt: str, **_kwargs) -> str:
        self.calls.append(prompt)
        marker = "CURRENT_CHAPTER_MARKDOWN:"
        return prompt.split(marker, 1)[-1].strip()


def test_identical_repair_exhaustion_rejects_the_formal_round() -> None:
    author = AlwaysIdenticalAuthor()
    pipeline = ContractGuidedPipeline(
        llm=author,
        reviewer_llm=ReviewerLLM(),
        max_repair_rounds=1,
        max_identical_repair_retries=1,
    )
    task = CourseTask(
        id="identical-exhausted",
        topic="Foundations",
        goal="Teach framing.",
        background="None.",
        chapter_min_characters=0,
        chapter_target_characters=0,
    )
    original = "# Foundations\n\nThe original inaccurate derivation."
    chapter = ChapterOutput(
        plan=ChapterPlan(title="Foundations", description="", purpose=""),
        contract=ChapterContract(chapter_title="Foundations"),
        content_markdown=original,
    )

    pipeline._review_and_repair(task, CourseBible(), RoutedMaterials(), chapter, set())

    assert chapter.content_markdown == original
    assert len(chapter.repair_history) == 1
    assert chapter.repair_history[0]["rejection_reason"] == "identical_repair_candidate"
    assert chapter.repair_history[0]["identical_retry_count"] == 1
    assert len(author.calls) == 2
