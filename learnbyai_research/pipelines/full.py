from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import (
    build_introduced_prefixes,
    check_contract_sequence,
    repair_improved,
    summarize_verifications,
    verify_chapter_contract,
)
from ..io_utils import extract_json_object, read_json, write_json
from ..llm import ChatModel, completion_history
from ..materials import route_materials
from ..prompts import (
    SCOPE_CONTROL_POLICY_ID,
    build_chapter_writer_prompt,
    build_contract_plan_prompt,
    build_full_length_enrichment_prompt,
    build_repair_dispute_prompt,
    build_repair_dispute_reconsideration_prompt,
    build_identical_repair_retry_prompt,
    build_repair_prompt,
    build_repair_validation_prompt,
    build_repair_validation_schema_retry_prompt,
    build_review_prompt,
)
from ..schemas import ChapterOutput, CourseOutput, CourseTask, RepairValidationResult, ReviewIssue, ReviewResult, RoutedMaterials, chapter_pre_review_target, to_plain, utc_now
from .common import bible_from_payload, contract_from_payload, plan_from_payload, repair_validation_from_payload, review_from_payload


MAX_REPAIR_CHAPTER_CHARACTERS = 30_000
REPAIR_STRATEGY_ID = "adaptive_reasoning_external_review_v4"


class ChapterIntegrityError(RuntimeError):
    """Raised after a blocking chapter checkpoint has been saved."""


@dataclass
class ContractGuidedPipeline:
    llm: ChatModel
    reviewer_llm: ChatModel | None = None
    repair_llm: ChatModel | None = None
    name: str = "full"
    use_contracts: bool = True
    use_material_routing: bool = True
    use_review_repair: bool = True
    max_repair_rounds: int = 2
    checkpoint_dir: str | None = None
    max_review_chars: int | None = None
    allow_review_failures: bool = False
    resume_from_checkpoints: bool = False
    max_validation_schema_retries: int = 1
    max_identical_repair_retries: int = 1
    max_json_repair_attempts: int = 2

    def run(self, task: CourseTask) -> CourseOutput:
        routed = route_materials(task, enabled=self.use_material_routing)
        checkpoint = self._load_resume_checkpoint(task)
        resumed = checkpoint is not None
        if checkpoint:
            plan_payload, chapters, resume_stage = checkpoint
        else:
            plan_payload = self._complete_json(
                self.llm,
                build_contract_plan_prompt(task, routed, use_contracts=self.use_contracts),
                system="You are a curriculum architect. Return valid JSON only.",
                temperature=0.2,
            )
            chapters = []
            resume_stage = None
        bible = bible_from_payload(plan_payload)
        raw_chapters = plan_payload.get("chapters", [])
        if not isinstance(raw_chapters, list) or not raw_chapters:
            raise ValueError("contract plan did not contain any chapters")
        if len(chapters) > len(raw_chapters):
            raise ValueError("checkpoint contains more chapters than the saved contract plan")
        if not resumed:
            self._write_checkpoint(
                "plan",
                {
                    "task_id": task.id,
                    "pipeline": self.name,
                    "raw_chapter_count": len(raw_chapters),
                    "plan": plan_payload,
                },
            )
        parsed_contracts = [
            contract_from_payload(raw.get("contract"), plan_from_payload(raw).title) if self.use_contracts else None
            for raw in raw_chapters
        ]
        _normalize_contract_transitions(raw_chapters, parsed_contracts)
        sequence_check = (
            check_contract_sequence(
                parsed_contracts,
                target_topics=target_topics_from_task(task),
                target_aliases=target_aliases_from_task(task),
                chapter_dependencies=bible.chapter_dependencies,
            )
            if self.use_contracts
            else None
        )
        introduced_prefixes = build_introduced_prefixes(parsed_contracts)
        bible.chapter_contracts = [contract for contract in parsed_contracts[: len(chapters)] if contract]
        start_index = _resume_start_index(resume_stage, len(chapters), self.use_review_repair) if resumed else 0
        if resumed and resume_stage and resume_stage.endswith("_reviewed") and chapters:
            _finalize_chapter_integrity(chapters[-1])
            _raise_for_blocking_chapter(chapters[-1], len(chapters))
        for index in range(start_index, len(raw_chapters)):
            raw = raw_chapters[index]
            plan = plan_from_payload(raw)
            contract = parsed_contracts[index] if index < len(parsed_contracts) else None
            if index < len(chapters):
                chapter_output = chapters[index]
                chapter_output.plan = plan
                chapter_output.contract = contract
                self._verify_contract(chapter_output, introduced_prefixes[index] if index < len(introduced_prefixes) else set())
                if self.use_review_repair:
                    self._review_and_repair(
                        task,
                        bible,
                        routed,
                        chapter_output,
                        introduced_prefixes[index] if index < len(introduced_prefixes) else set(),
                        chapter_number=index + 1,
                    )
                    _finalize_chapter_integrity(chapter_output)
                    self._write_partial_course(task, bible, chapters, routed, sequence_check, stage=f"chapter_{index + 1}_reviewed")
                    _raise_for_blocking_chapter(chapter_output, index + 1)
                continue
            if contract:
                bible.chapter_contracts.append(contract)
            markdown = self.llm.complete(
                build_chapter_writer_prompt(task, bible, plan, contract, routed, index, len(raw_chapters)),
                system="You are a textbook author. Return Markdown only.",
                temperature=0.35,
            ).strip()
            enrichment_history: list[dict[str, Any]] = []
            pre_review_target = chapter_pre_review_target(task)
            if pre_review_target and len(markdown) < pre_review_target:
                candidate = self.llm.complete(
                    build_full_length_enrichment_prompt(task, plan, contract, routed, markdown),
                    system="You are a textbook author. Return Markdown only.",
                    temperature=0.35,
                ).strip()
                enrichment_history.append(
                    {
                        "before_chars": len(markdown),
                        "after_chars": len(candidate),
                        "pre_review_target_characters": pre_review_target,
                        "accepted": len(candidate) > len(markdown),
                        "completion": self.llm.completion_metadata(),
                    }
                )
                if len(candidate) > len(markdown):
                    markdown = candidate
            chapter_output = ChapterOutput(plan=plan, contract=contract, content_markdown=markdown)
            chapter_output.length_enrichment_history = enrichment_history
            self._verify_contract(chapter_output, introduced_prefixes[index] if index < len(introduced_prefixes) else set())
            chapters.append(chapter_output)
            self._write_partial_course(task, bible, chapters, routed, sequence_check, stage=f"chapter_{index + 1}_written")
            if self.use_review_repair:
                self._review_and_repair(
                    task,
                    bible,
                    routed,
                    chapter_output,
                    introduced_prefixes[index] if index < len(introduced_prefixes) else set(),
                    chapter_number=index + 1,
                )
                _finalize_chapter_integrity(chapter_output)
                self._write_partial_course(task, bible, chapters, routed, sequence_check, stage=f"chapter_{index + 1}_reviewed")
                _raise_for_blocking_chapter(chapter_output, index + 1)
        verification_results = [
            verify_chapter_contract(
                chapter.content_markdown,
                chapter.contract,
                title=chapter.plan.title,
                introduced_before=introduced_prefixes[index] if index < len(introduced_prefixes) else set(),
            )
            for index, chapter in enumerate(chapters)
        ]
        integrity_failures = _course_integrity_failures(
            task,
            chapters,
            require_review=self.use_review_repair,
        )
        integrity_status = (
            "incomplete"
            if integrity_failures
            else "complete_with_warnings"
            if any(chapter.integrity_status == "complete_with_warnings" for chapter in chapters)
            else "complete"
        )
        return CourseOutput(
            task=task,
            pipeline=self.name,
            course_bible=bible,
            chapters=chapters,
            routed_materials=routed,
            metadata={
                "created_at": utc_now(),
                "llm_model": self.llm.model,
                "review_model": self._reviewer().model if self.use_review_repair else None,
                "repair_model": self._repairer().model if self.use_review_repair else None,
                "repair_strategy": REPAIR_STRATEGY_ID if self.use_review_repair else None,
                "scope_control_policy": SCOPE_CONTROL_POLICY_ID,
                "pre_review_target_characters": chapter_pre_review_target(task),
                "use_contracts": self.use_contracts,
                "use_material_routing": self.use_material_routing,
                "use_review_repair": self.use_review_repair,
                "contract_sequence": to_plain(sequence_check) if sequence_check else None,
                "contract_verification": summarize_verifications(verification_results),
                "integrity_status": integrity_status,
                "integrity_failures": integrity_failures,
                "resumed_from_checkpoint": resumed,
                "resume_stage": resume_stage,
                "author_completion_history": completion_history(self.llm),
                "repair_completion_history": (
                    completion_history(self._repairer())
                    if self.use_review_repair and self._repairer() is not self.llm
                    else []
                ),
                "reviewer_completion_history": (
                    completion_history(self._reviewer())
                    if self.use_review_repair and self._reviewer() is not self.llm
                    else []
                ),
                "author_reviewer_shared_client": self.use_review_repair and self._reviewer() is self.llm,
                "author_repair_shared_client": self.use_review_repair and self._repairer() is self.llm,
            },
        )

    def _load_resume_checkpoint(self, task: CourseTask) -> tuple[dict[str, Any], list[ChapterOutput], str | None] | None:
        if not self.resume_from_checkpoints or not self.checkpoint_dir:
            return None
        checkpoint_dir = Path(self.checkpoint_dir)
        plan_path = checkpoint_dir / "plan.json"
        if not plan_path.exists():
            return None
        plan_record = read_json(plan_path)
        if plan_record.get("task_id") != task.id or plan_record.get("pipeline") != self.name:
            raise ValueError("checkpoint plan belongs to a different task or pipeline")
        plan_payload = plan_record.get("plan")
        if not isinstance(plan_payload, dict):
            raise ValueError("checkpoint plan is missing its plan payload")

        latest_path = checkpoint_dir / "latest.json"
        if not latest_path.exists():
            return plan_payload, [], "plan"
        latest = read_json(latest_path)
        latest_task = latest.get("task", {})
        if latest_task.get("id") != task.id or latest.get("pipeline") != self.name:
            raise ValueError("latest checkpoint belongs to a different task or pipeline")
        raw_chapters = latest.get("chapters", [])
        if not isinstance(raw_chapters, list):
            raise ValueError("latest checkpoint chapters must be a list")
        chapters = [_chapter_from_checkpoint(raw) for raw in raw_chapters if isinstance(raw, dict)]
        if len(chapters) != len(raw_chapters):
            raise ValueError("latest checkpoint contains an invalid chapter record")
        stage = latest.get("metadata", {}).get("stage")
        return plan_payload, chapters, str(stage) if stage else None

    def _review_and_repair(
        self,
        task: CourseTask,
        bible,
        routed: RoutedMaterials,
        chapter_output: ChapterOutput,
        introduced_before: set[str],
        *,
        chapter_number: int | None = None,
    ) -> None:
        baseline_markdown = chapter_output.content_markdown
        baseline_verification = verify_chapter_contract(
            baseline_markdown,
            chapter_output.contract,
            title=chapter_output.plan.title,
            introduced_before=introduced_before,
        )
        chapter_output.evidence = to_plain(baseline_verification.evidence)
        chapter_output.verification = to_plain(baseline_verification)

        review_payload = self._review_chapter(task, bible, routed, chapter_output, 0)
        if review_payload is None:
            return
        review = _assign_reviewer_issue_ids(review_from_payload(review_payload))
        review, review_overrides = _reconcile_review_hard_metrics(task, baseline_markdown, review)
        chapter_output.review_reconciliation.extend(review_overrides)
        hard_metric_issues = _program_hard_metric_issues(task, baseline_markdown)
        all_issues = [*review.issues, *hard_metric_issues]
        needs_repair = not review.passed or bool(hard_metric_issues) or any(
            issue.severity in {"high", "medium"} for issue in review.issues
        )
        chapter_output.review = ReviewResult(
            passed=review.passed and not needs_repair,
            issues=all_issues,
            summary=review.summary,
        )
        if not needs_repair or self.max_repair_rounds <= 0:
            return

        repair_source = baseline_markdown
        source_verification = baseline_verification
        repair_feedback: dict[str, Any] = {
            "passed": False,
            "issues": [to_plain(issue) for issue in all_issues],
            "summary": review.summary,
            "contract_verification": to_plain(baseline_verification),
        }
        accepted_fallback = False
        for round_index in range(self.max_repair_rounds):
            repair_prompt = build_repair_prompt(
                task,
                bible,
                chapter_output.plan,
                chapter_output.contract,
                routed,
                repair_source,
                repair_feedback,
            )
            repair_attempt_artifacts: list[str] = []
            identical_retry_count = 0
            identical_retry_exhausted = False
            try:
                candidate = ""
                for attempt_index in range(self.max_identical_repair_retries + 1):
                    attempt_prompt = (
                        repair_prompt
                        if attempt_index == 0
                        else build_identical_repair_retry_prompt(repair_prompt)
                    )
                    candidate = self._repairer().complete(
                        attempt_prompt,
                        system="You are the original textbook author revising a chapter. Return complete Markdown only.",
                        temperature=0.2,
                    ).strip()
                    attempt_artifact = self._write_repair_attempt_artifact(
                        task,
                        chapter_output,
                        chapter_number,
                        round_index,
                        attempt_index,
                        repair_source,
                        candidate,
                    )
                    if attempt_artifact:
                        repair_attempt_artifacts.append(attempt_artifact)
                    if not candidate or not _repair_markdown_identical(repair_source, candidate):
                        break
                    if attempt_index >= self.max_identical_repair_retries:
                        identical_retry_exhausted = True
                        break
                    identical_retry_count += 1
            except Exception as error:
                chapter_output.review_error = {
                    "stage": "repair",
                    "round": round_index + 1,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "allowed": self.allow_review_failures or accepted_fallback,
                    "fallback_retained": accepted_fallback,
                }
                chapter_output.repair_history.append(
                    {
                        "round": round_index + 1,
                        "issue_count": len(all_issues),
                        "before_chars": len(repair_source),
                        "after_chars": 0,
                        "before_violation_score": source_verification.violation_score,
                        "after_violation_score": None,
                        "accepted": False,
                        "repair_model": self._repairer().model,
                        "review_model": self._reviewer().model,
                        "repair_attempt_artifacts": repair_attempt_artifacts,
                        "identical_retry_count": identical_retry_count,
                        "error": chapter_output.review_error,
                    }
                )
                if self.allow_review_failures or accepted_fallback:
                    return
                raise
            if not candidate:
                chapter_output.repair_history.append(
                    {
                        "round": round_index + 1,
                        "issue_count": len(all_issues),
                        "before_chars": len(repair_source),
                        "after_chars": 0,
                        "before_violation_score": source_verification.violation_score,
                        "after_violation_score": None,
                        "accepted": False,
                        "rejection_reason": "empty_repair_candidate",
                        "repair_model": self._repairer().model,
                        "review_model": self._reviewer().model,
                        "repair_attempt_artifacts": repair_attempt_artifacts,
                        "identical_retry_count": identical_retry_count,
                    }
                )
                return

            if identical_retry_exhausted:
                chapter_output.repair_history.append(
                    {
                        "round": round_index + 1,
                        "issue_count": len(all_issues),
                        "before_chars": len(repair_source),
                        "after_chars": len(candidate),
                        "before_violation_score": source_verification.violation_score,
                        "after_violation_score": source_verification.violation_score,
                        "accepted": False,
                        "rejection_reason": "identical_repair_candidate",
                        "repair_model": self._repairer().model,
                        "review_model": self._reviewer().model,
                        "repair_attempt_artifacts": repair_attempt_artifacts,
                        "identical_retry_count": identical_retry_count,
                    }
                )
                if round_index >= self.max_repair_rounds - 1:
                    return
                repair_feedback = {
                    "passed": False,
                    "issues": [to_plain(issue) for issue in all_issues],
                    "summary": "The author returned an unchanged chapter even after an internal retry.",
                    "program_rejection_reason": "identical_repair_candidate",
                    "contract_verification": to_plain(source_verification),
                }
                continue

            repair_artifact = self._write_repair_artifact(
                task,
                chapter_output,
                chapter_number,
                round_index,
                repair_source,
                candidate,
            )

            after_verification = verify_chapter_contract(
                candidate,
                chapter_output.contract,
                title=chapter_output.plan.title,
                introduced_before=introduced_before,
            )
            candidate_completion_metadata = self._repairer().completion_metadata()
            precheck_reason = _repair_candidate_precheck_reason(
                task,
                candidate,
                completion_metadata=candidate_completion_metadata,
            )
            if precheck_reason is not None:
                chapter_output.post_repair_verification = to_plain(after_verification)
                chapter_output.repair_history.append(
                    {
                        "round": round_index + 1,
                        "issue_count": len(all_issues),
                        "before_chars": len(repair_source),
                        "after_chars": len(candidate),
                        "before_violation_score": source_verification.violation_score,
                        "baseline_violation_score": baseline_verification.violation_score,
                        "after_violation_score": after_verification.violation_score,
                        "accepted": False,
                        "rejection_reason": precheck_reason,
                        "repair_model": self._repairer().model,
                        "review_model": self._reviewer().model,
                        "repair_artifact": repair_artifact,
                        "repair_attempt_artifacts": repair_attempt_artifacts,
                        "identical_retry_count": identical_retry_count,
                        "repair_completion_metadata": candidate_completion_metadata,
                    }
                )
                if round_index >= self.max_repair_rounds - 1:
                    return
                repair_feedback = {
                    "passed": False,
                    "issues": [to_plain(issue) for issue in all_issues],
                    "summary": "The program rejected the previous candidate before semantic validation.",
                    "program_rejection_reason": precheck_reason,
                    "contract_verification": to_plain(after_verification),
                }
                continue

            validation = self._validate_repair_candidate(
                task,
                bible,
                routed,
                chapter_output,
                all_issues,
                repair_source,
                candidate,
                round_index,
                chapter_number,
            )
            if validation is None:
                chapter_output.post_repair_verification = to_plain(after_verification)
                chapter_output.repair_history.append(
                    {
                        "round": round_index + 1,
                        "issue_count": len(all_issues),
                        "before_chars": len(repair_source),
                        "after_chars": len(candidate),
                        "before_violation_score": source_verification.violation_score,
                        "baseline_violation_score": baseline_verification.violation_score,
                        "after_violation_score": after_verification.violation_score,
                        "accepted": False,
                        "rejection_reason": "repair_validation_unavailable",
                        "repair_model": self._repairer().model,
                        "review_model": self._reviewer().model,
                        "repair_artifact": repair_artifact,
                        "repair_attempt_artifacts": repair_attempt_artifacts,
                        "identical_retry_count": identical_retry_count,
                        "error": chapter_output.review_error,
                    }
                )
                return

            validation = _assign_validation_new_issue_ids(validation)
            validation, validation_overrides = _reconcile_validation_hard_metrics(
                task,
                candidate,
                all_issues,
                validation,
                round_index,
            )
            chapter_output.review_reconciliation.extend(validation_overrides)
            validation, needs_adjudication, dispute_artifacts = self._resolve_new_high_disputes(
                task,
                bible,
                routed,
                chapter_output,
                all_issues,
                repair_source,
                candidate,
                validation,
                round_index,
                chapter_number,
            )
            validation, post_dispute_overrides = _reconcile_validation_hard_metrics(
                task,
                candidate,
                all_issues,
                validation,
                round_index,
            )
            chapter_output.review_reconciliation.extend(post_dispute_overrides)
            validation_passed = _repair_validation_passes_gate(all_issues, validation)
            semantic_improved = _semantic_repair_improved(all_issues, validation)
            residual_issues = _residual_validation_issues(all_issues, validation)
            accepted, rejection_reason = _accept_local_repair(
                task,
                repair_source,
                candidate,
                baseline_verification,
                after_verification,
                validation_passed=validation_passed,
                completion_metadata=candidate_completion_metadata,
            )
            cleanup_continued = (
                accepted
                and round_index < self.max_repair_rounds - 1
                and any(issue.severity == "medium" for issue in residual_issues)
            )
            acceptance_basis = (
                _repair_acceptance_basis(
                    baseline_verification,
                    after_verification,
                    semantic_improved=semantic_improved,
                )
                if accepted
                else None
            )
            chapter_output.post_repair_verification = to_plain(after_verification)
            chapter_output.repair_history.append(
                {
                    "round": round_index + 1,
                    "issue_count": len(all_issues),
                    "before_chars": len(repair_source),
                    "after_chars": len(candidate),
                    "before_violation_score": source_verification.violation_score,
                    "baseline_violation_score": baseline_verification.violation_score,
                    "after_violation_score": after_verification.violation_score,
                    "accepted": accepted,
                    "rejection_reason": rejection_reason,
                    "acceptance_basis": acceptance_basis,
                    "semantic_improved": semantic_improved,
                    "repair_model": self._repairer().model,
                    "review_model": self._reviewer().model,
                    "repair_validation": to_plain(validation),
                    "repair_artifact": repair_artifact,
                    "repair_attempt_artifacts": repair_attempt_artifacts,
                    "identical_retry_count": identical_retry_count,
                    "repair_completion_metadata": candidate_completion_metadata,
                    "dispute_artifacts": dispute_artifacts,
                    "needs_adjudication": needs_adjudication,
                    "residual_issue_count": len(residual_issues),
                    "cleanup_continued": cleanup_continued,
                }
            )
            if needs_adjudication:
                if accepted_fallback:
                    chapter_output.repair_history[-1]["rejection_reason"] = (
                        "cleanup_candidate_needs_adjudication"
                    )
                    chapter_output.repair_history[-1]["needs_adjudication"] = False
                    chapter_output.repair_history[-1]["fallback_retained"] = True
                    return
                chapter_output.repair_history[-1]["rejection_reason"] = "needs_adjudication"
                chapter_output.integrity_status = "needs_adjudication"
                chapter_output.integrity_failures = [
                    {
                        "type": "disputed_high_issue",
                        "round": round_index + 1,
                        "issue_ids": [issue.id for issue in validation.new_issues if issue.severity == "high"],
                    }
                ]
                return
            if accepted:
                chapter_output.content_markdown = candidate
                chapter_output.evidence = to_plain(after_verification.evidence)
                chapter_output.verification = to_plain(after_verification)
                chapter_output.review = _review_result_after_repair(all_issues, validation)
                if cleanup_continued:
                    accepted_fallback = True
                    all_issues = residual_issues
                    repair_source = candidate
                    source_verification = after_verification
                    repair_feedback = {
                        "passed": False,
                        "issues": [to_plain(issue) for issue in all_issues],
                        "summary": validation.summary,
                        "previous_validation": to_plain(validation),
                        "cleanup_reason": "residual_medium_issues",
                        "contract_verification": to_plain(after_verification),
                    }
                    continue
                return
            if round_index >= self.max_repair_rounds - 1:
                return

            all_issues = residual_issues
            repair_source = candidate
            source_verification = after_verification
            repair_feedback = {
                "passed": False,
                "issues": [to_plain(issue) for issue in all_issues],
                "summary": validation.summary,
                "previous_validation": to_plain(validation),
                "program_rejection_reason": rejection_reason,
                "contract_verification": to_plain(after_verification),
            }

    def _validate_repair_candidate(
        self,
        task: CourseTask,
        bible,
        routed: RoutedMaterials,
        chapter_output: ChapterOutput,
        original_issues: list[ReviewIssue],
        original_markdown: str,
        candidate: str,
        round_index: int,
        chapter_number: int | None,
    ) -> RepairValidationResult | None:
        if self.max_review_chars is not None and len(candidate) > self.max_review_chars:
            self._record_repair_validation_error(
                chapter_output,
                round_index,
                "ReviewSkippedLength",
                f"Repair validation skipped because candidate has {len(candidate)} characters.",
            )
            return None
        prompt = build_repair_validation_prompt(
            task,
            bible,
            chapter_output.plan,
            chapter_output.contract,
            routed,
            [to_plain(issue) for issue in original_issues],
            original_markdown,
            candidate,
        )
        try:
            return self._complete_validated_repair_review(
                prompt,
                original_issues,
                chapter_number,
                round_index,
                artifact_label="validation",
            )
        except Exception as error:
            self._record_repair_validation_error(chapter_output, round_index, type(error).__name__, str(error))
            if not self.allow_review_failures:
                raise
            return None

    def _complete_validated_repair_review(
        self,
        prompt: str,
        original_issues: list[ReviewIssue],
        chapter_number: int | None,
        round_index: int,
        *,
        artifact_label: str,
    ) -> RepairValidationResult:
        payload = self._complete_json(
            self._reviewer(),
            prompt,
            system="You validate textbook repairs. Return valid JSON only.",
            temperature=0.1,
        )
        expected_ids = [issue.id for issue in original_issues]
        for attempt in range(self.max_validation_schema_retries + 1):
            try:
                validation = _assign_validation_new_issue_ids(repair_validation_from_payload(payload))
                _validate_repair_validation(original_issues, validation)
                self._write_validation_artifact(
                    chapter_number,
                    round_index,
                    artifact_label,
                    attempt,
                    payload,
                    schema_error=None,
                    canonical=True,
                    normalized_validation=to_plain(validation),
                )
                return validation
            except ValueError as error:
                self._write_validation_artifact(
                    chapter_number,
                    round_index,
                    artifact_label,
                    attempt,
                    payload,
                    schema_error=str(error),
                    canonical=False,
                    normalized_validation=None,
                )
                if attempt >= self.max_validation_schema_retries:
                    raise ValueError(
                        f"repair validation schema remained invalid after {attempt + 1} attempts: {error}"
                    ) from error
                payload = self._complete_json(
                    self._reviewer(),
                    build_repair_validation_schema_retry_prompt(expected_ids, payload, str(error)),
                    system="You correct repair-validation JSON structure. Return valid JSON only.",
                    temperature=0.0,
                )

    def _resolve_new_high_disputes(
        self,
        task: CourseTask,
        bible,
        routed: RoutedMaterials,
        chapter_output: ChapterOutput,
        original_issues: list[ReviewIssue],
        original_markdown: str,
        candidate: str,
        validation: RepairValidationResult,
        round_index: int,
        chapter_number: int | None,
    ) -> tuple[RepairValidationResult, bool, list[str]]:
        disputed = [issue for issue in validation.new_issues if issue.severity == "high"]
        if not disputed:
            return validation, False, []

        response_payload = self._complete_json(
            self._repairer(),
            build_repair_dispute_prompt(
                task,
                bible,
                chapter_output.plan,
                chapter_output.contract,
                routed,
                candidate,
                [to_plain(issue) for issue in disputed],
            ),
            system="You respond to disputed textbook-review findings. Return valid JSON only.",
            temperature=0.0,
        )
        response_artifact = self._write_dispute_artifact(
            chapter_number,
            round_index,
            "author_response",
            response_payload,
        )
        responses = _validate_dispute_response(disputed, response_payload)
        rebutted_ids = {
            str(item.get("issue_id", ""))
            for item in responses
            if item.get("position") == "rebut"
        }
        if not rebutted_ids:
            return validation, False, [response_artifact] if response_artifact else []

        reconsidered = self._complete_validated_repair_review(
            build_repair_dispute_reconsideration_prompt(
                task,
                bible,
                chapter_output.plan,
                chapter_output.contract,
                routed,
                [to_plain(issue) for issue in original_issues],
                original_markdown,
                candidate,
                to_plain(validation),
                response_payload,
            ),
            original_issues,
            chapter_number,
            round_index,
            artifact_label="dispute_validation",
        )
        reconsidered = _assign_validation_new_issue_ids(reconsidered)
        remaining_high_ids = {
            issue.id for issue in reconsidered.new_issues if issue.severity == "high"
        }
        needs_adjudication = bool(remaining_high_ids)
        validation_artifact = self._artifact_relative_path(
            chapter_number,
            f"dispute_validation_round{round_index + 1}.json",
        )
        artifacts = [item for item in [response_artifact, validation_artifact] if item]
        return reconsidered, needs_adjudication, artifacts

    def _write_repair_artifact(
        self,
        task: CourseTask,
        chapter_output: ChapterOutput,
        chapter_number: int | None,
        round_index: int,
        source_markdown: str,
        candidate_markdown: str,
    ) -> str | None:
        if chapter_number is None:
            return None
        name = f"chapter_{chapter_number}_repair_round{round_index + 1}"
        self._write_checkpoint(
            name,
            {
                "task_id": task.id,
                "chapter": chapter_number,
                "chapter_title": chapter_output.plan.title,
                "round": round_index + 1,
                "repair_model": self._repairer().model,
                "created_at": utc_now(),
                "source_markdown": source_markdown,
                "candidate_markdown": candidate_markdown,
            },
        )
        return self._artifact_relative_path(chapter_number, f"repair_round{round_index + 1}.json")

    def _write_repair_attempt_artifact(
        self,
        task: CourseTask,
        chapter_output: ChapterOutput,
        chapter_number: int | None,
        round_index: int,
        attempt_index: int,
        source_markdown: str,
        candidate_markdown: str,
    ) -> str | None:
        if chapter_number is None:
            return None
        name = f"chapter_{chapter_number}_repair_round{round_index + 1}_attempt{attempt_index + 1}"
        self._write_checkpoint(
            name,
            {
                "task_id": task.id,
                "chapter": chapter_number,
                "chapter_title": chapter_output.plan.title,
                "round": round_index + 1,
                "attempt": attempt_index + 1,
                "repair_model": self._repairer().model,
                "created_at": utc_now(),
                "source_markdown": source_markdown,
                "candidate_markdown": candidate_markdown,
                "identical_to_source": _repair_markdown_identical(source_markdown, candidate_markdown),
            },
        )
        return self._artifact_relative_path(
            chapter_number,
            f"repair_round{round_index + 1}_attempt{attempt_index + 1}.json",
        )

    def _write_validation_artifact(
        self,
        chapter_number: int | None,
        round_index: int,
        label: str,
        attempt: int,
        payload: dict[str, Any],
        *,
        schema_error: str | None,
        canonical: bool,
        normalized_validation: dict[str, Any] | None,
    ) -> None:
        if chapter_number is None:
            return
        record = {
            "chapter": chapter_number,
            "round": round_index + 1,
            "attempt": attempt + 1,
            "review_model": self._reviewer().model,
            "created_at": utc_now(),
            "schema_valid": schema_error is None,
            "schema_error": schema_error,
            "validation": payload,
            "normalized_validation": normalized_validation,
        }
        self._write_checkpoint(
            f"chapter_{chapter_number}_{label}_round{round_index + 1}_attempt{attempt + 1}",
            record,
        )
        if canonical:
            self._write_checkpoint(f"chapter_{chapter_number}_{label}_round{round_index + 1}", record)

    def _write_dispute_artifact(
        self,
        chapter_number: int | None,
        round_index: int,
        label: str,
        payload: dict[str, Any],
    ) -> str | None:
        if chapter_number is None:
            return None
        name = f"chapter_{chapter_number}_dispute_{label}_round{round_index + 1}"
        self._write_checkpoint(
            name,
            {
                "chapter": chapter_number,
                "round": round_index + 1,
                "created_at": utc_now(),
                "payload": payload,
            },
        )
        return self._artifact_relative_path(chapter_number, f"dispute_{label}_round{round_index + 1}.json")

    def _artifact_relative_path(self, chapter_number: int | None, suffix: str) -> str | None:
        if chapter_number is None or not self.checkpoint_dir:
            return None
        return f"chapter_{chapter_number}_{suffix}"

    def _record_repair_validation_error(
        self,
        chapter_output: ChapterOutput,
        round_index: int,
        error_type: str,
        message: str,
    ) -> None:
        chapter_output.review_error = {
            "stage": "repair_validation",
            "round": round_index + 1,
            "error_type": error_type,
            "message": message,
            "allowed": self.allow_review_failures,
        }

    def _review_chapter(
        self,
        task: CourseTask,
        bible,
        routed: RoutedMaterials,
        chapter_output: ChapterOutput,
        round_index: int,
    ) -> dict[str, Any] | None:
        content_chars = len(chapter_output.content_markdown)
        if self.max_review_chars is not None and content_chars > self.max_review_chars:
            issue = ReviewIssue(
                severity="medium",
                category="review_skipped_length",
                message=f"LLM review skipped because chapter has {content_chars} characters.",
                suggestion="Use deterministic verification or rerun with a higher max_review_chars value.",
            )
            chapter_output.review = ReviewResult(
                passed=False,
                issues=[issue],
                summary="LLM review skipped by configured length guard.",
            )
            chapter_output.review_error = {
                "stage": "review",
                "round": round_index + 1,
                "error_type": "ReviewSkippedLength",
                "message": issue.message,
                "allowed": True,
            }
            return None
        try:
            return self._complete_json(
                self._reviewer(),
                build_review_prompt(
                    task,
                    bible,
                    chapter_output.plan,
                    chapter_output.contract,
                    routed,
                    chapter_output.content_markdown,
                ),
                system="You are a strict textbook reviewer. Return valid JSON only.",
                temperature=0.1,
            )
        except Exception as error:
            chapter_output.review = ReviewResult(
                passed=False,
                issues=[
                    ReviewIssue(
                        severity="high",
                        category="review_error",
                        message=f"LLM review failed: {type(error).__name__}",
                        suggestion="Inspect review_error metadata and rerun this chapter if needed.",
                    )
                ],
                summary="LLM review failed before returning a parseable result.",
            )
            chapter_output.review_error = {
                "stage": "review",
                "round": round_index + 1,
                "error_type": type(error).__name__,
                "message": str(error),
                "allowed": self.allow_review_failures,
            }
            if self.allow_review_failures:
                return None
            raise

    def _verify_contract(self, chapter_output: ChapterOutput, introduced_before: set[str]) -> None:
        result = verify_chapter_contract(
            chapter_output.content_markdown,
            chapter_output.contract,
            title=chapter_output.plan.title,
            introduced_before=introduced_before,
        )
        chapter_output.evidence = to_plain(result.evidence)
        chapter_output.verification = to_plain(result)

    def _reviewer(self) -> ChatModel:
        return self.reviewer_llm or self.llm

    def _repairer(self) -> ChatModel:
        return self.repair_llm or self.llm

    def _complete_json(self, llm: ChatModel, prompt: str, *, system: str, temperature: float) -> dict[str, Any]:
        candidate = llm.complete(prompt, system=system, temperature=temperature, response_format="json_object")
        for attempt in range(self.max_json_repair_attempts + 1):
            try:
                return extract_json_object(candidate)
            except ValueError as error:
                if attempt >= self.max_json_repair_attempts:
                    raise
                candidate = llm.complete(
                    build_json_repair_prompt(candidate, str(error)),
                    system="You repair invalid JSON. Return valid JSON only, preserving the original meaning.",
                    temperature=0.0,
                    response_format="json_object",
                )
        raise AssertionError("JSON repair loop exited unexpectedly")

    def _write_checkpoint(self, name: str, payload: dict[str, Any]) -> None:
        if not self.checkpoint_dir:
            return
        write_json(Path(self.checkpoint_dir) / f"{name}.json", payload)

    def _write_partial_course(
        self,
        task: CourseTask,
        bible,
        chapters: list[ChapterOutput],
        routed: RoutedMaterials,
        sequence_check,
        *,
        stage: str,
    ) -> None:
        if not self.checkpoint_dir:
            return
        partial = CourseOutput(
            task=task,
            pipeline=self.name,
            course_bible=bible,
            chapters=chapters,
            routed_materials=routed,
            metadata={
                "created_at": utc_now(),
                "stage": stage,
                "llm_model": self.llm.model,
                "review_model": self._reviewer().model if self.use_review_repair else None,
                "repair_model": self._repairer().model if self.use_review_repair else None,
                "repair_strategy": REPAIR_STRATEGY_ID if self.use_review_repair else None,
                "scope_control_policy": SCOPE_CONTROL_POLICY_ID,
                "pre_review_target_characters": chapter_pre_review_target(task),
                "use_contracts": self.use_contracts,
                "use_material_routing": self.use_material_routing,
                "use_review_repair": self.use_review_repair,
                "contract_sequence": to_plain(sequence_check) if sequence_check else None,
            },
        )
        self._write_checkpoint(stage, to_plain(partial))
        self._write_checkpoint("latest", to_plain(partial))


def _chapter_from_checkpoint(raw: dict[str, Any]) -> ChapterOutput:
    plan = plan_from_payload(raw.get("plan", {}))
    contract = contract_from_payload(raw.get("contract"), plan.title)
    review_payload = raw.get("review")
    review = review_from_payload(review_payload) if isinstance(review_payload, dict) else None
    return ChapterOutput(
        plan=plan,
        contract=contract,
        content_markdown=str(raw.get("content_markdown", "")),
        evidence=dict(raw.get("evidence", {})),
        verification=dict(raw.get("verification", {})),
        post_repair_verification=dict(raw.get("post_repair_verification", {})),
        review=review,
        review_error=dict(raw.get("review_error", {})),
        review_reconciliation=[
            dict(item) for item in raw.get("review_reconciliation", []) if isinstance(item, dict)
        ],
        repair_history=[dict(item) for item in raw.get("repair_history", []) if isinstance(item, dict)],
        length_enrichment_history=[
            dict(item) for item in raw.get("length_enrichment_history", []) if isinstance(item, dict)
        ],
        integrity_status=str(raw.get("integrity_status", "pending")),
        integrity_failures=[
            dict(item) for item in raw.get("integrity_failures", []) if isinstance(item, dict)
        ],
    )


def _resume_start_index(stage: str | None, chapter_count: int, use_review_repair: bool) -> int:
    if chapter_count == 0 or not use_review_repair:
        return chapter_count
    written_stage = f"chapter_{chapter_count}_written"
    reviewed_stage = f"chapter_{chapter_count}_reviewed"
    if stage == written_stage:
        return chapter_count - 1
    if stage in {"plan", reviewed_stage}:
        return chapter_count
    raise ValueError(f"checkpoint stage {stage!r} is inconsistent with {chapter_count} saved chapters")


def _accept_local_repair(
    task: CourseTask,
    before_markdown: str,
    candidate_markdown: str,
    before_verification: Any,
    after_verification: Any,
    *,
    validation_passed: bool = False,
    completion_metadata: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Accept only structurally valid candidates that pass Reviewer-2."""
    precheck_reason = _repair_candidate_precheck_reason(
        task,
        candidate_markdown,
        completion_metadata=completion_metadata,
    )
    if precheck_reason is not None:
        return False, precheck_reason
    if not validation_passed:
        return False, "repair_validation_failed"
    return True, None


def _repair_candidate_precheck_reason(
    task: CourseTask,
    candidate_markdown: str,
    *,
    completion_metadata: dict[str, Any] | None = None,
) -> str | None:
    if task.chapter_min_characters and len(candidate_markdown) < task.chapter_min_characters:
        return "candidate_below_chapter_minimum"
    if len(candidate_markdown) > MAX_REPAIR_CHAPTER_CHARACTERS:
        return "candidate_above_runaway_limit"
    stripped = candidate_markdown.strip()
    if not stripped.startswith("#"):
        return "candidate_missing_markdown_heading"
    if stripped.count("```") % 2:
        return "candidate_unclosed_code_fence"
    metadata = completion_metadata or {}
    finish_reason = str(metadata.get("finish_reason") or "").lower()
    if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
        return "candidate_truncated"
    if str(metadata.get("status") or "").lower() == "incomplete":
        return "candidate_truncated"
    return None


def _repair_markdown_identical(source_markdown: str, candidate_markdown: str) -> bool:
    def normalize(value: str) -> str:
        return "\n".join(line.rstrip() for line in value.strip().splitlines())

    return normalize(source_markdown) == normalize(candidate_markdown)


def _repair_acceptance_basis(
    before_verification: Any,
    after_verification: Any,
    *,
    semantic_improved: bool,
) -> str:
    if repair_improved(before_verification, after_verification):
        return "reviewer_validation_and_contract_improvement"
    if semantic_improved:
        return "reviewer_confirmed_semantic_improvement"
    return "reviewer_validation_passed"


def _program_hard_metric_issues(task: CourseTask, markdown: str) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    if task.chapter_min_characters and len(markdown) < task.chapter_min_characters:
        issues.append(
            ReviewIssue(
                id=f"D{len(issues) + 1}",
                severity="high",
                category="length",
                message=(
                    f"Chapter has {len(markdown)} characters, below the required minimum of "
                    f"{task.chapter_min_characters}."
                ),
                suggestion="Expand the complete replacement chapter until it meets the required minimum length.",
            )
        )
    if len(markdown) > MAX_REPAIR_CHAPTER_CHARACTERS:
        issues.append(
            ReviewIssue(
                id=f"D{len(issues) + 1}",
                severity="high",
                category="length",
                message=(
                    f"Chapter has {len(markdown)} characters, above the runaway limit of "
                    f"{MAX_REPAIR_CHAPTER_CHARACTERS}."
                ),
                suggestion="Return a complete chapter within the configured maximum length.",
            )
        )
    stripped = markdown.strip()
    if not stripped.startswith("#"):
        issues.append(
            ReviewIssue(
                id=f"D{len(issues) + 1}",
                severity="high",
                category="markdown_structure",
                message="Chapter does not begin with a Markdown heading.",
                suggestion="Return the complete chapter with a Markdown chapter heading.",
            )
        )
    if stripped.count("```") % 2:
        issues.append(
            ReviewIssue(
                id=f"D{len(issues) + 1}",
                severity="high",
                category="markdown_structure",
                message="Chapter contains an unclosed fenced code block.",
                suggestion="Close every fenced code block in the complete chapter.",
            )
        )
    return issues


def _residual_validation_issues(
    original_issues: list[ReviewIssue],
    validation: RepairValidationResult,
) -> list[ReviewIssue]:
    statuses = {resolution.issue_id: resolution.status for resolution in validation.resolutions}
    result = [issue for issue in original_issues if statuses.get(issue.id) != "resolved"]
    existing = {(issue.category, issue.message) for issue in result}
    next_index = 1
    for issue in validation.new_issues:
        if (issue.category, issue.message) in existing:
            continue
        while any(item.id == f"N{next_index}" for item in result):
            next_index += 1
        issue_id = issue.id
        if not issue_id or any(item.id == issue_id for item in result):
            issue_id = f"N{next_index}"
        result.append(
            ReviewIssue(
                id=issue_id,
                severity=issue.severity,
                category=issue.category,
                message=issue.message,
                suggestion=issue.suggestion,
            )
        )
        existing.add((issue.category, issue.message))
        next_index += 1
    return result


def _assign_reviewer_issue_ids(review: ReviewResult) -> ReviewResult:
    for index, issue in enumerate(review.issues, start=1):
        if not issue.id.strip():
            issue.id = f"R{index}"
    return review


def _assign_validation_new_issue_ids(validation: RepairValidationResult) -> RepairValidationResult:
    used: set[str] = set()
    next_index = 1
    for issue in validation.new_issues:
        issue_id = issue.id.strip()
        if not issue_id or issue_id in used:
            while f"N{next_index}" in used:
                next_index += 1
            issue_id = f"N{next_index}"
            next_index += 1
        issue.id = issue_id
        used.add(issue_id)
    return validation


def _reconcile_review_hard_metrics(
    task: CourseTask,
    markdown: str,
    review: ReviewResult,
) -> tuple[ReviewResult, list[dict[str, Any]]]:
    kept: list[ReviewIssue] = []
    overrides: list[dict[str, Any]] = []
    for issue in review.issues:
        if _program_overrides_length_issue(task, markdown, issue):
            overrides.append(_length_override_record("review_issue_removed", task, markdown, issue))
        else:
            kept.append(issue)
    if overrides and not kept:
        review.passed = True
    review.issues = kept
    return review, overrides


def _reconcile_validation_hard_metrics(
    task: CourseTask,
    candidate_markdown: str,
    original_issues: list[ReviewIssue],
    validation: RepairValidationResult,
    round_index: int,
) -> tuple[RepairValidationResult, list[dict[str, Any]]]:
    overrides: list[dict[str, Any]] = []
    original_by_id = {issue.id: issue for issue in original_issues}
    for resolution in validation.resolutions:
        issue = original_by_id.get(resolution.issue_id)
        if issue is None or not _program_overrides_length_issue(task, candidate_markdown, issue):
            continue
        if resolution.status != "resolved":
            overrides.append(
                {
                    **_length_override_record("resolution_forced_resolved", task, candidate_markdown, issue),
                    "round": round_index + 1,
                    "reviewer_status": resolution.status,
                    "reviewer_evidence": resolution.evidence,
                }
            )
            resolution.status = "resolved"
            resolution.evidence = (
                f"Program-authoritative character count is {len(candidate_markdown)}, meeting the "
                f"minimum {task.chapter_min_characters}."
            )

    kept_new_issues: list[ReviewIssue] = []
    for issue in validation.new_issues:
        if _program_overrides_length_issue(task, candidate_markdown, issue):
            overrides.append(
                {
                    **_length_override_record("new_issue_removed", task, candidate_markdown, issue),
                    "round": round_index + 1,
                }
            )
        else:
            kept_new_issues.append(issue)
    validation.new_issues = kept_new_issues

    if overrides:
        all_resolved = all(resolution.status == "resolved" for resolution in validation.resolutions)
        no_new_high = not any(issue.severity == "high" for issue in validation.new_issues)
        if all_resolved and no_new_high:
            validation.passed = True
    return validation, overrides


def _program_overrides_length_issue(task: CourseTask, markdown: str, issue: ReviewIssue) -> bool:
    minimum = task.chapter_min_characters
    if minimum and len(markdown) < minimum:
        return False
    category = issue.category.strip().lower().replace("-", "_").replace(" ", "_")
    message = issue.message.strip().lower()
    if category in {"length", "chapter_length", "review_skipped_length"}:
        return True
    explicit_markers = (
        "chapter_min_characters",
        "字数不足",
        "字符不足",
        "严重缺字",
        "低于最低要求",
        "below the required minimum",
        "below the minimum",
        "minimum character",
        "chapter has fewer",
    )
    return any(marker in message for marker in explicit_markers)


def _length_override_record(
    action: str,
    task: CourseTask,
    markdown: str,
    issue: ReviewIssue,
) -> dict[str, Any]:
    return {
        "stage": "program_hard_metric_reconciliation",
        "action": action,
        "metric": "chapter_characters",
        "character_count_scope": "complete_markdown_string",
        "chapter_characters": len(markdown),
        "chapter_min_characters": task.chapter_min_characters,
        "length_passed": not task.chapter_min_characters or len(markdown) >= task.chapter_min_characters,
        "reviewer_issue": to_plain(issue),
        "reason": "program-owned hard metric overrides reviewer length estimate",
    }


def _validate_dispute_response(
    disputed_issues: list[ReviewIssue],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    responses = payload.get("responses", [])
    if not isinstance(responses, list) or not all(isinstance(item, dict) for item in responses):
        raise ValueError("repair dispute response must contain a responses list")
    expected_ids = [issue.id for issue in disputed_issues]
    actual_ids = [str(item.get("issue_id", "")) for item in responses]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
        raise ValueError("repair dispute response must address every disputed high issue exactly once")
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("repair dispute response contains duplicate issue IDs")
    if any(item.get("position") not in {"accept", "rebut"} for item in responses):
        raise ValueError("repair dispute position must be accept or rebut")
    return responses


def _validate_repair_validation(original_issues: list[ReviewIssue], validation: RepairValidationResult) -> None:
    expected_ids = [issue.id for issue in original_issues]
    actual_ids = [resolution.issue_id for resolution in validation.resolutions]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
        raise ValueError("repair validation must resolve every original issue exactly once")
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("repair validation contains duplicate issue IDs")


def _normalize_contract_transitions(
    raw_chapters: list[Any],
    contracts: list[Any],
) -> None:
    """Make chapter order authoritative and keep long-range links in prepares_for."""
    for raw, contract in zip(raw_chapters, contracts):
        if contract is None or not isinstance(raw, dict):
            continue
        plan = plan_from_payload(raw)
        if plan.connection_from_previous.strip():
            contract.bridge_from_previous = plan.connection_from_previous.strip()
        immediate_transition = plan.setup_for_next.strip()
        if not immediate_transition:
            continue
        prior_bridge = contract.bridge_to_next.strip()
        if prior_bridge and prior_bridge != immediate_transition and prior_bridge not in contract.prepares_for:
            contract.prepares_for.append(prior_bridge)
        contract.bridge_to_next = immediate_transition
        contract.summary_for_next = immediate_transition


def _finalize_chapter_integrity(chapter: ChapterOutput) -> None:
    if chapter.integrity_status == "needs_adjudication":
        return
    failures: list[dict[str, Any]] = []
    if chapter.review is None:
        failures.append({"type": "review_missing"})
    else:
        high_ids = [issue.id for issue in chapter.review.issues if issue.severity == "high"]
        if high_ids:
            failures.append({"type": "unresolved_high_issue", "issue_ids": high_ids})
    chapter.integrity_failures = failures
    if failures:
        chapter.integrity_status = "incomplete"
    elif chapter.review is not None and chapter.review.passed:
        chapter.integrity_status = "complete"
    else:
        chapter.integrity_status = "complete_with_warnings"


def _raise_for_blocking_chapter(chapter: ChapterOutput, chapter_number: int) -> None:
    if chapter.integrity_status not in {"incomplete", "needs_adjudication"}:
        return
    reason = "needs adjudication" if chapter.integrity_status == "needs_adjudication" else "retains unresolved high issues"
    raise ChapterIntegrityError(
        f"chapter {chapter_number} {reason}; the reviewed checkpoint was saved and course generation stopped"
    )


def _semantic_repair_improved(original_issues: list[ReviewIssue], validation: RepairValidationResult) -> bool:
    if not _repair_validation_passes_gate(original_issues, validation):
        return False
    resolutions = {resolution.issue_id: resolution.status for resolution in validation.resolutions}
    meaningful = [issue for issue in original_issues if issue.severity in {"high", "medium"}]
    if not any(resolutions.get(issue.id) == "resolved" for issue in meaningful):
        return False
    return True


def _repair_validation_passes_gate(
    original_issues: list[ReviewIssue],
    validation: RepairValidationResult,
) -> bool:
    """Derive semantic acceptance from issue severity, not the reviewer's summary flag."""
    resolutions = {resolution.issue_id: resolution.status for resolution in validation.resolutions}
    if any(
        resolutions.get(issue.id) != "resolved"
        for issue in original_issues
        if issue.severity == "high"
    ):
        return False
    return not any(issue.severity == "high" for issue in validation.new_issues)


def _review_result_after_repair(original_issues: list[ReviewIssue], validation: RepairValidationResult) -> ReviewResult:
    statuses = {resolution.issue_id: resolution.status for resolution in validation.resolutions}
    unresolved = [issue for issue in original_issues if statuses.get(issue.id) != "resolved"]
    new_issues = [
        ReviewIssue(
            id=issue.id or f"N{index}",
            severity=issue.severity,
            category=issue.category,
            message=issue.message,
            suggestion=issue.suggestion,
        )
        for index, issue in enumerate(validation.new_issues, start=1)
    ]
    residual_issues = [*unresolved, *new_issues]
    return ReviewResult(
        passed=not residual_issues,
        issues=residual_issues,
        summary=validation.summary,
    )


def _course_integrity_failures(
    task: CourseTask,
    chapters: list[ChapterOutput],
    *,
    require_review: bool,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(chapters) != task.chapter_count:
        failures.append(
            {
                "type": "chapter_count",
                "expected": task.chapter_count,
                "actual": len(chapters),
            }
        )
    for index, chapter in enumerate(chapters, start=1):
        characters = len(chapter.content_markdown)
        if task.chapter_min_characters and characters < task.chapter_min_characters:
            failures.append(
                {
                    "type": "chapter_below_minimum",
                    "chapter": index,
                    "characters": characters,
                    "minimum": task.chapter_min_characters,
                }
            )
        if characters > MAX_REPAIR_CHAPTER_CHARACTERS:
            failures.append(
                {
                    "type": "chapter_above_maximum",
                    "chapter": index,
                    "characters": characters,
                    "maximum": MAX_REPAIR_CHAPTER_CHARACTERS,
                }
            )
        if not require_review:
            continue
        if chapter.review is None:
            failures.append({"type": "review_missing", "chapter": index})
            continue
        if any(issue.severity == "high" for issue in chapter.review.issues):
            failures.append({"type": "unresolved_high_issue", "chapter": index})
    return failures


def target_topics_from_task(task: CourseTask) -> list[str]:
    concepts = task.metadata.get("must_cover_concepts", [])
    if isinstance(concepts, list):
        return [str(item) for item in concepts]
    return []


def target_aliases_from_task(task: CourseTask) -> dict[str, list[str]]:
    aliases = task.metadata.get("concept_aliases", {})
    if not isinstance(aliases, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for key, values in aliases.items():
        if isinstance(values, list):
            normalized[str(key)] = [str(item) for item in values]
        elif values:
            normalized[str(key)] = [str(values)]
    return normalized


def build_json_repair_prompt(raw: str, error: str) -> str:
    return f"""JSON_REPAIR

The following text was intended to be one JSON object, but parsing failed.
Repair only JSON syntax. Preserve all fields, values, arrays, and object structure as much as possible.
Move comments or parenthetical notes into string values when needed.
Return valid JSON only.

PARSE_ERROR:
{error}

INVALID_JSON_TEXT:
{raw}"""
