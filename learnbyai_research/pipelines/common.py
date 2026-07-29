from __future__ import annotations

from typing import Any

from ..schemas import (
    ChapterContract,
    ChapterPlan,
    CourseBible,
    RepairIssueResolution,
    RepairValidationResult,
    ReviewIssue,
    ReviewResult,
    StudyTime,
)


def bible_from_payload(payload: dict[str, Any]) -> CourseBible:
    raw = payload.get("course_bible", payload)
    return CourseBible(
        target_learner=str(raw.get("target_learner", "")),
        final_outcomes=[str(item) for item in raw.get("final_outcomes", [])],
        teaching_style=str(raw.get("teaching_style", "")),
        prerequisites=[str(item) for item in raw.get("prerequisites", [])],
        global_narrative=str(raw.get("global_narrative", "")),
        terminology=list(raw.get("terminology", [])),
        chapter_dependencies=list(raw.get("chapter_dependencies", [])),
    )


def plan_from_payload(raw: dict[str, Any]) -> ChapterPlan:
    time = raw.get("time", {})
    return ChapterPlan(
        title=str(raw.get("title", "Untitled Chapter")),
        description=str(raw.get("description", "")),
        purpose=str(raw.get("purpose", raw.get("description", ""))),
        connection_from_previous=str(raw.get("connection_from_previous", raw.get("connectionFromPrevious", ""))),
        setup_for_next=str(raw.get("setup_for_next", raw.get("setupForNext", ""))),
        depth=raw.get("depth", "normal") if raw.get("depth") in {"core", "normal", "light"} else "normal",
        time=StudyTime(
            reading_minutes=int(time.get("reading_minutes", time.get("readingMinutes", 120))),
            exercise_minutes=int(time.get("exercise_minutes", time.get("exerciseMinutes", 60))),
            practice_minutes=int(time.get("practice_minutes", time.get("practiceMinutes", 60))),
            extension_minutes=int(time.get("extension_minutes", time.get("extensionMinutes", 30))),
        ),
    )


def contract_from_payload(raw: dict[str, Any] | None, title: str) -> ChapterContract | None:
    if not raw:
        return None
    return ChapterContract(
        chapter_title=str(raw.get("chapter_title", raw.get("chapterTitle", title))),
        required_topics=[str(item) for item in raw.get("required_topics", raw.get("requiredTopics", []))],
        introduced_concepts=[str(item) for item in raw.get("introduced_concepts", raw.get("introducedConcepts", []))],
        prerequisite_concepts=[str(item) for item in raw.get("prerequisite_concepts", raw.get("prerequisiteConcepts", []))],
        bridge_from_previous=str(raw.get("bridge_from_previous", raw.get("bridgeFromPrevious", ""))),
        bridge_to_next=str(raw.get("bridge_to_next", raw.get("bridgeToNext", ""))),
        forbidden_early_topics=[str(item) for item in raw.get("forbidden_early_topics", raw.get("forbiddenEarlyTopics", []))],
        required_examples=[str(item) for item in raw.get("required_examples", raw.get("requiredExamples", []))],
        required_formulas=[str(item) for item in raw.get("required_formulas", raw.get("requiredFormulas", []))],
        prepares_for=[str(item) for item in raw.get("prepares_for", raw.get("preparesFor", []))],
        assessment_targets=[str(item) for item in raw.get("assessment_targets", raw.get("assessmentTargets", []))],
        summary_for_next=str(raw.get("summary_for_next", raw.get("summaryForNext", ""))),
    )


def review_from_payload(raw: dict[str, Any]) -> ReviewResult:
    return ReviewResult(
        passed=bool(raw.get("passed", False)),
        issues=[
            ReviewIssue(
                severity=item.get("severity", "low") if item.get("severity") in {"low", "medium", "high"} else "low",
                category=str(item.get("category", "")),
                message=str(item.get("message", "")),
                suggestion=str(item.get("suggestion", "")),
                id=str(item.get("id", "")),
            )
            for item in raw.get("issues", [])
            if isinstance(item, dict)
        ],
        summary=str(raw.get("summary", "")),
    )


def repair_validation_from_payload(raw: dict[str, Any]) -> RepairValidationResult:
    return RepairValidationResult(
        passed=bool(raw.get("passed", False)),
        resolutions=[
            RepairIssueResolution(
                issue_id=str(item.get("issue_id", "")),
                status=item.get("status", "unresolved")
                if item.get("status") in {"resolved", "partial", "unresolved"}
                else "unresolved",
                evidence=str(item.get("evidence", "")),
            )
            for item in raw.get("resolutions", [])
            if isinstance(item, dict)
        ],
        new_issues=[
            ReviewIssue(
                severity=item.get("severity", "low") if item.get("severity") in {"low", "medium", "high"} else "low",
                category=str(item.get("category", "")),
                message=str(item.get("message", "")),
                suggestion=str(item.get("suggestion", "")),
                id=str(item.get("id", "")),
            )
            for item in raw.get("new_issues", [])
            if isinstance(item, dict)
        ],
        summary=str(raw.get("summary", "")),
    )
