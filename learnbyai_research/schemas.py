from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Literal


MaterialRole = Literal["requirements", "reference", "style"]
Difficulty = Literal["intro", "intermediate", "research"]
LearningMode = Literal["standard", "project", "exercise", "case"]
PipelineName = Literal[
    "direct_prompt",
    "single_agent",
    "full",
    "no_contract",
    "no_material_routing",
    "no_review_repair",
]

DEFAULT_CHAPTER_MIN_CHARACTERS = 8_000
DEFAULT_CHAPTER_TARGET_CHARACTERS = 10_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Material:
    name: str
    text: str
    role: MaterialRole | None = None
    source: str | None = None


@dataclass
class CourseTask:
    id: str
    topic: str
    goal: str
    background: str
    chapter_count: int = 6
    chapter_min_characters: int = DEFAULT_CHAPTER_MIN_CHARACTERS
    chapter_target_characters: int = DEFAULT_CHAPTER_TARGET_CHARACTERS
    difficulty: Difficulty = "intermediate"
    learning_mode: LearningMode = "standard"
    preference: str = ""
    styles: list[str] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def chapter_pre_review_target(task: CourseTask) -> int:
    """Return the optional instructional-depth floor used before review."""
    minimum = task.chapter_min_characters
    configured = task.metadata.get("pre_review_target_characters")
    if configured is None:
        return minimum
    target = int(configured)
    if target < minimum:
        raise ValueError("pre_review_target_characters must be at least chapter_min_characters")
    return target


@dataclass
class RoutedMaterials:
    requirements: str = ""
    reference: str = ""
    style: str = ""
    assignments: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StudyTime:
    reading_minutes: int = 120
    exercise_minutes: int = 60
    practice_minutes: int = 60
    extension_minutes: int = 30


@dataclass
class ChapterPlan:
    title: str
    description: str
    purpose: str
    connection_from_previous: str = ""
    setup_for_next: str = ""
    depth: Literal["core", "normal", "light"] = "normal"
    time: StudyTime = field(default_factory=StudyTime)


@dataclass
class ChapterContract:
    chapter_title: str
    required_topics: list[str] = field(default_factory=list)
    introduced_concepts: list[str] = field(default_factory=list)
    prerequisite_concepts: list[str] = field(default_factory=list)
    bridge_from_previous: str = ""
    bridge_to_next: str = ""
    forbidden_early_topics: list[str] = field(default_factory=list)
    required_examples: list[str] = field(default_factory=list)
    required_formulas: list[str] = field(default_factory=list)
    prepares_for: list[str] = field(default_factory=list)
    assessment_targets: list[str] = field(default_factory=list)
    summary_for_next: str = ""


@dataclass
class CourseBible:
    target_learner: str = ""
    final_outcomes: list[str] = field(default_factory=list)
    teaching_style: str = ""
    prerequisites: list[str] = field(default_factory=list)
    global_narrative: str = ""
    terminology: list[dict[str, str]] = field(default_factory=list)
    chapter_dependencies: list[dict[str, Any]] = field(default_factory=list)
    chapter_contracts: list[ChapterContract] = field(default_factory=list)


@dataclass
class ReviewIssue:
    severity: Literal["low", "medium", "high"]
    category: str
    message: str
    suggestion: str = ""
    id: str = ""


@dataclass
class ReviewResult:
    passed: bool
    issues: list[ReviewIssue] = field(default_factory=list)
    summary: str = ""


RepairResolutionStatus = Literal["resolved", "partial", "unresolved"]


@dataclass
class RepairIssueResolution:
    issue_id: str
    status: RepairResolutionStatus
    evidence: str = ""


@dataclass
class RepairValidationResult:
    passed: bool
    resolutions: list[RepairIssueResolution] = field(default_factory=list)
    new_issues: list[ReviewIssue] = field(default_factory=list)
    summary: str = ""


@dataclass
class ChapterOutput:
    plan: ChapterPlan
    contract: ChapterContract | None = None
    content_markdown: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    post_repair_verification: dict[str, Any] = field(default_factory=dict)
    review: ReviewResult | None = None
    review_error: dict[str, Any] = field(default_factory=dict)
    review_reconciliation: list[dict[str, Any]] = field(default_factory=list)
    repair_history: list[dict[str, Any]] = field(default_factory=list)
    length_enrichment_history: list[dict[str, Any]] = field(default_factory=list)
    integrity_status: str = "pending"
    integrity_failures: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CourseOutput:
    task: CourseTask
    pipeline: str
    course_bible: CourseBible | None = None
    chapters: list[ChapterOutput] = field(default_factory=list)
    routed_materials: RoutedMaterials | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PairwiseJudgment:
    task_id: str
    judge_model: str
    system_a: str
    system_b: str
    winner: Literal["A", "B", "Tie"]
    dimensions: dict[str, Literal["A", "B", "Tie"]] = field(default_factory=dict)
    dimension_evidence: dict[str, dict[str, str]] = field(default_factory=dict)
    rationale: str = ""
    presentation_order: Literal["A_then_B", "B_then_A"] = "A_then_B"
    completion_history: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float | None = None
    created_at: str = field(default_factory=utc_now)


def to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_plain(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    return value


def material_from_dict(data: dict[str, Any]) -> Material:
    return Material(
        name=str(data.get("name", "material")),
        text=str(data.get("text", "")),
        role=data.get("role"),
        source=data.get("source"),
    )


def task_from_dict(data: dict[str, Any]) -> CourseTask:
    chapter_min_characters = int(data.get("chapter_min_characters", DEFAULT_CHAPTER_MIN_CHARACTERS))
    chapter_target_characters = int(data.get("chapter_target_characters", DEFAULT_CHAPTER_TARGET_CHARACTERS))
    if chapter_min_characters < 0 or chapter_target_characters < 0:
        raise ValueError("chapter character budgets must be non-negative")
    if chapter_min_characters and chapter_target_characters and chapter_target_characters < chapter_min_characters:
        raise ValueError("chapter_target_characters must be greater than or equal to chapter_min_characters")
    return CourseTask(
        id=str(data["id"]),
        topic=str(data["topic"]),
        goal=str(data["goal"]),
        background=str(data.get("background", "")),
        chapter_count=int(data.get("chapter_count", 6)),
        chapter_min_characters=chapter_min_characters,
        chapter_target_characters=chapter_target_characters,
        difficulty=data.get("difficulty", "intermediate"),
        learning_mode=data.get("learning_mode", "standard"),
        preference=str(data.get("preference", "")),
        styles=list(data.get("styles", [])),
        materials=[material_from_dict(item) for item in data.get("materials", [])],
        metadata=dict(data.get("metadata", {})),
    )
