from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ViolationType = Literal[
    "missing_required_topic",
    "missing_required_example",
    "missing_required_formula",
    "missing_assessment_target",
    "missing_preparation_evidence",
    "missing_bridge_evidence",
    "premature_topic",
    "missing_prerequisite",
    "title_mismatch",
    "missing_target_topic",
    "unmet_contract_prerequisite",
    "unknown_dependency",
    "dependency_cycle",
]

ViolationSeverity = Literal["low", "medium", "high"]


@dataclass
class ContractEvidence:
    covered_topics: list[str] = field(default_factory=list)
    missing_topics: list[str] = field(default_factory=list)
    examples_found: list[str] = field(default_factory=list)
    examples_missing: list[str] = field(default_factory=list)
    formulas_found: list[str] = field(default_factory=list)
    formulas_missing: list[str] = field(default_factory=list)
    premature_topics_found: list[str] = field(default_factory=list)
    introduced_concepts_found: list[str] = field(default_factory=list)
    prerequisite_concepts_found: list[str] = field(default_factory=list)
    assessment_targets_found: list[str] = field(default_factory=list)
    assessment_targets_missing: list[str] = field(default_factory=list)
    prepares_for_found: list[str] = field(default_factory=list)
    prepares_for_missing: list[str] = field(default_factory=list)
    bridge_evidence_found: list[str] = field(default_factory=list)
    bridge_evidence_missing: list[str] = field(default_factory=list)


@dataclass
class ContractViolation:
    type: ViolationType
    item: str
    severity: ViolationSeverity
    message: str
    suggestion: str = ""


@dataclass
class VerificationResult:
    satisfied: bool
    score: float
    violation_score: int
    violations: list[ContractViolation] = field(default_factory=list)
    evidence: ContractEvidence = field(default_factory=ContractEvidence)


@dataclass
class ContractSequenceResult:
    satisfied: bool
    target_coverage_rate: float
    missing_targets: list[str] = field(default_factory=list)
    violations: list[ContractViolation] = field(default_factory=list)
    introduced_by_chapter: dict[str, int] = field(default_factory=dict)
    dependency_edges: list[dict[str, str]] = field(default_factory=list)
