from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .schema import ContractViolation, VerificationResult


SEVERITY_WEIGHTS = {"high": 3, "medium": 2, "low": 1}


def violation_score(violations: Iterable[ContractViolation]) -> int:
    return sum(SEVERITY_WEIGHTS.get(item.severity, 1) for item in violations)


def summarize_verifications(results: Iterable[VerificationResult]) -> dict[str, object]:
    items = list(results)
    if not items:
        return {
            "chapters": 0,
            "satisfaction_rate": 1.0,
            "avg_score": 1.0,
            "violation_score": 0,
            "topic_coverage_rate": 1.0,
            "example_coverage_rate": 1.0,
            "formula_coverage_rate": 1.0,
            "assessment_coverage_rate": 1.0,
            "preparation_coverage_rate": 1.0,
            "bridge_coverage_rate": 1.0,
            "coverage_counts": {},
        }
    by_type: Counter[str] = Counter()
    by_severity: Counter[str] = Counter()
    coverage = {
        "topic": [0, 0],
        "example": [0, 0],
        "formula": [0, 0],
        "assessment": [0, 0],
        "preparation": [0, 0],
        "bridge": [0, 0],
    }
    for result in items:
        for violation in result.violations:
            by_type[violation.type] += 1
            by_severity[violation.severity] += 1
        add_coverage(coverage["topic"], len(result.evidence.covered_topics), len(result.evidence.missing_topics))
        add_coverage(coverage["example"], len(result.evidence.examples_found), len(result.evidence.examples_missing))
        add_coverage(coverage["formula"], len(result.evidence.formulas_found), len(result.evidence.formulas_missing))
        add_coverage(
            coverage["assessment"],
            len(result.evidence.assessment_targets_found),
            len(result.evidence.assessment_targets_missing),
        )
        add_coverage(coverage["preparation"], len(result.evidence.prepares_for_found), len(result.evidence.prepares_for_missing))
        add_coverage(coverage["bridge"], len(result.evidence.bridge_evidence_found), len(result.evidence.bridge_evidence_missing))
    return {
        "chapters": len(items),
        "satisfied_chapters": sum(1 for item in items if item.satisfied),
        "satisfaction_rate": round(sum(1 for item in items if item.satisfied) / len(items), 4),
        "avg_score": round(sum(item.score for item in items) / len(items), 4),
        "violation_score": sum(item.violation_score for item in items),
        "topic_coverage_rate": coverage_rate(coverage["topic"]),
        "example_coverage_rate": coverage_rate(coverage["example"]),
        "formula_coverage_rate": coverage_rate(coverage["formula"]),
        "assessment_coverage_rate": coverage_rate(coverage["assessment"]),
        "preparation_coverage_rate": coverage_rate(coverage["preparation"]),
        "bridge_coverage_rate": coverage_rate(coverage["bridge"]),
        "coverage_counts": coverage_counts(coverage),
        "violations_by_type": dict(by_type),
        "violations_by_severity": dict(by_severity),
    }


def repair_improved(before: VerificationResult, after: VerificationResult) -> bool:
    if after.violation_score < before.violation_score:
        return True
    if after.violation_score == before.violation_score and after.score > before.score:
        return True
    if before.violation_score == 0 and after.violation_score == 0:
        return True
    return False


def add_coverage(bucket: list[int], found: int, missing: int) -> None:
    bucket[0] += found
    bucket[1] += found + missing


def coverage_rate(bucket: list[int]) -> float:
    found, total = bucket
    if total == 0:
        return 1.0
    return round(found / total, 4)


def coverage_counts(coverage: dict[str, list[int]]) -> dict[str, dict[str, int]]:
    return {name: {"found": bucket[0], "total": bucket[1]} for name, bucket in coverage.items()}
