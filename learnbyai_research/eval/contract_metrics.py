from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_generation_outputs(paths: list[str]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for path in paths:
        source = Path(path)
        if source.is_dir():
            for child in sorted(source.rglob("*.json")):
                outputs.append(json.loads(child.read_text(encoding="utf-8")))
        else:
            outputs.append(json.loads(source.read_text(encoding="utf-8")))
    return outputs


def aggregate_contract_metrics(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for output in outputs:
        grouped[str(output.get("pipeline", "unknown"))].append(output)

    rows: list[dict[str, Any]] = []
    for pipeline, items in sorted(grouped.items()):
        chapter_total = 0
        satisfied_total = 0
        score_total = 0.0
        violation_score_total = 0
        coverage_totals = {
            "topic": [0, 0],
            "example": [0, 0],
            "formula": [0, 0],
            "assessment": [0, 0],
            "preparation": [0, 0],
            "bridge": [0, 0],
        }
        sequence_outputs = 0
        sequence_satisfied = 0
        sequence_target_rate_total = 0.0
        sequence_violation_score_total = 0
        by_type: Counter[str] = Counter()
        by_severity: Counter[str] = Counter()
        sequence_by_type: Counter[str] = Counter()
        for output in items:
            metadata = output.get("metadata", {})
            summary = metadata.get("contract_verification", {})
            chapter_total += int(summary.get("chapters", 0))
            satisfied_total += int(summary.get("satisfied_chapters", 0))
            score_total += float(summary.get("avg_score", 0.0)) * int(summary.get("chapters", 0))
            violation_score_total += int(summary.get("violation_score", 0))
            for name, counts in summary.get("coverage_counts", {}).items():
                if name in coverage_totals:
                    coverage_totals[name][0] += int(counts.get("found", 0))
                    coverage_totals[name][1] += int(counts.get("total", 0))
            by_type.update(summary.get("violations_by_type", {}))
            by_severity.update(summary.get("violations_by_severity", {}))
            sequence = metadata.get("contract_sequence")
            if sequence:
                sequence_outputs += 1
                sequence_satisfied += 1 if sequence.get("satisfied") else 0
                sequence_target_rate_total += float(sequence.get("target_coverage_rate", 0.0))
                for violation in sequence.get("violations", []):
                    sequence_violation_score_total += severity_weight(str(violation.get("severity", "low")))
                    sequence_by_type[str(violation.get("type", "unknown"))] += 1
        denominator = max(1, chapter_total)
        sequence_denominator = max(1, sequence_outputs)
        rows.append(
            {
                "pipeline": pipeline,
                "outputs": len(items),
                "chapters": chapter_total,
                "satisfied_chapters": satisfied_total,
                "satisfaction_rate": round(satisfied_total / denominator, 4),
                "avg_score": round(score_total / denominator, 4),
                "violation_score": violation_score_total,
                "topic_coverage_rate": coverage_rate(coverage_totals["topic"]),
                "example_coverage_rate": coverage_rate(coverage_totals["example"]),
                "formula_coverage_rate": coverage_rate(coverage_totals["formula"]),
                "assessment_coverage_rate": coverage_rate(coverage_totals["assessment"]),
                "preparation_coverage_rate": coverage_rate(coverage_totals["preparation"]),
                "bridge_coverage_rate": coverage_rate(coverage_totals["bridge"]),
                "sequence_outputs": sequence_outputs,
                "sequence_satisfaction_rate": round(sequence_satisfied / sequence_denominator, 4),
                "sequence_target_coverage_rate": round(sequence_target_rate_total / sequence_denominator, 4),
                "sequence_violation_score": sequence_violation_score_total,
                "violations_by_type": json.dumps(dict(by_type), ensure_ascii=False, sort_keys=True),
                "violations_by_severity": json.dumps(dict(by_severity), ensure_ascii=False, sort_keys=True),
                "sequence_violations_by_type": json.dumps(dict(sequence_by_type), ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def write_contract_metrics_csv(path: str, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pipeline",
        "outputs",
        "chapters",
        "satisfied_chapters",
        "satisfaction_rate",
        "avg_score",
        "violation_score",
        "topic_coverage_rate",
        "example_coverage_rate",
        "formula_coverage_rate",
        "assessment_coverage_rate",
        "preparation_coverage_rate",
        "bridge_coverage_rate",
        "sequence_outputs",
        "sequence_satisfaction_rate",
        "sequence_target_coverage_rate",
        "sequence_violation_score",
        "violations_by_type",
        "violations_by_severity",
        "sequence_violations_by_type",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def severity_weight(severity: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(severity, 1)


def coverage_rate(bucket: list[int]) -> float:
    found, total = bucket
    if total == 0:
        return 1.0
    return round(found / total, 4)
