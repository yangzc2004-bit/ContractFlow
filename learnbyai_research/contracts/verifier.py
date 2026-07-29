from __future__ import annotations

from collections.abc import Sequence

from ..schemas import ChapterContract
from .evidence import extract_contract_evidence
from .metrics import violation_score
from .schema import ContractViolation, VerificationResult


def build_introduced_prefixes(contracts: Sequence[ChapterContract | None]) -> list[set[str]]:
    prefixes: list[set[str]] = []
    seen: set[str] = set()
    for contract in contracts:
        prefixes.append(set(seen))
        if contract:
            seen.update(contract.introduced_concepts)
            seen.update(contract.required_topics)
    return prefixes


def verify_chapter_contract(
    content: str,
    contract: ChapterContract | None,
    *,
    title: str | None = None,
    introduced_before: set[str] | None = None,
) -> VerificationResult:
    if contract is None:
        return VerificationResult(satisfied=True, score=1.0, violation_score=0)

    evidence = extract_contract_evidence(content, contract)
    violations: list[ContractViolation] = []

    if title and normalize_title(title) != normalize_title(contract.chapter_title):
        violations.append(
            ContractViolation(
                type="title_mismatch",
                item=contract.chapter_title,
                severity="high",
                message=f"章节标题应为 {contract.chapter_title}，实际规划标题为 {title}。",
                suggestion="保持章节标题与 contract.chapter_title 完全一致。",
            )
        )

    for item in evidence.missing_topics:
        violations.append(
            ContractViolation(
                type="missing_required_topic",
                item=item,
                severity="high",
                message=f"缺少必讲知识点：{item}。",
                suggestion=f"在本章正文中加入对 {item} 的明确解释。",
            )
        )

    for item in evidence.examples_missing:
        violations.append(
            ContractViolation(
                type="missing_required_example",
                item=item,
                severity="medium",
                message=f"缺少必需例子或实践：{item}。",
                suggestion=f"补充一个围绕 {item} 的例子或练习。",
            )
        )

    for item in evidence.formulas_missing:
        violations.append(
            ContractViolation(
                type="missing_required_formula",
                item=item,
                severity="medium",
                message=f"缺少必需公式或推导：{item}。",
                suggestion=f"补充并解释公式 {item}。",
            )
        )

    for item in evidence.premature_topics_found:
        violations.append(
            ContractViolation(
                type="premature_topic",
                item=item,
                severity="high",
                message=f"提前展开了后续主题：{item}。",
                suggestion=f"删减 {item} 的细节，只保留必要的一句铺垫。",
            )
        )

    previous = introduced_before or set()
    for item in contract.prerequisite_concepts:
        if item not in previous and item not in evidence.prerequisite_concepts_found:
            violations.append(
                ContractViolation(
                    type="missing_prerequisite",
                    item=item,
                    severity="medium",
                    message=f"本章依赖的前置概念尚未在前文或本章中明确出现：{item}。",
                    suggestion=f"在进入本章核心内容前回顾或补充 {item}。",
                )
            )

    for item in evidence.assessment_targets_missing:
        violations.append(
            ContractViolation(
                type="missing_assessment_target",
                item=item,
                severity="low",
                message=f"缺少可观察的学习目标或练习目标：{item}。",
                suggestion=f"在练习、总结或自测中加入与 {item} 对应的任务。",
            )
        )

    for item in evidence.prepares_for_missing:
        violations.append(
            ContractViolation(
                type="missing_preparation_evidence",
                item=item,
                severity="low",
                message=f"缺少对后续内容的铺垫证据：{item}。",
                suggestion=f"在章节总结或过渡段说明本章如何为 {item} 做准备。",
            )
        )

    for item in evidence.bridge_evidence_missing:
        violations.append(
            ContractViolation(
                type="missing_bridge_evidence",
                item=item,
                severity="low",
                message=f"缺少章节衔接证据：{item}。",
                suggestion="在章节开头或结尾加入与相邻章节的明确过渡。",
            )
        )

    score = satisfaction_score(contract, violations)
    total = violation_score(violations)
    return VerificationResult(
        satisfied=not violations,
        score=score,
        violation_score=total,
        violations=violations,
        evidence=evidence,
    )


def satisfaction_score(contract: ChapterContract, violations: list[ContractViolation]) -> float:
    obligations = (
        len(contract.required_topics)
        + len(contract.required_examples)
        + len(contract.required_formulas)
        + len(contract.forbidden_early_topics)
        + len(contract.prerequisite_concepts)
        + len(contract.assessment_targets)
        + len(contract.prepares_for)
        + sum(1 for item in [contract.bridge_from_previous, contract.bridge_to_next, contract.summary_for_next] if item.strip())
    )
    if obligations == 0:
        return 1.0 if not violations else 0.0
    penalty = sum({"high": 1.0, "medium": 0.6, "low": 0.3}[item.severity] for item in violations)
    return max(0.0, round(1.0 - penalty / max(1.0, obligations), 4))


def normalize_title(value: str) -> str:
    return "".join(value.lower().split())
