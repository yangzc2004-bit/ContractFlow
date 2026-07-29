from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..schemas import ChapterContract
from .evidence import item_present, normalize
from .schema import ContractSequenceResult, ContractViolation


def check_contract_sequence(
    contracts: Sequence[ChapterContract | None],
    *,
    target_topics: Sequence[str] | None = None,
    target_aliases: dict[str, Sequence[str]] | None = None,
    chapter_dependencies: Sequence[dict[str, Any]] | None = None,
) -> ContractSequenceResult:
    concrete = [contract for contract in contracts if contract is not None]
    introduced_by_chapter = build_introduced_map(concrete)
    violations: list[ContractViolation] = []

    missing_targets = missing_target_topics(concrete, target_topics or [], target_aliases=target_aliases)
    for target in missing_targets:
        violations.append(
            ContractViolation(
                type="missing_target_topic",
                item=target,
                severity="high",
                message=f"全书 contract 没有覆盖目标知识点：{target}。",
                suggestion=f"把 {target} 分配到某一章的 required_topics 或 introduced_concepts。",
            )
        )

    previous: set[str] = set()
    for index, contract in enumerate(concrete, start=1):
        current = set(contract.required_topics) | set(contract.introduced_concepts)
        previous_text = normalize(" ".join(previous))
        current_text = normalize(" ".join(current))
        for prerequisite in contract.prerequisite_concepts:
            if not item_present(prerequisite, previous_text) and not item_present(prerequisite, current_text):
                violations.append(
                    ContractViolation(
                        type="unmet_contract_prerequisite",
                        item=f"chapter {index}: {prerequisite}",
                        severity="medium",
                        message=f"第 {index} 章 contract 依赖 {prerequisite}，但前序章节未分配该概念。",
                        suggestion=f"将 {prerequisite} 放入更早章节，或在第 {index} 章显式安排 prerequisite review。",
                    )
                )
        previous.update(current)

    dependency_edges, dependency_violations = check_dependency_graph(chapter_dependencies or [])
    violations.extend(dependency_violations)

    target_total = len(target_topics or [])
    target_coverage_rate = 1.0 if target_total == 0 else round((target_total - len(missing_targets)) / target_total, 4)
    return ContractSequenceResult(
        satisfied=not violations,
        target_coverage_rate=target_coverage_rate,
        missing_targets=missing_targets,
        violations=violations,
        introduced_by_chapter=introduced_by_chapter,
        dependency_edges=dependency_edges,
    )


def build_introduced_map(contracts: Sequence[ChapterContract]) -> dict[str, int]:
    introduced: dict[str, int] = {}
    for index, contract in enumerate(contracts, start=1):
        for concept in [*contract.introduced_concepts, *contract.required_topics]:
            introduced.setdefault(concept, index)
    return introduced


def missing_target_topics(
    contracts: Sequence[ChapterContract],
    target_topics: Sequence[str],
    *,
    target_aliases: dict[str, Sequence[str]] | None = None,
) -> list[str]:
    if not target_topics:
        return []
    contract_text = normalize(
        " ".join(
            " ".join(
                [
                    contract.chapter_title,
                    *contract.required_topics,
                    *contract.introduced_concepts,
                    *contract.prerequisite_concepts,
                    *contract.prepares_for,
                    *contract.assessment_targets,
                ]
            )
            for contract in contracts
        )
    )
    aliases = target_aliases or {}
    return [target for target in target_topics if not target_present(target, aliases.get(target, []), contract_text)]


def target_present(target: str, aliases: Sequence[str], normalized_text: str) -> bool:
    return any(item_present(candidate, normalized_text) for candidate in [target, *aliases])


def check_dependency_graph(dependencies: Sequence[dict[str, Any]]) -> tuple[list[dict[str, str]], list[ContractViolation]]:
    titles = {normalize_title(str(item.get("chapter_title", ""))) for item in dependencies if item.get("chapter_title")}
    edges: list[dict[str, str]] = []
    violations: list[ContractViolation] = []
    graph: dict[str, list[str]] = {title: [] for title in titles}

    for item in dependencies:
        target = normalize_title(str(item.get("chapter_title", "")))
        if not target:
            continue
        for raw_source in item.get("depends_on", []):
            source_label = str(raw_source)
            source = normalize_title(source_label)
            edges.append({"source": source_label, "target": str(item.get("chapter_title", ""))})
            if source not in titles:
                violations.append(
                    ContractViolation(
                        type="unknown_dependency",
                        item=source_label,
                        severity="medium",
                        message=f"章节依赖引用了未知章节：{source_label}。",
                        suggestion="保持 chapter_dependencies 中的 depends_on 与章节标题一致。",
                    )
                )
                continue
            graph.setdefault(source, []).append(target)

    cycle = find_cycle(graph)
    if cycle:
        violations.append(
            ContractViolation(
                type="dependency_cycle",
                item=" -> ".join(cycle),
                severity="high",
                message=f"章节依赖图存在循环：{' -> '.join(cycle)}。",
                suggestion="调整章节顺序或依赖边，使课程依赖图成为 DAG。",
            )
        )
    return edges, violations


def find_cycle(graph: dict[str, list[str]]) -> list[str]:
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def dfs(node: str) -> list[str]:
        visiting.add(node)
        stack.append(node)
        for child in graph.get(node, []):
            if child in visiting:
                start = stack.index(child)
                return [*stack[start:], child]
            if child not in visited:
                found = dfs(child)
                if found:
                    return found
        visiting.remove(node)
        visited.add(node)
        stack.pop()
        return []

    for node in graph:
        if node not in visited:
            found = dfs(node)
            if found:
                return found
    return []


def normalize_title(value: str) -> str:
    return "".join(value.lower().split())
