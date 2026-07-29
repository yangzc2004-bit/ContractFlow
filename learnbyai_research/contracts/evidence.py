from __future__ import annotations

import re
from collections.abc import Iterable

from ..schemas import ChapterContract
from .schema import ContractEvidence


def extract_contract_evidence(content: str, contract: ChapterContract) -> ContractEvidence:
    text = normalize(content)
    covered = [item for item in contract.required_topics if item_present(item, text)]
    missing = [item for item in contract.required_topics if item not in covered]
    examples = [item for item in contract.required_examples if item_present(item, text)]
    missing_examples = [item for item in contract.required_examples if item not in examples]
    formulas = [item for item in contract.required_formulas if formula_present(item, text)]
    missing_formulas = [item for item in contract.required_formulas if item not in formulas]
    premature = [item for item in contract.forbidden_early_topics if expanded_topic_present(item, text)]
    introduced = [item for item in contract.introduced_concepts if item_present(item, text)]
    prerequisites = [item for item in contract.prerequisite_concepts if item_present(item, text)]
    assessments = [item for item in contract.assessment_targets if assessment_present(item, text)]
    missing_assessments = [item for item in contract.assessment_targets if item not in assessments]
    preparations = [item for item in contract.prepares_for if preparation_present(item, text)]
    missing_preparations = [item for item in contract.prepares_for if item not in preparations]
    bridge_found, bridge_missing = bridge_evidence(content, contract)
    return ContractEvidence(
        covered_topics=covered,
        missing_topics=missing,
        examples_found=examples,
        examples_missing=missing_examples,
        formulas_found=formulas,
        formulas_missing=missing_formulas,
        premature_topics_found=premature,
        introduced_concepts_found=introduced,
        prerequisite_concepts_found=prerequisites,
        assessment_targets_found=assessments,
        assessment_targets_missing=missing_assessments,
        prepares_for_found=preparations,
        prepares_for_missing=missing_preparations,
        bridge_evidence_found=bridge_found,
        bridge_evidence_missing=bridge_missing,
    )


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower())


def item_present(item: str, normalized_text: str) -> bool:
    candidates = search_candidates(item)
    return any(candidate and candidate in normalized_text for candidate in candidates)


def formula_present(item: str, normalized_text: str) -> bool:
    candidates = set(search_candidates(item))
    compact = re.sub(r"\s+", "", item.lower())
    symbolic = re.sub(r"[^a-z0-9_+\-*/=<>|^(){}\[\]\\\\]+", "", compact)
    candidates.update(value for value in [compact, symbolic] if len(value) >= 2)
    text_compact = re.sub(r"\s+", "", normalized_text)
    return any(candidate and (candidate in normalized_text or candidate in text_compact) for candidate in candidates)


def assessment_present(item: str, normalized_text: str) -> bool:
    if item_present(item, normalized_text):
        return True
    assessment_markers = [
        "exercise",
        "practice",
        "question",
        "problem",
        "quiz",
        "assessment",
        "练习",
        "习题",
        "问题",
        "思考",
        "测验",
        "目标",
    ]
    if not any(marker in normalized_text for marker in assessment_markers):
        return False
    return any(candidate in normalized_text for candidate in key_candidates(item))


def preparation_present(item: str, normalized_text: str) -> bool:
    if item_present(item, normalized_text):
        return True
    preparation_markers = [
        "prepare",
        "next",
        "later",
        "subsequent",
        "sets up",
        "foundation for",
        "后续",
        "下一章",
        "之后",
        "铺垫",
        "准备",
    ]
    return any(marker in normalized_text for marker in preparation_markers) and any(
        candidate in normalized_text for candidate in key_candidates(item)
    )


def bridge_evidence(content: str, contract: ChapterContract) -> tuple[list[str], list[str]]:
    text = normalize(content)
    beginning = normalize(content[:1500])
    ending = normalize(content[-2000:])
    checks = [
        ("bridge_from_previous", contract.bridge_from_previous, beginning, previous_bridge_present),
        ("bridge_to_next", contract.bridge_to_next, ending, next_bridge_present),
        ("summary_for_next", contract.summary_for_next, ending, next_bridge_present),
    ]
    found: list[str] = []
    missing: list[str] = []
    for label, requirement, scope, predicate in checks:
        if not requirement.strip():
            continue
        if item_present(requirement, text) or predicate(requirement, scope):
            found.append(label)
        else:
            missing.append(label)
    return found, missing


def previous_bridge_present(requirement: str, normalized_text: str) -> bool:
    markers = ["previous", "prior", "earlier", "last chapter", "recall", "review", "before", "前面", "上一章", "回顾", "此前"]
    return any(marker in normalized_text for marker in markers) and any(candidate in normalized_text for candidate in key_candidates(requirement))


def next_bridge_present(requirement: str, normalized_text: str) -> bool:
    markers = ["next", "later", "subsequent", "prepare", "sets up", "summary", "下一章", "后续", "之后", "总结", "铺垫"]
    return any(marker in normalized_text for marker in markers) and any(candidate in normalized_text for candidate in key_candidates(requirement))


def expanded_topic_present(item: str, normalized_text: str) -> bool:
    # Forbidden topics require stricter matching than coverage obligations. Splitting
    # "policy gradient" into "policy" would falsely flag ordinary policy discussion.
    for candidate in forbidden_topic_candidates(item):
        for match in re.finditer(re.escape(candidate), normalized_text):
            sentence = sentence_window(normalized_text, match.start(), match.end())
            if not forward_pointer_present(candidate, sentence):
                return True
    return False


def forbidden_topic_candidates(item: str) -> list[str]:
    raw = normalize(item)
    candidates = {raw}
    # Parenthetical forms are explicit aliases, unlike individual words in a phrase.
    candidates.update(part.strip() for part in re.findall(r"[（(]([^()（）]+)[）)]", raw))
    candidates.update(part.strip() for part in re.split(r"[,，;；/、]", raw))
    return sorted((candidate for candidate in candidates if len(candidate) >= 3), key=len, reverse=True)


def sentence_window(text: str, start: int, end: int) -> str:
    left = max(text.rfind(marker, 0, start) for marker in ".!?。！？") + 1
    right_candidates = [position for marker in ".!?。！？" if (position := text.find(marker, end)) >= 0]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left:right]


def forward_pointer_present(topic: str, sentence: str) -> bool:
    markers = "后续|之后|下一章|后面|拓展|later|next chapter|subsequent|future"
    return bool(
        re.search(rf"{re.escape(topic)}[^。.!?]{{0,180}}({markers})", sentence)
        or re.search(rf"({markers})[^。.!?]{{0,180}}{re.escape(topic)}", sentence)
    )


def search_candidates(item: str) -> list[str]:
    raw = normalize(item)
    parts = {raw}
    parts.update(split_aliases(raw))
    parts.update(token for token in re.split(r"[,，;；/、()（）\s]+", raw) if len(token) >= 2)
    return sorted(parts, key=len, reverse=True)


def key_candidates(item: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "as",
        "for",
        "from",
        "in",
        "into",
        "of",
        "or",
        "the",
        "to",
        "with",
        "write",
        "explain",
        "show",
        "describe",
    }
    return [
        candidate
        for candidate in search_candidates(item)
        if len(candidate) >= 4 and candidate not in stopwords
    ]


def split_aliases(value: str) -> Iterable[str]:
    aliases = re.findall(r"[a-z][a-z0-9_\- ]{1,40}", value)
    for alias in aliases:
        cleaned = alias.strip()
        if len(cleaned) >= 2:
            yield cleaned
