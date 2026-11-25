from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from .io_utils import extract_json_object, read_json, write_json
from .llm import ChatModel, completion_history


CONTRACTFLOW_V2_VERSION = "contractflow_v2"
RuleScope = Literal["all", "single", "range", "every"]


@dataclass
class ContractRule:
    scope: RuleScope
    obligation: str
    index: int | None = None
    start: int | None = None
    end: int | None = None
    step: int | None = None
    derivation: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "scope": self.scope,
            "obligation": self.obligation,
        }
        if self.index is not None:
            payload["index"] = self.index
        if self.start is not None:
            payload["start"] = self.start
        if self.end is not None:
            payload["end"] = self.end
        if self.step is not None:
            payload["step"] = self.step
        if self.derivation:
            payload["derivation"] = self.derivation
        return payload


@dataclass
class UnitContract:
    index: int
    title: str
    min_words: int
    max_words: int
    obligations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "min_words": self.min_words,
            "max_words": self.max_words,
            "obligations": list(self.obligations),
        }


@dataclass
class DocumentContract:
    global_brief: str
    global_invariants: list[str]
    unit_label: str
    total_units: int
    rules: list[ContractRule]
    units: list[UnitContract]
    compiler_mode: str = "sparse_ir"

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_brief": self.global_brief,
            "global_invariants": list(self.global_invariants),
            "unit_label": self.unit_label,
            "total_units": self.total_units,
            "compiler_mode": self.compiler_mode,
            "rules": [rule.to_dict() for rule in self.rules],
            "units": [unit.to_dict() for unit in self.units],
        }


@dataclass
class FlowConfig:
    batch_size: int = 8
    plan_max_output_tokens: int = 2_048
    plan_retry_cap_tokens: int = 8_192
    generation_max_output_tokens: int = 6_144
    review_max_output_tokens: int = 4_096
    repair_max_output_tokens: int = 4_096
    max_singleton_retries: int = 2
    length_retry_ratio: float = 0.7
    rolling_state_chars: int = 800
    use_contracts: bool = True
    use_review_repair: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.length_retry_ratio <= 1.0:
            raise ValueError("length_retry_ratio must be between 0.0 and 1.0")


# ---------------------------------------------------------------------------
# Dynamic, bounded token budgets (generous enough to avoid truncation,
# tightly capped to prevent runaway repetition)
# ---------------------------------------------------------------------------

def effective_batch_size(
    requested_batch_size: int,
    max_words: int,
    generation_cap_tokens: int,
) -> int:
    per_unit_tokens = math.ceil(max(1, max_words) * 2.2) + 48
    available = max(1, generation_cap_tokens - 256)
    fitting = max(1, available // max(1, per_unit_tokens))
    return max(1, min(max(1, requested_batch_size), fitting))


def draft_token_budget(num_units: int, max_words: int, cap_tokens: int) -> int:
    estimated = math.ceil(max(1, num_units) * (max(1, max_words) * 2.2 + 48) + 256)
    return max(512, min(max(512, cap_tokens), estimated))


def review_token_budget(num_units: int, cap_tokens: int) -> int:
    estimated = max(1, num_units) * 256 + 512
    return max(1_024, min(max(1_024, cap_tokens), estimated))


def repair_token_budget(num_units: int, max_words: int, cap_tokens: int) -> int:
    estimated = math.ceil(max(1, num_units) * (max(1, max_words) * 2.4 + 48) + 256)
    return max(512, min(max(512, cap_tokens), estimated))


# ---------------------------------------------------------------------------
# Mode B: Sparse Declarative Contract Compiler + Universal 15-line Expander
# ---------------------------------------------------------------------------

def expand_contract_rules(
    total_units: int,
    rules: list[ContractRule],
) -> tuple[list[str], dict[int, list[str]]]:
    """Expand sparse declarative scope rules into global invariants and per-unit obligations."""
    global_invariants: list[str] = []
    obligations: dict[int, list[str]] = {index: [] for index in range(1, total_units + 1)}
    for rule in rules:
        text = re.sub(r"\s+", " ", rule.obligation).strip()
        if not text:
            continue
        if rule.scope == "all":
            if text not in global_invariants:
                global_invariants.append(text)
            continue
        targets: Iterable[int] = ()
        if rule.scope == "single" and rule.index is not None:
            targets = (rule.index,)
        elif rule.scope == "range" and rule.start is not None and rule.end is not None:
            lo, hi = (rule.start, rule.end) if rule.start <= rule.end else (rule.end, rule.start)
            targets = range(max(1, lo), min(total_units, hi) + 1)
        elif rule.scope == "every" and rule.start is not None and rule.step is not None and rule.step > 0:
            stop = min(total_units, rule.end) if rule.end is not None else total_units
            if rule.start <= stop:
                targets = range(rule.start, stop + 1, rule.step)
        valid_targets = [index for index in targets if 1 <= index <= total_units]
        if total_units > 1 and len(set(valid_targets)) == total_units:
            if text not in global_invariants:
                global_invariants.append(text)
            continue
        for index in valid_targets:
            if text not in obligations[index]:
                obligations[index].append(text)
    return global_invariants, obligations


def build_sparse_contract_prompt(
    prompt: str,
    *,
    total_units: int | None = None,
    unit_label: str | None = None,
) -> str:
    unit_hint = ""
    if total_units is not None or unit_label is not None:
        parts = []
        if total_units is not None:
            parts.append(f"total_units = {total_units}")
        if unit_label is not None:
            parts.append(f'unit_label = "{unit_label}"')
        unit_hint = f"\nKNOWN_STRUCTURE: {', '.join(parts)}\n"

    return f"""CONTRACTFLOW_SPARSE_CONTRACT_JSON

Extract a compact, declarative execution contract from USER_REQUEST.
Do NOT enumerate ordinary/unconstrained units and do NOT write document prose.

Return valid JSON only with this exact schema:
{{
  "global_brief": "2-3 concise sentences summarizing the document goal, persona/tone, and ordinary content (omit formatting marker instructions and long examples).",
  "total_units": 12,
  "unit_label": "Chapter",
  "unit_titles": {{}},
  "global_invariants": [
    "short rule that must hold across every unit"
  ],
  "rules": [
    {{
      "scope": "single",
      "derivation": "optional short check if calculating an index",
      "index": 1,
      "obligation": "exact concrete content required at this unit"
    }},
    {{
      "scope": "range",
      "derivation": "optional short check",
      "start": 4,
      "end": 6,
      "obligation": "exact concrete content required across this inclusive range"
    }},
    {{
      "scope": "every",
      "derivation": "optional short check",
      "start": 3,
      "step": 3,
      "end": null,
      "obligation": "exact concrete content required at every step from start through end (null means through total_units)"
    }}
  ]
}}

Rules for accuracy:
- Use only four scopes: "all", "single", "range", or "every".
- Extract EVERY explicit localized requirement stated in USER_REQUEST into "rules".
- Put general style/consistency/domain rules into "global_invariants" (or scope "all"), NOT into localized rules.
- All unit indices are 1-based (1 through total_units).
- Read starting baselines, numbering order, and inclusive bounds carefully.
- For "every" rules, leave "end" as null whenever USER_REQUEST does not state an explicit stopping bound; do NOT guess or invent a cutoff.
- If a requirement involves converting a date, coordinate, or grid/schedule position into a 1-based index, write a brief step-by-step arithmetic check in "derivation" BEFORE choosing the integers.
- Double-check that every integer field ("index", "start", "end", "step") matches the exact numbers in your "derivation" and "obligation".
- If a rule spans positions separated by a constant stride (e.g., jumping by a fixed offset along a schedule or 2D grid), use scope "every" with exact "start", "end", and "step".
- Keep each "obligation" concrete, faithful to USER_REQUEST, and free of invented details.
- Leave "unit_titles" empty ({{}}) unless specific chapter/section titles are requested or helpful for open-ended writing.
{unit_hint}
USER_REQUEST:
{prompt}
"""


def parse_sparse_contract_payload(
    payload: dict[str, Any],
    *,
    fallback_prompt: str,
    total_units: int | None = None,
    unit_label: str | None = None,
    min_words: int = 100,
    max_words: int = 120,
) -> DocumentContract:
    resolved_total = total_units if total_units is not None else _as_positive_int(payload.get("total_units"), 1)
    resolved_label = (
        unit_label
        if unit_label is not None
        else (str(payload.get("unit_label", "")).strip() or "Section")
    )
    global_brief = str(payload.get("global_brief", "")).strip() or _fallback_brief(
        fallback_prompt, resolved_total, resolved_label
    )

    raw_invariants = payload.get("global_invariants", [])
    invariants: list[str] = []
    if isinstance(raw_invariants, list):
        for item in raw_invariants:
            cleaned = re.sub(r"\s+", " ", str(item)).strip()
            if cleaned and cleaned not in invariants:
                invariants.append(cleaned)

    raw_titles = payload.get("unit_titles", {})
    titles_by_index: dict[int, str] = {}
    if isinstance(raw_titles, dict):
        for key, value in raw_titles.items():
            idx = _as_int(key)
            title = str(value).strip()
            if idx is not None and 1 <= idx <= resolved_total and title:
                titles_by_index[idx] = title

    rules: list[ContractRule] = []
    raw_rules = payload.get("rules", [])
    if isinstance(raw_rules, list):
        for raw in raw_rules:
            if not isinstance(raw, dict):
                continue
            scope_raw = str(raw.get("scope", "")).strip().lower()
            obligation = re.sub(r"\s+", " ", str(raw.get("obligation", ""))).strip()
            if not obligation:
                continue
            derivation = str(raw.get("derivation", "")).strip()
            if scope_raw in {"all", "global"}:
                rules.append(ContractRule(scope="all", obligation=obligation, derivation=derivation))
            elif scope_raw in {"single", "once", "point"}:
                idx = _as_int(raw.get("index"))
                if idx is None:
                    idx = _as_int(raw.get("start"))
                if idx is not None:
                    rules.append(
                        ContractRule(scope="single", obligation=obligation, index=idx, derivation=derivation)
                    )
            elif scope_raw == "range":
                start = _as_int(raw.get("start"))
                end = _as_int(raw.get("end"))
                if start is not None and end is not None:
                    rules.append(
                        ContractRule(
                            scope="range",
                            obligation=obligation,
                            start=start,
                            end=end,
                            derivation=derivation,
                        )
                    )
            elif scope_raw in {"every", "periodic", "stride"}:
                start = _as_int(raw.get("start"))
                step = _as_int(raw.get("step"))
                end = _as_int(raw.get("end"))
                if start is not None and step is not None and step > 0:
                    rules.append(
                        ContractRule(
                            scope="every",
                            obligation=obligation,
                            start=start,
                            end=end,
                            step=step,
                            derivation=derivation,
                        )
                    )

    expanded_invariants, unit_obligations = expand_contract_rules(resolved_total, rules)
    for inv in expanded_invariants:
        if inv not in invariants:
            invariants.append(inv)

    units = [
        UnitContract(
            index=index,
            title=titles_by_index.get(index, f"{resolved_label} {index}"),
            min_words=min_words,
            max_words=max_words,
            obligations=unit_obligations.get(index, []),
        )
        for index in range(1, resolved_total + 1)
    ]
    return DocumentContract(
        global_brief=global_brief,
        global_invariants=invariants,
        unit_label=resolved_label,
        total_units=resolved_total,
        rules=rules,
        units=units,
        compiler_mode="sparse_ir",
    )


def compile_sparse_contract(
    prompt: str,
    llm: ChatModel,
    *,
    total_units: int | None = None,
    unit_label: str | None = None,
    min_words: int = 100,
    max_words: int = 120,
    max_output_tokens: int = 2_048,
    retry_cap_tokens: int = 8_192,
) -> DocumentContract:
    payload = _complete_json_resilient(
        llm,
        build_sparse_contract_prompt(prompt, total_units=total_units, unit_label=unit_label),
        system="You are a precise declarative contract compiler. Return valid JSON only.",
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        retry_cap_tokens=retry_cap_tokens,
    )
    return parse_sparse_contract_payload(
        payload,
        fallback_prompt=prompt,
        total_units=total_units,
        unit_label=unit_label,
        min_words=min_words,
        max_words=max_words,
    )


def build_unconstrained_contract(
    prompt: str,
    *,
    total_units: int,
    unit_label: str,
    min_words: int,
    max_words: int,
) -> DocumentContract:
    units = [
        UnitContract(
            index=index,
            title=f"{unit_label} {index}",
            min_words=min_words,
            max_words=max_words,
            obligations=[],
        )
        for index in range(1, total_units + 1)
    ]
    return DocumentContract(
        global_brief=prompt.strip(),
        global_invariants=[],
        unit_label=unit_label,
        total_units=total_units,
        rules=[],
        units=units,
        compiler_mode="none",
    )


# ---------------------------------------------------------------------------
# Context-Isolated Drafting, Generic 3-Level + Tag-Free Singleton Parsing
# ---------------------------------------------------------------------------

def build_window_draft_prompt(
    contract: DocumentContract,
    window: list[UnitContract],
    rolling_state: str,
    *,
    use_contracts: bool = True,
) -> str:
    if len(window) == 1:
        return build_singleton_draft_prompt(
            contract, window[0], rolling_state, use_contracts=use_contracts
        )

    indices = [unit.index for unit in window]
    min_w = window[0].min_words
    max_w = window[0].max_words
    invariants_block = (
        "\n".join(f"- {inv}" for inv in contract.global_invariants)
        if use_contracts and contract.global_invariants
        else "- Keep content concrete, coherent, and varied."
    )
    unit_specs = [
        {
            "index": unit.index,
            "title": unit.title,
            **({"obligations": unit.obligations} if use_contracts else {}),
        }
        for unit in window
    ]
    example_idx = indices[0]
    return f"""CONTRACTFLOW_DRAFT_UNITS

Write ONLY the requested units: {indices} (out of {contract.total_units} total {contract.unit_label} units).
Do not write any earlier or later unit.

GLOBAL_BRIEF:
{contract.global_brief}

GLOBAL_INVARIANTS:
{invariants_block}

ROLLING_STATE (recent context for continuity):
{rolling_state or "(start of document)"}

REQUIREMENTS:
- Output every requested index exactly once.
- Each unit must contain {min_w} to {max_w} words of final reader-facing prose.
- Fulfill every listed obligation naturally in its assigned unit.
- Do not include wrapper markdown headings or benchmark markers inside the prose.
- Wrap each unit in exact tags, replacing {example_idx} with each requested index:

<<<UNIT {example_idx}>>>
unit prose
<<<END UNIT {example_idx}>>>

UNITS_TO_WRITE_JSON:
{json.dumps(unit_specs, ensure_ascii=False)}
"""


def build_singleton_draft_prompt(
    contract: DocumentContract,
    unit: UnitContract,
    rolling_state: str,
    *,
    use_contracts: bool = True,
    shortfall_note: str = "",
) -> str:
    invariants_block = (
        "\n".join(f"- {inv}" for inv in contract.global_invariants)
        if use_contracts and contract.global_invariants
        else "- Keep content concrete, coherent, and varied."
    )
    obligations_block = (
        "\n".join(f"- {item}" for item in unit.obligations)
        if use_contracts and unit.obligations
        else "- Ordinary coherent content consistent with GLOBAL_BRIEF."
    )
    note_section = f"\nRECOVERY_NOTE:\n{shortfall_note}\n" if shortfall_note else ""
    return f"""CONTRACTFLOW_DRAFT_SINGLE_UNIT

Write ONLY unit {unit.index} ({unit.title}) out of {contract.total_units}.

GLOBAL_BRIEF:
{contract.global_brief}

GLOBAL_INVARIANTS:
{invariants_block}

ROLLING_STATE (recent context for continuity):
{rolling_state or "(start of document)"}

UNIT_OBLIGATIONS (must be clearly included):
{obligations_block}
{note_section}
REQUIREMENTS:
- Write {unit.min_words} to {unit.max_words} words of final reader-facing prose.
- Return ONLY the prose for unit {unit.index}. Do NOT add wrapper tags, headings, or commentary.
"""


def build_window_review_prompt(
    contract: DocumentContract,
    targets: list[UnitContract],
    unit_texts: dict[int, str],
) -> str:
    items = [
        {
            "index": unit.index,
            "title": unit.title,
            "obligations": unit.obligations,
            "text": unit_texts.get(unit.index, ""),
        }
        for unit in targets
    ]
    return f"""CONTRACTFLOW_REVIEW_UNITS_JSON

Check whether each unit clearly fulfills every listed obligation in its `obligations` field.
Treat each unit's `obligations` list as authoritative: verify ONLY whether the unit's text satisfies those obligations.
Never question, recalculate, or reject an assigned obligation, and never ask to remove content that fulfills a listed obligation.
Accept clear paraphrases. Do not fail a unit for minor style preferences or harmless extra detail.

Return valid JSON only:
{{
  "results": [
    {{"index": 1, "passed": true, "missing_obligations": [], "issues": []}}
  ]
}}

GLOBAL_BRIEF:
{contract.global_brief}

UNITS_JSON:
{json.dumps(items, ensure_ascii=False)}
"""


def build_unit_repair_prompt(
    contract: DocumentContract,
    targets: list[UnitContract],
    unit_texts: dict[int, str],
    issues: dict[int, dict[str, Any]],
) -> str:
    if len(targets) == 1:
        unit = targets[0]
        issue = issues.get(unit.index, {})
        return f"""CONTRACTFLOW_REPAIR_SINGLE_UNIT

Rewrite unit {unit.index} ({unit.title}) to fix every listed issue while preserving valid details.

GLOBAL_BRIEF:
{contract.global_brief}

UNIT_OBLIGATIONS:
{json.dumps(unit.obligations, ensure_ascii=False)}

CURRENT_TEXT:
{unit_texts.get(unit.index, "")}

DIAGNOSED_ISSUES:
{json.dumps(issue, ensure_ascii=False)}

REQUIREMENTS:
- Write {unit.min_words} to {unit.max_words} words of final reader-facing prose.
- Treat UNIT_OBLIGATIONS as authoritative: clearly satisfy every listed obligation and NEVER remove or omit any listed obligation even if DIAGNOSED_ISSUES questions it.
- Return ONLY the revised prose for unit {unit.index}. Do NOT add wrapper tags, headings, or commentary.
"""

    indices = [unit.index for unit in targets]
    min_w = targets[0].min_words
    max_w = targets[0].max_words
    items = [
        {
            "index": unit.index,
            "title": unit.title,
            "obligations": unit.obligations,
            "current_text": unit_texts.get(unit.index, ""),
            "issues": issues.get(unit.index, {}),
        }
        for unit in targets
    ]
    example_idx = indices[0]
    return f"""CONTRACTFLOW_REPAIR_UNITS

Rewrite ONLY units {indices} to resolve every listed issue and include every listed obligation.
Treat each unit's `obligations` list as authoritative: clearly satisfy every listed obligation and NEVER remove or omit any listed obligation even if `issues` questions it.
Each revised unit must contain {min_w} to {max_w} words of reader-facing prose.

Use exact tags:
<<<UNIT {example_idx}>>>
revised prose
<<<END UNIT {example_idx}>>>

GLOBAL_BRIEF:
{contract.global_brief}

FAILING_UNITS_JSON:
{json.dumps(items, ensure_ascii=False)}
"""


def parse_unit_batch(
    expected: list[UnitContract],
    raw: str,
    *,
    unit_label: str,
) -> dict[int, str]:
    """Generic 3-level parser: complete tags -> unclosed tags -> generic numbered headings."""
    expected_indices = {unit.index for unit in expected}
    if not raw or not raw.strip():
        return {}

    result: dict[int, str] = {}

    # Level 1: complete <<<UNIT i>>> ... <<<END UNIT i>>> (or <<<SEGMENT i>>>)
    complete_pattern = re.compile(
        r"<<<(?:UNIT|SEGMENT)\s+(\d+)>>>\s*(.*?)\s*<<<END\s+(?:UNIT|SEGMENT)\s+\1>>>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in complete_pattern.finditer(raw):
        index = int(match.group(1))
        text = _clean_unit_prose(match.group(2), unit_label, index)
        if index in expected_indices and text and index not in result:
            result[index] = text

    if len(result) == len(expected_indices):
        return result

    remainder = complete_pattern.sub("\n", raw)

    # Levels 2 & 3: combine unclosed <<<UNIT i>>> tags and generic numbered headings
    # by position so a missing closing tag stops cleanly before the next unit heading.
    open_pattern = re.compile(r"<<<(?:UNIT|SEGMENT)\s+(\d+)>>>", flags=re.IGNORECASE)
    escaped_label = re.escape(unit_label)
    heading_pattern = re.compile(
        rf"(?:^|\n)[ \t]*(?:"
        rf"(?:#\*#|#{{1,6}}|\*\*|[-*])[ \t]*{escaped_label}[ \t]+(\d+)\b[^:\n]{{0,60}}(?::(?:\*\*)?[ \t]*|(?:\*\*)?[ \t]*(?=\r?$))"
        rf"|{escaped_label}[ \t]+(\d+)\b[^:\n.!?]{{0,60}}:[ \t]*"
        rf")",
        flags=re.IGNORECASE | re.MULTILINE,
    )

    markers: list[tuple[int, int, int, int]] = []  # (start, end, priority, index)
    for match in open_pattern.finditer(remainder):
        markers.append((match.start(), match.end(), 0, int(match.group(1))))
    for match in heading_pattern.finditer(remainder):
        markers.append((match.start(), match.end(), 1, int(match.group(1) or match.group(2))))

    markers.sort(key=lambda item: (item[0], item[2]))
    deduped: list[tuple[int, int, int]] = []
    for start, end, _prio, index in markers:
        if deduped and start < deduped[-1][1]:
            continue
        deduped.append((start, end, index))

    for pos, (_start, end, index) in enumerate(deduped):
        if index not in expected_indices or index in result:
            continue
        next_start = deduped[pos + 1][0] if pos + 1 < len(deduped) else len(remainder)
        body = re.split(
            r"<<<END\s+(?:UNIT|SEGMENT)\s+\d+>>>",
            remainder[end:next_start],
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        text = _clean_unit_prose(body, unit_label, index)
        if text:
            result[index] = text

    return result


def parse_singleton_unit(raw: str, unit: UnitContract, *, unit_label: str) -> str:
    """Tag-free extraction for single-unit drafting/recovery/repair."""
    parsed = parse_unit_batch([unit], raw, unit_label=unit_label)
    if unit.index in parsed:
        return parsed[unit.index]
    return _clean_unit_prose(raw, unit_label, unit.index)


def _clean_unit_prose(text: str, unit_label: str, index: int) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^<<<(?:UNIT|SEGMENT)\s+\d+>>>\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*<<<END\s+(?:UNIT|SEGMENT)\s+\d+>>>$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.split(r"\*\*\*\s*finished", cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    escaped_label = re.escape(unit_label)
    cleaned = re.sub(
        rf"^(?:#\*#|#{{1,6}}|\*\*)?[ \t]*{escaped_label}[ \t]+{index}\b[^:\n.!?]{{0,60}}(?::(?:\*\*)?[ \t]*\n*|(?:(?:\*\*)?[ \t]*\n+))",
        "",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Online Stateful Flow Engine
# ---------------------------------------------------------------------------

def run_contractflow(
    prompt: str,
    llm: ChatModel,
    *,
    config: FlowConfig | None = None,
    total_units: int | None = None,
    unit_label: str | None = None,
    min_words: int = 100,
    max_words: int = 120,
    precompiled_contract: DocumentContract | None = None,
    checkpoint_path: str | Path | None = None,
    review_llm: ChatModel | None = None,
    repair_llm: ChatModel | None = None,
    renderer: Callable[[DocumentContract, dict[int, str]], str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = config or FlowConfig()
    reviewer = review_llm or llm
    repairer = repair_llm or llm
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    checkpoint = _load_v2_checkpoint(checkpoint_path, prompt_hash)
    if checkpoint is not None:
        contract = _contract_from_dict(checkpoint["contract"])
        unit_texts = {
            int(k): str(v)
            for k, v in checkpoint.get("unit_texts", {}).items()
            if str(v).strip()
        }
        completed_windows = set(int(i) for i in checkpoint.get("completed_windows", []))
        window_records = list(checkpoint.get("window_records", []))
        review_records = list(checkpoint.get("review_records", []))
        repair_records = list(checkpoint.get("repair_records", []))
    else:
        if precompiled_contract is not None:
            contract = precompiled_contract
        elif cfg.use_contracts:
            contract = compile_sparse_contract(
                prompt,
                llm,
                total_units=total_units,
                unit_label=unit_label,
                min_words=min_words,
                max_words=max_words,
                max_output_tokens=cfg.plan_max_output_tokens,
                retry_cap_tokens=cfg.plan_retry_cap_tokens,
            )
        else:
            if total_units is None or unit_label is None:
                raise ValueError("total_units and unit_label are required when use_contracts=False")
            contract = build_unconstrained_contract(
                prompt,
                total_units=total_units,
                unit_label=unit_label,
                min_words=min_words,
                max_words=max_words,
            )
        unit_texts = {}
        completed_windows = set()
        window_records = []
        review_records = []
        repair_records = []
        _write_v2_checkpoint(
            checkpoint_path,
            prompt_hash=prompt_hash,
            contract=contract,
            unit_texts=unit_texts,
            completed_windows=completed_windows,
            window_records=window_records,
            review_records=review_records,
            repair_records=repair_records,
            stage="compiled",
        )

    unit_max_words = max((unit.max_words for unit in contract.units), default=max_words)
    eff_batch = effective_batch_size(
        cfg.batch_size, unit_max_words, cfg.generation_max_output_tokens
    )
    windows = [list(w) for w in _chunked(contract.units, eff_batch)]

    for window_idx, window in enumerate(windows):
        if window_idx in completed_windows:
            continue

        rolling_state = _rolling_state(unit_texts, max_chars=cfg.rolling_state_chars)
        window_draft_log = _draft_window(
            contract,
            window,
            llm,
            unit_texts=unit_texts,
            rolling_state=rolling_state,
            cfg=cfg,
        )
        window_records.append(
            {
                "window_index": window_idx,
                "indices": [u.index for u in window],
                **window_draft_log,
            }
        )

        # Online Tier-2 contract verification + localized repair for THIS window before advancing state
        if cfg.use_contracts and cfg.use_review_repair:
            constrained = [u for u in window if u.obligations and u.index in unit_texts]
            if constrained:
                batch_review = _review_units_resilient(
                    contract,
                    constrained,
                    unit_texts,
                    reviewer,
                    review_cap_tokens=cfg.review_max_output_tokens,
                )
                failing: list[UnitContract] = []
                failing_issues: dict[int, dict[str, Any]] = {}
                for unit in constrained:
                    rec = batch_review.get(unit.index)
                    if rec is None:
                        rec = {
                            "passed": False,
                            "missing_obligations": list(unit.obligations),
                            "issues": ["review result missing"],
                        }
                    review_records.append({"index": unit.index, **rec})
                    if not rec["passed"]:
                        failing.append(unit)
                        failing_issues[unit.index] = rec

                if failing:
                    repairs = _repair_units_monotonic(
                        contract,
                        failing,
                        unit_texts,
                        failing_issues,
                        repairer,
                        cfg=cfg,
                    )
                    repair_records.extend(repairs)

        completed_windows.add(window_idx)
        _write_v2_checkpoint(
            checkpoint_path,
            prompt_hash=prompt_hash,
            contract=contract,
            unit_texts=unit_texts,
            completed_windows=completed_windows,
            window_records=window_records,
            review_records=review_records,
            repair_records=repair_records,
            stage=f"window_{window_idx + 1}_of_{len(windows)}",
        )

    final_structural_issues = check_structural_issues(contract.units, unit_texts)
    render_fn = renderer or render_markdown_document
    final_text = render_fn(contract, unit_texts)

    all_history = completion_history(llm)
    if reviewer is not llm:
        all_history.extend(completion_history(reviewer))
    if repairer is not llm and repairer is not reviewer:
        all_history.extend(completion_history(repairer))

    return {
        "system": v2_system_name(cfg.use_contracts, cfg.use_review_repair),
        "final_text": final_text,
        "units": [
            {
                "index": unit.index,
                "title": unit.title,
                "text": unit_texts[unit.index],
                "words": word_count(unit_texts[unit.index]),
            }
            for unit in contract.units
            if unit.index in unit_texts
        ],
        "contract": contract.to_dict(),
        "metadata": {
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "model": llm.model,
            "pipeline_version": CONTRACTFLOW_V2_VERSION,
            "requested_batch_size": cfg.batch_size,
            "effective_batch_size": eff_batch,
            "length_retry_ratio": cfg.length_retry_ratio,
            "use_contracts": cfg.use_contracts,
            "use_review_repair": cfg.use_review_repair,
            "compiler_mode": contract.compiler_mode,
            "rule_count": len(contract.rules),
            "constrained_unit_count": sum(bool(u.obligations) for u in contract.units),
            "window_records": window_records,
            "review_results": review_records,
            "repair_records": repair_records,
            "final_structural_issues": final_structural_issues,
            "completion_history": all_history,
            "resumed_from_checkpoint": checkpoint is not None,
        },
    }


def _draft_window(
    contract: DocumentContract,
    window: list[UnitContract],
    llm: ChatModel,
    *,
    unit_texts: dict[int, str],
    rolling_state: str,
    cfg: FlowConfig,
) -> dict[str, Any]:
    pending: list[list[UnitContract]] = [
        [unit for unit in window if unit.index not in unit_texts]
    ]
    pending = [batch for batch in pending if batch]
    singleton_attempts: dict[int, int] = {}
    calls = 0
    recovered_indices: list[int] = []

    while pending:
        batch = pending.pop(0)
        calls += 1
        max_w = max(u.max_words for u in batch)
        budget = draft_token_budget(len(batch), max_w, cfg.generation_max_output_tokens)
        local_state = _rolling_state(unit_texts, max_chars=cfg.rolling_state_chars) or rolling_state

        if len(batch) == 1:
            unit = batch[0]
            prior = unit_texts.get(unit.index, "")
            shortfall = (
                f"Previous draft was too short ({word_count(prior)} words); expand cleanly to {unit.min_words}-{unit.max_words} words."
                if prior
                else ""
            )
            prompt = build_singleton_draft_prompt(
                contract,
                unit,
                local_state,
                use_contracts=cfg.use_contracts,
                shortfall_note=shortfall,
            )
            raw = llm.complete(
                prompt,
                system="Return only clean final prose for the requested unit.",
                temperature=0.2,
                max_output_tokens=budget,
            )
            text = parse_singleton_unit(raw, unit, unit_label=contract.unit_label)
            parsed = {unit.index: text} if text else {}
        else:
            prompt = build_window_draft_prompt(
                contract,
                batch,
                local_state,
                use_contracts=cfg.use_contracts,
            )
            raw = llm.complete(
                prompt,
                system="Write only the requested units using the required tags.",
                temperature=0.2,
                max_output_tokens=budget,
            )
            parsed = parse_unit_batch(batch, raw, unit_label=contract.unit_label)

        need_retry: list[UnitContract] = []
        for unit in batch:
            candidate = parsed.get(unit.index, "").strip()
            existing = unit_texts.get(unit.index, "")
            if candidate and word_count(candidate) >= word_count(existing):
                unit_texts[unit.index] = candidate
                if calls > 1:
                    recovered_indices.append(unit.index)
            current = unit_texts.get(unit.index, "")
            retry_floor = _length_retry_floor(unit.min_words, cfg.length_retry_ratio)
            if not current or word_count(current) < retry_floor:
                need_retry.append(unit)

        # In open-loop ablation (use_review_repair=False), stop after the single initial draft call.
        if not cfg.use_review_repair or not need_retry:
            continue

        if len(need_retry) > 1:
            mid = max(1, len(need_retry) // 2)
            pending.insert(0, need_retry[mid:])
            pending.insert(0, need_retry[:mid])
        else:
            idx = need_retry[0].index
            singleton_attempts[idx] = singleton_attempts.get(idx, 0) + 1
            if singleton_attempts[idx] <= cfg.max_singleton_retries:
                pending.insert(0, need_retry)

    return {
        "draft_calls": calls,
        "recovered_indices": sorted(set(recovered_indices)),
        "present_indices": [u.index for u in window if u.index in unit_texts],
    }


def _review_units_resilient(
    contract: DocumentContract,
    targets: list[UnitContract],
    unit_texts: dict[int, str],
    llm: ChatModel,
    *,
    review_cap_tokens: int,
) -> dict[int, dict[str, Any]]:
    budget = review_token_budget(len(targets), review_cap_tokens)
    try:
        payload = _complete_json_resilient(
            llm,
            build_window_review_prompt(contract, targets, unit_texts),
            system="You verify document units against explicit obligations. Return valid JSON only.",
            temperature=0.0,
            max_output_tokens=budget,
            retry_cap_tokens=review_cap_tokens,
        )
        return _parse_review_payload(payload)
    except ValueError:
        if len(targets) > 1:
            mid = len(targets) // 2
            left = _review_units_resilient(
                contract, targets[:mid], unit_texts, llm, review_cap_tokens=review_cap_tokens
            )
            right = _review_units_resilient(
                contract, targets[mid:], unit_texts, llm, review_cap_tokens=review_cap_tokens
            )
            return {**left, **right}
        unit = targets[0]
        return {
            unit.index: {
                "passed": False,
                "missing_obligations": list(unit.obligations),
                "issues": ["review json parsing failed"],
            }
        }


def _repair_units_monotonic(
    contract: DocumentContract,
    failing: list[UnitContract],
    unit_texts: dict[int, str],
    issues: dict[int, dict[str, Any]],
    llm: ChatModel,
    *,
    cfg: FlowConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    repair_chunk_size = max(1, cfg.batch_size // 2)
    queue = [list(chunk) for chunk in _chunked(failing, repair_chunk_size)]

    while queue:
        batch = queue.pop(0)
        max_w = max(u.max_words for u in batch)
        budget = repair_token_budget(len(batch), max_w, cfg.repair_max_output_tokens)
        prompt = build_unit_repair_prompt(contract, batch, unit_texts, issues)
        system_msg = (
            "Return only revised clean prose for the requested unit."
            if len(batch) == 1
            else "Repair only the requested units using the required tags."
        )
        raw = llm.complete(
            prompt,
            system=system_msg,
            temperature=0.1,
            max_output_tokens=budget,
        )
        if len(batch) == 1:
            unit = batch[0]
            text = parse_singleton_unit(raw, unit, unit_label=contract.unit_label)
            repaired = {unit.index: text} if text else {}
        else:
            repaired = parse_unit_batch(batch, raw, unit_label=contract.unit_label)

        unparsed: list[UnitContract] = []
        for unit in batch:
            before_text = unit_texts.get(unit.index, "")
            before_words = word_count(before_text)
            candidate = repaired.get(unit.index, "").strip()
            after_words = word_count(candidate)
            min_accept_words = _length_retry_floor(unit.min_words, cfg.length_retry_ratio)
            accepted = bool(candidate) and (
                after_words >= min_accept_words or after_words > before_words
            )
            if accepted:
                unit_texts[unit.index] = candidate
            elif len(batch) > 1 and not candidate:
                unparsed.append(unit)
            records.append(
                {
                    "index": unit.index,
                    "accepted": accepted,
                    "before_words": before_words,
                    "after_words": after_words,
                }
            )

        if unparsed:
            for unit in reversed(unparsed):
                queue.insert(0, [unit])

    return records


# ---------------------------------------------------------------------------
# Rendering & Helpers
# ---------------------------------------------------------------------------

def render_markdown_document(contract: DocumentContract, unit_texts: dict[int, str]) -> str:
    blocks = []
    for unit in contract.units:
        text = unit_texts.get(unit.index, "").strip()
        if not text:
            continue
        blocks.append(f"## {unit.title}\n\n{text}")
    return "\n\n".join(blocks)


def v2_system_name(use_contracts: bool, use_review_repair: bool) -> str:
    if use_contracts and use_review_repair:
        return "v2_contractflow"
    if not use_contracts and use_review_repair:
        return "v2_no_contract"
    if use_contracts and not use_review_repair:
        return "v2_no_review_repair"
    return "v2_neither"


def word_count(text: str) -> int:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9'-]*\b", text))
    return cjk + latin


def _length_retry_floor(min_words: int, ratio: float) -> int:
    if min_words <= 0 or ratio <= 0.0:
        return 0
    return max(1, math.ceil(round(min_words * ratio, 6)))


def check_structural_issues(
    units: list[UnitContract],
    unit_texts: dict[int, str],
) -> dict[int, dict[str, Any]]:
    issues: dict[int, dict[str, Any]] = {}
    for unit in units:
        text = unit_texts.get(unit.index, "")
        count = word_count(text)
        msgs = []
        if not text:
            msgs.append("unit is missing")
        elif count < unit.min_words:
            msgs.append(f"word count {count} is below {unit.min_words}")
        if msgs:
            issues[unit.index] = {
                "passed": False,
                "missing_obligations": list(unit.obligations),
                "issues": msgs,
            }
    return issues


def _complete_json_resilient(
    llm: ChatModel,
    prompt: str,
    *,
    system: str,
    temperature: float,
    max_output_tokens: int,
    retry_cap_tokens: int,
    max_retries: int = 1,
) -> dict[str, Any]:
    raw = ""
    error: Exception | None = None
    truncated = False
    budget = max_output_tokens
    for attempt in range(max_retries + 1):
        request_prompt = prompt
        if attempt:
            if truncated:
                budget = min(max(budget, retry_cap_tokens), max(budget * 2, budget + 1_024))
                request_prompt = (
                    "Previous JSON response was truncated at the token limit. "
                    "Return compact, valid JSON only without repeating items.\n\n" + prompt
                )
            else:
                request_prompt = (
                    "Repair the malformed JSON below into valid JSON without losing content. "
                    f"Return valid JSON only.\n\nMALFORMED_RESPONSE:\n{raw}\n\nPARSER_ERROR:\n{error}\n"
                )
        raw = llm.complete(
            request_prompt,
            system=system,
            temperature=0.0 if attempt else temperature,
            response_format="json_object",
            max_output_tokens=budget,
        )
        truncated = _was_truncated(llm)
        try:
            return extract_json_object(raw)
        except ValueError as caught:
            error = caught
    raise ValueError(f"JSON completion failed: {error}")


def _was_truncated(llm: ChatModel) -> bool:
    getter = getattr(llm, "completion_metadata", None)
    if not callable(getter):
        return False
    meta = getter()
    return isinstance(meta, dict) and meta.get("finish_reason") == "length"


def _parse_review_payload(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        return results
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        idx = _as_int(raw.get("index"))
        if idx is None:
            continue
        missing = raw.get("missing_obligations")
        if missing is None:
            missing = raw.get("missing_events", [])
        issues = raw.get("issues", [])
        results[idx] = {
            "passed": bool(raw.get("passed", False)),
            "missing_obligations": [str(x).strip() for x in missing if str(x).strip()]
            if isinstance(missing, list)
            else [],
            "issues": [str(x).strip() for x in issues if str(x).strip()]
            if isinstance(issues, list)
            else [],
        }
    return results


def _rolling_state(unit_texts: dict[int, str], *, max_chars: int) -> str:
    if not unit_texts or max_chars <= 0:
        return ""
    latest_index = max(unit_texts)
    latest_text = unit_texts[latest_index].strip()
    if len(latest_text) <= max_chars:
        return f"[Latest Unit {latest_index}]\n{latest_text}"
    return f"[Latest Unit {latest_index} tail]\n...{latest_text[-max_chars:]}"


def _fallback_brief(prompt: str, total_units: int, unit_label: str) -> str:
    normalized = re.sub(r"\s+", " ", prompt).strip()
    first = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0]
    return f"{first} Produce {total_units} coherent {unit_label} units."


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_positive_int(value: Any, default: int) -> int:
    parsed = _as_int(value)
    return parsed if parsed is not None and parsed > 0 else default


def _chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    step = max(1, size)
    for start in range(0, len(items), step):
        yield items[start : start + step]


def _contract_from_dict(data: dict[str, Any]) -> DocumentContract:
    rules = [
        ContractRule(
            scope=item["scope"],
            obligation=item["obligation"],
            index=item.get("index"),
            start=item.get("start"),
            end=item.get("end"),
            step=item.get("step"),
            derivation=item.get("derivation", ""),
        )
        for item in data.get("rules", [])
        if isinstance(item, dict)
    ]
    units = [
        UnitContract(
            index=int(item["index"]),
            title=str(item.get("title", "")),
            min_words=int(item.get("min_words", 100)),
            max_words=int(item.get("max_words", 120)),
            obligations=[str(x) for x in item.get("obligations", [])],
        )
        for item in data.get("units", [])
        if isinstance(item, dict)
    ]
    return DocumentContract(
        global_brief=str(data.get("global_brief", "")),
        global_invariants=[str(x) for x in data.get("global_invariants", [])],
        unit_label=str(data.get("unit_label", "Section")),
        total_units=int(data.get("total_units", len(units))),
        rules=rules,
        units=units,
        compiler_mode=str(data.get("compiler_mode", "sparse_ir")),
    )


def _load_v2_checkpoint(path: str | Path | None, prompt_hash: str) -> dict[str, Any] | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    data = read_json(file_path)
    if data.get("pipeline_version") != CONTRACTFLOW_V2_VERSION:
        raise ValueError(f"incompatible v2 checkpoint version: {data.get('pipeline_version')}")
    if data.get("prompt_hash") != prompt_hash:
        raise ValueError("v2 checkpoint belongs to a different prompt")
    return data


def _write_v2_checkpoint(
    path: str | Path | None,
    *,
    prompt_hash: str,
    contract: DocumentContract,
    unit_texts: dict[int, str],
    completed_windows: set[int],
    window_records: list[dict[str, Any]],
    review_records: list[dict[str, Any]],
    repair_records: list[dict[str, Any]],
    stage: str,
) -> None:
    if path is None:
        return
    write_json(
        path,
        {
            "pipeline_version": CONTRACTFLOW_V2_VERSION,
            "prompt_hash": prompt_hash,
            "stage": stage,
            "contract": contract.to_dict(),
            "unit_texts": {str(k): v for k, v in sorted(unit_texts.items())},
            "completed_windows": sorted(completed_windows),
            "window_records": window_records,
            "review_records": review_records,
            "repair_records": repair_records,
        },
    )
