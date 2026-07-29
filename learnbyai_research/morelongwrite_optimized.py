from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .io_utils import extract_json_object
from .llm import ChatModel, completion_history
from .longbench_write_transfer import (
    ADAPTERS,
    MORELONGWRITE_OPTIMIZED_PROFILE,
    LongBenchWriteTask,
    build_contract_plan_prompt,
    build_contract_plan_retry_prompt,
    generation_record,
    official_length_count,
    parse_contract_plan,
    read_json,
    write_json,
)


TYPE_RULES = {
    "Academic and Technical Writing": [
        "Do not invent citations, experiments, datasets, quotations, or precise findings.",
        "Distinguish established knowledge, author analysis, assumptions, and examples.",
        "Qualify named legal, regulatory, medical, and technical claims; do not turn "
        "contested interpretations into universal rules.",
        "When the request does not supply an exact legal proposition, present it as a "
        "jurisdiction-dependent governance consideration rather than a legal mandate.",
        "Do not say studies or evidence show something unless the request supplies that "
        "evidence; describe unsupported synthesis explicitly as analysis.",
        "Avoid unsupported universals such as consistently outperforms, generally more "
        "accurate, guarantees, eliminates, or always scales; verify complexity claims.",
        "When relevant, cover evaluation, security and privacy, governance, human factors, "
        "implementation constraints, and limitations rather than giving only an overview.",
        "Keep terminology, notation, claims, and limitations consistent across sections.",
    ],
    "Education and Training": [
        "Keep examples technically correct and executable at the stated level.",
        "Maintain a deliberate prerequisite progression and connect exercises to outcomes.",
        "Do not leave code, exercises, troubleshooting steps, or projects incomplete.",
    ],
    "Popular Science and Expository": [
        "Prefer robust explanations over unsupported precision.",
        "Define terminology once and preserve the same meaning throughout.",
        "Avoid repeating the same explanation, analogy, opening, or conclusion.",
    ],
    "Functional and Professional Writing": [
        "Keep roles, timelines, budgets, phases, metrics, and escalation rules consistent.",
        "Every referenced appendix, template, table, or decision tool must actually appear.",
        "Use implementable procedures and qualify legal, policy, and technical claims.",
    ],
    "News and Investigative Reporting": [
        "Do not invent interviews, quotations, confidential documents, field observations, "
        "organizations, reports, cases, or precise statistics.",
        "Clearly distinguish established evidence, allegation, interpretation, and uncertainty.",
        "Use source-neutral descriptions when a claim cannot be verified from the request.",
    ],
    "Literature and Creative Writing": [
        "Treat character identity, age, relationships, timeline, setting, and world rules as canon.",
        "Do not resolve major conflict through an unprepared discovery, sudden agreement, or rescue.",
        "Avoid repeated motifs, scene templates, emotional conclusions, and exposition.",
        "Every section must advance plot, character state, or the causal world state.",
    ],
}


def run_morelongwrite_optimized(
    task: LongBenchWriteTask,
    adapter_name: str,
    planning_client: ChatModel,
    author_client_factory,
    review_client_factory,
    repair_client_factory,
    *,
    checkpoint_dir: str | Path,
    planning_temperature: float,
    writing_temperature: float,
    review_temperature: float,
    repair_temperature: float,
    planning_max_output_tokens: int,
    section_max_output_tokens: int,
    review_max_output_tokens: int,
    repair_max_output_tokens: int,
    global_review_max_output_tokens: int,
    max_plan_attempts: int,
    section_workers: int,
    narrative_section_workers: int,
    review_workers: int,
    repair_workers: int,
    review_batch_size: int,
    max_repair_fraction: float,
    recent_context_characters: int,
    state_excerpt_characters: int,
    minimum_candidate_ratio: float,
    maximum_candidate_ratio: float,
) -> dict[str, Any]:
    if adapter_name not in ADAPTERS:
        raise ValueError(f"unknown task-family adapter: {adapter_name}")
    if min(section_workers, narrative_section_workers, review_workers, repair_workers) < 1:
        raise ValueError("optimized worker counts must be positive")
    if not 0.0 <= max_repair_fraction <= 1.0:
        raise ValueError("max_repair_fraction must be between zero and one")

    started = time.perf_counter()
    adapter = ADAPTERS[adapter_name]
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    histories: list[dict[str, Any]] = []
    plan = _load_or_create_plan(
        task,
        adapter_name,
        adapter,
        planning_client,
        checkpoint_root,
        planning_temperature,
        planning_max_output_tokens,
        max_plan_attempts,
    )
    histories.extend(read_json(checkpoint_root / "contract_plan.json").get("completion_history", []))

    workers = narrative_section_workers if adapter_name == "narrative" else section_workers
    sections, section_histories = _generate_sections(
        task,
        adapter_name,
        adapter,
        plan,
        checkpoint_root,
        author_client_factory,
        workers=workers,
        temperature=writing_temperature,
        hard_max_output_tokens=section_max_output_tokens,
        recent_context_characters=recent_context_characters,
        state_excerpt_characters=state_excerpt_characters,
    )
    histories.extend(section_histories)

    local_reviews, local_histories = _review_sections(
        task,
        adapter,
        plan,
        sections,
        checkpoint_root / "reviews",
        review_client_factory,
        workers=review_workers,
        batch_size=review_batch_size,
        temperature=review_temperature,
        max_output_tokens=review_max_output_tokens,
        state_excerpt_characters=state_excerpt_characters,
    )
    histories.extend(local_histories)

    global_review = _load_or_create_global_review(
        task,
        adapter,
        plan,
        sections,
        local_reviews,
        checkpoint_root / "global_review.json",
        review_client_factory,
        temperature=review_temperature,
        max_output_tokens=global_review_max_output_tokens,
        state_excerpt_characters=state_excerpt_characters,
    )
    histories.extend(global_review.get("completion_history", []))
    review_map = _merge_reviews(local_reviews, global_review)

    final_sections, repair_histories, repair_stats = _repair_sections(
        task,
        adapter,
        plan,
        sections,
        review_map,
        checkpoint_root,
        repair_client_factory,
        review_client_factory,
        workers=repair_workers,
        review_workers=review_workers,
        review_batch_size=review_batch_size,
        repair_temperature=repair_temperature,
        review_temperature=review_temperature,
        repair_max_output_tokens=repair_max_output_tokens,
        review_max_output_tokens=review_max_output_tokens,
        max_repair_fraction=max_repair_fraction,
        minimum_candidate_ratio=minimum_candidate_ratio,
        maximum_candidate_ratio=maximum_candidate_ratio,
        state_excerpt_characters=state_excerpt_characters,
    )
    histories.extend(repair_histories)

    final_text = _assemble_document(task, final_sections)
    return generation_record(
        task,
        "contractflow",
        final_text,
        started,
        histories,
        plan=plan,
        units=final_sections,
        adaptations=[
            f"MoreLongWrite benchmark profile: {MORELONGWRITE_OPTIMIZED_PROFILE}",
            f"frozen task-family adapter: {adapter_name}",
            "unchanged ContractFlow stages: contract, write, review, repair, assemble",
            f"contract-guided wave generation with up to {workers} parallel author calls",
            f"parallel local review plus one document-level contract review",
            "document-budget control with language-calibrated output limits",
            "selective parallel repair with deterministic acceptance checks",
            "deterministic exact-title assembly and resumable checkpoints",
        ],
        extra_metadata={
            "adapter": adapter_name,
            "profile": MORELONGWRITE_OPTIMIZED_PROFILE,
            "section_workers": workers,
            "review_workers": review_workers,
            "repair_workers": repair_workers,
            "review_batch_size": review_batch_size,
            "max_repair_fraction": max_repair_fraction,
            "recent_context_characters": recent_context_characters,
            "state_excerpt_characters": state_excerpt_characters,
            "repair_stats": repair_stats,
            "global_review_summary": global_review.get("summary", ""),
        },
    )


def _load_or_create_plan(
    task: LongBenchWriteTask,
    adapter_name: str,
    adapter: dict[str, Any],
    client: ChatModel,
    checkpoint_root: Path,
    temperature: float,
    max_output_tokens: int,
    max_attempts: int,
) -> dict[str, Any]:
    path = checkpoint_root / "contract_plan.json"
    if path.exists():
        return dict(read_json(path)["plan"])
    base_prompt = build_contract_plan_prompt(
        task,
        adapter_name,
        adapter,
        profile=MORELONGWRITE_OPTIMIZED_PROFILE,
    )
    prompt = base_prompt
    last_error: Exception | None = None
    raw = ""
    for _ in range(max_attempts):
        raw = client.complete(
            prompt,
            system="You compile document contracts from user requests. Return valid JSON only.",
            temperature=temperature,
            response_format="json_object",
            max_output_tokens=max_output_tokens,
        )
        try:
            plan = parse_contract_plan(
                raw,
                task.target_words,
                profile=MORELONGWRITE_OPTIMIZED_PROFILE,
            )
            break
        except (ValueError, json.JSONDecodeError) as error:
            last_error = error
            prompt = build_contract_plan_retry_prompt(
                base_prompt,
                raw,
                str(error),
                task.target_words,
                profile=MORELONGWRITE_OPTIMIZED_PROFILE,
            )
    else:
        raise RuntimeError(f"optimized contract planning failed: {last_error}") from last_error
    write_json(
        path,
        {
            "profile": MORELONGWRITE_OPTIMIZED_PROFILE,
            "plan": plan,
            "completion_history": completion_history(client),
        },
    )
    return plan


def _generate_sections(
    task: LongBenchWriteTask,
    adapter_name: str,
    adapter: dict[str, Any],
    plan: dict[str, Any],
    checkpoint_root: Path,
    author_client_factory,
    *,
    workers: int,
    temperature: float,
    hard_max_output_tokens: int,
    recent_context_characters: int,
    state_excerpt_characters: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    section_dir = checkpoint_root / "sections"
    section_dir.mkdir(parents=True, exist_ok=True)
    records: dict[int, dict[str, Any]] = {}
    histories: list[dict[str, Any]] = []
    for path in section_dir.glob("*.json"):
        record = read_json(path)
        records[int(record["index"])] = record
        histories.extend(record.get("completion_history", []))

    contracts = list(plan["sections"])
    for start in range(0, len(contracts), workers):
        wave = contracts[start : start + workers]
        missing = [section for section in wave if int(section["index"]) not in records]
        if not missing:
            continue
        prior = [
            records[index]
            for index in sorted(records)
            if index < int(wave[0]["index"])
        ]
        targets = _effective_targets(task, contracts, records, missing)
        with ThreadPoolExecutor(max_workers=min(workers, len(missing))) as executor:
            futures = {
                executor.submit(
                    _generate_one_section,
                    task,
                    adapter_name,
                    adapter,
                    plan,
                    section,
                    prior,
                    targets[int(section["index"])],
                    author_client_factory,
                    temperature,
                    hard_max_output_tokens,
                    recent_context_characters,
                    state_excerpt_characters,
                ): section
                for section in missing
            }
            for future in as_completed(futures):
                record = future.result()
                index = int(record["index"])
                records[index] = record
                histories.extend(record.get("completion_history", []))
                write_json(section_dir / f"{index:03d}.json", record)
    return [records[int(section["index"])] for section in contracts], histories


def _effective_targets(
    task: LongBenchWriteTask,
    contracts: list[dict[str, Any]],
    records: dict[int, dict[str, Any]],
    missing_wave: list[dict[str, Any]],
) -> dict[int, int]:
    completed_words = sum(int(record["actual_words"]) for record in records.values())
    remaining_budget = max(1, task.target_words - completed_words)
    remaining_contracts = [
        section
        for section in contracts
        if int(section["index"]) not in records
    ]
    remaining_contract_total = max(
        1,
        sum(int(section["target_words"]) for section in remaining_contracts),
    )
    scale = remaining_budget / remaining_contract_total
    targets: dict[int, int] = {}
    for section in missing_wave:
        original = int(section["target_words"])
        adjusted = round(original * scale)
        targets[int(section["index"])] = min(
            round(original * 1.35),
            max(round(original * 0.65), adjusted),
        )
    return targets


def _generate_one_section(
    task: LongBenchWriteTask,
    adapter_name: str,
    adapter: dict[str, Any],
    plan: dict[str, Any],
    section: dict[str, Any],
    prior: list[dict[str, Any]],
    effective_target: int,
    author_client_factory,
    temperature: float,
    hard_max_output_tokens: int,
    recent_context_characters: int,
    state_excerpt_characters: int,
) -> dict[str, Any]:
    client = author_client_factory()
    segment_count = _author_segment_count(task)
    desired_segments = _distribute_units(effective_target, segment_count)
    segment_records: list[dict[str, Any]] = []
    segment_texts: list[str] = []
    for segment_index, desired_segment in enumerate(desired_segments, start=1):
        author_target = _calibrated_author_target(
            task,
            desired_segment,
            segmented=segment_count > 1,
        )
        prompt = _section_prompt(
            task,
            adapter_name,
            adapter,
            plan,
            section,
            prior,
            author_target,
            recent_context_characters,
            state_excerpt_characters,
            segment_index=segment_index,
            segment_count=segment_count,
            prior_segment_text="\n\n".join(segment_texts)[-recent_context_characters:],
        )
        segment_text = client.complete(
            prompt,
            system=(
                f"You are the author of a {adapter['artifact']}. "
                "Return reader-facing prose only."
            ),
            temperature=temperature,
            max_output_tokens=_dynamic_token_budget(
                task,
                author_target,
                hard_max_output_tokens,
            ),
        ).strip()
        if not segment_text:
            raise RuntimeError(
                f"optimized Harness produced an empty segment at "
                f"{section['index']}:{segment_index}"
            )
        metadata = client.completion_metadata()
        segment_texts.append(segment_text)
        segment_records.append(
            {
                "index": segment_index,
                "desired_words": desired_segment,
                "author_target_words": author_target,
                "actual_words": official_length_count(segment_text),
                "finish_reason": metadata.get("finish_reason"),
            }
        )
    text = "\n\n".join(segment_texts).strip()
    if not text:
        raise RuntimeError(f"optimized Harness produced an empty section at {section['index']}")
    finish_reasons = [
        str(segment.get("finish_reason", "")).lower()
        for segment in segment_records
    ]
    return {
        "index": int(section["index"]),
        "title": str(section.get("title", "")),
        "target_words": int(section["target_words"]),
        "effective_target_words": effective_target,
        "author_target_words": sum(
            int(segment["author_target_words"]) for segment in segment_records
        ),
        "segment_count": segment_count,
        "segments": segment_records,
        "actual_words": official_length_count(text),
        "finish_reason": "length" if "length" in finish_reasons else "stop",
        "text": text,
        "completion_history": completion_history(client),
    }


def _section_prompt(
    task: LongBenchWriteTask,
    adapter_name: str,
    adapter: dict[str, Any],
    plan: dict[str, Any],
    section: dict[str, Any],
    prior: list[dict[str, Any]],
    author_target: int,
    recent_context_characters: int,
    state_excerpt_characters: int,
    *,
    segment_index: int,
    segment_count: int,
    prior_segment_text: str,
) -> str:
    minimum = round(author_target * 0.9)
    maximum = round(author_target * 1.08)
    state = _state_ledger(prior, state_excerpt_characters)
    recent = "\n\n".join(record["text"] for record in prior)[-recent_context_characters:]
    assigned_points = _segment_points(section, segment_index, segment_count)
    segment_role = _segment_role(segment_index, segment_count)
    return f"""WRITE_MORELONGWRITE_CONTRACT_V2_SECTION

ORIGINAL_USER_REQUEST:
{task.prompt}

TASK_FAMILY: {adapter_name}
DOCUMENT_CONTRACT:
{json.dumps(plan['document_contract'], ensure_ascii=False)}

FULL_SECTION_PLAN:
{json.dumps(plan['sections'], ensure_ascii=False)}

COMPLETED_STATE_LEDGER:
{json.dumps(state, ensure_ascii=False)}

RECENT_CONTINUITY_TEXT:
{recent or "[None]"}

CURRENT_SECTION_CONTRACT:
{json.dumps(section, ensure_ascii=False)}

CURRENT_SECTION_SEGMENT: {segment_index} of {segment_count}
SEGMENT_ROLE: {segment_role}
SEGMENT_ASSIGNED_POINTS:
{json.dumps(assigned_points, ensure_ascii=False)}

PRIOR_TEXT_WITHIN_CURRENT_SECTION:
{prior_segment_text or "[None]"}

TYPE_SPECIFIC_RULES:
{json.dumps(TYPE_RULES.get(task.task_type, adapter['review_focus']), ensure_ascii=False)}

Write only this prose segment of the current section. Its working budget is
{minimum}-{maximum} benchmark units, counted as Chinese characters plus English words.
Do not output the document title or a section heading; deterministic assembly supplies
both. Cover the assigned points with concrete development, preserve the document
contract and established state, and do not cover material assigned to sibling or later
sections. Continue naturally from PRIOR_TEXT_WITHIN_CURRENT_SECTION without repeating
its opening, examples, claims, or summary. Only the final segment may conclude the
section; earlier segments must end at a natural internal boundary without summarizing
the whole section. Do not output planning notes, word counts, or commentary. Silently
check that the segment is near the requested working budget before returning it."""


def _state_ledger(
    sections: list[dict[str, Any]],
    excerpt_characters: int,
) -> list[dict[str, Any]]:
    return [
        {
            "index": int(section["index"]),
            "title": section.get("title", ""),
            "actual_words": int(section["actual_words"]),
            "opening": str(section["text"])[: excerpt_characters // 2],
            "closing": str(section["text"])[-excerpt_characters:],
        }
        for section in sections
    ]


def _review_sections(
    task: LongBenchWriteTask,
    adapter: dict[str, Any],
    plan: dict[str, Any],
    sections: list[dict[str, Any]],
    review_dir: Path,
    review_client_factory,
    *,
    workers: int,
    batch_size: int,
    temperature: float,
    max_output_tokens: int,
    state_excerpt_characters: int,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    review_dir.mkdir(parents=True, exist_ok=True)
    effective_batch_size = min(batch_size, 2) if task.task_type in {
        "Academic and Technical Writing",
        "Functional and Professional Writing",
        "News and Investigative Reporting",
    } else batch_size
    batches = [
        sections[start : start + effective_batch_size]
        for start in range(0, len(sections), effective_batch_size)
    ]
    records: list[dict[str, Any]] = []
    pending: list[tuple[list[dict[str, Any]], Path]] = []
    for batch in batches:
        path = review_dir / f"{batch[0]['index']:03d}_{batch[-1]['index']:03d}.json"
        if path.exists():
            records.append(read_json(path))
        else:
            pending.append((batch, path))
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending)))) as executor:
        futures = {
            executor.submit(
                _review_batch,
                task,
                adapter,
                plan,
                batch,
                sections,
                review_client_factory,
                temperature,
                max_output_tokens,
                state_excerpt_characters,
            ): path
            for batch, path in pending
        }
        for future in as_completed(futures):
            record = future.result()
            write_json(futures[future], record)
            records.append(record)
    reviews: dict[int, dict[str, Any]] = {}
    histories: list[dict[str, Any]] = []
    for record in records:
        histories.extend(record.get("completion_history", []))
        for review in record["reviews"]:
            reviews[int(review["index"])] = review
    return reviews, histories


def _review_batch(
    task: LongBenchWriteTask,
    adapter: dict[str, Any],
    plan: dict[str, Any],
    batch: list[dict[str, Any]],
    all_sections: list[dict[str, Any]],
    review_client_factory,
    temperature: float,
    max_output_tokens: int,
    state_excerpt_characters: int,
) -> dict[str, Any]:
    indices = [int(section["index"]) for section in batch]
    preceding = [
        section for section in all_sections if int(section["index"]) < indices[0]
    ]
    following = [
        section for section in all_sections if int(section["index"]) > indices[-1]
    ]
    prompt = f"""REVIEW_MORELONGWRITE_CONTRACT_V2_BATCH_JSON

Act as a strict independent ContractFlow reviewer. Diagnose each section against
its contract, type-specific rules, and document-wide invariants. Do not rewrite.

ORIGINAL_USER_REQUEST:
{task.prompt}

DOCUMENT_CONTRACT:
{json.dumps(plan['document_contract'], ensure_ascii=False)}

TYPE_SPECIFIC_RULES:
{json.dumps(TYPE_RULES.get(task.task_type, adapter['review_focus']), ensure_ascii=False)}

PRECEDING_STATE:
{json.dumps(_state_ledger(preceding, state_excerpt_characters), ensure_ascii=False)}

SECTIONS_TO_REVIEW:
{json.dumps(batch, ensure_ascii=False)}

FOLLOWING_STATE:
{json.dumps(_state_ledger(following, state_excerpt_characters), ensure_ascii=False)}

Return exactly one JSON object:
{{
  "reviews": [
    {{
      "index": {indices[0]},
      "passed": true,
      "issues": [
        {{
          "severity": "low|medium|high",
          "category": "...",
          "message": "...",
          "suggestion": "..."
        }}
      ],
      "summary": "...",
      "state_summary": {{
        "entities_and_roles": ["..."],
        "facts_and_claims": ["..."],
        "timeline_and_world_state": ["..."],
        "open_loops_and_dependencies": ["..."]
      }}
    }}
  ]
}}

Return one review for every index in {indices}, in that order. Use high only for
a serious factual, logical, continuity, truncation, or explicit-requirement failure.
Treat unsupported quotations, precise statistics, cases, or sources as high when the
request forbids invention. For named laws, regulations, medical guidance, or technical
capabilities, flag oversimplified universal claims and unsupported claims of empirical
evidence. Do not mark deviation from a section's working target as high when the section
is complete, remains substantial, and the whole document is on track. Identify repeated
or incomplete prose explicitly."""
    client = review_client_factory()
    raw_attempts: list[str] = []
    last_error: Exception | None = None
    reviews: list[dict[str, Any]] | None = None
    for attempt in range(3):
        request = prompt if attempt == 0 else _review_retry_prompt(
            raw_attempts[-1],
            str(last_error),
            indices,
        )
        raw = client.complete(
            request,
            system="You review long-form documents against explicit contracts. Return valid JSON only.",
            temperature=temperature if attempt == 0 else 0.0,
            response_format="json_object",
            max_output_tokens=max_output_tokens,
        )
        raw_attempts.append(raw)
        try:
            parsed = extract_json_object(raw)
            candidate = parsed.get("reviews") if isinstance(parsed, dict) else None
            if not isinstance(candidate, list):
                raise ValueError("review response is missing reviews")
            returned = [int(item.get("index", 0)) for item in candidate]
            if returned != indices:
                raise ValueError(f"review indices must be {indices}, got {returned}")
            if any(not isinstance(item.get("issues", []), list) for item in candidate):
                raise ValueError("review issues must be lists")
            reviews = candidate
            break
        except (ValueError, TypeError) as error:
            last_error = error
    if reviews is None:
        raise RuntimeError(f"optimized batch review failed: {last_error}")
    return {
        "reviews": reviews,
        "raw_response_attempts": raw_attempts,
        "completion_history": completion_history(client),
    }


def _review_retry_prompt(raw: str, error: str, indices: list[int]) -> str:
    return f"""RETRY_REVIEW_MORELONGWRITE_CONTRACT_V2_JSON

Repair the previous response into one JSON object with a "reviews" list.
Return one review for every index in this exact order: {indices}. Each review
must contain index, passed, issues, summary, and state_summary. Do not output Markdown.

PARSE_ERROR:
{error}

INVALID_RESPONSE:
{raw}"""


def _load_or_create_global_review(
    task: LongBenchWriteTask,
    adapter: dict[str, Any],
    plan: dict[str, Any],
    sections: list[dict[str, Any]],
    local_reviews: dict[int, dict[str, Any]],
    path: Path,
    review_client_factory,
    *,
    temperature: float,
    max_output_tokens: int,
    state_excerpt_characters: int,
) -> dict[str, Any]:
    if path.exists():
        return read_json(path)
    document_map = [
        {
            "index": int(section["index"]),
            "title": section.get("title", ""),
            "target_words": section.get("effective_target_words", section["target_words"]),
            "actual_words": section["actual_words"],
            "opening": section["text"][:state_excerpt_characters],
            "closing": section["text"][-state_excerpt_characters:],
            "local_summary": local_reviews.get(int(section["index"]), {}).get("summary", ""),
            "state_summary": local_reviews.get(int(section["index"]), {}).get(
                "state_summary",
                {},
            ),
        }
        for section in sections
    ]
    prompt = f"""REVIEW_MORELONGWRITE_CONTRACT_V2_GLOBAL_JSON

Review the document map for long-range contradictions, unsupported factual claims,
repeated content, missing title or required components, unresolved dependencies,
timeline drift, character or role drift, and incomplete endings. Diagnose only.

ORIGINAL_USER_REQUEST:
{task.prompt}

DOCUMENT_CONTRACT:
{json.dumps(plan['document_contract'], ensure_ascii=False)}

TYPE_SPECIFIC_RULES:
{json.dumps(TYPE_RULES.get(task.task_type, adapter['review_focus']), ensure_ascii=False)}

DOCUMENT_MAP:
{json.dumps(document_map, ensure_ascii=False)}

Return exactly:
{{
  "reviews": [
    {{
      "index": 1,
      "issues": [
        {{
          "severity": "low|medium|high",
          "category": "...",
          "message": "...",
          "suggestion": "..."
        }}
      ],
      "summary": "..."
    }}
  ],
  "summary": "document-level assessment"
}}

Include only sections with document-level issues. Use high only for a serious
contract, factual, continuity, truncation, or internal-consistency failure. Prioritize
truncation, unsupported or oversimplified factual claims, contradictions, missing
requirements, and repeated prose over small section-level length deviations. A section
that is complete and substantial should not receive a high issue merely for missing its
working target when the total document length is on track."""
    client = review_client_factory()
    raw_attempts: list[str] = []
    parsed: dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(3):
        request = prompt if attempt == 0 else f"""RETRY_GLOBAL_REVIEW_JSON

Repair the previous response into exactly one valid JSON object with a "reviews"
list and a "summary" string. Preserve the diagnoses, but do not output Markdown.

PARSE_ERROR:
{last_error}

INVALID_RESPONSE:
{raw_attempts[-1]}"""
        raw = client.complete(
            request,
            system="You perform document-level ContractFlow review. Return valid JSON only.",
            temperature=temperature if attempt == 0 else 0.0,
            response_format="json_object",
            max_output_tokens=max_output_tokens,
        )
        raw_attempts.append(raw)
        try:
            candidate = extract_json_object(raw)
            if not isinstance(candidate, dict) or not isinstance(
                candidate.get("reviews", []),
                list,
            ):
                raise ValueError("global review must contain a reviews list")
            parsed = candidate
            break
        except (ValueError, TypeError) as error:
            last_error = error
    if parsed is None:
        raise RuntimeError(f"optimized global review returned invalid JSON: {last_error}")
    record = {
        "reviews": parsed.get("reviews", []),
        "summary": parsed.get("summary", ""),
        "raw_response_attempts": raw_attempts,
        "completion_history": completion_history(client),
    }
    write_json(path, record)
    return record


def _merge_reviews(
    local_reviews: dict[int, dict[str, Any]],
    global_review: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    merged = {
        index: {
            **review,
            "issues": list(review.get("issues", [])),
        }
        for index, review in local_reviews.items()
    }
    for review in global_review.get("reviews", []):
        index = int(review.get("index", 0))
        if index not in merged:
            merged[index] = {
                "index": index,
                "passed": False,
                "issues": [],
                "summary": "",
                "state_summary": {},
            }
        merged[index]["issues"].extend(review.get("issues", []))
        if review.get("summary"):
            merged[index]["summary"] = (
                f"{merged[index].get('summary', '')} {review['summary']}"
            ).strip()
    return merged


def _repair_sections(
    task: LongBenchWriteTask,
    adapter: dict[str, Any],
    plan: dict[str, Any],
    sections: list[dict[str, Any]],
    review_map: dict[int, dict[str, Any]],
    checkpoint_root: Path,
    repair_client_factory,
    review_client_factory,
    *,
    workers: int,
    review_workers: int,
    review_batch_size: int,
    repair_temperature: float,
    review_temperature: float,
    repair_max_output_tokens: int,
    review_max_output_tokens: int,
    max_repair_fraction: float,
    minimum_candidate_ratio: float,
    maximum_candidate_ratio: float,
    state_excerpt_characters: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    repair_dir = checkpoint_root / "repairs"
    repair_dir.mkdir(parents=True, exist_ok=True)
    priorities = _repair_priorities(task, sections, review_map)
    maximum_repairs = min(
        4,
        max(1, math.ceil(len(sections) * max_repair_fraction)),
    )
    selected = {index for _, index in priorities[:maximum_repairs]}
    candidates: dict[int, dict[str, Any]] = {}
    histories: list[dict[str, Any]] = []
    pending = []
    for section in sections:
        index = int(section["index"])
        if index not in selected:
            continue
        path = repair_dir / f"{index:03d}.json"
        if path.exists():
            record = read_json(path)
            candidates[index] = record
            histories.extend(record.get("completion_history", []))
        else:
            pending.append(section)
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending)))) as executor:
        futures = {
            executor.submit(
                _repair_one,
                task,
                adapter,
                plan,
                section,
                review_map.get(int(section["index"]), {}),
                repair_client_factory,
                repair_temperature,
                repair_max_output_tokens,
            ): section
            for section in pending
        }
        for future in as_completed(futures):
            record = future.result()
            index = int(record["index"])
            candidates[index] = record
            histories.extend(record.get("completion_history", []))
            write_json(repair_dir / f"{index:03d}.json", record)

    final_sections = []
    accepted = 0
    for section in sections:
        index = int(section["index"])
        candidate = candidates.get(index)
        use_candidate = False
        if candidate and candidate.get("deterministic_candidate"):
            use_candidate = (
                minimum_candidate_ratio
                <= candidate["after_words"]
                / max(1, int(section.get("effective_target_words", section["target_words"])))
                <= maximum_candidate_ratio
            )
            candidate["accepted"] = use_candidate
            candidate["post_repair_review"] = {
                "skipped": True,
                "reason": "deterministic acceptance avoids an extra benchmark-only model call",
            }
            write_json(repair_dir / f"{index:03d}.json", candidate)
        if use_candidate:
            accepted += 1
            final_sections.append(
                {
                    **section,
                    "actual_words": candidate["after_words"],
                    "finish_reason": candidate.get("finish_reason"),
                    "text": candidate["text"],
                    "review": review_map.get(index, {}),
                    "repair_history": [candidate],
                }
            )
        else:
            final_sections.append(
                {
                    **section,
                    "review": review_map.get(index, {}),
                    "repair_history": [candidate] if candidate else [],
                }
            )
    return (
        final_sections,
        histories,
        {
            "selected": len(selected),
            "attempted": len(candidates),
            "accepted": accepted,
            "acceptance_rate": round(accepted / len(candidates), 4) if candidates else 1.0,
        },
    )


def _repair_priorities(
    task: LongBenchWriteTask,
    sections: list[dict[str, Any]],
    review_map: dict[int, dict[str, Any]],
) -> list[tuple[float, int]]:
    total_words = sum(int(section["actual_words"]) for section in sections)
    document_ratio = total_words / task.target_words
    priorities: list[tuple[float, int]] = []
    for section in sections:
        index = int(section["index"])
        target = max(1, int(section.get("effective_target_words", section["target_words"])))
        ratio = int(section["actual_words"]) / target
        review = review_map.get(index, {})
        high = _high_issue_count(review)
        truncated = str(section.get("finish_reason", "")).lower() == "length"
        score = _issue_priority(review) + high * 2.0 + (10.0 if truncated else 0.0)
        if ratio < 0.55 or ratio > 1.55:
            score += 3.0 + abs(math.log(max(ratio, 0.01)))
        if document_ratio > 1.15 and ratio > 1.0:
            score += ratio - 1.0
        if document_ratio < 0.85 and ratio < 1.0:
            score += 1.0 - ratio
        if score > 0:
            priorities.append((score, index))
    return sorted(priorities, reverse=True)


def _issue_priority(review: dict[str, Any]) -> float:
    score = 0.0
    for issue in review.get("issues", []):
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity", "")).lower()
        if severity not in {"medium", "high"}:
            continue
        category = str(issue.get("category", "")).lower()
        message = str(issue.get("message", "")).lower()
        evidence = f"{category} {message}"
        weight = 1.0 if severity == "medium" else 2.0
        if any(term in evidence for term in ("truncat", "incomplete", "mid-sentence")):
            weight += 8.0
        elif any(
            term in evidence
            for term in (
                "accuracy",
                "factual",
                "unsupported",
                "fabricat",
                "citation",
                "quotation",
                "contradiction",
                "continuity",
                "consistency",
                "timeline",
                "requirement",
            )
        ):
            weight += 6.0
        elif "length" in evidence:
            weight += 0.5
        else:
            weight += 3.0
        score += weight
    return score


def _repair_one(
    task: LongBenchWriteTask,
    adapter: dict[str, Any],
    plan: dict[str, Any],
    section: dict[str, Any],
    review: dict[str, Any],
    repair_client_factory,
    temperature: float,
    hard_max_output_tokens: int,
) -> dict[str, Any]:
    index = int(section["index"])
    target = int(section.get("effective_target_words", section["target_words"]))
    actual = int(section["actual_words"])
    finish_reason = str(section.get("finish_reason", "")).lower()
    if finish_reason == "length":
        mode = "complete the truncated section and compress repetition"
    elif actual > target * 1.35:
        mode = "compress the section while preserving required content"
    elif actual < target * 0.7:
        mode = "expand only the missing contracted content"
    else:
        mode = "repair the identified factual, logical, continuity, or completeness issues"
    minimum = round(target * 0.8)
    maximum = round(target * 1.2)
    prompt = f"""REPAIR_MORELONGWRITE_CONTRACT_V2_SECTION

Perform this repair operation: {mode}.
Return a complete replacement section of {minimum}-{maximum} benchmark units.
Preserve correct content and resolve every high-severity issue. Do not add new
quotations, statistics, sources, cases, entities, world rules, or policy commitments
unless they are supported by the original request or established document state.

ORIGINAL_USER_REQUEST:
{task.prompt}

DOCUMENT_CONTRACT:
{json.dumps(plan['document_contract'], ensure_ascii=False)}

CURRENT_SECTION_CONTRACT:
{json.dumps(next(item for item in plan['sections'] if int(item['index']) == index), ensure_ascii=False)}

TYPE_SPECIFIC_RULES:
{json.dumps(TYPE_RULES.get(task.task_type, adapter['review_focus']), ensure_ascii=False)}

REVIEW:
{json.dumps(review, ensure_ascii=False)}

CURRENT_SECTION:
{section['text']}

Return only the full revised section. End at a natural complete boundary."""
    client = repair_client_factory()
    text = client.complete(
        prompt,
        system=f"You repair one contracted section of a {adapter['artifact']}. Return prose only.",
        temperature=temperature,
        max_output_tokens=_dynamic_token_budget(task, target, hard_max_output_tokens),
    ).strip()
    metadata = client.completion_metadata()
    after_words = official_length_count(text)
    deterministic = bool(
        text
        and str(metadata.get("finish_reason", "")).lower() != "length"
        and round(target * 0.65) <= after_words <= round(target * 1.35)
    )
    return {
        "index": index,
        "mode": mode,
        "before_words": actual,
        "after_words": after_words,
        "target_words": target,
        "minimum_words": minimum,
        "maximum_words": maximum,
        "finish_reason": metadata.get("finish_reason"),
        "deterministic_candidate": deterministic,
        "accepted": False,
        "text": text,
        "completion_history": completion_history(client),
    }


def _high_issue_count(review: dict[str, Any]) -> int:
    return sum(
        str(issue.get("severity", "")).lower() == "high"
        for issue in review.get("issues", [])
        if isinstance(issue, dict)
    )


def _dynamic_token_budget(
    task: LongBenchWriteTask,
    target_units: int,
    hard_max_output_tokens: int,
) -> int:
    chinese = len(re.findall(r"[\u4e00-\u9fff]", task.prompt))
    english = len(re.findall(r"\b[a-zA-Z]+\b", task.prompt))
    tokens_per_unit = 0.82 if chinese > english else 1.45
    headroom = 1.18 if chinese > english else 1.35
    estimated = math.ceil(target_units * tokens_per_unit * headroom)
    return min(hard_max_output_tokens, max(1024, estimated))


def _calibrated_author_target(
    task: LongBenchWriteTask,
    desired_units: int,
    *,
    segmented: bool,
) -> int:
    chinese = len(re.findall(r"[\u4e00-\u9fff]", task.prompt))
    english = len(re.findall(r"\b[a-zA-Z]+\b", task.prompt))
    if chinese > english:
        return max(desired_units, round(desired_units * 1.45))
    multiplier = 1.45 if segmented else 1.7
    minimum = 900 if segmented else desired_units
    return max(desired_units, minimum, round(desired_units * multiplier))


def _author_segment_count(task: LongBenchWriteTask) -> int:
    return 1


def _distribute_units(total: int, count: int) -> list[int]:
    base, remainder = divmod(total, count)
    return [base + (1 if index < remainder else 0) for index in range(count)]


def _segment_points(
    section: dict[str, Any],
    segment_index: int,
    segment_count: int,
) -> list[str]:
    points = [str(point) for point in section.get("required_points", []) if str(point)]
    if not points:
        points = [str(section.get("purpose", "Develop the contracted section."))]
    assignments = [[] for _ in range(segment_count)]
    for point_index, point in enumerate(points):
        assignments[min(segment_count - 1, point_index * segment_count // len(points))].append(
            point
        )
    selected = assignments[segment_index - 1]
    if selected:
        return selected
    return [
        "Deepen the section's argument with distinct evidence, mechanisms, implications, "
        "or scene development without repeating another segment."
    ]


def _segment_role(segment_index: int, segment_count: int) -> str:
    if segment_count == 1:
        return "complete the full section, including its natural opening and conclusion"
    if segment_index == 1:
        return "open the section and develop its first assigned claims; do not conclude"
    if segment_index == segment_count:
        return "develop the final assigned claims and close the section without recap padding"
    return "deepen the middle of the section with new analysis or causal development"


def _requested_title(prompt: str) -> str | None:
    chinese = re.search(r"\u300a([^\u300b]+)\u300b", prompt)
    if chinese:
        return chinese.group(1).strip()
    patterns = [
        r'\btitled\s+["\u201c]([^"\u201d]+)["\u201d]',
        r'\btitle\s*[:\-]\s*["\u201c]?([^"\u201d\n.]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _assemble_document(
    task: LongBenchWriteTask,
    sections: list[dict[str, Any]],
) -> str:
    title = _requested_title(task.prompt)
    assembled: list[str] = []
    for section in sections:
        index = int(section["index"])
        section_title = str(section.get("title", "")).strip() or f"Section {index}"
        body = _strip_generated_heading(
            str(section["text"]),
            index=index,
            section_title=section_title,
            document_title=title,
        )
        assembled.append(f"## {index}. {section_title}\n\n{body}".strip())
    body = "\n\n".join(assembled).strip()
    return f"# {title}\n\n{body}" if title else body


def _strip_generated_heading(
    text: str,
    *,
    index: int,
    section_title: str,
    document_title: str | None,
) -> str:
    body = text.strip()
    candidates = [section_title]
    if document_title:
        candidates.insert(0, document_title)
    for candidate in candidates:
        body = re.sub(
            rf"^\s*(?:#{{1,6}}\s*)?(?:\*\*)?{re.escape(candidate)}"
            rf"(?:\*\*)?\s*\r?\n+",
            "",
            body,
            count=1,
            flags=re.IGNORECASE,
        )
    body = re.sub(
        rf"^\s*(?:#{{1,6}}\s*)?(?:\*\*)?{index}\s*[.)-]?\s*"
        rf"{re.escape(section_title)}(?:\*\*)?\s*\r?\n+",
        "",
        body,
        count=1,
        flags=re.IGNORECASE,
    )
    return body.strip()
