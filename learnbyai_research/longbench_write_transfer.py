from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import extract_json_object
from .llm import ChatModel, completion_history


AGENTWRITE_SOURCE_REPOSITORY = ""
AGENTWRITE_SOURCE_COMMIT = "447539b356a8b09760b51eca876e19b6fc1f2dd7"
MORELONGWRITE_CONTRACT_PROFILE = "morelongwrite_contract_v1"
MORELONGWRITE_OPTIMIZED_PROFILE = "morelongwrite_contract_v2"


@dataclass(frozen=True)
class LongBenchWriteTask:
    index: int
    prompt: str
    task_type: str
    target_words: int

    @property
    def task_id(self) -> str:
        return f"longbench_write_{self.index:03d}"


@dataclass(frozen=True)
class AgentWriteStep:
    index: int
    raw: str
    target_words: int


ADAPTERS: dict[str, dict[str, Any]] = {
    "expository": {
        "artifact": "long-form expository article",
        "contract_fields": [
            "audience and assumed background",
            "coverage obligations",
            "explanation order",
            "terminology invariants",
            "examples or comparisons",
            "factual-risk constraints",
            "conclusion obligations",
        ],
        "review_focus": [
            "topic coverage",
            "factual accuracy",
            "explanatory progression",
            "terminology consistency",
            "redundancy",
        ],
    },
    "academic": {
        "artifact": "academic or technical report",
        "contract_fields": [
            "research purpose and scope",
            "required sections",
            "claim dependencies",
            "technical definitions and notation",
            "evidence and qualification obligations",
            "limitations",
            "conclusion obligations",
        ],
        "review_focus": [
            "argument structure",
            "technical accuracy",
            "notation consistency",
            "unsupported claims",
            "cross-section dependencies",
        ],
    },
    "narrative": {
        "artifact": "long-form narrative",
        "contract_fields": [
            "characters and relationships",
            "setting and timeline",
            "point of view",
            "required plot events",
            "character-state transitions",
            "genre and tone",
            "ending obligations",
        ],
        "review_focus": [
            "plot progression",
            "character consistency",
            "timeline and world-state consistency",
            "point-of-view consistency",
            "ending satisfaction",
        ],
    },
}


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL record at line {line_number}: {error}") from error
            records.append(record)
    return records


def task_from_record(index: int, record: dict[str, Any]) -> LongBenchWriteTask:
    return LongBenchWriteTask(
        index=index,
        prompt=str(record["prompt"]),
        task_type=str(record["type"]),
        target_words=int(record["length"]),
    )


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def official_length_count(text: str) -> int:
    chinese_characters = re.findall(r"[\u4e00-\u9fff]", text)
    english_words = re.findall(r"\b[a-zA-Z]+\b", text)
    return len(chinese_characters) + len(english_words)


def contract_section_word_bounds(
    target_words: int,
    profile: str = "legacy_longbench_write",
) -> tuple[int, int]:
    if profile == MORELONGWRITE_OPTIMIZED_PROFILE:
        if target_words <= 16000:
            return 1000, 1800
        if target_words <= 32000:
            return 1200, 2200
        return 1200, 2200
    if profile == MORELONGWRITE_CONTRACT_PROFILE:
        if target_words <= 16000:
            return 1200, 2800
        if target_words <= 32000:
            return 2000, 3500
        return 3200, 4800
    if target_words <= 20000:
        return 350, 1000
    return 1000, 4000


def contract_section_bounds(
    target_words: int,
    profile: str = "legacy_longbench_write",
) -> tuple[int, int]:
    if profile == MORELONGWRITE_OPTIMIZED_PROFILE:
        if target_words <= 16000:
            return 12, 12
        if target_words <= 32000:
            return 20, 20
        return 40, 40
    if profile == MORELONGWRITE_CONTRACT_PROFILE:
        if target_words <= 16000:
            return 8, 8
        if target_words <= 32000:
            return 12, 12
        return 16, 16
    minimum_words, maximum_words = contract_section_word_bounds(target_words, profile)
    minimum_sections = max(5, (target_words + maximum_words - 1) // maximum_words)
    if target_words <= 10000:
        return minimum_sections, 10
    if target_words <= 20000:
        return minimum_sections, min(
            20,
            max(minimum_sections, target_words // minimum_words),
        )
    maximum_sections = min(
        64,
        max(minimum_sections, target_words // minimum_words),
    )
    return minimum_sections, maximum_sections


def run_direct(
    task: LongBenchWriteTask,
    client: ChatModel,
    *,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    text = client.complete(
        task.prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    ).strip()
    if not text:
        raise RuntimeError("Direct produced an empty response")
    return generation_record(
        task,
        "direct",
        text,
        started,
        completion_history(client),
        adaptations=["none; the user message is the original benchmark prompt verbatim"],
    )


def run_agentwrite(
    task: LongBenchWriteTask,
    planning_client: ChatModel,
    writing_client_factory,
    *,
    source_root: str | Path,
    checkpoint_dir: str | Path,
    planning_temperature: float,
    writing_temperature: float,
    planning_max_output_tokens: int,
    paragraph_max_output_tokens: int,
    max_plan_attempts: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    source_root = Path(source_root)
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    plan_template = (source_root / "agentwrite" / "prompts" / "plan.txt").read_text(encoding="utf-8")
    write_template = (source_root / "agentwrite" / "prompts" / "write.txt").read_text(encoding="utf-8")
    plan_path = checkpoint_root / "plan.json"

    if plan_path.exists():
        plan_record = read_json(plan_path)
        plan = str(plan_record["plan"])
        steps = [AgentWriteStep(**item) for item in plan_record["steps"]]
        histories = list(plan_record.get("completion_history", []))
    else:
        plan_prompt = plan_template.replace("$INST$", task.prompt)
        last_error: Exception | None = None
        for _ in range(max_plan_attempts):
            plan = planning_client.complete(
                plan_prompt,
                temperature=planning_temperature,
                max_output_tokens=planning_max_output_tokens,
            ).strip()
            try:
                steps = parse_agentwrite_plan(plan)
                break
            except ValueError as error:
                last_error = error
        else:
            raise RuntimeError(f"AgentWrite planning failed: {last_error}") from last_error
        histories = completion_history(planning_client)
        write_json(
            plan_path,
            {
                "plan": plan,
                "steps": [step.__dict__ for step in steps],
                "completion_history": histories,
            },
        )

    if len(steps) > 50:
        raise RuntimeError(f"AgentWrite official 50-step guard exceeded: {len(steps)}")

    text = ""
    responses: list[str] = []
    segment_dir = checkpoint_root / "paragraphs"
    segment_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        step_path = segment_dir / f"{step.index:03d}.json"
        if step_path.exists():
            record = read_json(step_path)
            response = str(record["response"])
            step_history = list(record.get("completion_history", []))
        else:
            prompt = (
                write_template.replace("$INST$", task.prompt)
                .replace("$PLAN$", plan.strip())
                .replace("$TEXT$", text.strip())
                .replace("$STEP$", step.raw.strip())
            )
            client = writing_client_factory()
            response = client.complete(
                prompt,
                temperature=writing_temperature,
                max_output_tokens=paragraph_max_output_tokens,
            ).strip()
            if not response:
                raise RuntimeError(f"AgentWrite produced an empty paragraph at step {step.index}")
            step_history = completion_history(client)
            write_json(
                step_path,
                {
                    "step": step.__dict__,
                    "response": response,
                    "completion_history": step_history,
                },
            )
        responses.append(response)
        text += response + "\n\n"
        histories.extend(step_history)

    return generation_record(
        task,
        "agentwrite",
        text.strip(),
        started,
        histories,
        plan=plan,
        units=[{"index": step.index, "target_words": step.target_words, "text": response}
               for step, response in zip(steps, responses)],
        adaptations=[
            "official plan.txt and write.txt loaded verbatim from the pinned source tree",
            "OpenAI-compatible API and model configuration",
            "resumable checkpoints",
            "configured output limits and retries",
        ],
        extra_metadata={
            "source_repository": AGENTWRITE_SOURCE_REPOSITORY,
            "source_commit": AGENTWRITE_SOURCE_COMMIT,
            "source_files": [
                "agentwrite/plan.py",
                "agentwrite/write.py",
                "agentwrite/prompts/plan.txt",
                "agentwrite/prompts/write.txt",
            ],
        },
    )


def run_harness(
    task: LongBenchWriteTask,
    adapter_name: str,
    author_client: ChatModel,
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
    max_plan_attempts: int,
    max_repair_rounds_per_section: int,
    minimum_section_length_ratio: float,
    maximum_section_length_ratio: float = 1.2,
    profile: str = "legacy_longbench_write",
    review_batch_size: int = 1,
    repair_severities: list[str] | None = None,
    continuity_context_characters: int = 12000,
) -> dict[str, Any]:
    if profile == MORELONGWRITE_CONTRACT_PROFILE:
        return run_morelongwrite_harness(
            task,
            adapter_name,
            author_client,
            review_client_factory,
            repair_client_factory,
            checkpoint_dir=checkpoint_dir,
            planning_temperature=planning_temperature,
            writing_temperature=writing_temperature,
            review_temperature=review_temperature,
            repair_temperature=repair_temperature,
            planning_max_output_tokens=planning_max_output_tokens,
            section_max_output_tokens=section_max_output_tokens,
            review_max_output_tokens=review_max_output_tokens,
            repair_max_output_tokens=repair_max_output_tokens,
            max_plan_attempts=max_plan_attempts,
            max_repair_rounds_per_section=max_repair_rounds_per_section,
            minimum_section_length_ratio=minimum_section_length_ratio,
            maximum_section_length_ratio=maximum_section_length_ratio,
            review_batch_size=review_batch_size,
            repair_severities=repair_severities or ["high"],
            continuity_context_characters=continuity_context_characters,
        )
    return run_legacy_longbench_write_harness(
        task,
        adapter_name,
        author_client,
        review_client_factory,
        repair_client_factory,
        checkpoint_dir=checkpoint_dir,
        planning_temperature=planning_temperature,
        writing_temperature=writing_temperature,
        review_temperature=review_temperature,
        repair_temperature=repair_temperature,
        planning_max_output_tokens=planning_max_output_tokens,
        section_max_output_tokens=section_max_output_tokens,
        review_max_output_tokens=review_max_output_tokens,
        repair_max_output_tokens=repair_max_output_tokens,
        max_plan_attempts=max_plan_attempts,
        max_repair_rounds_per_section=max_repair_rounds_per_section,
        minimum_section_length_ratio=minimum_section_length_ratio,
    )


def run_legacy_longbench_write_harness(
    task: LongBenchWriteTask,
    adapter_name: str,
    author_client: ChatModel,
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
    max_plan_attempts: int,
    max_repair_rounds_per_section: int,
    minimum_section_length_ratio: float,
) -> dict[str, Any]:
    if adapter_name not in ADAPTERS:
        raise ValueError(f"unknown task-family adapter: {adapter_name}")
    started = time.perf_counter()
    adapter = ADAPTERS[adapter_name]
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    plan_path = checkpoint_root / "contract_plan.json"
    histories: list[dict[str, Any]] = []

    if plan_path.exists():
        plan = read_json(plan_path)
    else:
        base_prompt = build_contract_plan_prompt(task, adapter_name, adapter)
        prompt = base_prompt
        last_error: Exception | None = None
        for _ in range(max_plan_attempts):
            raw = author_client.complete(
                prompt,
                system="You compile document contracts from user requests. Return valid JSON only.",
                temperature=planning_temperature,
                response_format="json_object",
                max_output_tokens=planning_max_output_tokens,
            )
            try:
                plan = parse_contract_plan(raw, task.target_words)
                break
            except (ValueError, json.JSONDecodeError) as error:
                last_error = error
                prompt = build_contract_plan_retry_prompt(
                    base_prompt,
                    raw,
                    str(error),
                    task.target_words,
                )
        else:
            raise RuntimeError(f"Harness contract planning failed: {last_error}") from last_error
        histories.extend(completion_history(author_client))
        write_json(plan_path, plan)

    sections: list[dict[str, Any]] = []
    section_dir = checkpoint_root / "sections"
    section_dir.mkdir(parents=True, exist_ok=True)
    for section in plan["sections"]:
        index = int(section["index"])
        section_path = section_dir / f"{index:03d}.json"
        if section_path.exists():
            record = read_json(section_path)
            sections.append(record)
            histories.extend(record.get("completion_history", []))
            continue

        prior_text = "\n\n".join(item["text"] for item in sections)
        text = author_client.complete(
            build_section_prompt(task, adapter_name, plan, section, prior_text),
            system=f"You are the author of a {adapter['artifact']}. Return reader-facing prose only.",
            temperature=writing_temperature,
            max_output_tokens=section_max_output_tokens,
        ).strip()
        author_history = [author_client.completion_metadata()]
        review_history: list[dict[str, Any]] = []
        repair_history: list[dict[str, Any]] = []
        review = review_section(
            task,
            adapter,
            plan,
            section,
            prior_text,
            text,
            review_client_factory,
            temperature=review_temperature,
            max_output_tokens=review_max_output_tokens,
        )
        review_history.extend(review.pop("_completion_history", []))
        minimum_words = round(int(section["target_words"]) * minimum_section_length_ratio)
        needs_repair = official_length_count(text) < minimum_words or any(
            issue.get("severity") in {"high", "medium"} for issue in review.get("issues", [])
        )
        for repair_round in range(max_repair_rounds_per_section if needs_repair else 0):
            repair_client = repair_client_factory()
            candidate = repair_client.complete(
                build_repair_prompt(task, adapter, plan, section, prior_text, text, review, minimum_words),
                system=f"You revise one complete section of a {adapter['artifact']}. Return the full revised section only.",
                temperature=repair_temperature,
                max_output_tokens=repair_max_output_tokens,
            ).strip()
            repair_entry = {
                "round": repair_round + 1,
                "before_words": official_length_count(text),
                "after_words": official_length_count(candidate),
                "accepted": bool(candidate),
                "issues": review.get("issues", []),
                "completion_history": completion_history(repair_client),
            }
            repair_history.append(repair_entry)
            if candidate:
                text = candidate
            break

        record = {
            "index": index,
            "title": section["title"],
            "target_words": int(section["target_words"]),
            "actual_words": official_length_count(text),
            "text": text,
            "review": review,
            "repair_history": repair_history,
            "completion_history": [*author_history, *review_history, *[
                item for repair in repair_history for item in repair["completion_history"]
            ]],
        }
        sections.append(record)
        histories.extend(record["completion_history"])
        write_json(section_path, record)

    final_text = "\n\n".join(section["text"] for section in sections)
    return generation_record(
        task,
        "harness_full",
        final_text,
        started,
        histories,
        plan=plan,
        units=sections,
        adaptations=[
            f"frozen task-family adapter: {adapter_name}",
            "automatic document-contract instantiation from the original prompt",
            "contract-guided section generation with prior-text context",
            "independent section review and at most one complete-section repair",
            "no task-specific manual contract editing",
        ],
        extra_metadata={"adapter": adapter_name},
    )


def run_morelongwrite_harness(
    task: LongBenchWriteTask,
    adapter_name: str,
    author_client: ChatModel,
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
    max_plan_attempts: int,
    max_repair_rounds_per_section: int,
    minimum_section_length_ratio: float,
    maximum_section_length_ratio: float,
    review_batch_size: int,
    repair_severities: list[str],
    continuity_context_characters: int,
) -> dict[str, Any]:
    if adapter_name not in ADAPTERS:
        raise ValueError(f"unknown task-family adapter: {adapter_name}")
    if review_batch_size < 1:
        raise ValueError("review_batch_size must be positive")
    started = time.perf_counter()
    adapter = ADAPTERS[adapter_name]
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    plan_path = checkpoint_root / "contract_plan.json"
    histories: list[dict[str, Any]] = []

    if plan_path.exists():
        plan_record = read_json(plan_path)
        plan = dict(plan_record["plan"])
        histories.extend(plan_record.get("completion_history", []))
    else:
        base_prompt = build_contract_plan_prompt(
            task,
            adapter_name,
            adapter,
            profile=MORELONGWRITE_CONTRACT_PROFILE,
        )
        prompt = base_prompt
        last_error: Exception | None = None
        for _ in range(max_plan_attempts):
            raw = author_client.complete(
                prompt,
                system="You compile document contracts from user requests. Return valid JSON only.",
                temperature=planning_temperature,
                response_format="json_object",
                max_output_tokens=planning_max_output_tokens,
            )
            try:
                plan = parse_contract_plan(
                    raw,
                    task.target_words,
                    profile=MORELONGWRITE_CONTRACT_PROFILE,
                )
                break
            except (ValueError, json.JSONDecodeError) as error:
                last_error = error
                prompt = build_contract_plan_retry_prompt(
                    base_prompt,
                    raw,
                    str(error),
                    task.target_words,
                    profile=MORELONGWRITE_CONTRACT_PROFILE,
                )
        else:
            raise RuntimeError(
                f"MoreLongWrite contract planning failed: {last_error}"
            ) from last_error
        plan_history = completion_history(author_client)
        histories.extend(plan_history)
        write_json(
            plan_path,
            {
                "profile": MORELONGWRITE_CONTRACT_PROFILE,
                "plan": plan,
                "completion_history": plan_history,
            },
        )

    sections: list[dict[str, Any]] = []
    section_dir = checkpoint_root / "sections"
    section_dir.mkdir(parents=True, exist_ok=True)
    for section in plan["sections"]:
        index = int(section["index"])
        section_path = section_dir / f"{index:03d}.json"
        if section_path.exists():
            record = read_json(section_path)
            sections.append(record)
            histories.extend(record.get("completion_history", []))
            continue

        prior_text = "\n\n".join(item["text"] for item in sections)
        text = author_client.complete(
            build_morelongwrite_section_prompt(
                task,
                adapter_name,
                plan,
                section,
                sections,
                prior_text[-continuity_context_characters:],
            ),
            system=f"You are the author of a {adapter['artifact']}. Return reader-facing prose only.",
            temperature=writing_temperature,
            max_output_tokens=section_max_output_tokens,
        ).strip()
        if not text:
            raise RuntimeError(
                f"MoreLongWrite Harness produced an empty section at {index}"
            )
        record = {
            "index": index,
            "title": section["title"],
            "target_words": int(section["target_words"]),
            "actual_words": official_length_count(text),
            "text": text,
            "completion_history": [author_client.completion_metadata()],
        }
        sections.append(record)
        histories.extend(record["completion_history"])
        write_json(section_path, record)

    review_map: dict[int, dict[str, Any]] = {}
    review_dir = checkpoint_root / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(sections), review_batch_size):
        batch = sections[start : start + review_batch_size]
        batch_path = review_dir / f"{batch[0]['index']:03d}_{batch[-1]['index']:03d}.json"
        if batch_path.exists():
            batch_record = read_json(batch_path)
        else:
            preceding_text = "\n\n".join(
                item["text"] for item in sections[:start]
            )[-continuity_context_characters:]
            following_text = "\n\n".join(
                item["text"] for item in sections[start + len(batch) :]
            )[:continuity_context_characters]
            batch_record = review_section_batch(
                task,
                adapter,
                plan,
                batch,
                preceding_text,
                following_text,
                review_client_factory,
                temperature=review_temperature,
                max_output_tokens=review_max_output_tokens,
            )
            write_json(batch_path, batch_record)
        histories.extend(batch_record.get("completion_history", []))
        for review in batch_record["reviews"]:
            review_map[int(review["index"])] = review

    repair_dir = checkpoint_root / "repairs"
    repair_dir.mkdir(parents=True, exist_ok=True)
    final_sections: list[dict[str, Any]] = []
    repair_severity_set = {value.lower() for value in repair_severities}
    for section, contract in zip(sections, plan["sections"]):
        index = int(section["index"])
        target_words = int(contract["target_words"])
        minimum_words = round(target_words * minimum_section_length_ratio)
        maximum_words = round(target_words * maximum_section_length_ratio)
        review = review_map.get(
            index,
            {
                "index": index,
                "passed": False,
                "issues": [
                    {
                        "severity": "high",
                        "category": "review_missing",
                        "message": "The batch reviewer omitted this section.",
                        "suggestion": "Check the section against its complete contract.",
                    }
                ],
                "summary": "Missing review result.",
            },
        )
        severe_issues = [
            issue
            for issue in review.get("issues", [])
            if str(issue.get("severity", "")).lower() in repair_severity_set
        ]
        original_words = official_length_count(section["text"])
        length_out_of_bounds = (
            original_words < minimum_words or original_words > maximum_words
        )
        needs_repair = bool(severe_issues or length_out_of_bounds)
        repair_history: list[dict[str, Any]] = []
        text = str(section["text"])
        repair_path = repair_dir / f"{index:03d}.json"

        if needs_repair and max_repair_rounds_per_section:
            if repair_path.exists():
                repair_record = read_json(repair_path)
                repair_history = [repair_record]
                histories.extend(repair_record.get("completion_history", []))
                if repair_record.get("accepted"):
                    text = str(repair_record["text"])
            else:
                repair_client = repair_client_factory()
                candidate = repair_client.complete(
                    build_morelongwrite_repair_prompt(
                        task,
                        adapter,
                        plan,
                        contract,
                        text,
                        review,
                        minimum_words,
                        maximum_words,
                    ),
                    system=f"You revise one complete section of a {adapter['artifact']}. Return the full revised section only.",
                    temperature=repair_temperature,
                    max_output_tokens=repair_max_output_tokens,
                ).strip()
                candidate_words = official_length_count(candidate)
                accepted = bool(
                    candidate
                    and minimum_words <= candidate_words <= maximum_words
                )
                repair_record = {
                    "index": index,
                    "before_words": original_words,
                    "after_words": candidate_words,
                    "minimum_words": minimum_words,
                    "maximum_words": maximum_words,
                    "accepted": accepted,
                    "triggered_by_length": length_out_of_bounds,
                    "triggered_issue_severities": [
                        issue.get("severity") for issue in severe_issues
                    ],
                    "text": candidate,
                    "completion_history": completion_history(repair_client),
                }
                write_json(repair_path, repair_record)
                repair_history = [repair_record]
                histories.extend(repair_record["completion_history"])
                if accepted:
                    text = candidate
        final_sections.append(
            {
                **section,
                "actual_words": official_length_count(text),
                "text": text,
                "review": review,
                "repair_history": repair_history,
            }
        )

    final_text = "\n\n".join(section["text"] for section in final_sections)
    return generation_record(
        task,
        "harness_full",
        final_text,
        started,
        histories,
        plan=plan,
        units=final_sections,
        adaptations=[
            f"MoreLongWrite benchmark profile: {MORELONGWRITE_CONTRACT_PROFILE}",
            f"frozen task-family adapter: {adapter_name}",
            "contract-guided sequential section generation",
            f"batched independent review with batch size {review_batch_size}",
            "at most one complete-section repair for configured severe issues or deterministic length violations",
            "global contract plus section ledger plus rolling continuity context",
            "no task-specific manual contract editing",
        ],
        extra_metadata={
            "adapter": adapter_name,
            "profile": MORELONGWRITE_CONTRACT_PROFILE,
            "review_batch_size": review_batch_size,
            "repair_severities": sorted(repair_severity_set),
            "minimum_section_length_ratio": minimum_section_length_ratio,
            "maximum_section_length_ratio": maximum_section_length_ratio,
            "continuity_context_characters": continuity_context_characters,
        },
    )


def parse_agentwrite_plan(plan: str) -> list[AgentWriteStep]:
    lines = [line.strip() for line in plan.splitlines() if line.strip()]
    pattern = re.compile(
        r"^Paragraph\s+(\d+)\s*-\s*Main Point:\s*(.*?)\s*-\s*Word Count:\s*(\d+)\s*words?\.?$",
        flags=re.IGNORECASE,
    )
    steps: list[AgentWriteStep] = []
    for line in lines:
        match = pattern.match(line)
        if not match:
            raise ValueError(f"unparseable AgentWrite plan line: {line[:160]}")
        steps.append(
            AgentWriteStep(
                index=int(match.group(1)),
                raw=line,
                target_words=int(match.group(3)),
            )
        )
    if not steps:
        raise ValueError("AgentWrite returned no plan steps")
    if [step.index for step in steps] != list(range(1, len(steps) + 1)):
        raise ValueError("AgentWrite paragraph indices are not consecutive")
    return steps


def build_contract_plan_prompt(
    task: LongBenchWriteTask,
    adapter_name: str,
    adapter: dict[str, Any],
    *,
    profile: str = "legacy_longbench_write",
) -> str:
    minimum_sections, maximum_sections = contract_section_bounds(
        task.target_words,
        profile,
    )
    minimum_section_words, maximum_section_words = contract_section_word_bounds(
        task.target_words,
        profile,
    )
    return f"""DOCUMENT_CONTRACT_PLAN_JSON

Compile a document contract and section plan from the original user request.
The task-family adapter is frozen before evaluation. Do not invent task-specific
requirements that are unsupported by the request. Operationalize ambiguous requests
conservatively. The final document should target approximately {task.target_words}
benchmark length units, counted as Chinese characters plus English words.

TASK_FAMILY: {adapter_name}
ARTIFACT: {adapter['artifact']}
CONTRACT_VOCABULARY:
{json.dumps(adapter['contract_fields'], ensure_ascii=False)}

ORIGINAL_USER_REQUEST:
{task.prompt}

Return exactly one JSON object:
{{
  "document_contract": {{
    "artifact_type": "...",
    "global_goal": "...",
    "audience": "...",
    "tone_and_style": ["..."],
    "required_content": ["..."],
    "explicit_constraints": ["..."],
    "global_invariants": ["..."],
    "forbidden_or_unsupported": ["..."],
    "completion_criteria": ["..."]
  }},
  "sections": [
    {{
      "index": 1,
      "title": "...",
      "purpose": "...",
      "target_words": 600,
      "required_points": ["..."],
      "depends_on": [],
      "continuity_checks": ["..."]
    }}
  ]
}}

Rules:
- Use {minimum_sections} to {maximum_sections} sections.
- Section indices must be consecutive starting at 1.
- Each section target must be between {minimum_section_words} and {maximum_section_words} benchmark length units.
- Section targets must sum to between 95% and 105% of {task.target_words}.
- Cover every explicit user requirement.
- Preserve genre-specific continuity obligations across sections.
- Do not output prose outside the JSON object."""


def build_contract_plan_retry_prompt(
    original_prompt: str,
    invalid_response: str,
    error: str,
    target_words: int,
    *,
    profile: str = "legacy_longbench_write",
) -> str:
    minimum_sections, maximum_sections = contract_section_bounds(
        target_words,
        profile,
    )
    minimum_section_words, maximum_section_words = contract_section_word_bounds(
        target_words,
        profile,
    )
    return f"""{original_prompt}

SCHEMA_RETRY

The prior contract plan failed deterministic validation:
{error}

Repair the plan while preserving the same content obligations. Return the complete JSON
object again. For a {target_words}-word document under the frozen
{minimum_section_words}-{maximum_section_words} words per
section bounds, use between {minimum_sections} and {maximum_sections} sections. Ensure
the section targets sum to 95%-105% of {target_words}. Do not remove explicit user
requirements merely to satisfy the numeric constraints.

INVALID_PRIOR_RESPONSE:
{invalid_response}"""


def parse_contract_plan(
    raw: str,
    target_words: int,
    *,
    profile: str = "legacy_longbench_write",
) -> dict[str, Any]:
    plan = extract_json_object(raw)
    if not isinstance(plan, dict):
        raise ValueError("contract plan is not an object")
    contract = plan.get("document_contract")
    sections = plan.get("sections")
    if not isinstance(contract, dict) or not isinstance(sections, list):
        raise ValueError("contract plan is missing document_contract or sections")
    minimum_sections, maximum_sections = contract_section_bounds(
        target_words,
        profile,
    )
    minimum_section_words, maximum_section_words = contract_section_word_bounds(
        target_words,
        profile,
    )
    if not minimum_sections <= len(sections) <= maximum_sections:
        raise ValueError(
            "contract plan must contain "
            f"{minimum_sections}-{maximum_sections} sections, got {len(sections)}"
        )
    indices = [int(section.get("index", 0)) for section in sections if isinstance(section, dict)]
    if indices != list(range(1, len(sections) + 1)):
        raise ValueError("contract section indices are not consecutive")
    targets = [int(section.get("target_words", 0)) for section in sections]
    if any(value <= 0 for value in targets):
        raise ValueError(f"contract section target must be positive: {targets}")
    total = sum(targets)
    if (
        any(
            value < minimum_section_words or value > maximum_section_words
            for value in targets
        )
        or total < target_words * 0.95
        or total > target_words * 1.05
    ):
        normalized = normalize_section_targets(
            targets,
            target_words,
            profile=profile,
        )
        for section, value in zip(sections, normalized):
            section["target_words"] = value
        plan["budget_normalization"] = {
            "original_section_targets": targets,
            "original_total": total,
            "normalized_section_targets": normalized,
            "normalized_total": sum(normalized),
            "policy": "proportional deterministic normalization to the benchmark target",
        }
    return plan


def normalize_section_targets(
    targets: list[int],
    target_words: int,
    *,
    profile: str = "legacy_longbench_write",
) -> list[int]:
    minimum_words, maximum_words = contract_section_word_bounds(
        target_words,
        profile,
    )
    if (
        not targets
        or target_words < minimum_words * len(targets)
        or target_words > maximum_words * len(targets)
    ):
        raise ValueError(
            f"cannot normalize {len(targets)} section targets to {target_words} words "
            f"within the frozen {minimum_words}-{maximum_words} per-section bounds"
        )
    scale = target_words / sum(targets)
    normalized = [
        min(maximum_words, max(minimum_words, round(value * scale)))
        for value in targets
    ]
    delta = target_words - sum(normalized)
    direction = 1 if delta > 0 else -1
    while delta:
        changed = False
        for index in range(len(normalized)):
            candidate = normalized[index] + direction
            if minimum_words <= candidate <= maximum_words:
                normalized[index] = candidate
                delta -= direction
                changed = True
                if delta == 0:
                    break
        if not changed:
            raise ValueError("section budget normalization could not satisfy the frozen bounds")
    return normalized


def build_section_prompt(
    task: LongBenchWriteTask,
    adapter_name: str,
    plan: dict[str, Any],
    section: dict[str, Any],
    prior_text: str,
) -> str:
    return f"""WRITE_DOCUMENT_SECTION

ORIGINAL_USER_REQUEST:
{task.prompt}

TASK_FAMILY: {adapter_name}
DOCUMENT_CONTRACT:
{json.dumps(plan['document_contract'], ensure_ascii=False)}

FULL_SECTION_PLAN:
{json.dumps(plan['sections'], ensure_ascii=False)}

CURRENT_SECTION:
{json.dumps(section, ensure_ascii=False)}

TEXT_ALREADY_WRITTEN:
{prior_text or "[This is the first section.]"}

Write only the complete current section. Aim for {section['target_words']} benchmark
length units, counted as Chinese characters plus English words, and
do not fall below 80% of that target. Satisfy the current section obligations while
preserving all global invariants and continuity with prior text. Do not repeat prior
sections. Use a concise section heading when appropriate. Do not include planning notes,
contract labels, word counts, or meta-commentary."""


def build_morelongwrite_section_prompt(
    task: LongBenchWriteTask,
    adapter_name: str,
    plan: dict[str, Any],
    section: dict[str, Any],
    prior_sections: list[dict[str, Any]],
    recent_prior_text: str,
) -> str:
    ledger = [
        {
            "index": item["index"],
            "title": item["title"],
            "target_words": item["target_words"],
            "actual_words": item["actual_words"],
        }
        for item in prior_sections
    ]
    target_words = int(section["target_words"])
    minimum_words = round(target_words * 0.9)
    maximum_words = round(target_words * 1.1)
    return f"""WRITE_MORELONGWRITE_SECTION

ORIGINAL_USER_REQUEST:
{task.prompt}

TASK_FAMILY: {adapter_name}
DOCUMENT_CONTRACT:
{json.dumps(plan['document_contract'], ensure_ascii=False)}

FULL_SECTION_PLAN:
{json.dumps(plan['sections'], ensure_ascii=False)}

COMPLETED_SECTION_LEDGER:
{json.dumps(ledger, ensure_ascii=False)}

RECENT_PRIOR_TEXT:
{recent_prior_text or "[This is the first section.]"}

CURRENT_SECTION:
{json.dumps(section, ensure_ascii=False)}

Write only the complete current section. The hard working range is
{minimum_words}-{maximum_words} benchmark length units, counted as Chinese
characters plus English words. Cover every current-section obligation, preserve
the global contract and long-range dependencies, and continue naturally from
the recent text. Do not repeat completed sections or add material assigned to a
later section. Stop when the current section is complete. Do not include
planning notes, contract labels, word counts, or meta-commentary."""


def review_section_batch(
    task: LongBenchWriteTask,
    adapter: dict[str, Any],
    plan: dict[str, Any],
    sections: list[dict[str, Any]],
    preceding_text: str,
    following_text: str,
    review_client_factory,
    *,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    contract_by_index = {
        int(section["index"]): section for section in plan["sections"]
    }
    review_payload = [
        {
            "contract": contract_by_index[int(section["index"])],
            "actual_words": section["actual_words"],
            "text": section["text"],
        }
        for section in sections
    ]
    expected_indices = [int(section["index"]) for section in sections]
    prompt = f"""REVIEW_MORELONGWRITE_SECTION_BATCH_JSON

Act as a strict independent reviewer. Review each section against its own
contract and the document-wide invariants. Diagnose only; do not rewrite.

ORIGINAL_USER_REQUEST:
{task.prompt}

DOCUMENT_CONTRACT:
{json.dumps(plan['document_contract'], ensure_ascii=False)}

REVIEW_FOCUS:
{json.dumps(adapter['review_focus'], ensure_ascii=False)}

PRECEDING_CONTINUITY_WINDOW:
{preceding_text or "[None]"}

SECTIONS_TO_REVIEW:
{json.dumps(review_payload, ensure_ascii=False)}

FOLLOWING_CONTINUITY_WINDOW:
{following_text or "[None]"}

Return exactly one JSON object:
{{
  "reviews": [
    {{
      "index": {expected_indices[0]},
      "passed": true,
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
  ]
}}

Return exactly one review for every index in {expected_indices}. Use high only
for a serious factual, logical, continuity, safety, or explicit-requirement
failure. Use medium for a material but non-blocking gap. Do not fail solely for
minor style preferences or small length deviations."""
    client = review_client_factory()
    raw_responses: list[str] = []
    last_error: Exception | None = None
    reviews: list[dict[str, Any]] | None = None
    for attempt in range(3):
        request = (
            prompt
            if attempt == 0
            else build_batch_review_json_retry_prompt(
                raw_responses[-1],
                str(last_error),
                expected_indices,
            )
        )
        raw = client.complete(
            request,
            system=(
                "You are an independent long-form document reviewer. Return valid JSON only."
                if attempt == 0
                else "You repair invalid reviewer JSON. Return the required JSON object only."
            ),
            temperature=temperature if attempt == 0 else 0.0,
            response_format="json_object",
            max_output_tokens=max_output_tokens,
        )
        raw_responses.append(raw)
        try:
            parsed = extract_json_object(raw)
            candidate = parsed.get("reviews") if isinstance(parsed, dict) else None
            if not isinstance(candidate, list):
                raise ValueError("batch reviewer did not return a reviews list")
            indices = [
                int(item.get("index", 0))
                for item in candidate
                if isinstance(item, dict)
            ]
            if indices != expected_indices:
                raise ValueError(
                    f"batch review indices must be {expected_indices}, got {indices}"
                )
            if any(
                not isinstance(item.get("issues", []), list)
                for item in candidate
            ):
                raise ValueError("batch reviewer returned invalid issues")
            reviews = candidate
            break
        except (ValueError, TypeError) as error:
            last_error = error
    if reviews is None:
        raise RuntimeError(
            f"batch reviewer failed JSON schema after 3 attempts: {last_error}"
        )
    return {
        "reviews": reviews,
        "raw_response_attempts": raw_responses,
        "completion_history": completion_history(client),
    }


def build_batch_review_json_retry_prompt(
    raw: str,
    error: str,
    expected_indices: list[int],
) -> str:
    return f"""RETRY_REVIEW_MORELONGWRITE_SECTION_BATCH_JSON

Repair the prior response into exactly one JSON object with a "reviews" list.
Return exactly one review for each index in this order: {expected_indices}.
Every review must contain index, passed, issues, and summary. Every issue must
contain severity, category, message, and suggestion. Do not output Markdown.

PARSE_ERROR:
{error}

INVALID_RESPONSE:
{raw}"""


def build_morelongwrite_repair_prompt(
    task: LongBenchWriteTask,
    adapter: dict[str, Any],
    plan: dict[str, Any],
    section: dict[str, Any],
    text: str,
    review: dict[str, Any],
    minimum_words: int,
    maximum_words: int,
) -> str:
    return f"""REPAIR_MORELONGWRITE_SECTION

Return a complete replacement for the current section. Preserve correct
content, resolve every high-severity issue, and fit the deterministic length
range of {minimum_words}-{maximum_words} benchmark units. Do not expand beyond
the current section's assigned scope.

ORIGINAL_USER_REQUEST:
{task.prompt}

DOCUMENT_CONTRACT:
{json.dumps(plan['document_contract'], ensure_ascii=False)}

CURRENT_SECTION_CONTRACT:
{json.dumps(section, ensure_ascii=False)}

REVIEW_FOCUS:
{json.dumps(adapter['review_focus'], ensure_ascii=False)}

REVIEW:
{json.dumps(review, ensure_ascii=False)}

CURRENT_SECTION:
{text}

Return only the complete revised section, with no review commentary, planning
notes, or word-count statement."""


def review_section(
    task: LongBenchWriteTask,
    adapter: dict[str, Any],
    plan: dict[str, Any],
    section: dict[str, Any],
    prior_text: str,
    text: str,
    review_client_factory,
    *,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    client = review_client_factory()
    prompt = f"""REVIEW_DOCUMENT_SECTION_JSON

Act as a strict reviewer. Diagnose the current section; do not rewrite it.

ORIGINAL_USER_REQUEST:
{task.prompt}

DOCUMENT_CONTRACT:
{json.dumps(plan['document_contract'], ensure_ascii=False)}

CURRENT_SECTION_CONTRACT:
{json.dumps(section, ensure_ascii=False)}

REVIEW_FOCUS:
{json.dumps(adapter['review_focus'], ensure_ascii=False)}

PRIOR_TEXT:
{prior_text or "[None]"}

CURRENT_SECTION:
{text}

Return one JSON object:
{{
  "passed": true,
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

Use high only for a serious factual, logical, continuity, safety, or explicit-requirement
failure. Use medium for a material gap. Do not fail solely for minor style preferences."""
    raw_responses: list[str] = []
    last_error: Exception | None = None
    review: dict[str, Any] | None = None
    for attempt in range(3):
        request = prompt if attempt == 0 else build_json_retry_prompt(raw_responses[-1], str(last_error))
        raw = client.complete(
            request,
            system=(
                "You are an independent long-form document reviewer. Return valid JSON only."
                if attempt == 0
                else "You repair invalid reviewer JSON. Return the required JSON object only."
            ),
            temperature=temperature if attempt == 0 else 0.0,
            response_format="json_object",
            max_output_tokens=max_output_tokens,
        )
        raw_responses.append(raw)
        try:
            parsed = extract_json_object(raw)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("issues", []), list):
                raise ValueError("section reviewer returned an invalid object")
            review = parsed
            break
        except ValueError as error:
            last_error = error
    if review is None:
        raise RuntimeError(f"section reviewer failed JSON schema after 3 attempts: {last_error}")
    review["_raw_response_attempts"] = raw_responses
    review["_completion_history"] = completion_history(client)
    return review


def build_json_retry_prompt(raw: str, error: str) -> str:
    return f"""RETRY_REVIEW_DOCUMENT_SECTION_JSON

The prior response was invalid. Return exactly one valid JSON object with this schema:
{{
  "passed": true,
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

Preserve any usable review meaning. Do not output Markdown or commentary.

PARSE_ERROR:
{error}

INVALID_RESPONSE:
{raw}"""


def build_repair_prompt(
    task: LongBenchWriteTask,
    adapter: dict[str, Any],
    plan: dict[str, Any],
    section: dict[str, Any],
    prior_text: str,
    text: str,
    review: dict[str, Any],
    minimum_words: int,
) -> str:
    return f"""REPAIR_DOCUMENT_SECTION

Revise the entire current section. Preserve correct content, resolve every medium or
high issue, and do not introduce new facts or continuity errors. The returned section
must be at least {minimum_words} words.

ORIGINAL_USER_REQUEST:
{task.prompt}

DOCUMENT_CONTRACT:
{json.dumps(plan['document_contract'], ensure_ascii=False)}

CURRENT_SECTION_CONTRACT:
{json.dumps(section, ensure_ascii=False)}

REVIEW_FOCUS:
{json.dumps(adapter['review_focus'], ensure_ascii=False)}

PRIOR_TEXT:
{prior_text or "[None]"}

REVIEW:
{json.dumps(review, ensure_ascii=False)}

CURRENT_SECTION:
{text}

Return only the complete revised section, with no review commentary."""


def generation_record(
    task: LongBenchWriteTask,
    system: str,
    final_text: str,
    started: float,
    histories: list[dict[str, Any]],
    *,
    plan: Any = None,
    units: Any = None,
    adaptations: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "word_count": official_length_count(final_text),
        "length_count_rule": "Chinese characters plus regex-matched English words",
        "target_words": task.target_words,
        "length_ratio": round(official_length_count(final_text) / task.target_words, 4),
        "completion_history": histories,
        "adaptations": adaptations or [],
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "task": {
            "index": task.index,
            "task_id": task.task_id,
            "prompt": task.prompt,
            "type": task.task_type,
            "target_words": task.target_words,
        },
        "system": system,
        "final_text": final_text,
        "plan": plan,
        "units": units,
        "metadata": metadata,
    }


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
