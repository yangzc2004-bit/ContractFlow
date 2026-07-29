from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .io_utils import extract_json_object
from .llm import ChatModel, completion_history
from .longgenbench import LongGenTask, task_id, task_record, word_count


COGWRITER_SOURCE_REPOSITORY = ""
COGWRITER_SOURCE_COMMIT = "dc3bf084e8733c951172cddd89fa4d7337121fdd"


@dataclass(frozen=True)
class CogWriterSpec:
    segment_type: str
    expert_role: str
    document_description: str
    plan_key: str
    revised_plan_key: str
    id_key: str
    summary_key: str
    content_key: str
    special_label: str
    special_item_key: str
    special_location_key: str


SPECS = {
    "Week": CogWriterSpec(
        segment_type="Week",
        expert_role="expert writer",
        document_description="weekly diary containing 52 weeks",
        plan_key="weekly_plan",
        revised_plan_key="revised_weekly_plan",
        id_key="week_id",
        summary_key="events",
        content_key="diary_entry",
        special_label="special_events",
        special_item_key="event_name",
        special_location_key="week_numb",
    ),
    "Menu Week": CogWriterSpec(
        segment_type="Menu Week",
        expert_role="expert chef",
        document_description="weekly menu plan containing 52 weeks",
        plan_key="weekly_plan",
        revised_plan_key="revised_weekly_plan",
        id_key="week_id",
        summary_key="dishes",
        content_key="week_menu",
        special_label="special_dishes",
        special_item_key="dish_name",
        special_location_key="week_numb",
    ),
    "Floor": CogWriterSpec(
        segment_type="Floor",
        expert_role="expert architect",
        document_description="floor-by-floor plan for a skyscraper with 100 floors",
        plan_key="floor_plan",
        revised_plan_key="revised_floor_plan",
        id_key="floor_id",
        summary_key="purpose",
        content_key="plan",
        special_label="special_floors",
        special_item_key="special_purpose",
        special_location_key="floor_number",
    ),
    "Block": CogWriterSpec(
        segment_type="Block",
        expert_role="expert designer",
        document_description="block-by-block plan for a city with a 10x10 grid",
        plan_key="block_plan",
        revised_plan_key="revised_block_plan",
        id_key="block_id",
        summary_key="use",
        content_key="plan",
        special_label="special_blocks",
        special_item_key="special_use",
        special_location_key="block_number",
    ),
}


def run_cogwriter(
    task: LongGenTask,
    client_factory: Callable[[], ChatModel],
    *,
    planning_client_factory: Callable[[], ChatModel] | None = None,
    checkpoint_dir: str | Path,
    segment_workers: int = 4,
    planning_max_output_tokens: int = 16_384,
    segment_max_output_tokens: int = 2_048,
    max_parse_attempts: int = 3,
    max_refinements: int = 3,
) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    spec = SPECS[task.segment_type]
    plan_client_factory = planning_client_factory or client_factory

    planning_history: list[dict[str, Any]] = []
    initial_path = checkpoint_root / "plan_initial.json"
    if initial_path.exists():
        initial_plan = read_json(initial_path)["plan"]
    else:
        planning_client = plan_client_factory()
        initial_plan = request_plan(
            planning_client,
            build_initial_plan_prompt(task, spec),
            plan_key=spec.plan_key,
            max_output_tokens=planning_max_output_tokens,
            max_attempts=max_parse_attempts,
        )
        planning_history.extend(completion_history(planning_client))
        write_json(initial_path, {"plan": initial_plan})

    revised_path = checkpoint_root / "plan_revised.json"
    if revised_path.exists():
        plan = read_json(revised_path)["plan"]
    else:
        revision_client = plan_client_factory()
        plan = request_plan(
            revision_client,
            build_revision_prompt(task, spec, initial_plan),
            plan_key=spec.revised_plan_key,
            max_output_tokens=planning_max_output_tokens,
            max_attempts=max_parse_attempts,
        )
        planning_history.extend(completion_history(revision_client))
        write_json(revised_path, {"plan": plan})

    segment_dir = checkpoint_root / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    segment_results: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, dict[str, Any]]] = []
    for ordinal, item in enumerate(plan, start=1):
        segment_path = segment_dir / f"{ordinal:04d}.json"
        if segment_path.exists():
            segment_results[ordinal] = read_json(segment_path)
        else:
            pending.append((ordinal, item))

    segment_failures: list[tuple[int, Exception]] = []
    with ThreadPoolExecutor(max_workers=segment_workers) as executor:
        futures = {
            executor.submit(
                generate_segment,
                task,
                spec,
                plan,
                ordinal,
                item,
                client_factory,
                segment_max_output_tokens,
                max_parse_attempts,
                max_refinements,
            ): (ordinal, item)
            for ordinal, item in pending
        }
        completed = 0
        for future in as_completed(futures):
            ordinal, _ = futures[future]
            try:
                result = future.result()
            except Exception as error:
                segment_failures.append((ordinal, error))
                print(
                    f"CogWriter {task_id(task)}: segment {ordinal} failed: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )
                continue
            segment_results[ordinal] = result
            write_json(segment_dir / f"{ordinal:04d}.json", result)
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                print(
                    f"CogWriter {task_id(task)}: generated {completed}/{len(pending)} pending segments",
                    flush=True,
                )

    if segment_failures:
        first_ordinal, first_error = segment_failures[0]
        raise RuntimeError(
            f"CogWriter {task_id(task)} has {len(segment_failures)} failed segment(s); "
            f"first failure at segment {first_ordinal}: {first_error}"
        ) from first_error

    ordered_results = [segment_results[index] for index in sorted(segment_results)]
    final_text = assemble_cogwriter_document(ordered_results)
    histories = list(planning_history)
    for result in ordered_results:
        histories.extend(result.get("completion_history", []))

    parsed_segments = []
    for result in ordered_results:
        index = parse_segment_index(result["segment_id"], task.segment_type)
        if index is not None:
            parsed_segments.append({"index": index, "text": result["text"]})

    return {
        "task": task_record(task),
        "system": "cogwriter",
        "final_text": final_text,
        "segments": parsed_segments,
        "plan": plan,
        "metadata": {
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "model": client_factory().model,
            "segment_workers": segment_workers,
            "planning_calls_expected": 2,
            "source_repository": COGWRITER_SOURCE_REPOSITORY,
            "source_commit": COGWRITER_SOURCE_COMMIT,
            "adaptations": [
                "OpenAI-compatible ModelArts MaaS endpoint",
                "bounded JSON parse retries",
                "bounded word-count refinements",
                "per-segment resumable checkpoints",
                "partial-progress checkpointing when another segment fails",
                "separate streaming client for long planning requests",
            ],
            "completion_history": histories,
        },
    }


def request_plan(
    llm: ChatModel,
    prompt: str,
    *,
    plan_key: str,
    max_output_tokens: int,
    max_attempts: int,
) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            raw = llm.complete(
                prompt,
                temperature=0.2,
                response_format="json_object",
                max_output_tokens=max_output_tokens,
            )
            payload = extract_json_object(raw)
            plan = payload.get(plan_key)
            if not isinstance(plan, list) or not plan:
                raise ValueError(f"CogWriter response does not contain a non-empty {plan_key}")
            return [dict(item) for item in plan if isinstance(item, dict)]
        except (ValueError, KeyError, TypeError) as error:
            last_error = error
    raise RuntimeError(f"CogWriter planning failed: {last_error}") from last_error


def generate_segment(
    task: LongGenTask,
    spec: CogWriterSpec,
    full_plan: list[dict[str, Any]],
    ordinal: int,
    item: dict[str, Any],
    client_factory: Callable[[], ChatModel],
    max_output_tokens: int,
    max_parse_attempts: int,
    max_refinements: int,
) -> dict[str, Any]:
    segment_id = str(item.get(spec.id_key, f"{task.segment_type} {ordinal}")).strip()
    summary = str(item.get(spec.summary_key, "")).strip()
    llm = client_factory()
    last_error: Exception | None = None
    text = ""
    for _ in range(max_parse_attempts):
        try:
            raw = llm.complete(
                build_generation_prompt(task, spec, full_plan, segment_id, summary),
                temperature=0.2,
                response_format="json_object",
                max_output_tokens=max_output_tokens,
            )
            payload = extract_json_object(raw)
            text = str(payload[spec.content_key]).strip()
            if not text:
                raise ValueError("CogWriter generated empty segment text")
            break
        except (ValueError, KeyError, TypeError) as error:
            last_error = error
    else:
        raise RuntimeError(
            f"CogWriter segment generation failed for {segment_id}: {last_error}"
        ) from last_error

    refinement_count = 0
    while refinement_count < max_refinements:
        current_words = word_count(text)
        difference = abs(task.minimum_words - current_words)
        if difference <= task.minimum_words * 0.1:
            break
        action = "shorten" if current_words > task.minimum_words else "lengthen"
        text = llm.complete(
            build_refinement_prompt(text, action, difference),
            temperature=0.2,
            max_output_tokens=max_output_tokens,
        ).strip()
        refinement_count += 1

    return {
        "ordinal": ordinal,
        "segment_id": segment_id,
        "summary": summary,
        "text": text,
        "word_count": word_count(text),
        "refinement_count": refinement_count,
        "completion_history": completion_history(llm),
    }


def build_initial_plan_prompt(task: LongGenTask, spec: CogWriterSpec) -> str:
    item_example = _plan_item_example(spec)
    return f"""You are an {spec.expert_role}, and your task is to create a {spec.document_description}.
User requirements:
{task.prompt}

Think step by step. Analyse the user requirements to identify special requirements and
their exact segment numbers and list them in "{spec.special_label}". If a requirement is
periodic, list every occurrence separately and specify which segments it belongs in.
Then follow your analysis to create the complete segment-by-segment plan.

Return your analysis and plan in ONLY this exact JSON format:
{{
  "analysis": "",
  "{spec.special_label}": [
    {{
      "{spec.special_item_key}": "special requirement",
      "{spec.special_location_key}": "exact segment"
    }}
  ],
  "{spec.plan_key}": [
    {item_example}
  ]
}}"""


def build_revision_prompt(
    task: LongGenTask,
    spec: CogWriterSpec,
    plan: list[dict[str, Any]],
) -> str:
    item_example = _plan_item_example(spec)
    return f"""You are an {spec.expert_role}, and your task is to revise a {spec.document_description}.
Current plan:
{json.dumps(plan, ensure_ascii=False)}

User requirements:
{task.prompt}

Think step by step. The current plan may contain wrong or missing information. Identify
every one-time, range, and periodic requirement, correct its exact segment assignment,
and revise the complete plan. Strictly preserve every required segment.

Return your analysis and revised plan in ONLY this exact JSON format:
{{
  "analysis": "",
  "{spec.revised_plan_key}": [
    {item_example}
  ]
}}"""


def build_generation_prompt(
    task: LongGenTask,
    spec: CogWriterSpec,
    full_plan: list[dict[str, Any]],
    segment_id: str,
    summary: str,
) -> str:
    return f"""You are an {spec.expert_role}.
Write a {task.minimum_words}-word {task.segment_type.lower()} entry for {segment_id}.
The planned content for this segment is: {summary}
You should consider the coherence of the entry by referring to the complete plan:
{json.dumps(full_plan, ensure_ascii=False)}

You should consider the user requirements:
{task.prompt}

Check whether any special requirement belongs in this exact segment. If so, include it
clearly. Otherwise write a suitable general entry. Return ONLY this JSON format:
{{
  "{spec.id_key}": "{segment_id}",
  "check": "reason and check if the user requirements are met",
  "{spec.content_key}": "Your {task.minimum_words}-word entry here"
}}"""


def build_refinement_prompt(text: str, action: str, word_difference: int) -> str:
    return f"""You are an expert editor. The provided text needs to be {action}ed by
approximately {word_difference} English words while maintaining its original meaning,
all required details, and coherence.

Text:
{text}

Return only the refined text."""


def assemble_cogwriter_document(results: list[dict[str, Any]]) -> str:
    blocks = [
        f"#*# {result['segment_id']}:\n{result['text'].strip()}"
        for result in results
        if result.get("text", "").strip()
    ]
    return "\n\n".join(blocks) + "\n\n*** finished ***"


def parse_segment_index(segment_id: str, segment_type: str) -> int | None:
    match = re.search(rf"{re.escape(segment_type)}\s+(\d+)", segment_id, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _plan_item_example(spec: CogWriterSpec) -> str:
    return (
        "{"
        f'"{spec.id_key}": "{spec.segment_type} 1", '
        f'"{spec.summary_key}": "Briefly describe this segment"'
        "}"
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
