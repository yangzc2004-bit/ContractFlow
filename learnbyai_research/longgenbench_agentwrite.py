from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .llm import ChatModel, completion_history
from .longgenbench import LongGenTask, assemble_document, task_id, task_record, word_count


AGENTWRITE_SOURCE_REPOSITORY = ""
AGENTWRITE_SOURCE_COMMIT = "447539b356a8b09760b51eca876e19b6fc1f2dd7"
AGENTWRITE_ADAPTER_VERSION = "longgenbench_agentwrite_v1"


@dataclass(frozen=True)
class AgentWriteStep:
    index: int
    main_point: str
    word_count: int
    raw: str


def run_agentwrite(
    task: LongGenTask,
    planning_client_factory: Callable[[], ChatModel],
    writing_client_factory: Callable[[], ChatModel],
    *,
    checkpoint_dir: str | Path,
    planning_max_output_tokens: int = 16_384,
    segment_max_output_tokens: int = 4_096,
    max_plan_attempts: int = 3,
) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    model_name = "unknown"

    plan_path = checkpoint_root / "plan.json"
    if plan_path.exists():
        plan_payload = read_json(plan_path)
        plan_text = str(plan_payload["plan_text"])
        steps = [AgentWriteStep(**item) for item in plan_payload["steps"]]
        planning_history = list(plan_payload.get("completion_history", []))
        model_name = str(plan_payload.get("model", model_name))
    else:
        planning_client = planning_client_factory()
        model_name = planning_client.model
        plan_text, steps = request_agentwrite_plan(
            task,
            planning_client,
            max_output_tokens=planning_max_output_tokens,
            max_attempts=max_plan_attempts,
        )
        planning_history = completion_history(planning_client)
        write_json(
            plan_path,
            {
                "plan_text": plan_text,
                "steps": [step.__dict__ for step in steps],
                "model": planning_client.model,
                "completion_history": planning_history,
            },
        )

    segment_dir = checkpoint_root / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    segment_results: dict[int, dict[str, Any]] = {}
    for step in steps:
        segment_path = segment_dir / f"{step.index:04d}.json"
        if segment_path.exists():
            segment_results[step.index] = read_json(segment_path)
            continue

        previous_text = agentwrite_previous_text(task, segment_results)
        client = writing_client_factory()
        model_name = client.model
        text = client.complete(
            build_agentwrite_step_prompt(
                task,
                plan_text=plan_text,
                previous_text=previous_text,
                step=step,
            ),
            temperature=1.0,
            max_output_tokens=segment_max_output_tokens,
        ).strip()
        if not text:
            raise RuntimeError(
                f"AgentWrite generated empty text for {task_id(task)} segment {step.index}"
            )
        result = {
            "index": step.index,
            "step": step.__dict__,
            "text": text,
            "word_count": word_count(text),
            "completion_history": completion_history(client),
        }
        segment_results[step.index] = result
        write_json(segment_path, result)
        if step.index % 10 == 0 or step.index == task.segment_count:
            print(
                f"AgentWrite {task_id(task)}: generated {step.index}/{task.segment_count} segments",
                flush=True,
            )

    segments = {
        index: str(segment_results[index]["text"])
        for index in range(1, task.segment_count + 1)
    }
    histories = list(planning_history)
    for index in range(1, task.segment_count + 1):
        histories.extend(segment_results[index].get("completion_history", []))

    return {
        "task": task_record(task),
        "system": "agentwrite",
        "final_text": assemble_document(task, segments),
        "segments": [
            {"index": index, "text": text}
            for index, text in sorted(segments.items())
        ],
        "plan": [step.__dict__ for step in steps],
        "metadata": {
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "model": model_name,
            "adapter_version": AGENTWRITE_ADAPTER_VERSION,
            "source_repository": AGENTWRITE_SOURCE_REPOSITORY,
            "source_commit": AGENTWRITE_SOURCE_COMMIT,
            "source_files": [
                "agentwrite/plan.py",
                "agentwrite/write.py",
                "agentwrite/prompts/plan.txt",
                "agentwrite/prompts/write.txt",
            ],
            "adaptations": [
                "OpenAI-compatible MaaS endpoint",
                "one required plan step per LongGenBench segment",
                "support for plans longer than the upstream 50-step guard",
                "LongGenBench marker assembly",
                "per-step resumable checkpoints",
            ],
            "completion_history": histories,
        },
    }


def request_agentwrite_plan(
    task: LongGenTask,
    client: ChatModel,
    *,
    max_output_tokens: int,
    max_attempts: int,
) -> tuple[str, list[AgentWriteStep]]:
    last_error: Exception | None = None
    for _ in range(max_attempts):
        plan_text = client.complete(
            build_agentwrite_plan_prompt(task),
            temperature=1.0,
            max_output_tokens=max_output_tokens,
        ).strip()
        try:
            return plan_text, parse_agentwrite_plan(plan_text, task.segment_count)
        except ValueError as error:
            last_error = error
    raise RuntimeError(f"AgentWrite planning failed: {last_error}") from last_error


def parse_agentwrite_plan(plan_text: str, expected_count: int) -> list[AgentWriteStep]:
    pattern = re.compile(
        r"(?:^|\n)\s*(?:Paragraph|Segment)\s+(\d+)\s*-\s*"
        r"Main Point:\s*(.*?)\s*-\s*Word Count:\s*(\d+)\s*words?\.?\s*"
        r"(?=\n\s*(?:Paragraph|Segment)\s+\d+\s*-|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    steps = [
        AgentWriteStep(
            index=int(match.group(1)),
            main_point=" ".join(match.group(2).split()),
            word_count=int(match.group(3)),
            raw=" ".join(match.group(0).split()),
        )
        for match in pattern.finditer(plan_text)
    ]
    expected = list(range(1, expected_count + 1))
    actual = [step.index for step in steps]
    if actual != expected:
        raise ValueError(
            f"expected AgentWrite steps {expected[0]}..{expected[-1]}, got {actual}"
        )
    return steps


def build_agentwrite_plan_prompt(task: LongGenTask) -> str:
    return f"""I need you to help me break down the following long-form writing instruction
into multiple subtasks. Each subtask will guide the writing of exactly one required
{task.segment_type} segment and must include its main points and word count requirement.

The writing instruction is as follows:

{task.prompt}

Return exactly {task.segment_count} lines in this format:

Segment 1 - Main Point: [Detailed content and every requirement assigned to segment 1] - Word Count: {task.minimum_words} words
Segment 2 - Main Point: [Detailed content and every requirement assigned to segment 2] - Word Count: {task.minimum_words} words

Continue consecutively through Segment {task.segment_count}. Every segment number must
appear exactly once. Make every subtask clear and specific, cover the entire instruction,
and place one-time, range, and periodic requirements at their exact segment numbers.
Do not output any other content."""


def build_agentwrite_step_prompt(
    task: LongGenTask,
    *,
    plan_text: str,
    previous_text: str,
    step: AgentWriteStep,
) -> str:
    return f"""You are an excellent writing assistant. I will give you an original writing
instruction, my planned writing steps, and the text already written. Continue with only
the next planned segment.

Writing instruction:

{task.prompt}

Writing steps:

{plan_text}

Already written text:

{previous_text or "[No text has been written yet.]"}

Now write this step:

{step.raw}

Write at least {task.minimum_words} English words for {task.segment_type} {step.index}.
Include every requirement assigned to this exact segment. Output only the reader-facing
segment body, without repeating previous text, without the #*# marker, and without
open-ended conclusions or rhetorical hooks."""


def agentwrite_previous_text(
    task: LongGenTask,
    segment_results: dict[int, dict[str, Any]],
) -> str:
    return "\n\n".join(
        f"#*# {task.segment_type} {index}:\n{str(result['text']).strip()}"
        for index, result in sorted(segment_results.items())
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
