from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from .llm import ChatModel, completion_history
from .longgenbench import LongGenTask, parse_document_blocks, task_record


SELF_REFINE_SOURCE_REPOSITORY = ""
SELF_REFINE_SOURCE_COMMIT = "9a206d41e5d2d0c241bb441f41eeadb945afaa55"
SELF_REFINE_ADAPTER_VERSION = "longgenbench_self_refine_v1"


def run_self_refine(
    task: LongGenTask,
    draft_text: str,
    feedback_client_factory: Callable[[], ChatModel],
    refine_client_factory: Callable[[], ChatModel],
    *,
    checkpoint_dir: str | Path,
    iterations: int = 1,
    feedback_max_output_tokens: int = 8_192,
    refine_max_output_tokens: int = 32_768,
) -> dict[str, Any]:
    if not draft_text.strip():
        raise ValueError("Self-Refine requires a non-empty initial draft")
    if iterations < 1:
        raise ValueError("Self-Refine iterations must be at least 1")

    started = time.perf_counter()
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    current_text = draft_text.strip()
    histories: list[dict[str, Any]] = []
    iteration_records: list[dict[str, Any]] = []
    model_name = "unknown"

    for iteration in range(1, iterations + 1):
        feedback_path = checkpoint_root / f"feedback_{iteration:02d}.json"
        if feedback_path.exists():
            feedback_payload = read_json(feedback_path)
            feedback = str(feedback_payload["feedback"])
            histories.extend(feedback_payload.get("completion_history", []))
            model_name = str(feedback_payload.get("model", model_name))
        else:
            feedback_client = feedback_client_factory()
            model_name = feedback_client.model
            feedback = feedback_client.complete(
                build_self_refine_feedback_prompt(task, current_text),
                temperature=0.7,
                max_output_tokens=feedback_max_output_tokens,
            ).strip()
            if not feedback:
                raise RuntimeError(
                    f"Self-Refine produced empty feedback at iteration {iteration}"
                )
            feedback_history = completion_history(feedback_client)
            histories.extend(feedback_history)
            write_json(
                feedback_path,
                {
                    "iteration": iteration,
                    "feedback": feedback,
                    "model": feedback_client.model,
                    "completion_history": feedback_history,
                },
            )

        refined_path = checkpoint_root / f"refined_{iteration:02d}.json"
        if refined_path.exists():
            refined_payload = read_json(refined_path)
            current_text = str(refined_payload["text"])
            histories.extend(refined_payload.get("completion_history", []))
            model_name = str(refined_payload.get("model", model_name))
        else:
            refine_client = refine_client_factory()
            model_name = refine_client.model
            refined_text = refine_client.complete(
                build_self_refine_iteration_prompt(
                    task,
                    draft_text=current_text,
                    feedback=feedback,
                ),
                temperature=0.7,
                max_output_tokens=refine_max_output_tokens,
            ).strip()
            if not refined_text:
                raise RuntimeError(
                    f"Self-Refine produced empty refined output at iteration {iteration}"
                )
            refine_history = completion_history(refine_client)
            histories.extend(refine_history)
            current_text = refined_text
            write_json(
                refined_path,
                {
                    "iteration": iteration,
                    "text": current_text,
                    "model": refine_client.model,
                    "completion_history": refine_history,
                },
            )
        iteration_records.append(
            {
                "iteration": iteration,
                "feedback_checkpoint": feedback_path.name,
                "refined_checkpoint": refined_path.name,
            }
        )

    parsed = parse_document_blocks(current_text, task.segment_type)
    return {
        "task": task_record(task),
        "system": "self_refine",
        "final_text": current_text,
        "segments": [
            {"index": index, "text": text}
            for index, text in sorted(parsed.items())
        ],
        "iterations": iteration_records,
        "metadata": {
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "model": model_name,
            "adapter_version": SELF_REFINE_ADAPTER_VERSION,
            "initial_draft_sha256": hashlib.sha256(
                draft_text.encode("utf-8")
            ).hexdigest(),
            "iteration_count": iterations,
            "source_repository": SELF_REFINE_SOURCE_REPOSITORY,
            "source_commit": SELF_REFINE_SOURCE_COMMIT,
            "source_pattern": "task-specific Init, Feedback, and Iterate prompts",
            "adaptations": [
                "reuse frozen LongGenBench direct output as Init",
                "LongGenBench-specific feedback rubric",
                "one full-document refinement per iteration",
                "OpenAI-compatible MaaS endpoint",
                "per-iteration resumable checkpoints",
            ],
            "completion_history": histories,
        },
    }


def build_self_refine_feedback_prompt(task: LongGenTask, draft_text: str) -> str:
    return f"""We want to iteratively improve a response to a constrained long-form writing
task. Give specific, actionable feedback on the draft. Do not rewrite the document yet.

Original task:

{task.prompt}

Draft:

{draft_text}

Evaluate the draft against every requirement in the original task. In particular check:
1. It contains exactly {task.segment_count} uniquely numbered {task.segment_type} segments.
2. It uses the required "#*# {task.segment_type} N:" marker format in numerical order.
3. Every segment contains at least {task.minimum_words} English words.
4. Every one-time requirement appears at its exact segment.
5. Every range requirement appears throughout its complete required range.
6. Every periodic requirement appears at every required interval and nowhere important is omitted.
7. The document remains coherent and does not duplicate or contradict requirements.

List concrete missing, misplaced, short, duplicated, or malformed segments and explain
exactly how to fix them. Keep the feedback concise enough to guide one complete revision."""


def build_self_refine_iteration_prompt(
    task: LongGenTask,
    *,
    draft_text: str,
    feedback: str,
) -> str:
    return f"""Improve the response using the feedback below.

Original task:

{task.prompt}

Current response:

{draft_text}

Feedback:

{feedback}

Produce the complete revised document, not a patch or commentary. Preserve correct
material while fixing every identified issue. Output exactly {task.segment_count}
segments in numerical order, each with at least {task.minimum_words} English words and
the exact marker "#*# {task.segment_type} N:". End with "*** finished ***".
Output only the complete reader-facing document."""


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
