from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .contractflow_v2 import (
    DocumentContract,
    FlowConfig,
    compile_sparse_contract,
    run_contractflow,
)
from .llm import ChatModel
from .longgenbench import (
    LongGenTask,
    assemble_document,
    task_record,
    task_word_range,
)


def render_longgen_document(
    task: LongGenTask,
    unit_texts: dict[int, str],
) -> str:
    """Thin boundary renderer for official LongGenBench #*# output format."""
    sanitized = {
        index: re.sub(r"\s*#\*#\s*", " ", text).strip()
        for index, text in unit_texts.items()
    }
    return assemble_document(task, sanitized)


def compile_longgen_v2_contract(
    task: LongGenTask,
    llm: ChatModel,
    *,
    max_output_tokens: int = 2_048,
    retry_cap_tokens: int = 8_192,
) -> DocumentContract:
    min_words, max_words = task_word_range(task)
    return compile_sparse_contract(
        task.prompt,
        llm,
        total_units=task.segment_count,
        unit_label=task.segment_type,
        min_words=min_words,
        max_words=max_words,
        max_output_tokens=max_output_tokens,
        retry_cap_tokens=retry_cap_tokens,
    )


def run_longgen_v2(
    task: LongGenTask,
    llm: ChatModel,
    *,
    config: FlowConfig | None = None,
    checkpoint_path: str | Path | None = None,
    review_llm: ChatModel | None = None,
    repair_llm: ChatModel | None = None,
    precompiled_contract: DocumentContract | None = None,
) -> dict[str, Any]:
    min_words, max_words = task_word_range(task)
    result = run_contractflow(
        task.prompt,
        llm,
        config=config,
        total_units=task.segment_count,
        unit_label=task.segment_type,
        min_words=min_words,
        max_words=max_words,
        precompiled_contract=precompiled_contract,
        checkpoint_path=checkpoint_path,
        review_llm=review_llm,
        repair_llm=repair_llm,
        renderer=lambda _contract, texts: render_longgen_document(task, texts),
    )
    # Match LongGenBench generation record schema so eval_longgenbench.py works seamlessly.
    return {
        "task": task_record(task),
        "system": result["system"],
        "final_text": result["final_text"],
        "segments": [
            {"index": unit["index"], "text": unit["text"]}
            for unit in result["units"]
        ],
        "contracts": [
            {
                "index": unit["index"],
                "title": unit["title"],
                "required_events": unit["obligations"],
            }
            for unit in result["contract"]["units"]
        ],
        "contract_v2": result["contract"],
        "metadata": result["metadata"],
    }
