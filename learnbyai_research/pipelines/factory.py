from __future__ import annotations

import os

from ..llm import ChatModel
from ..schemas import PipelineName
from .baselines import DirectPromptPipeline, SingleAgentPipeline
from .full import ContractGuidedPipeline


def build_pipeline(
    name: PipelineName,
    llm: ChatModel,
    reviewer_llm: ChatModel | None = None,
    repair_llm: ChatModel | None = None,
):
    if name == "direct_prompt":
        return DirectPromptPipeline(llm=llm)
    if name == "single_agent":
        return SingleAgentPipeline(
            llm=llm,
            checkpoint_dir=os.environ.get("SINGLE_CHECKPOINT_DIR") or None,
            resume_from_checkpoints=_env_bool("SINGLE_RESUME", default=False),
        )
    if name == "full":
        return ContractGuidedPipeline(
            llm=llm,
            reviewer_llm=reviewer_llm,
            repair_llm=repair_llm,
            name="full",
            **_full_options(),
        )
    if name == "no_contract":
        return ContractGuidedPipeline(
            llm=llm,
            reviewer_llm=reviewer_llm,
            repair_llm=repair_llm,
            name="no_contract",
            use_contracts=False,
            **_full_options(),
        )
    if name == "no_material_routing":
        return ContractGuidedPipeline(
            llm=llm,
            reviewer_llm=reviewer_llm,
            repair_llm=repair_llm,
            name="no_material_routing",
            use_material_routing=False,
            **_full_options(),
        )
    if name == "no_review_repair":
        return ContractGuidedPipeline(llm=llm, name="no_review_repair", use_review_repair=False, **_full_options())
    raise ValueError(f"Unknown pipeline: {name}")


def _full_options() -> dict:
    return {
        "checkpoint_dir": os.environ.get("FULL_CHECKPOINT_DIR") or None,
        "max_review_chars": _optional_int("FULL_MAX_REVIEW_CHARS"),
        "allow_review_failures": _env_bool("FULL_ALLOW_REVIEW_FAILURES", default=False),
        "resume_from_checkpoints": _env_bool("FULL_RESUME", default=False),
    }


def _optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
