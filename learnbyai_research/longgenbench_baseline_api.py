from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from .llm import OpenAICompatibleClient


ProviderMode = Literal[
    "auto",
    "deepseek-official",
    "modelarts-v2",
    "modelarts-openai",
    "openai-compatible",
]
PROVIDER_MODES: tuple[ProviderMode, ...] = (
    "auto",
    "deepseek-official",
    "modelarts-v2",
    "modelarts-openai",
    "openai-compatible",
)


@dataclass(frozen=True)
class ResolvedAPI:
    api_key: str
    api_key_env: str
    base_url: str
    model: str
    provider_mode: ProviderMode


def validate_indices(dataset: list[dict], indices: list[int]) -> None:
    if len(set(indices)) != len(indices):
        raise ValueError("--indices contains duplicates")
    invalid = [index for index in indices if index < 0 or index >= len(dataset)]
    if invalid:
        raise ValueError(f"dataset indices out of range: {invalid}")


def resolve_api(
    *,
    api_key_env: str,
    base_url: str,
    model: str,
    provider_mode: ProviderMode,
) -> ResolvedAPI:
    candidate_names = list(
        dict.fromkeys(
            [
                api_key_env,
                "LONGGENBENCH_API_KEY",
                "OPENAI_API_KEY",
                "MODELARTS_MAAS_KEY",
            ]
        )
    )
    for name in candidate_names:
        value = os.environ.get(name)
        if value:
            return ResolvedAPI(
                api_key=value,
                api_key_env=name,
                base_url=base_url.rstrip("/"),
                model=model,
                provider_mode=infer_provider_mode(base_url, provider_mode),
            )
    raise RuntimeError(
        "API key is required. Set LONGGENBENCH_API_KEY, OPENAI_API_KEY, "
        f"MODELARTS_MAAS_KEY, or the variable named by --api-key-env ({api_key_env})."
    )


def infer_provider_mode(
    base_url: str,
    requested: ProviderMode = "auto",
) -> ProviderMode:
    if requested != "auto":
        return requested
    normalized = base_url.rstrip("/").lower()
    if "modelarts-maas.com" in normalized and normalized.endswith("/v2"):
        return "modelarts-v2"
    if "modelarts-maas.com" in normalized and normalized.endswith("/openai/v1"):
        return "modelarts-openai"
    return "openai-compatible"


def make_baseline_client(
    resolved: ResolvedAPI,
    *,
    timeout: int,
    max_retries: int,
    stream: bool,
    include_default_system: bool = True,
) -> OpenAICompatibleClient:
    max_output_token_field = None
    thinking = None
    chat_template_kwargs = None
    if resolved.provider_mode == "deepseek-official":
        max_output_token_field = "max_tokens"
        thinking = "disabled"
    elif resolved.provider_mode == "modelarts-v2":
        max_output_token_field = "max_completion_tokens"
        thinking = "disabled"
    elif resolved.provider_mode == "modelarts-openai":
        max_output_token_field = "max_tokens"
        chat_template_kwargs = {"thinking": False}

    return OpenAICompatibleClient(
        model=resolved.model,
        api_key=resolved.api_key,
        base_url=resolved.base_url,
        wire_api="chat",
        timeout=timeout,
        max_retries=max_retries,
        max_output_token_field=max_output_token_field,
        thinking=thinking,
        chat_template_kwargs=chat_template_kwargs,
        stream=stream,
        stream_transport="urllib",
        include_default_system=include_default_system,
    )
