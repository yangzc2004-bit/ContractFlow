from __future__ import annotations

import json
import os
import re
import socket
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from http.client import IncompleteRead, RemoteDisconnected
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol


class ChatModel(Protocol):
    model: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        response_format: Literal["json_object"] | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        ...

    def completion_metadata(self) -> dict[str, Any]:
        ...

    def completion_history(self) -> list[dict[str, Any]]:
        ...


@dataclass
class OpenAICompatibleClient:
    model: str
    api_key: str
    base_url: str = ""
    wire_api: Literal["chat", "responses"] = "chat"
    timeout: int = 120
    max_retries: int = 2
    reasoning_effort: str | None = None
    store: bool | None = None
    ssl_verify: bool = True
    temperature_override: float | None = None
    max_output_tokens: int | None = None
    max_output_token_field: Literal["max_tokens", "max_completion_tokens"] | None = None
    thinking: Literal["enabled", "disabled"] | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    stream: bool = False
    stream_transport: Literal["curl", "urllib"] = "curl"
    include_default_system: bool = True
    _last_completion_metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _completion_history: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _current_attempts: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_env(cls, prefix: str = "OPENAI") -> "OpenAICompatibleClient":
        api_key = os.environ.get(f"{prefix}_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(f"{prefix}_API_KEY is required unless --mock is used.")
        model = os.environ.get(f"{prefix}_MODEL") or os.environ.get("OPENAI_MODEL")
        if not model:
            raise RuntimeError(f"{prefix}_MODEL is required unless --mock is used.")
        base_url = os.environ.get(f"{prefix}_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise RuntimeError(f"{prefix}_BASE_URL is required unless --mock is used.")
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            wire_api=clean_wire_api(os.environ.get(f"{prefix}_WIRE_API") or os.environ.get("OPENAI_WIRE_API", "chat")),
            timeout=int(os.environ.get(f"{prefix}_TIMEOUT") or os.environ.get("OPENAI_TIMEOUT", "120")),
            max_retries=int(os.environ.get(f"{prefix}_MAX_RETRIES") or os.environ.get("OPENAI_MAX_RETRIES", "2")),
            reasoning_effort=os.environ.get(f"{prefix}_REASONING_EFFORT") or os.environ.get("OPENAI_REASONING_EFFORT"),
            store=parse_bool(os.environ.get(f"{prefix}_STORE") or os.environ.get("OPENAI_STORE")),
            ssl_verify=parse_bool(os.environ.get(f"{prefix}_SSL_VERIFY") or os.environ.get("OPENAI_SSL_VERIFY")) is not False,
            temperature_override=parse_float(os.environ.get(f"{prefix}_TEMPERATURE") or os.environ.get("OPENAI_TEMPERATURE")),
            thinking=parse_thinking(os.environ.get(f"{prefix}_THINKING") or os.environ.get("OPENAI_THINKING")),
            chat_template_kwargs=parse_json_object(
                os.environ.get(f"{prefix}_CHAT_TEMPLATE_KWARGS")
                or os.environ.get("OPENAI_CHAT_TEMPLATE_KWARGS")
            ),
            stream=parse_bool(os.environ.get(f"{prefix}_STREAM") or os.environ.get("OPENAI_STREAM")) is True,
            stream_transport=clean_stream_transport(
                os.environ.get(f"{prefix}_STREAM_TRANSPORT")
                or os.environ.get("OPENAI_STREAM_TRANSPORT", "curl")
            ),
            max_output_token_field=parse_max_output_token_field(
                os.environ.get(f"{prefix}_MAX_OUTPUT_TOKEN_FIELD") or os.environ.get("OPENAI_MAX_OUTPUT_TOKEN_FIELD")
            ),
            max_output_tokens=parse_positive_int(
                os.environ.get(f"{prefix}_MAX_OUTPUT_TOKENS")
                or os.environ.get(f"{prefix}_MAX_TOKENS")
                or os.environ.get("OPENAI_MAX_OUTPUT_TOKENS")
                or os.environ.get("OPENAI_MAX_TOKENS")
            ),
        )

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        response_format: Literal["json_object"] | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        budget = max_output_tokens if max_output_tokens is not None else self.max_output_tokens
        started_at = utc_timestamp()
        started = time.perf_counter()
        self._current_attempts = 0
        self._last_completion_metadata = {}
        try:
            if self.wire_api == "responses":
                text = self._complete_responses(
                    prompt,
                    system=system,
                    temperature=temperature,
                    response_format=response_format,
                    max_output_tokens=budget,
                )
            else:
                text = self._complete_chat(
                    prompt,
                    system=system,
                    temperature=temperature,
                    response_format=response_format,
                    max_output_tokens=budget,
                )
        except Exception as error:
            self._last_completion_metadata = self._instrument_completion(
                started_at,
                started,
                budget,
                response_format,
                success=False,
                error=error,
            )
            self._completion_history.append(dict(self._last_completion_metadata))
            raise
        self._last_completion_metadata = self._instrument_completion(
            started_at,
            started,
            budget,
            response_format,
            success=True,
        )
        self._completion_history.append(dict(self._last_completion_metadata))
        return text

    def completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def completion_history(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._completion_history]

    def _complete_chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        response_format: Literal["json_object"] | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        elif self.include_default_system:
            messages.append(
                {
                    "role": "system",
                    "content": "You are a precise research assistant.",
                }
            )
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature(temperature),
        }
        if response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if self.thinking is not None:
            payload["thinking"] = {"type": self.thinking}
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        max_output_token_field = self._chat_max_output_token_field()
        if max_output_tokens is not None:
            payload[max_output_token_field] = max_output_tokens
        if self.stream:
            payload["stream"] = True
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if self.stream else "application/json",
                "User-Agent": "contractflow-research/0.1",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._current_attempts = attempt + 1
            try:
                if self.stream and self.stream_transport == "curl":
                    text, data = curl_chat_completion_stream(
                        url=f"{self.base_url}/chat/completions",
                        api_key=self.api_key,
                        body=body,
                        timeout=self.timeout,
                        ssl_verify=self.ssl_verify,
                    )
                    self._last_completion_metadata = chat_completion_metadata(
                        data,
                        max_output_tokens,
                        max_output_token_field,
                        self.thinking,
                    )
                    return text
                with urllib.request.urlopen(request, timeout=self.timeout, context=self.ssl_context()) as response:
                    if self.stream:
                        text, data = read_chat_completion_stream(response)
                        self._last_completion_metadata = chat_completion_metadata(
                            data,
                            max_output_tokens,
                            max_output_token_field,
                            self.thinking,
                        )
                        return text
                    data = json.loads(response.read().decode("utf-8"))
                    self._last_completion_metadata = chat_completion_metadata(
                        data,
                        max_output_tokens,
                        max_output_token_field,
                        self.thinking,
                    )
                    return data["choices"][0]["message"]["content"]
            except transient_request_errors() as error:
                last_error = error
                if attempt >= self.max_retries or not is_retryable_request_error(error):
                    break
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM request failed: {last_error}") from last_error

    def _complete_responses(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        response_format: Literal["json_object"] | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        input_messages: list[dict[str, str]] = []
        if system is not None:
            input_messages.append({"role": "system", "content": system})
        elif self.include_default_system:
            input_messages.append(
                {"role": "system", "content": "You are a precise research assistant."}
            )
        input_messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
            "temperature": self._temperature(temperature),
        }
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.store is not None:
            payload["store"] = self.store
        if response_format == "json_object":
            payload["text"] = {"format": {"type": "json_object"}}
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "contractflow-research/0.1",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._current_attempts = attempt + 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout, context=self.ssl_context()) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    self._last_completion_metadata = responses_completion_metadata(data, max_output_tokens)
                    return extract_responses_text(data)
            except transient_request_errors() as error:
                last_error = error
                if attempt >= self.max_retries or not is_retryable_request_error(error):
                    break
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM request failed: {last_error}") from last_error

    def ssl_context(self) -> ssl.SSLContext | None:
        if self.ssl_verify:
            return None
        return ssl._create_unverified_context()

    def _temperature(self, requested: float) -> float:
        if self.model.strip().lower().startswith("kimi-"):
            return 0.6
        return self.temperature_override if self.temperature_override is not None else requested

    def _chat_max_output_token_field(self) -> Literal["max_tokens", "max_completion_tokens"]:
        if self.max_output_token_field is not None:
            return self.max_output_token_field
        if self.model.strip().lower().startswith("kimi-"):
            return "max_completion_tokens"
        return "max_tokens"

    def _instrument_completion(
        self,
        started_at: str,
        started: float,
        budget: int | None,
        response_format: Literal["json_object"] | None,
        *,
        success: bool,
        error: Exception | None = None,
    ) -> dict[str, Any]:
        metadata = dict(self._last_completion_metadata)
        attempts = self._current_attempts
        metadata.update(
            {
                "started_at": started_at,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "success": success,
                "attempts": attempts,
                "retry_count": max(0, attempts - 1),
                "model": self.model,
                "base_url": self.base_url,
                "wire_api": self.wire_api,
                "stream": self.stream if self.wire_api == "chat" else False,
                "stream_transport": self.stream_transport if self.stream and self.wire_api == "chat" else None,
                "response_format": response_format,
                "request_max_output_tokens": budget,
            }
        )
        if error is not None:
            metadata.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
        return metadata


def clean_wire_api(value: str) -> Literal["chat", "responses"]:
    return "responses" if value.strip().lower() == "responses" else "chat"


def clean_stream_transport(value: str) -> Literal["curl", "urllib"]:
    normalized = value.strip().lower()
    if normalized not in {"curl", "urllib"}:
        raise ValueError("stream transport must be either 'curl' or 'urllib'")
    return "urllib" if normalized == "urllib" else "curl"


def parse_json_object(value: str | None) -> dict[str, Any] | None:
    if value is None or value.strip() == "":
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("JSON configuration value must be an object")
    return parsed


def transient_request_errors() -> tuple[type[BaseException], ...]:
    """Errors for which re-sending an idempotent completion request is reasonable."""
    return (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        socket.timeout,
        IncompleteRead,
        RemoteDisconnected,
        ConnectionError,
        OSError,
    )


def is_retryable_request_error(error: BaseException) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 429 or error.code >= 500
    return True


def read_chat_completion_stream(response: Any) -> tuple[str, dict[str, Any]]:
    """Consume a Chat Completions SSE stream and return only a complete response."""
    content_parts: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    saw_done = False
    event_data: list[str] = []

    def consume_event() -> None:
        nonlocal finish_reason, usage, saw_done
        if not event_data:
            return
        raw_data = "\n".join(event_data).strip()
        event_data.clear()
        if raw_data == "[DONE]":
            saw_done = True
            return
        try:
            chunk = json.loads(raw_data)
        except json.JSONDecodeError as error:
            raise ConnectionError("invalid JSON in streaming Chat Completions response") from error
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices", [])
        if not choices or not isinstance(choices[0], dict):
            return
        choice = choices[0]
        delta = choice.get("delta", {})
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            content_parts.append(delta["content"])
        if choice.get("finish_reason") is not None:
            finish_reason = str(choice["finish_reason"])

    while True:
        line = response.readline()
        if not line:
            consume_event()
            break
        decoded = line.decode("utf-8").rstrip("\r\n")
        if decoded == "":
            consume_event()
        elif decoded.startswith("data:"):
            event_data.append(decoded[5:].lstrip())

    if not saw_done and finish_reason is None:
        raise ConnectionError("stream ended before a completion marker")
    if not content_parts:
        raise ConnectionError("stream completed without assistant content")
    text = "".join(content_parts)
    data = {
        "choices": [{"message": {"content": text}, "finish_reason": finish_reason or "stop"}],
        "usage": usage,
    }
    return text, data


def curl_chat_completion_stream(
    *,
    url: str,
    api_key: str,
    body: bytes,
    timeout: int,
    ssl_verify: bool,
) -> tuple[str, dict[str, Any]]:
    """Read Chat Completions SSE through curl without exposing credentials in argv."""
    descriptor, body_path = tempfile.mkstemp(prefix="contractflow-chat-", suffix=".json")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
        args = [
            "curl",
            "--config",
            "-",
            "--http1.1",
            "--no-buffer",
            "--silent",
            "--show-error",
            "--fail-with-body",
            "--connect-timeout",
            str(min(30, timeout)),
            "--max-time",
            str(timeout),
            "--request",
            "POST",
            "--header",
            "Content-Type: application/json",
            "--header",
            "Accept: text/event-stream",
            "--header",
            "User-Agent: contractflow-research/0.1",
            "--data-binary",
            f"@{body_path}",
            url,
        ]
        if not ssl_verify:
            args.insert(3, "--insecure")
        process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise ConnectionError("curl stream did not expose the required pipes")
        authorization = _curl_config_quote(f"Authorization: Bearer {api_key}")
        process.stdin.write(f'header = "{authorization}"\n'.encode("utf-8"))
        process.stdin.close()
        try:
            text, data = read_chat_completion_stream(process.stdout)
        except BaseException as error:
            process.kill()
            stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
            process.wait()
            detail = f": {stderr[:500]}" if stderr else ""
            raise ConnectionError(f"curl streaming response failed{detail}") from error
        return_code = process.wait()
        stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
        if return_code != 0:
            detail = f": {stderr[:500]}" if stderr else ""
            raise ConnectionError(f"curl exited with status {return_code}{detail}")
        return text, data
    finally:
        try:
            os.unlink(body_path)
        except FileNotFoundError:
            pass


def _curl_config_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "")


def parse_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def parse_positive_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("max output token budget must be positive")
    return parsed


def parse_thinking(value: str | None) -> Literal["enabled", "disabled"] | None:
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower()
    if normalized not in {"enabled", "disabled"}:
        raise ValueError("thinking must be either 'enabled' or 'disabled'")
    return "enabled" if normalized == "enabled" else "disabled"


def parse_max_output_token_field(value: str | None) -> Literal["max_tokens", "max_completion_tokens"] | None:
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower()
    if normalized not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("max output token field must be 'max_tokens' or 'max_completion_tokens'")
    return "max_tokens" if normalized == "max_tokens" else "max_completion_tokens"

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def completion_history(model: ChatModel) -> list[dict[str, Any]]:
    getter = getattr(model, "completion_history", None)
    if not callable(getter):
        return []
    return [dict(item) for item in getter()]


def chat_completion_metadata(
    data: dict[str, Any],
    requested_max_output_tokens: int | None,
    requested_max_output_token_field: str,
    requested_thinking: Literal["enabled", "disabled"] | None,
) -> dict[str, Any]:
    choices = data.get("choices", [])
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    return {
        "request_max_output_tokens": requested_max_output_tokens,
        "request_max_output_token_field": requested_max_output_token_field,
        "request_thinking": requested_thinking,
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage", {}),
    }


def responses_completion_metadata(data: dict[str, Any], requested_max_output_tokens: int | None) -> dict[str, Any]:
    return {
        "request_max_output_tokens": requested_max_output_tokens,
        "status": data.get("status"),
        "incomplete_details": data.get("incomplete_details"),
        "usage": data.get("usage", {}),
    }


def extract_responses_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text
    chunks: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    if chunks:
        return "\n".join(chunks)
    choices = data.get("choices", [])
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message", {})
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
    raise KeyError("No text content found in Responses API payload.")


@dataclass
class MockLLM:
    model: str = "mock-llm"
    _last_completion_metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _completion_history: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        response_format: Literal["json_object"] | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        self._last_completion_metadata = {
            "request_max_output_tokens": max_output_tokens,
            "finish_reason": "mock",
            "started_at": utc_timestamp(),
            "elapsed_seconds": 0.0,
            "success": True,
            "attempts": 1,
            "retry_count": 0,
            "model": self.model,
            "wire_api": "mock",
            "response_format": response_format,
        }
        self._completion_history.append(dict(self._last_completion_metadata))
        return "{}"

    def completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def completion_history(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._completion_history]
