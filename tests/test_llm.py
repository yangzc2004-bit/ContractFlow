from __future__ import annotations

import json
from http.client import IncompleteRead, RemoteDisconnected
from urllib.error import HTTPError
from urllib import request as urllib_request

import pytest

from learnbyai_research.llm import (
    MockLLM,
    OpenAICompatibleClient,
    is_retryable_request_error,
    transient_request_errors,
)

TEST_BASE_URL = "https" + "://example.invalid/v1"


def test_client_reads_prefixed_temperature_override(monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_API_KEY", "x")
    monkeypatch.setenv("REVIEW_MODEL", "kimi-k2.6")
    monkeypatch.setenv("REVIEW_BASE_URL", TEST_BASE_URL)
    monkeypatch.setenv("REVIEW_TEMPERATURE", "1")

    client = OpenAICompatibleClient.from_env("REVIEW")

    assert client.model == "kimi-k2.6"
    assert client.temperature_override == 1.0
    assert client._temperature(0.1) == 0.6


def test_client_reads_prefixed_output_token_budget(monkeypatch) -> None:
    monkeypatch.setenv("AUTHOR_API_KEY", "x")
    monkeypatch.setenv("AUTHOR_MODEL", "test-model")
    monkeypatch.setenv("AUTHOR_BASE_URL", TEST_BASE_URL)
    monkeypatch.setenv("AUTHOR_MAX_OUTPUT_TOKENS", "32768")
    monkeypatch.setenv("AUTHOR_THINKING", "disabled")

    client = OpenAICompatibleClient.from_env("AUTHOR")

    assert client.max_output_tokens == 32768
    assert client.thinking == "disabled"


def test_client_reads_chat_template_kwargs(monkeypatch) -> None:
    monkeypatch.setenv("AUTHOR_API_KEY", "x")
    monkeypatch.setenv("AUTHOR_MODEL", "test-model")
    monkeypatch.setenv("AUTHOR_BASE_URL", TEST_BASE_URL)
    monkeypatch.setenv("AUTHOR_CHAT_TEMPLATE_KWARGS", '{"thinking": false}')

    client = OpenAICompatibleClient.from_env("AUTHOR")

    assert client.chat_template_kwargs == {"thinking": False}


def test_client_reads_prefixed_stream_setting(monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_API_KEY", "x")
    monkeypatch.setenv("REVIEW_MODEL", "test-model")
    monkeypatch.setenv("REVIEW_BASE_URL", TEST_BASE_URL)
    monkeypatch.setenv("REVIEW_STREAM", "true")

    client = OpenAICompatibleClient.from_env("REVIEW")

    assert client.stream is True
    assert client.stream_transport == "curl"


def test_kimi_uses_current_completion_token_parameter() -> None:
    client = OpenAICompatibleClient(model="kimi-k2.6", api_key="x", base_url=TEST_BASE_URL)

    assert client._chat_max_output_token_field() == "max_completion_tokens"
    assert client._temperature(0.1) == 0.6


def test_incomplete_read_is_retryable() -> None:
    assert IncompleteRead in transient_request_errors()


def test_http_retryability_excludes_non_rate_limit_4xx() -> None:
    assert not is_retryable_request_error(HTTPError(TEST_BASE_URL, 403, "Forbidden", {}, None))
    assert is_retryable_request_error(HTTPError(TEST_BASE_URL, 429, "Rate limited", {}, None))
    assert is_retryable_request_error(HTTPError(TEST_BASE_URL, 503, "Unavailable", {}, None))


def test_non_retryable_http_error_fails_after_one_attempt(monkeypatch) -> None:
    attempts = 0

    def forbidden(request, **kwargs):
        nonlocal attempts
        attempts += 1
        raise HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(urllib_request, "urlopen", forbidden)
    monkeypatch.setattr(
        "learnbyai_research.llm.time.sleep",
        lambda seconds: pytest.fail("403 response should not be retried"),
    )
    client = OpenAICompatibleClient(
        model="deepseek-v4-flash",
        api_key="x",
        base_url=TEST_BASE_URL,
        max_retries=6,
    )

    with pytest.raises(RuntimeError, match="403"):
        client.complete("test")

    assert attempts == 1
    assert client.completion_metadata()["attempts"] == 1


def test_mock_records_completion_history_without_prompt_content() -> None:
    client = MockLLM()

    client.complete("secret prompt", response_format="json_object", max_output_tokens=42)

    history = client.completion_history()

    assert len(history) == 1
    assert history[0]["request_max_output_tokens"] == 42
    assert history[0]["response_format"] == "json_object"
    assert history[0]["attempts"] == 1
    assert history[0]["retry_count"] == 0
    assert "secret prompt" not in str(history[0])


class _FakeStreamingResponse:
    def __init__(self, lines: list[bytes], error_after: int | None = None) -> None:
        self.lines = lines
        self.error_after = error_after
        self.index = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def readline(self) -> bytes:
        if self.error_after is not None and self.index >= self.error_after:
            raise RemoteDisconnected("stream interrupted")
        if self.index >= len(self.lines):
            return b""
        line = self.lines[self.index]
        self.index += 1
        return line


def _sse_chunk(payload: dict) -> list[bytes]:
    return [f"data: {json.dumps(payload)}\n".encode("utf-8"), b"\n"]


def test_chat_stream_joins_content_and_records_metadata(monkeypatch) -> None:
    lines = [
        *_sse_chunk({"choices": [{"delta": {"content": '{"passed":'}, "finish_reason": None}]}),
        *_sse_chunk({"choices": [{"delta": {"content": "true}"}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n",
        b"\n",
    ]
    captured_request = {}

    def fake_urlopen(request, **kwargs):
        captured_request["payload"] = json.loads(request.data.decode("utf-8"))
        captured_request["accept"] = request.headers["Accept"]
        return _FakeStreamingResponse(lines)

    monkeypatch.setattr(urllib_request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        model="test-model",
        api_key="x",
        base_url=TEST_BASE_URL,
        stream=True,
        stream_transport="urllib",
    )

    text = client.complete("review", response_format="json_object")

    assert text == '{"passed":true}'
    assert captured_request["payload"]["stream"] is True
    assert captured_request["accept"] == "text/event-stream"
    assert client.completion_metadata()["finish_reason"] == "stop"
    assert client.completion_metadata()["stream"] is True


def test_chat_can_omit_default_system_message_for_upstream_reproduction(
    monkeypatch,
) -> None:
    lines = [
        *_sse_chunk(
            {
                "choices": [
                    {
                        "delta": {"content": "complete"},
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
        b"data: [DONE]\n",
        b"\n",
    ]
    captured_request = {}

    def fake_urlopen(request, **kwargs):
        captured_request["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeStreamingResponse(lines)

    monkeypatch.setattr(urllib_request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient(
        model="deepseek-v4-flash",
        api_key="x",
        base_url=TEST_BASE_URL,
        stream=True,
        stream_transport="urllib",
        include_default_system=False,
    )

    assert client.complete("official prompt") == "complete"
    assert captured_request["payload"]["messages"] == [
        {"role": "user", "content": "official prompt"}
    ]


def test_chat_stream_discards_partial_content_and_retries(monkeypatch) -> None:
    interrupted = _FakeStreamingResponse(
        _sse_chunk({"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]}),
        error_after=2,
    )
    complete_lines = [
        *_sse_chunk({"choices": [{"delta": {"content": "complete"}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n",
        b"\n",
    ]
    responses = iter([interrupted, _FakeStreamingResponse(complete_lines)])
    monkeypatch.setattr(urllib_request, "urlopen", lambda request, **kwargs: next(responses))
    monkeypatch.setattr("learnbyai_research.llm.time.sleep", lambda seconds: None)
    client = OpenAICompatibleClient(
        model="test-model",
        api_key="x",
        base_url=TEST_BASE_URL,
        stream=True,
        stream_transport="urllib",
        max_retries=1,
    )

    text = client.complete("review")

    assert text == "complete"
    assert client.completion_metadata()["attempts"] == 2
    assert client.completion_metadata()["retry_count"] == 1


def test_chat_stream_routes_through_curl_transport(monkeypatch) -> None:
    captured = {}

    def fake_curl_stream(**kwargs):
        captured.update(kwargs)
        return "complete", {
            "choices": [{"message": {"content": "complete"}, "finish_reason": "stop"}],
            "usage": {},
        }

    monkeypatch.setattr("learnbyai_research.llm.curl_chat_completion_stream", fake_curl_stream)
    client = OpenAICompatibleClient(
        model="test-model",
        api_key="x",
        base_url=TEST_BASE_URL,
        stream=True,
    )

    text = client.complete("review")

    assert text == "complete"
    assert captured["api_key"] == "x"
    assert json.loads(captured["body"].decode("utf-8"))["stream"] is True
    assert client.completion_metadata()["stream_transport"] == "curl"
