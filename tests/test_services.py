from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.exceptions import LLMServiceError, STTServiceError
from app.core.intent_parser import IntentParser
from app.services.llm_service import LLMService
from app.services.stt_service import STTService


class FakeWhisperPipeline:
    def __init__(self, response: object | Exception) -> None:
        self._response = response

    def __call__(self, inputs: str, **kwargs) -> object:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class FakeOllamaResponse:
    def __init__(self, payload: dict | Exception) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        if isinstance(self._payload, Exception):
            raise self._payload

    def json(self) -> dict:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeOllamaClient:
    def __init__(self, responses: list[object | Exception]) -> None:
        self._responses = responses
        self.calls = 0

    def post(self, url: str, *, json: dict) -> object:
        response = self._responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return FakeOllamaResponse(response)


def make_chat_response(content: str) -> dict:
    return {"message": {"content": content}}


def build_settings() -> Settings:
    return Settings(
        stt_model_id="openai/whisper-base",
        ollama_model="qwen2.5:3b",
        llm_max_retries=1,
    )


def test_stt_service_transcribes_audio() -> None:
    service = STTService(
        settings=build_settings(),
        client=FakeWhisperPipeline({"text": "hello world"}),
    )

    result = service.transcribe(
        audio_bytes=b"audio",
        filename="sample.wav",
        content_type="audio/wav",
    )

    assert result.text == "hello world"
    assert result.model == "openai/whisper-base"


def test_stt_service_raises_on_provider_failure() -> None:
    service = STTService(
        settings=build_settings(),
        client=FakeWhisperPipeline(RuntimeError("boom")),
    )

    with pytest.raises(STTServiceError):
        service.transcribe(
            audio_bytes=b"audio",
            filename="sample.wav",
            content_type="audio/wav",
        )


def test_llm_service_retries_and_falls_back_to_chat() -> None:
    fake_client = FakeOllamaClient(
        responses=[
            make_chat_response("not json"),
            make_chat_response("still not json"),
        ]
    )
    service = LLMService(
        settings=build_settings(),
        parser=IntentParser(),
        client=fake_client,
    )

    result = service.analyze_transcript("say hello")

    assert result.used_fallback is True
    assert result.attempts == 2
    assert result.decision.intent.value == "chat"
    assert result.decision.payload.source_text == "say hello"


def test_llm_service_returns_valid_intent_after_retry() -> None:
    fake_client = FakeOllamaClient(
        responses=[
            make_chat_response("not json"),
            make_chat_response(
                '{"intent":"summarize","requires_confirmation":true,"payload":{"source_text":"abc"}}'
            ),
        ]
    )
    service = LLMService(
        settings=build_settings(),
        parser=IntentParser(),
        client=fake_client,
    )

    result = service.analyze_transcript("summarize this")

    assert result.used_fallback is False
    assert result.attempts == 2
    assert result.decision.intent.value == "summarize"
    assert result.decision.requires_confirmation is False


def test_llm_service_raises_on_chat_provider_failure() -> None:
    service = LLMService(
        settings=build_settings(),
        parser=IntentParser(),
        client=FakeOllamaClient([RuntimeError("boom")]),
    )

    with pytest.raises(LLMServiceError):
        service.generate_chat_response("hello")


def test_llm_service_generates_summary() -> None:
    service = LLMService(
        settings=build_settings(),
        parser=IntentParser(),
        client=FakeOllamaClient([make_chat_response("Short summary.")]),
    )

    result = service.summarize_text("Long text")

    assert result == "Short summary."
