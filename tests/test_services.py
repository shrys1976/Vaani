from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.exceptions import LLMServiceError, STTServiceError
from app.core.intent_parser import IntentParser
from app.services.llm_service import LLMService
from app.services.stt_service import STTService


class FakeTranscriptionsAPI:
    def __init__(self, response: object | Exception) -> None:
        self._response = response

    def create(self, *, file: tuple[str, object, str], model: str) -> object:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class FakeOpenAIClient:
    def __init__(self, response: object | Exception) -> None:
        self.audio = SimpleNamespace(transcriptions=FakeTranscriptionsAPI(response))


class FakeCompletionsAPI:
    def __init__(self, responses: list[object | Exception]) -> None:
        self._responses = responses
        self.calls = 0

    def create(self, *, model: str, temperature: float, messages: list[dict[str, str]]) -> object:
        response = self._responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


class FakeGroqClient:
    def __init__(self, responses: list[object | Exception]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletionsAPI(responses))


def make_choice_response(content: str) -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def build_settings() -> Settings:
    return Settings(
        openai_api_key="test-openai",
        groq_api_key="test-groq",
        llm_max_retries=1,
    )


def test_stt_service_transcribes_audio() -> None:
    service = STTService(
        settings=build_settings(),
        client=FakeOpenAIClient(SimpleNamespace(text="hello world")),
    )

    result = service.transcribe(
        audio_bytes=b"audio",
        filename="sample.wav",
        content_type="audio/wav",
    )

    assert result.text == "hello world"
    assert result.model == "whisper-1"


def test_stt_service_raises_on_provider_failure() -> None:
    service = STTService(
        settings=build_settings(),
        client=FakeOpenAIClient(RuntimeError("boom")),
    )

    with pytest.raises(STTServiceError):
        service.transcribe(
            audio_bytes=b"audio",
            filename="sample.wav",
            content_type="audio/wav",
        )


def test_llm_service_retries_and_falls_back_to_chat() -> None:
    fake_client = FakeGroqClient(
        responses=[
            make_choice_response("not json"),
            make_choice_response("still not json"),
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
    fake_client = FakeGroqClient(
        responses=[
            make_choice_response("not json"),
            make_choice_response(
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
        client=FakeGroqClient([RuntimeError("boom")]),
    )

    with pytest.raises(LLMServiceError):
        service.generate_chat_response("hello")
