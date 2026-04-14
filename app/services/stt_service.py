from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol

from openai import OpenAI

from app.core.config import Settings
from app.core.exceptions import STTServiceError


class OpenAITranscriptionsAPI(Protocol):
    def create(self, *, file: tuple[str, BytesIO, str], model: str) -> Any:
        """Create a transcription through the OpenAI API."""


class OpenAIAudioAPI(Protocol):
    transcriptions: OpenAITranscriptionsAPI


class OpenAIClientProtocol(Protocol):
    audio: OpenAIAudioAPI


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    model: str


class STTService:
    """Speech-to-text service backed by OpenAI Whisper."""

    def __init__(self, settings: Settings, client: OpenAIClientProtocol | None = None) -> None:
        self._settings = settings
        self._client = client or self._build_client()

    def transcribe(self, *, audio_bytes: bytes, filename: str, content_type: str) -> TranscriptionResult:
        if not audio_bytes:
            raise STTServiceError("Audio payload is empty.")

        try:
            response = self._client.audio.transcriptions.create(
                file=(filename, BytesIO(audio_bytes), content_type),
                model=self._settings.openai_transcription_model,
            )
        except Exception as exc:
            raise STTServiceError("Failed to transcribe audio.") from exc

        text = getattr(response, "text", None)
        if not text or not isinstance(text, str):
            raise STTServiceError("Transcription provider returned an empty response.")

        return TranscriptionResult(
            text=text.strip(),
            model=self._settings.openai_transcription_model,
        )

    def _build_client(self) -> OpenAI:
        if not self._settings.openai_api_key:
            raise STTServiceError("OPENAI_API_KEY is not configured.")
        return OpenAI(api_key=self._settings.openai_api_key)
