from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

from app.core.config import Settings
from app.core.exceptions import STTServiceError


class WhisperPipelineProtocol(Protocol):
    def __call__(self, inputs: str, **kwargs: Any) -> dict[str, Any]:
        """Run local automatic speech recognition."""


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    model: str


class STTService:
    """Speech-to-text service backed by a local Hugging Face Whisper pipeline."""

    def __init__(self, settings: Settings, client: WhisperPipelineProtocol | None = None) -> None:
        self._settings = settings
        self._client = client or self._build_client()

    def transcribe(self, *, audio_bytes: bytes, filename: str, content_type: str) -> TranscriptionResult:
        if not audio_bytes:
            raise STTServiceError("Audio payload is empty.")

        suffix = Path(filename).suffix or self._suffix_from_content_type(content_type)
        with NamedTemporaryFile(suffix=suffix, delete=True) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio.flush()

            try:
                response = self._client(
                    temp_audio.name,
                    chunk_length_s=self._settings.stt_chunk_length_seconds,
                    generate_kwargs={"language": "en"},
                )
            except Exception as exc:
                raise STTServiceError("Failed to transcribe audio with the local Whisper model.") from exc

        text = response.get("text") if isinstance(response, dict) else None
        if not text or not isinstance(text, str):
            raise STTServiceError("Local Whisper returned an empty transcription.")

        return TranscriptionResult(
            text=text.strip(),
            model=self._settings.stt_model_id,
        )

    def _build_client(self) -> WhisperPipelineProtocol:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise STTServiceError(
                "Transformers is not installed. Run `uv sync --extra dev` to install local STT dependencies."
            ) from exc

        device = self._resolve_device()
        try:
            return pipeline(
                task="automatic-speech-recognition",
                model=self._settings.stt_model_id,
                device=device,
            )
        except Exception as exc:
            raise STTServiceError("Failed to initialize the local Whisper pipeline.") from exc

    def _resolve_device(self) -> int | str:
        if self._settings.stt_device.lower() == "cpu":
            return -1
        return self._settings.stt_device

    @staticmethod
    def _suffix_from_content_type(content_type: str) -> str:
        mapping = {
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/mp4": ".mp4",
            "audio/webm": ".webm",
        }
        return mapping.get(content_type.lower(), ".wav")
