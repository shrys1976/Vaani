from typing import Any, Protocol

import httpx

from app.core.config import Settings
from app.core.exceptions import LLMServiceError
from app.core.intent_parser import IntentParser
from app.schemas.request_models import IntentAnalysisResult

BASE_SYSTEM_PROMPT = """
You are an AI agent that converts user input into structured JSON.

You MUST return ONLY valid JSON. No explanations.

Available intents:
1. create_file
2. write_code
3. summarize
4. chat

Rules:
- Always choose one intent
- If file writing or code execution is involved, set requires_confirmation = true
- Keep responses deterministic and structured

JSON format:
{
  "intent": "...",
  "requires_confirmation": true/false,
  "payload": {
    "filename": "...",
    "content": "...",
    "language": "...",
    "source_text": "..."
  }
}
""".strip()

RETRY_SYSTEM_PROMPT = """
Return one JSON object only.
Do not wrap it in markdown.
Do not include commentary.
Do not add fields outside: intent, requires_confirmation, payload.
""".strip()


class OllamaClientProtocol(Protocol):
    def post(self, url: str, *, json: dict[str, Any]) -> Any:
        """Send a chat request to Ollama."""


class LLMService:
    """Ollama-backed LLM service for intent classification and chat fallback."""

    def __init__(
        self,
        settings: Settings,
        parser: IntentParser | None = None,
        client: OllamaClientProtocol | None = None,
    ) -> None:
        self._settings = settings
        self._parser = parser or IntentParser()
        self._client = client or self._build_client()

    def analyze_transcript(self, transcript: str) -> IntentAnalysisResult:
        last_raw_output = ""

        for attempt in range(1, self._settings.llm_max_retries + 2):
            raw_output = self._request_intent_completion(
                transcript=transcript,
                retry_mode=attempt > 1,
            )
            last_raw_output = raw_output

            try:
                decision = self._parser.parse(raw_output)
                return IntentAnalysisResult(
                    decision=decision,
                    raw_output=raw_output,
                    attempts=attempt,
                    used_fallback=False,
                )
            except Exception:
                continue

        return IntentAnalysisResult(
            decision=self._parser.fallback_chat_decision(transcript),
            raw_output=last_raw_output,
            attempts=self._settings.llm_max_retries + 1,
            used_fallback=True,
        )

    def generate_chat_response(self, text: str) -> str:
        content = self._send_chat(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant for a local voice agent UI.",
                },
                {"role": "user", "content": text},
            ],
            error_message="Failed to generate chat response.",
        )
        if not content:
            raise LLMServiceError("Ollama returned an empty chat response.")
        return content

    def summarize_text(self, text: str) -> str:
        content = self._send_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You summarize text for a local voice agent. "
                        "Return a concise plain-text summary only, no bullets and no markdown."
                    ),
                },
                {"role": "user", "content": text},
            ],
            error_message="Failed to generate summary response.",
        )
        if not content:
            raise LLMServiceError("Ollama returned an empty summary response.")
        return content

    def _request_intent_completion(self, *, transcript: str, retry_mode: bool) -> str:
        prompt = BASE_SYSTEM_PROMPT
        if retry_mode:
            prompt = f"{BASE_SYSTEM_PROMPT}\n\n{RETRY_SYSTEM_PROMPT}"

        content = self._send_chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": transcript},
            ],
            error_message="Failed to analyze transcript with the LLM.",
        )
        if not content:
            raise LLMServiceError("Ollama returned an empty structured response.")
        return content

    def _send_chat(self, messages: list[dict[str, str]], *, error_message: str) -> str:
        try:
            response = self._client.post(
                "/api/chat",
                json={
                    "model": self._settings.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": self._settings.llm_temperature},
                },
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise LLMServiceError(error_message) from exc

        return self._extract_content(payload)

    @staticmethod
    def _extract_content(payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        message = payload.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content", "")
        return content.strip() if isinstance(content, str) else ""

    def _build_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self._settings.ollama_base_url.rstrip("/"),
            timeout=self._settings.request_timeout_seconds,
        )
