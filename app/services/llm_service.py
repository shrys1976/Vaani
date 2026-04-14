from typing import Any, Protocol

from groq import Groq

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


class GroqCompletionsAPI(Protocol):
    def create(self, *, model: str, temperature: float, messages: list[dict[str, str]]) -> Any:
        """Create a completion through the Groq API."""


class GroqChatAPI(Protocol):
    completions: GroqCompletionsAPI


class GroqClientProtocol(Protocol):
    chat: GroqChatAPI


class LLMService:
    """Groq-backed LLM service for intent classification and chat fallback."""

    def __init__(
        self,
        settings: Settings,
        parser: IntentParser | None = None,
        client: GroqClientProtocol | None = None,
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
        try:
            response = self._client.chat.completions.create(
                model=self._settings.groq_model,
                temperature=self._settings.llm_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for a local voice agent UI.",
                    },
                    {"role": "user", "content": text},
                ],
            )
        except Exception as exc:
            raise LLMServiceError("Failed to generate chat response.") from exc

        content = self._extract_content(response)
        if not content:
            raise LLMServiceError("LLM provider returned an empty chat response.")
        return content

    def summarize_text(self, text: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._settings.groq_model,
                temperature=self._settings.llm_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You summarize text for a local voice agent. "
                            "Return a concise plain-text summary only, no bullets and no markdown."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
            )
        except Exception as exc:
            raise LLMServiceError("Failed to generate summary response.") from exc

        content = self._extract_content(response)
        if not content:
            raise LLMServiceError("LLM provider returned an empty summary response.")
        return content

    def _request_intent_completion(self, *, transcript: str, retry_mode: bool) -> str:
        prompt = BASE_SYSTEM_PROMPT
        if retry_mode:
            prompt = f"{BASE_SYSTEM_PROMPT}\n\n{RETRY_SYSTEM_PROMPT}"

        try:
            response = self._client.chat.completions.create(
                model=self._settings.groq_model,
                temperature=self._settings.llm_temperature,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": transcript},
                ],
            )
        except Exception as exc:
            raise LLMServiceError("Failed to analyze transcript with the LLM.") from exc

        content = self._extract_content(response)
        if not content:
            raise LLMServiceError("LLM provider returned an empty structured response.")
        return content

    @staticmethod
    def _extract_content(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        if not message:
            return ""

        content = getattr(message, "content", "")
        return content.strip() if isinstance(content, str) else ""

    def _build_client(self) -> Groq:
        if not self._settings.groq_api_key:
            raise LLMServiceError("GROQ_API_KEY is not configured.")
        return Groq(api_key=self._settings.groq_api_key)
