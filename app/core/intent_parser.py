import json
import re

from pydantic import ValidationError

from app.core.exceptions import IntentValidationError
from app.schemas.request_models import ActionPayload, IntentDecision, IntentType


class IntentParser:
    """Parses and normalizes structured intent responses from the LLM."""

    _json_block_pattern = re.compile(r"\{.*\}", re.DOTALL)

    def parse(self, raw_output: str) -> IntentDecision:
        candidate = self._extract_json(raw_output)

        try:
            payload = json.loads(candidate)
            decision = IntentDecision.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise IntentValidationError("Invalid structured LLM output.") from exc

        return self._normalize(decision)

    def fallback_chat_decision(self, transcript: str | None = None) -> IntentDecision:
        return IntentDecision(
            intent=IntentType.CHAT,
            requires_confirmation=False,
            payload=ActionPayload(source_text=transcript),
        )

    def _extract_json(self, raw_output: str) -> str:
        stripped = raw_output.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`").strip()
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()

        match = self._json_block_pattern.search(stripped)
        if not match:
            raise IntentValidationError("No JSON object found in LLM output.")
        return match.group(0)

    def _normalize(self, decision: IntentDecision) -> IntentDecision:
        if decision.intent in {IntentType.CREATE_FILE, IntentType.WRITE_CODE}:
            decision.requires_confirmation = True
        elif decision.intent in {IntentType.SUMMARIZE, IntentType.CHAT}:
            decision.requires_confirmation = False
        else:
            raise IntentValidationError(f"Unsupported intent: {decision.intent}")

        return decision
