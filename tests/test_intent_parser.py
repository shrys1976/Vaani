import pytest

from app.core.exceptions import IntentValidationError
from app.core.intent_parser import IntentParser
from app.schemas.request_models import IntentType


def test_parser_normalizes_write_intent_to_require_confirmation() -> None:
    parser = IntentParser()

    decision = parser.parse(
        """
        {
          "intent": "write_code",
          "requires_confirmation": false,
          "payload": {
            "filename": "retry.py",
            "content": "def retry(): pass",
            "language": "python"
          }
        }
        """
    )

    assert decision.intent is IntentType.WRITE_CODE
    assert decision.requires_confirmation is True


def test_parser_extracts_json_from_markdown_wrapper() -> None:
    parser = IntentParser()

    decision = parser.parse(
        """```json
        {"intent":"chat","requires_confirmation":false,"payload":{"source_text":"hello"}}
        ```"""
    )

    assert decision.intent is IntentType.CHAT
    assert decision.payload.source_text == "hello"


def test_parser_rejects_invalid_output() -> None:
    parser = IntentParser()

    with pytest.raises(IntentValidationError):
        parser.parse("not json at all")
