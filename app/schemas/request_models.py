from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IntentType(str, Enum):
    CREATE_FILE = "create_file"
    WRITE_CODE = "write_code"
    SUMMARIZE = "summarize"
    CHAT = "chat"


class ActionPayload(BaseModel):
    filename: str | None = None
    content: str | None = None
    language: str | None = None
    source_text: str | None = None

    model_config = ConfigDict(extra="forbid")


class IntentDecision(BaseModel):
    intent: IntentType
    requires_confirmation: bool
    payload: ActionPayload = Field(default_factory=ActionPayload)

    model_config = ConfigDict(extra="forbid")


class ExecuteActionRequest(BaseModel):
    transcript: str | None = None
    decision: IntentDecision
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")
