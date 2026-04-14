from typing import Any

from pydantic import BaseModel, Field

from app.schemas.request_models import IntentDecision


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class PipelineResponse(BaseModel):
    transcript: str | None = None
    decision: IntentDecision | None = None
    requires_confirmation: bool = False
    action_status: str = "pending"
    result: str | None = None
    error: ErrorResponse | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResult(BaseModel):
    action: str
    status: str
    message: str
    output_path: str | None = None
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
