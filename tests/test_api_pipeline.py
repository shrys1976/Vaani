from io import BytesIO

import asyncio
import pytest

import app.api.routes as route_module
from app.api.routes import execute_action, process_audio
from app.schemas.request_models import ActionPayload, ExecuteActionRequest, IntentDecision, IntentType
from app.schemas.response_models import PipelineResponse


class FakeUploadFile:
    def __init__(self, filename: str, content: bytes, content_type: str | None = None) -> None:
        self.filename = filename
        self.content_type = content_type
        self._buffer = BytesIO(content)

    async def read(self) -> bytes:
        return self._buffer.getvalue()


class StubPipelineService:
    def __init__(self) -> None:
        self.last_context_file = None
        self.last_execute_request = None

    def process_audio(self, *, audio_bytes: bytes, filename: str, content_type: str, context_file=None):
        self.last_context_file = context_file
        return PipelineResponse(
            transcript="Create a file",
            decision=IntentDecision(
                intent=IntentType.CREATE_FILE,
                requires_confirmation=True,
                payload=ActionPayload(filename="notes.txt", content="hello"),
            ),
            requires_confirmation=True,
            action_status="awaiting_confirmation",
            result="Approval required before execution.",
            metadata={"audio_filename": filename, "content_type": content_type},
        )

    def execute_action(self, request: ExecuteActionRequest):
        self.last_execute_request = request
        return PipelineResponse(
            transcript=request.transcript,
            decision=request.decision,
            requires_confirmation=request.decision.requires_confirmation,
            action_status="completed",
            result="done",
        )


@pytest.fixture
def immediate_threadpool(monkeypatch):
    async def fake_run_in_threadpool(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(route_module, "run_in_threadpool", fake_run_in_threadpool)


def test_process_audio_endpoint_returns_structured_response(immediate_threadpool) -> None:
    stub_service = StubPipelineService()

    async def run_request():
        audio = FakeUploadFile(filename="sample.wav", content=b"fake-audio", content_type="audio/wav")
        return await process_audio(
            audio=audio,
            context_text_file=None,
            context_text="summarize this extra text",
            pipeline_service=stub_service,
        )

    response = asyncio.run(run_request())

    assert response.action_status == "awaiting_confirmation"
    assert response.requires_confirmation is True
    assert response.decision is not None
    assert response.decision.intent == IntentType.CREATE_FILE
    assert stub_service.last_context_file is not None
    assert stub_service.last_context_file.content == "summarize this extra text"


def test_execute_action_endpoint_executes_approved_payload(immediate_threadpool) -> None:
    stub_service = StubPipelineService()

    async def run_request():
        return await execute_action(
            request=ExecuteActionRequest(
                transcript="Create a file",
                decision=IntentDecision(
                    intent=IntentType.CREATE_FILE,
                    requires_confirmation=True,
                    payload=ActionPayload(filename="notes.txt", content="hello"),
                ),
                metadata={"source": "ui"},
            ),
            pipeline_service=stub_service,
        )

    response = asyncio.run(run_request())

    assert response.action_status == "completed"
    assert response.result == "done"
    assert stub_service.last_execute_request is not None
    assert stub_service.last_execute_request.metadata == {"source": "ui"}
