from types import SimpleNamespace

from app.core.router import ToolRouter
from app.schemas.request_models import ActionPayload, ExecuteActionRequest, IntentAnalysisResult, IntentDecision, IntentType
from app.services.pipeline_service import PipelineService, UploadedTextContext
from app.tools.code_tool import CodeTool
from app.tools.file_tool import FileTool
from app.tools.summary_tool import SummaryTool


class StubSTTService:
    def __init__(self, transcript: str = "hello") -> None:
        self._transcript = transcript

    def transcribe(self, *, audio_bytes: bytes, filename: str, content_type: str):
        return SimpleNamespace(text=self._transcript, model="whisper-1")


class StubLLMService:
    def __init__(self, analysis: IntentAnalysisResult, chat_response: str = "chat reply") -> None:
        self._analysis = analysis
        self._chat_response = chat_response

    def analyze_transcript(self, transcript: str) -> IntentAnalysisResult:
        return self._analysis

    def generate_chat_response(self, text: str) -> str:
        return self._chat_response


def build_router(tmp_path) -> ToolRouter:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return ToolRouter(
        file_tool=FileTool(output_dir),
        code_tool=CodeTool(output_dir),
        summary_tool=SummaryTool(),
    )


def test_process_audio_returns_awaiting_confirmation_for_write_intent(tmp_path) -> None:
    decision = IntentDecision(
        intent=IntentType.CREATE_FILE,
        requires_confirmation=True,
        payload=ActionPayload(filename="todo.txt", content="buy milk"),
    )
    pipeline = PipelineService(
        stt_service=StubSTTService("create a file"),
        llm_service=StubLLMService(
            IntentAnalysisResult(decision=decision, raw_output="{}", attempts=1, used_fallback=False)
        ),
        tool_router=build_router(tmp_path),
    )

    response = pipeline.process_audio(
        audio_bytes=b"audio",
        filename="sample.wav",
        content_type="audio/wav",
    )

    assert response.action_status == "awaiting_confirmation"
    assert response.requires_confirmation is True
    assert response.intent == "create_file"
    assert response.action == "create_file"


def test_process_audio_executes_summarize_with_uploaded_context(tmp_path) -> None:
    decision = IntentDecision(
        intent=IntentType.SUMMARIZE,
        requires_confirmation=False,
        payload=ActionPayload(),
    )
    pipeline = PipelineService(
        stt_service=StubSTTService("summarize this"),
        llm_service=StubLLMService(
            IntentAnalysisResult(decision=decision, raw_output="{}", attempts=1, used_fallback=False)
        ),
        tool_router=build_router(tmp_path),
    )

    response = pipeline.process_audio(
        audio_bytes=b"audio",
        filename="sample.wav",
        content_type="audio/wav",
        context_file=UploadedTextContext(filename="notes.txt", content="Alpha. Beta. Gamma."),
    )

    assert response.action_status == "completed"
    assert response.result == "Alpha. Beta."
    assert response.metadata["context_filename"] == "notes.txt"
    assert response.intent == "summarize"
    assert response.action == "summarize"


def test_execute_action_runs_approved_payload(tmp_path) -> None:
    pipeline = PipelineService(
        stt_service=StubSTTService(),
        llm_service=StubLLMService(
            IntentAnalysisResult(
                decision=IntentDecision(
                    intent=IntentType.CHAT,
                    requires_confirmation=False,
                    payload=ActionPayload(source_text="hello"),
                ),
                raw_output="{}",
            )
        ),
        tool_router=build_router(tmp_path),
    )

    response = pipeline.execute_action(
        ExecuteActionRequest(
            transcript="write a file",
            decision=IntentDecision(
                intent=IntentType.WRITE_CODE,
                requires_confirmation=True,
                payload=ActionPayload(filename="app.py", content="print('hi')", language="python"),
            ),
            metadata={"approved": True},
        )
    )

    assert response.action_status == "completed"
    assert response.metadata["language"] == "python"
    assert response.action == "write_code"
