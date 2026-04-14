from dataclasses import dataclass

from app.core.exceptions import LLMServiceError, STTServiceError, ToolExecutionError
from app.core.router import ToolRouter
from app.schemas.request_models import ActionPayload, ExecuteActionRequest, IntentDecision, IntentType
from app.schemas.response_models import ErrorResponse, PipelineResponse, ToolExecutionResult
from app.services.llm_service import LLMService
from app.services.stt_service import STTService


@dataclass(slots=True)
class UploadedTextContext:
    filename: str
    content: str


class PipelineService:
    """Coordinates transcription, intent analysis, confirmation gating, and execution."""

    def __init__(
        self,
        *,
        stt_service: STTService,
        llm_service: LLMService,
        tool_router: ToolRouter,
    ) -> None:
        self._stt_service = stt_service
        self._llm_service = llm_service
        self._tool_router = tool_router

    def process_audio(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        content_type: str,
        context_file: UploadedTextContext | None = None,
    ) -> PipelineResponse:
        try:
            transcription = self._stt_service.transcribe(
                audio_bytes=audio_bytes,
                filename=filename,
                content_type=content_type,
            )
        except STTServiceError as exc:
            return PipelineResponse(
                action_status="error",
                error=ErrorResponse(code="stt_error", message=str(exc)),
            )

        transcript = transcription.text

        try:
            analysis = self._llm_service.analyze_transcript(transcript)
        except LLMServiceError as exc:
            return PipelineResponse(
                transcript=transcript,
                intent="chat",
                action_status="error",
                error=ErrorResponse(code="llm_error", message=str(exc)),
            )

        decision = analysis.decision
        if context_file and decision.intent is IntentType.SUMMARIZE:
            decision = self._merge_uploaded_text(decision, context_file)

        if decision.requires_confirmation:
            return PipelineResponse(
                transcript=transcript,
                intent=decision.intent.value,
                action=decision.intent.value,
                decision=decision,
                requires_confirmation=True,
                action_status="awaiting_confirmation",
                result="Approval required before execution.",
                metadata={
                    "attempts": analysis.attempts,
                    "used_fallback": analysis.used_fallback,
                    "context_filename": context_file.filename if context_file else None,
                },
            )

        return self._execute_decision(
            transcript=transcript,
            decision=decision,
            metadata={
                "attempts": analysis.attempts,
                "used_fallback": analysis.used_fallback,
                "context_filename": context_file.filename if context_file else None,
            },
        )

    def execute_action(self, request: ExecuteActionRequest) -> PipelineResponse:
        return self._execute_decision(
            transcript=request.transcript,
            decision=request.decision,
            metadata=request.metadata,
        )

    def _execute_decision(
        self,
        *,
        transcript: str | None,
        decision: IntentDecision,
        metadata: dict[str, object] | None = None,
    ) -> PipelineResponse:
        metadata = dict(metadata or {})

        if decision.intent is IntentType.CHAT:
            source_text = decision.payload.source_text or transcript or ""
            try:
                reply = self._llm_service.generate_chat_response(source_text)
            except LLMServiceError as exc:
                return PipelineResponse(
                    transcript=transcript,
                    intent=decision.intent.value,
                    action="chat_response",
                    decision=decision,
                    requires_confirmation=False,
                    action_status="error",
                    error=ErrorResponse(code="chat_error", message=str(exc)),
                    metadata=metadata,
                )

            return PipelineResponse(
                transcript=transcript,
                intent=decision.intent.value,
                action="chat_response",
                decision=decision,
                requires_confirmation=False,
                action_status="completed",
                result=reply,
                metadata=metadata,
            )

        try:
            execution_result = self._tool_router.execute(decision)
        except ToolExecutionError as exc:
            return PipelineResponse(
                transcript=transcript,
                intent=decision.intent.value,
                action=decision.intent.value,
                decision=decision,
                requires_confirmation=decision.requires_confirmation,
                action_status="error",
                error=ErrorResponse(code="tool_error", message=str(exc)),
                metadata=metadata,
            )

        return self._pipeline_response_from_execution(
            transcript=transcript,
            decision=decision,
            execution_result=execution_result,
            metadata=metadata,
        )

    def _pipeline_response_from_execution(
        self,
        *,
        transcript: str | None,
        decision: IntentDecision,
        execution_result: ToolExecutionResult,
        metadata: dict[str, object],
    ) -> PipelineResponse:
        merged_metadata = dict(metadata)
        merged_metadata.update(execution_result.metadata)
        if execution_result.output_path:
            merged_metadata["output_path"] = execution_result.output_path

        return PipelineResponse(
            transcript=transcript,
            intent=decision.intent.value,
            action=execution_result.action,
            decision=decision,
            requires_confirmation=decision.requires_confirmation,
            action_status=execution_result.status,
            result=execution_result.content or execution_result.message,
            metadata=merged_metadata,
        )

    @staticmethod
    def _merge_uploaded_text(
        decision: IntentDecision,
        context_file: UploadedTextContext,
    ) -> IntentDecision:
        payload = ActionPayload.model_validate(decision.payload.model_dump())
        if not payload.source_text:
            payload.source_text = context_file.content
        return IntentDecision(
            intent=decision.intent,
            requires_confirmation=decision.requires_confirmation,
            payload=payload,
        )
