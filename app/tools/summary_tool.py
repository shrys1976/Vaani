from collections.abc import Callable

from app.core.exceptions import LLMServiceError
from app.core.exceptions import ToolExecutionError
from app.schemas.request_models import ActionPayload
from app.schemas.response_models import ToolExecutionResult


class SummaryTool:
    """Produces a summary for transcript or provided text."""

    def __init__(self, summarizer: Callable[[str], str] | None = None) -> None:
        self._summarizer = summarizer

    def run(self, payload: ActionPayload) -> ToolExecutionResult:
        source_text = payload.source_text or payload.content
        if not source_text or not source_text.strip():
            raise ToolExecutionError("Source text is required for summarization.")

        normalized = " ".join(source_text.split())
        if self._summarizer is not None:
            try:
                summary = self._summarizer(normalized).strip()
            except LLMServiceError as exc:
                raise ToolExecutionError(str(exc)) from exc
        else:
            sentences = [chunk.strip() for chunk in normalized.split(".") if chunk.strip()]
            summary = ". ".join(sentences[:2]).strip()
            if summary and not summary.endswith("."):
                summary = f"{summary}."
            if not summary:
                summary = normalized[:240]

        return ToolExecutionResult(
            action="summarize",
            status="completed",
            message="Generated summary.",
            content=summary,
            metadata={"source_length": len(source_text)},
        )
