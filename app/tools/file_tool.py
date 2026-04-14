from pathlib import Path

from app.core.exceptions import ToolExecutionError
from app.schemas.request_models import ActionPayload
from app.schemas.response_models import ToolExecutionResult


class FileTool:
    """Creates text files within the configured output directory."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir.resolve()

    def create(self, payload: ActionPayload) -> ToolExecutionResult:
        filename = payload.filename
        content = payload.content
        if not filename:
            raise ToolExecutionError("Filename is required for file creation.")
        if content is None:
            raise ToolExecutionError("Content is required for file creation.")

        destination = self._resolve_output_path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

        return ToolExecutionResult(
            action="create_file",
            status="completed",
            message=f"Created file at {destination}",
            output_path=str(destination),
            content=content,
        )

    def _resolve_output_path(self, filename: str) -> Path:
        candidate = (self._output_dir / filename).resolve()
        if candidate == self._output_dir:
            raise ToolExecutionError("Filename must point to a file inside the output directory.")
        if self._output_dir not in candidate.parents:
            raise ToolExecutionError("File operations are restricted to the output directory.")
        return candidate
