from app.core.exceptions import ToolExecutionError
from app.schemas.request_models import IntentDecision, IntentType
from app.schemas.response_models import ToolExecutionResult
from app.tools.code_tool import CodeTool
from app.tools.file_tool import FileTool
from app.tools.summary_tool import SummaryTool


class ToolRouter:
    """Routes validated intents to the appropriate local tool."""

    def __init__(
        self,
        *,
        file_tool: FileTool,
        code_tool: CodeTool,
        summary_tool: SummaryTool,
    ) -> None:
        self._file_tool = file_tool
        self._code_tool = code_tool
        self._summary_tool = summary_tool

    def execute(self, decision: IntentDecision) -> ToolExecutionResult:
        payload = decision.payload

        if decision.intent is IntentType.CREATE_FILE:
            return self._file_tool.create(payload)
        if decision.intent is IntentType.WRITE_CODE:
            return self._code_tool.write(payload)
        if decision.intent is IntentType.SUMMARIZE:
            return self._summary_tool.run(payload)
        if decision.intent is IntentType.CHAT:
            raise ToolExecutionError("Chat intent should not be executed by the tool router.")

        raise ToolExecutionError(f"Unsupported intent: {decision.intent}")
