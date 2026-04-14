from pathlib import Path

import pytest

from app.core.exceptions import ToolExecutionError
from app.core.router import ToolRouter
from app.schemas.request_models import ActionPayload, IntentDecision, IntentType
from app.tools.code_tool import CodeTool
from app.tools.file_tool import FileTool
from app.tools.summary_tool import SummaryTool


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "output"
    directory.mkdir()
    return directory


def build_router(output_dir: Path) -> ToolRouter:
    return ToolRouter(
        file_tool=FileTool(output_dir),
        code_tool=CodeTool(output_dir),
        summary_tool=SummaryTool(),
    )


def test_file_tool_creates_file_in_output(output_dir: Path) -> None:
    tool = FileTool(output_dir)

    result = tool.create(ActionPayload(filename="notes.txt", content="hello"))

    destination = output_dir / "notes.txt"
    assert destination.read_text(encoding="utf-8") == "hello"
    assert result.output_path == str(destination.resolve())


def test_file_tool_can_create_folder_in_output(output_dir: Path) -> None:
    tool = FileTool(output_dir)

    result = tool.create(ActionPayload(filename="notes", content=None))

    destination = output_dir / "notes"
    assert destination.is_dir()
    assert result.action == "create_folder"


def test_file_tool_rejects_path_traversal(output_dir: Path) -> None:
    tool = FileTool(output_dir)

    with pytest.raises(ToolExecutionError):
        tool.create(ActionPayload(filename="../escape.txt", content="nope"))


def test_code_tool_writes_code_and_language_metadata(output_dir: Path) -> None:
    tool = CodeTool(output_dir)

    result = tool.write(
        ActionPayload(
            filename="retry.py",
            content="def retry():\n    return True\n",
            language="python",
        )
    )

    destination = output_dir / "retry.py"
    assert destination.read_text(encoding="utf-8").startswith("def retry")
    assert result.metadata["language"] == "python"


def test_summary_tool_summarizes_source_text() -> None:
    tool = SummaryTool(summarizer=lambda text: "Model summary.")

    result = tool.run(
        ActionPayload(
            source_text="Sentence one. Sentence two. Sentence three."
        )
    )

    assert result.content == "Model summary."


def test_router_executes_create_file(output_dir: Path) -> None:
    router = build_router(output_dir)

    result = router.execute(
        IntentDecision(
            intent=IntentType.CREATE_FILE,
            requires_confirmation=True,
            payload=ActionPayload(filename="todo.txt", content="buy milk"),
        )
    )

    assert result.action == "create_file"
    assert (output_dir / "todo.txt").exists()


def test_router_executes_write_code(output_dir: Path) -> None:
    router = build_router(output_dir)

    result = router.execute(
        IntentDecision(
            intent=IntentType.WRITE_CODE,
            requires_confirmation=True,
            payload=ActionPayload(filename="app.py", content="print('hi')", language="python"),
        )
    )

    assert result.action == "write_code"
    assert (output_dir / "app.py").read_text(encoding="utf-8") == "print('hi')"


def test_router_executes_summarize(output_dir: Path) -> None:
    router = build_router(output_dir)

    result = router.execute(
        IntentDecision(
            intent=IntentType.SUMMARIZE,
            requires_confirmation=False,
            payload=ActionPayload(source_text="Alpha. Beta. Gamma."),
        )
    )

    assert result.action == "summarize"
    assert result.content == "Alpha. Beta."


def test_router_rejects_chat_execution(output_dir: Path) -> None:
    router = build_router(output_dir)

    with pytest.raises(ToolExecutionError):
        router.execute(
            IntentDecision(
                intent=IntentType.CHAT,
                requires_confirmation=False,
                payload=ActionPayload(source_text="hello"),
            )
        )
