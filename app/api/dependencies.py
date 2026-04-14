from functools import lru_cache

from app.core.config import get_settings
from app.core.router import ToolRouter
from app.services.llm_service import LLMService
from app.services.pipeline_service import PipelineService
from app.services.stt_service import STTService
from app.tools.code_tool import CodeTool
from app.tools.file_tool import FileTool
from app.tools.summary_tool import SummaryTool


@lru_cache(maxsize=1)
def get_stt_service() -> STTService:
    return STTService(get_settings())


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService(get_settings())


@lru_cache(maxsize=1)
def get_tool_router() -> ToolRouter:
    settings = get_settings()
    return ToolRouter(
        file_tool=FileTool(settings.resolved_output_dir),
        code_tool=CodeTool(settings.resolved_output_dir),
        summary_tool=SummaryTool(),
    )


def get_pipeline_service() -> PipelineService:
    return PipelineService(
        stt_service=get_stt_service(),
        llm_service=get_llm_service(),
        tool_router=get_tool_router(),
    )
