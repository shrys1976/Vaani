from functools import partial

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.dependencies import get_pipeline_service
from app.schemas.request_models import ExecuteActionRequest
from app.schemas.response_models import HealthResponse, PipelineResponse
from app.services.pipeline_service import PipelineService, UploadedTextContext

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/process-audio", response_model=PipelineResponse, tags=["pipeline"])
async def process_audio(
    audio: UploadFile = File(...),
    context_text_file: UploadFile | None = File(default=None),
    context_text: str | None = Form(default=None),
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    uploaded_context = None
    if context_text_file is not None:
        raw_context = await context_text_file.read()
        uploaded_context = UploadedTextContext(
            filename=context_text_file.filename or "context.txt",
            content=_decode_text_payload(raw_context),
        )
    elif context_text:
        uploaded_context = UploadedTextContext(
            filename="context.txt",
            content=context_text,
        )

    audio_bytes = await audio.read()
    response = await run_in_threadpool(
        partial(
            pipeline_service.process_audio,
            audio_bytes=audio_bytes,
            filename=audio.filename or "audio.wav",
            content_type=audio.content_type or "application/octet-stream",
            context_file=uploaded_context,
        )
    )
    return response


@router.post("/execute-action", response_model=PipelineResponse, tags=["pipeline"])
async def execute_action(
    request: ExecuteActionRequest,
    pipeline_service: PipelineService = Depends(get_pipeline_service),
):
    response = await run_in_threadpool(
        partial(
            pipeline_service.execute_action,
            request=request,
        )
    )
    return response


def _decode_text_payload(raw_content: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw_content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_content.decode("utf-8", errors="ignore")
