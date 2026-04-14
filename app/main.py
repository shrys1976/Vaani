from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.logger import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Local-first AI voice agent backend for Vaani.",
    )
    application.include_router(api_router)
    return application


app = create_app()
