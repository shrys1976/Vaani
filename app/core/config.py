from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Vaani"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = ""
    output_dir: Path = Field(default_factory=lambda: Path("output"))

    stt_model_id: str = "openai/whisper-base"
    stt_device: str = "cpu"
    stt_chunk_length_seconds: int = 30

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b"

    llm_temperature: float = 0.0
    llm_max_retries: int = 2
    request_timeout_seconds: float = 60.0
    gradio_backend_url: str = "http://127.0.0.1:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def resolved_output_dir(self) -> Path:
        return self.output_dir.resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings
