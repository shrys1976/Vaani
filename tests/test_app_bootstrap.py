import asyncio
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import app


def test_health_check() -> None:
    async def run_request() -> tuple[int, dict[str, str]]:
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/health")
            return response.status_code, response.json()

    status_code, payload = asyncio.run(run_request())

    assert status_code == 200
    assert payload == {"status": "ok"}


def test_output_dir_exists() -> None:
    settings = get_settings()

    assert settings.output_dir.exists()
    assert settings.output_dir.is_dir()
