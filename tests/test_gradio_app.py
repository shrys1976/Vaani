import json
from pathlib import Path

import gradio_app


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("bad response")

    def json(self) -> dict:
        return self._payload


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = responses
        self.requests: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url: str, files=None, data=None, json=None):
        self.requests.append({"url": url, "files": files, "data": data, "json": json})
        return self._responses.pop(0)


def test_analyze_audio_returns_pending_state(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")

    fake_client = FakeClient(
        [
            FakeResponse(
                {
                    "transcript": "Create a file",
                    "decision": {
                        "intent": "create_file",
                        "requires_confirmation": True,
                        "payload": {"filename": "notes.txt", "content": "hello"},
                    },
                    "requires_confirmation": True,
                    "action_status": "awaiting_confirmation",
                    "result": "Approval required before execution.",
                    "metadata": {"source": "backend"},
                    "error": None,
                }
            )
        ]
    )

    monkeypatch.setattr(gradio_app.httpx, "Client", lambda timeout: fake_client)
    monkeypatch.setattr(gradio_app, "_build_backend_url", lambda path: f"http://test{path}")

    transcript, intent, status, result, decision_json, error_json, pending_state = gradio_app.analyze_audio(
        str(audio_path),
        None,
        None,
    )

    assert transcript == "Create a file"
    assert intent == "create_file"
    assert status == "awaiting_confirmation"
    assert result == "Approval required before execution."
    assert json.loads(decision_json)["payload"]["filename"] == "notes.txt"
    assert error_json == ""
    assert pending_state["decision"]["intent"] == "create_file"


def test_approve_action_clears_pending_state(monkeypatch) -> None:
    fake_client = FakeClient(
        [
            FakeResponse(
                {
                    "action_status": "completed",
                    "result": "done",
                    "error": None,
                }
            )
        ]
    )

    monkeypatch.setattr(gradio_app.httpx, "Client", lambda timeout: fake_client)
    monkeypatch.setattr(gradio_app, "_build_backend_url", lambda path: f"http://test{path}")

    status, result, error, pending_state_json = gradio_app.approve_action(
        {
            "transcript": "Create a file",
            "decision": {
                "intent": "create_file",
                "requires_confirmation": True,
                "payload": {"filename": "notes.txt", "content": "hello"},
            },
            "metadata": {"source": "ui"},
        }
    )

    assert status == "completed"
    assert result == "done"
    assert error == ""
    assert json.loads(pending_state_json) == {}


def test_reject_action_resets_state() -> None:
    status, result, error, pending_state_json = gradio_app.reject_action()

    assert status == "rejected"
    assert result == "Action rejected by user."
    assert error == ""
    assert json.loads(pending_state_json) == {}
