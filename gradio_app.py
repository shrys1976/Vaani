from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

from app.core.config import get_settings


def _build_backend_url(path: str) -> str:
    settings = get_settings()
    return f"{settings.gradio_backend_url.rstrip('/')}{path}"


def _serialize_decision(decision: dict[str, Any] | None) -> str:
    if not decision:
        return "{}"
    return json.dumps(decision, indent=2, sort_keys=True)


def _serialize_error(error: dict[str, Any] | None) -> str:
    if not error:
        return ""
    return json.dumps(error, indent=2, sort_keys=True)


def _empty_pending_state() -> dict[str, Any]:
    return {}


def _serialize_metadata(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return "{}"
    return json.dumps(metadata, indent=2, sort_keys=True)


def analyze_audio(
    audio_path: str | None,
    context_text: str | None,
    context_file_path: str | None,
) -> tuple[str, str, str, str, str, str, str, dict[str, Any]]:
    if not audio_path:
        return (
            "",
            "",
            "",
            "error",
            "",
            "{}",
            "Please provide audio input before analyzing.",
            "",
            _empty_pending_state(),
        )

    files = {
        "audio": _open_upload(audio_path, default_content_type="audio/wav"),
    }
    data: dict[str, str] = {}

    if context_text:
        data["context_text"] = context_text
    elif context_file_path:
        files["context_text_file"] = _open_upload(
            context_file_path,
            default_content_type="text/plain",
        )

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                _build_backend_url("/process-audio"),
                files=files,
                data=data,
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        return (
            "",
            "",
            "",
            "error",
            "",
            "{}",
            f"Backend request failed: {exc}",
            "",
            _empty_pending_state(),
        )
    finally:
        _close_files(files)

    decision = payload.get("decision") or {}
    pending_state = _empty_pending_state()
    if payload.get("requires_confirmation"):
        pending_state = {
            "transcript": payload.get("transcript"),
            "decision": decision,
            "metadata": payload.get("metadata", {}),
        }

    return (
        payload.get("transcript") or "",
        decision.get("intent") or "",
        payload.get("action") or "",
        payload.get("action_status") or "",
        payload.get("result") or "",
        _serialize_metadata(payload.get("metadata")),
        _serialize_decision(decision),
        _serialize_error(payload.get("error")),
        pending_state,
    )


def approve_action(pending_state: dict[str, Any] | None) -> tuple[str, str, str, str, str]:
    if not pending_state:
        return "No pending action to approve.", "", "{}", "", json.dumps(_empty_pending_state())

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                _build_backend_url("/execute-action"),
                json=pending_state,
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        return f"Approval request failed: {exc}", "", "{}", "", json.dumps(pending_state, indent=2)

    return (
        payload.get("action_status") or "",
        payload.get("result") or "",
        _serialize_metadata(payload.get("metadata")),
        _serialize_error(payload.get("error")),
        json.dumps(_empty_pending_state()),
    )


def reject_action() -> tuple[str, str, str, str, str]:
    return "rejected", "Action rejected by user.", "{}", "", json.dumps(_empty_pending_state())


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Vaani") as demo:
        pending_state = gr.State(_empty_pending_state())

        gr.Markdown("## Vaani\nLocal-first voice agent with explicit approval before file writes.")

        with gr.Row():
            audio_input = gr.Audio(
                label="Audio Input",
                sources=["microphone", "upload"],
                type="filepath",
            )
            with gr.Column():
                context_text = gr.Textbox(
                    label="Optional Context Text",
                    lines=4,
                    placeholder="Paste text here for summarize requests.",
                )
                context_file = gr.File(
                    label="Optional Context Text File",
                    type="filepath",
                    file_count="single",
                )

        analyze_button = gr.Button("Analyze Audio", variant="primary")

        with gr.Row():
            transcript_output = gr.Textbox(label="Transcript", lines=4)
            intent_output = gr.Textbox(label="Detected Intent")

        with gr.Row():
            action_output = gr.Textbox(label="Planned Action")
            status_output = gr.Textbox(label="Action Status")
            result_output = gr.Textbox(label="Result", lines=6)

        metadata_output = gr.Code(label="Metadata", language="json")
        decision_output = gr.Code(label="Decision Preview", language="json")
        error_output = gr.Code(label="Error", language="json")

        with gr.Row():
            approve_button = gr.Button("Approve", variant="secondary")
            reject_button = gr.Button("Reject")

        analyze_button.click(
            fn=analyze_audio,
            inputs=[audio_input, context_text, context_file],
            outputs=[
                transcript_output,
                intent_output,
                action_output,
                status_output,
                result_output,
                metadata_output,
                decision_output,
                error_output,
                pending_state,
            ],
        )

        approve_button.click(
            fn=approve_action,
            inputs=[pending_state],
            outputs=[status_output, result_output, metadata_output, error_output, pending_state],
        )

        reject_button.click(
            fn=reject_action,
            outputs=[status_output, result_output, metadata_output, error_output, pending_state],
        )

    return demo


def _open_upload(path: str, *, default_content_type: str) -> tuple[str, Any, str]:
    file_path = Path(path)
    return (file_path.name, file_path.open("rb"), default_content_type)


def _close_files(files: dict[str, tuple[str, Any, str]]) -> None:
    for _, file_obj, _ in files.values():
        file_obj.close()


if __name__ == "__main__":
    build_ui().launch()
