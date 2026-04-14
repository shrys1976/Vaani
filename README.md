# Vaani

Vaani is a local-only AI voice agent with a FastAPI backend and a separate Gradio client. It accepts audio input, transcribes speech with OpenAI Whisper, classifies intent with Groq, validates the result as structured JSON, and safely routes approved actions to local tools.

## Features

- Audio ingestion through a backend `POST /process-audio` pipeline
- Speech-to-text via OpenAI Whisper
- Intent classification and chat fallback via Groq
- Strict JSON parsing and safe fallback behavior
- Human-in-the-loop approval before any file-writing action
- Local tool execution restricted to the `output/` directory
- Model-backed summarization through the Groq service
- `create_file` support for both files and folders inside `output/`
- Separate Gradio interface for analysis, review, approval, and rejection

## Architecture

The backend is organized under `app/`:

- `app/api`: FastAPI routes and dependency wiring
- `app/core`: config, logging, exceptions, intent parsing, and routing
- `app/services`: STT, LLM, and pipeline orchestration
- `app/tools`: local tools for files, code generation, and summarization
- `app/schemas`: request and response contracts

The UI lives in `gradio_app.py` and communicates with the backend over HTTP.

## API Overview

### `POST /process-audio`

Accepts multipart audio and optional summarize context.

Returns a structured response with:

- `transcript`
- `intent`
- `action`
- `decision`
- `requires_confirmation`
- `action_status`
- `result`
- `error`
- `metadata`

### `POST /execute-action`

Accepts an approved action payload and executes it after the UI approval step.

## Setup

1. Create a `.env` file from `.env.example`.
2. Install dependencies:

```bash
uv sync --extra dev
```

3. Set the required API keys:

- `OPENAI_API_KEY`
- `GROQ_API_KEY`

## Running The Backend

```bash
uv run uvicorn main:app --reload
```

The backend runs by default at `http://127.0.0.1:8000`.

## Running The Gradio UI

In a second terminal:

```bash
uv run python gradio_app.py
```

The Gradio app uses `GRADIO_BACKEND_URL` from your environment and defaults to `http://127.0.0.1:8000`.

## Testing

Run the full test suite with:

```bash
./.venv/bin/python -m pytest -q
```

## Safety Model

- File and code writes are restricted to `output/`
- Path traversal attempts are rejected
- Invalid LLM JSON is retried and then downgraded to a safe chat fallback
- Write-style intents require confirmation before execution

## Current Local Scope

- `create_file`: creates folders when only a path is provided, or writes generic text files inside `output/`
- `write_code`: writes code text to `output/`
- `summarize`: summarizes transcript or supplied text context with the LLM service
- `chat`: returns a plain LLM response without tool execution
- approval remains UI-driven by design, with the Gradio client resubmitting approved payloads
