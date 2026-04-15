# Vaani

Vaani is a local-first AI voice agent with a FastAPI backend and a separate
Gradio client. It processes local audio input, transcribes speech with a local
Whisper model, classifies intent with a local Ollama model, validates the
result as structured JSON, and executes approved actions through safe local
tools.

> This project currently targets a **local-only architecture**.

## What It Does

- Accepts microphone or uploaded audio in the Gradio UI
- Sends audio to `POST /process-audio` for transcription and intent analysis
- Produces structured decisions for `create_file`, `write_code`, `summarize`,
  or `chat`
- Requires explicit approval before any write-style action
- Executes tools only inside the local `output/` directory

## Current Architecture

### Backend (`app/`)

- `app/api`: FastAPI routes (`/health`, `/process-audio`, `/execute-action`)
- `app/core`: settings, logging, exceptions, intent parsing, tool routing
- `app/services`: STT service, LLM service, pipeline orchestration
- `app/tools`: file, code, and summary tools
- `app/schemas`: request and response models

### Frontend

- `gradio_app.py`: local Gradio UI for analyze, decision review, and
  approve/reject flow

## Processing Flow

1. Audio is uploaded/recorded in Gradio.
2. Backend transcribes audio with local Whisper (`transformers` pipeline).
3. Ollama returns a structured intent response.
4. Intent parser validates and normalizes JSON output.
5. If confirmation is required, UI shows decision and waits for approval.
6. Approved decisions are executed via `/execute-action`.
7. Tool result is returned and displayed in the UI.

## API Endpoints

### `GET /health`

Basic service health response.

### `POST /process-audio`

Multipart request:

- `audio` (required)
- `context_text_file` (optional)
- `context_text` (optional)

Response includes:

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

Accepts approved `decision` payload plus optional transcript/metadata and
executes the action.

## Local Tool Scope

- `create_file`: creates folders or text files inside `output/`
- `write_code`: writes code files inside `output/`
- `summarize`: summarizes transcript or provided context text
- `chat`: returns non-tool conversational output from the LLM

## Safety Guarantees

- File writes are restricted to `output/`
- Path traversal and unsafe targets are rejected
- Invalid LLM JSON is retried, then safely downgraded to chat fallback
- Write intents require explicit user approval before execution

## Setup

1. Copy environment variables:

```bash
cp .env.example .env
```

2. Install dependencies:

```bash
uv sync --extra dev
```

3. Pull and run the local Ollama model:

```bash
ollama pull qwen2.5:3b
ollama serve
```

## Run

Start backend:

```bash
uv run uvicorn main:app --reload
```

Start Gradio UI in a second terminal:

```bash
uv run python gradio_app.py
```

Defaults:

- Backend: `http://127.0.0.1:8000`
- Gradio backend target: `GRADIO_BACKEND_URL` (default
  `http://127.0.0.1:8000`)

## Configuration

Key local-first settings in `.env`:

- `STT_MODEL_ID` (default: `openai/whisper-base`)
- `STT_DEVICE` (default: `cpu`)
- `STT_CHUNK_LENGTH_SECONDS`
- `OLLAMA_BASE_URL` (default: `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` (default: `qwen2.5:3b`)
- `GRADIO_BACKEND_URL`

## Testing

Run all tests:

```bash
./.venv/bin/python -m pytest -q
```
