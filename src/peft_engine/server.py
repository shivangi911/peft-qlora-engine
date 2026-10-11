"""FastAPI server streaming tokens over Server-Sent Events."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from peft_engine.config import settings

app = FastAPI(title="PEFT Engine Streaming Server")

_lock = threading.Lock()
_state: dict[str, Any] = {}


class StreamRequest(BaseModel):
    prompt: str
    system: str = "You are an expert AI assistant that thinks step-by-step."
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


def _load() -> tuple[Any, Any]:
    if "model" not in _state:
        from mlx_lm import load

        _state["model"], _state["tokenizer"] = load(settings.server_model)
    return _state["model"], _state["tokenizer"]


def generate_tokens(req: StreamRequest) -> Iterator[str]:
    """Yield text chunks for the request. Isolated so tests can replace it."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = _load()
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": req.system}, {"role": "user", "content": req.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    sampler = make_sampler(temp=req.temperature)
    with _lock:
        for resp in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=req.max_tokens, sampler=sampler
        ):
            yield resp.text


def sse_events(req: StreamRequest) -> Iterator[str]:
    try:
        for chunk in generate_tokens(req):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
    except Exception as exc:  # surface errors to the client instead of dropping the stream
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": settings.server_model, "loaded": "model" in _state}


@app.post("/v1/chat/completions/stream")
def stream(req: StreamRequest) -> StreamingResponse:
    return StreamingResponse(sse_events(req), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.server_host, port=settings.server_port)


if __name__ == "__main__":
    main()
