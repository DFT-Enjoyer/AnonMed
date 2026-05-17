from __future__ import annotations

from dataclasses import asdict
from typing import Any

from asr_integer_extractor import IntegerExtractor


def create_app() -> Any:
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as exc:
        message: str = "Install API dependencies with `pip install asr-integer-extractor[api]`."
        raise RuntimeError(message) from exc

    class ParseRequest(BaseModel):  # type: ignore[no-redef]
        text: str
        replace: bool = False
        include_candidates: bool = True

    app = FastAPI(title="ASR Integer Extractor", version="0.1.0")
    extractor = IntegerExtractor()

    @app.post("/v1/asr-integer/parse")
    def parse(request: ParseRequest) -> dict[str, Any]:
        spans = extractor.extract(request.text)
        payload: list[dict[str, Any]] = [asdict(span) for span in spans]
        if not request.include_candidates:
            for item in payload:
                item.pop("candidates", None)
        replaced: str | None = extractor.replace(request.text) if request.replace else None
        response: dict[str, Any] = {
            "items": payload,
            "replaced_text": replaced,
        }
        return response

    return app


__all__: list[str] = ["create_app"]
