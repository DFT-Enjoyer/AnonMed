from __future__ import annotations

from dataclasses import asdict
from typing import Any

from anonmed.preprocessing.asr import ASRNormalizationPipeline, IntegerExtractor, PunctuationRemover


def create_app() -> Any:
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as exc:
        message: str = "Install API dependencies with `pip install anonmed[api]`."
        raise RuntimeError(message) from exc

    class ParseRequest(BaseModel):  # type: ignore[no-redef]
        text: str
        replace: bool = False
        include_candidates: bool = True

    class PunctuationCleanRequest(BaseModel):  # type: ignore[no-redef]
        text: str
        include_spans: bool = True

    class RunRequest(BaseModel):  # type: ignore[no-redef]
        text: str
        include_spans: bool = True
        remove_punctuation: bool = True
        deduplicate_repetitions: bool = False

    app = FastAPI(title="AnonMed Preprocessing API", version="0.2.0")
    extractor = IntegerExtractor()
    punctuation_remover = PunctuationRemover()

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

    def punctuation_clean(request: PunctuationCleanRequest) -> dict[str, Any]:
        result = punctuation_remover.clean(request.text)
        response: dict[str, Any] = {
            "original_text": result.original_text,
            "cleaned_text": result.text,
        }
        if request.include_spans:
            response["removed_spans"] = [asdict(span) for span in result.removed_spans]
            response["protected_spans"] = [asdict(span) for span in result.protected_spans]
        return response

    def run(request: RunRequest) -> dict[str, Any]:
        pipeline = ASRNormalizationPipeline(
            remove_punctuation=request.remove_punctuation,
            deduplicate_repetitions=request.deduplicate_repetitions,
        )
        result = pipeline.run(request.text)
        response: dict[str, Any] = {
            "original_text": result.original_text,
            "disfluency_cleaned_text": result.disfluency_cleaned_text,
            "punctuation_cleaned_text": result.punctuation_cleaned_text,
            "repetition_cleaned_text": result.repetition_cleaned_text,
            "cleaned_text": result.cleaned_text,
            "normalized_text": result.normalized_text,
        }
        if request.include_spans:
            response["removed_spans"] = [asdict(span) for span in result.removed_spans]
            response["punctuation_removed_spans"] = [
                asdict(span) for span in result.punctuation_removed_spans
            ]
            response["punctuation_protected_spans"] = [
                asdict(span) for span in result.punctuation_protected_spans
            ]
            response["repetition_suppressed_indexes"] = list(
                result.repetition_suppressed_indexes
            )
            response["integer_spans"] = [asdict(span) for span in result.integer_spans]
        return response

    app.post("/v1/asr-integer/parse")(parse)
    app.post("/v1/preprocessing/asr/parse")(parse)
    app.post("/v1/asr-integer/punctuation-clean")(punctuation_clean)
    app.post("/v1/preprocessing/asr/punctuation-clean")(punctuation_clean)
    app.post("/v1/asr-integer/run")(run)
    app.post("/v1/preprocessing/asr/run")(run)

    return app


__all__: list[str] = ["create_app"]
