"""REST API v2 — с поддержкой GPU-конфигурации."""

from __future__ import annotations
import os, time, logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import (InflectRequest, InflectResponse, BatchRequest,
                      BatchResponse, ParadigmRequest, HealthResponse)
from ..service import DeclensionService

logger = logging.getLogger(__name__)
_service: DeclensionService | None = None

def get_service() -> DeclensionService:
    global _service
    if _service is None:
        _service = DeclensionService(
            cache_size=int(os.getenv("CACHE_SIZE", "100000")),
            rut5_model_path=os.getenv("RUT5_MODEL_PATH"),
            bert_validator_model=os.getenv("BERT_VALIDATOR_MODEL"),
            animacy_model_path=os.getenv("ANIMACY_MODEL_PATH"),
            ensemble_model_path=os.getenv("ENSEMBLE_MODEL_PATH"),
            gpu_device=os.getenv("GPU_DEVICE", "auto"),
        )
    return _service

def create_app() -> FastAPI:
    app = FastAPI(
        title="Russian Declension API v2",
        description="Склонение русских слов, фраз, ФИО, числительных, организаций. CPU + GPU.",
        version="0.4.0", docs_url="/docs", redoc_url="/redoc")

    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.on_event("startup")
    async def startup():
        get_service()
        logger.info("API v2 ready.")

    @app.post("/api/v1/inflect", response_model=InflectResponse,
              summary="Склонить текст", tags=["Склонение"])
    async def inflect(req: InflectRequest) -> InflectResponse:
        return InflectResponse(**get_service().inflect(
            text=req.text, target_case=req.target_case,
            entity_type=req.entity_type, target_number=req.target_number,
            gender=req.gender, context=req.context))

    @app.post("/api/v1/inflect/batch", response_model=BatchResponse,
              summary="Пакетное склонение", tags=["Склонение"])
    async def inflect_batch(req: BatchRequest) -> BatchResponse:
        t0 = time.perf_counter()
        results = get_service().inflect_batch([it.model_dump() for it in req.items])
        return BatchResponse(results=[InflectResponse(**r) for r in results],
                             total_elapsed_ms=round((time.perf_counter()-t0)*1000, 2))

    @app.post("/api/v1/paradigm", summary="Полная парадигма", tags=["Анализ"])
    async def paradigm(req: ParadigmRequest) -> dict:
        r = get_service().paradigm(req.word)
        if r is None:
            raise HTTPException(404, f"Парадигма не найдена: '{req.word}'")
        return r

    @app.get("/api/v1/health", response_model=HealthResponse,
             summary="Healthcheck", tags=["Мониторинг"])
    async def health() -> HealthResponse:
        return HealthResponse(**get_service().health())

    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": "Russian Declension API v2", "version": "0.4.0",
                "docs": "/docs", "gpu": "configured via env vars"}

    return app

app = create_app()
