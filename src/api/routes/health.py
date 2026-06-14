"""Health check and readiness endpoints."""

import time

from fastapi import APIRouter
from pydantic import BaseModel

from src.models.predictor import predictor

router = APIRouter(tags=["Health"])

START_TIME = time.time()


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_ready: bool
    version: str


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - START_TIME, 1),
        model_ready=predictor.is_ready,
        version="1.0.0",
    )


@router.get("/ready", summary="Readiness check (returns 503 if model not loaded)")
async def ready():
    from fastapi import HTTPException

    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ready"}
