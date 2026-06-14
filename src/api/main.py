"""FastAPI application — ChurnGuard Customer Churn Intelligence Platform."""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.routes import analytics, health, predictions
from src.models.predictor import predictor

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.joblib")
METADATA_PATH = os.getenv("METADATA_PATH", "models/metadata.json")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
APP_ENV = os.getenv("APP_ENV", "development")

os.makedirs("logs", exist_ok=True)
logger.add("logs/api.log", rotation="50 MB", retention="30 days", level=LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting ChurnGuard API | env={APP_ENV}")
    try:
        predictor.load(model_path=MODEL_PATH, metadata_path=METADATA_PATH)
        logger.success("Model loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning(f"Model not found on startup: {exc}")
        logger.warning("Prediction endpoints will return 503 until the model is trained.")
    yield
    logger.info("ChurnGuard API shutting down.")


app = FastAPI(
    title="ChurnGuard API",
    description=(
        "## Customer Churn Intelligence Platform\n\n"
        "Real-time ML-powered churn prediction with:\n"
        "- **Single & batch inference** endpoints\n"
        "- **Statistical hypothesis testing** results\n"
        "- **Model performance metrics** and SHAP feature importances\n\n"
        "Train the model first: `python scripts/train_model.py --generate`"
    ),
    version="1.0.0",
    contact={"name": "ChurnGuard Team", "email": "support@churnguard.ai"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_and_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed}ms"
    logger.debug(
        f"[{request_id}] {request.method} {request.url.path} → {response.status_code} ({elapsed}ms)"
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."},
    )


PREFIX = "/api/v1"
app.include_router(health.router, prefix=PREFIX)
app.include_router(predictions.router, prefix=PREFIX)
app.include_router(analytics.router, prefix=PREFIX)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "ChurnGuard API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
