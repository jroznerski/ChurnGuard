"""Prediction endpoints — single and batch inference."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from src.api.schemas.customer import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    PredictionResponse,
)
from src.models.predictor import ModelNotLoadedError, predictor

router = APIRouter(prefix="/predictions", tags=["Predictions"])


def _risk_explanation(risk: str, proba: float) -> str:
    explanations = {
        "High": (
            f"This customer has a {proba * 100:.1f}% churn probability — HIGH RISK. "
            "Immediate retention action recommended: consider a loyalty discount or personal outreach."
        ),
        "Medium": (
            f"This customer has a {proba * 100:.1f}% churn probability — MEDIUM RISK. "
            "Monitor engagement and offer a service upgrade."
        ),
        "Low": (
            f"This customer has a {proba * 100:.1f}% churn probability — LOW RISK. "
            "Standard retention program is sufficient."
        ),
    }
    return explanations.get(risk, "")


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict churn for a single customer",
)
async def predict_single(customer: CustomerFeatures) -> PredictionResponse:
    try:
        result = predictor.predict_single(customer.model_dump())
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")

    return PredictionResponse(
        **result,
        explanation=_risk_explanation(result["risk_level"], result["churn_probability"]),
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict churn for up to 1 000 customers",
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    t0 = time.perf_counter()
    records = [c.model_dump() for c in request.customers]
    try:
        results = predictor.predict_batch(records)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error(f"Batch prediction error: {exc}")
        raise HTTPException(status_code=500, detail="Internal prediction error.")

    high = sum(1 for r in results if r["risk_level"] == "High")
    medium = sum(1 for r in results if r["risk_level"] == "Medium")
    low = sum(1 for r in results if r["risk_level"] == "Low")

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(
        f"Batch inference: {len(results)} records in {elapsed}s | "
        f"High={high} Medium={medium} Low={low}"
    )

    return BatchPredictionResponse(
        total=len(results),
        predictions=results,
        summary={
            "high_risk": high,
            "medium_risk": medium,
            "low_risk": low,
            "avg_churn_probability": round(
                sum(r["churn_probability"] for r in results) / len(results), 4
            ),
            "inference_time_seconds": elapsed,
        },
    )
