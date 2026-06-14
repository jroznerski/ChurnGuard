"""Analytics endpoints — model metadata and hypothesis test results."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger

from src.analysis.hypothesis_testing import HypothesisTester
from src.models.predictor import predictor

router = APIRouter(prefix="/analytics", tags=["Analytics"])

DATA_PATH = os.getenv("DATA_RAW_PATH", "data/raw/customers.csv")
METADATA_PATH = os.getenv("METADATA_PATH", "models/metadata.json")


@router.get("/model/metrics", summary="Model performance metrics and feature importances")
async def model_metrics():
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return predictor.metadata


@router.get("/hypothesis-tests", summary="Run all statistical hypothesis tests on the dataset")
async def hypothesis_tests():
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found at {DATA_PATH}. Run the data generation script first.",
        )
    try:
        df = pd.read_csv(data_path)
        tester = HypothesisTester(df)
        results = tester.run_all()
        return {
            "alpha": 0.05,
            "n_tests": len(results),
            "n_rejected": sum(1 for r in results if r.rejected),
            "results": [r.to_dict() for r in results],
        }
    except Exception as exc:
        logger.error(f"Hypothesis testing error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/dataset/summary", summary="Descriptive statistics of the raw dataset")
async def dataset_summary():
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = pd.read_csv(data_path)
    churn_rate = df["churn"].mean() if "churn" in df.columns else None

    return {
        "total_customers": len(df),
        "churn_rate": round(float(churn_rate), 4) if churn_rate is not None else None,
        "features": len(df.columns) - 2,
        "missing_values": int(df.isnull().sum().sum()),
        "churn_distribution": df["churn"].value_counts().to_dict() if "churn" in df.columns else {},
        "contract_distribution": df["contract"].value_counts().to_dict() if "contract" in df.columns else {},
        "internet_service_distribution": (
            df["internet_service"].value_counts().to_dict()
            if "internet_service" in df.columns else {}
        ),
    }
