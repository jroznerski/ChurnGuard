"""Model loader and inference wrapper — singleton pattern for production use."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger


class ModelNotLoadedError(Exception):
    pass


class ChurnPredictor:
    _instance: ChurnPredictor | None = None
    _lock = threading.Lock()

    def __new__(cls) -> "ChurnPredictor":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def load(
        self,
        model_path: str = "models/churn_model.joblib",
        metadata_path: str = "models/metadata.json",
    ) -> None:
        model_path = Path(model_path)
        metadata_path = Path(metadata_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run: python scripts/train_model.py --generate"
            )

        self._pipeline = joblib.load(model_path)
        self._threshold = 0.45

        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata = json.load(f)
            self._threshold = self._metadata.get("threshold", 0.45)
        else:
            self._metadata = {}

        self._initialized = True
        logger.info(f"Model loaded from {model_path} | threshold={self._threshold}")

    def _check_loaded(self) -> None:
        if not getattr(self, "_initialized", False):
            raise ModelNotLoadedError("Model not loaded. Call predictor.load() first.")

    def predict_single(self, features: dict[str, Any]) -> dict[str, Any]:
        self._check_loaded()
        df = pd.DataFrame([features])
        proba = self._pipeline.predict_proba(df)[0, 1]
        label = int(proba >= self._threshold)
        risk = "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
        return {
            "churn_probability": round(float(proba), 4),
            "churn_prediction": label,
            "risk_level": risk,
            "threshold_used": self._threshold,
        }

    def predict_batch(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._check_loaded()
        df = pd.DataFrame(records)
        probas = self._pipeline.predict_proba(df)[:, 1]
        results = []
        for i, proba in enumerate(probas):
            label = int(proba >= self._threshold)
            risk = "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
            results.append(
                {
                    "index": i,
                    "churn_probability": round(float(proba), 4),
                    "churn_prediction": label,
                    "risk_level": risk,
                }
            )
        return results

    @property
    def metadata(self) -> dict[str, Any]:
        self._check_loaded()
        return self._metadata

    @property
    def is_ready(self) -> bool:
        return getattr(self, "_initialized", False)


predictor = ChurnPredictor()
