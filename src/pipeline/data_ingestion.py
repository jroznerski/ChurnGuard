"""Data ingestion layer — loads raw CSV, validates schema, returns a clean DataFrame."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

REQUIRED_COLUMNS = {
    "customer_id",
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
    "churn",
}


class DataIngestionError(Exception):
    pass


class DataIngestionPipeline:
    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)

    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise DataIngestionError(f"Missing required columns: {missing}")

    def _validate_non_empty(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise DataIngestionError("Dataset is empty.")

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["senior_citizen"] = df["senior_citizen"].astype(int)
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").astype("Int64")
        df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce")
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
        if df["churn"].dtype == object:
            df["churn"] = df["churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
        df["churn"] = df["churn"].astype(int)
        return df

    def load(self, nrows: Optional[int] = None) -> pd.DataFrame:
        if not self.path.exists():
            raise DataIngestionError(f"Data file not found: {self.path}")

        logger.info(f"Loading data from {self.path}")
        df = pd.read_csv(self.path, nrows=nrows)

        self._validate_non_empty(df)
        self._validate_schema(df)
        df = self._cast_types(df)

        logger.info(
            f"Loaded {len(df):,} rows | churn rate: {df['churn'].mean() * 100:.1f}% "
            f"| missing: {df.isnull().sum().sum()}"
        )
        return df
