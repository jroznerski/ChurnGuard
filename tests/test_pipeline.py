"""Tests for data ingestion and preprocessing pipeline."""

import io
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.generate_data import generate_customers
from src.pipeline.data_ingestion import DataIngestionPipeline, DataIngestionError
from src.pipeline.preprocessing import FeatureEngineer, build_preprocessor


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    return generate_customers(n=500, seed=0)


class TestDataGeneration:
    def test_row_count(self, sample_df):
        assert len(sample_df) == 500

    def test_required_columns(self, sample_df):
        expected = {"customer_id", "churn", "monthly_charges", "contract", "tenure"}
        assert expected.issubset(set(sample_df.columns))

    def test_churn_binary(self, sample_df):
        assert set(sample_df["churn"].unique()).issubset({0, 1})

    def test_churn_rate_reasonable(self, sample_df):
        rate = sample_df["churn"].mean()
        assert 0.10 <= rate <= 0.45, f"Unexpected churn rate: {rate:.2%}"

    def test_monthly_charges_range(self, sample_df):
        assert sample_df["monthly_charges"].between(0, 500).all()

    def test_tenure_range(self, sample_df):
        assert sample_df["tenure"].between(0, 72).all()


class TestFeatureEngineer:
    def test_adds_tenure_group(self, sample_df):
        eng = FeatureEngineer()
        out = eng.fit_transform(sample_df.drop(columns=["churn", "customer_id"]))
        assert "tenure_group" in out.columns

    def test_adds_service_count(self, sample_df):
        eng = FeatureEngineer()
        out = eng.fit_transform(sample_df.drop(columns=["churn", "customer_id"]))
        assert "service_count" in out.columns
        assert out["service_count"].between(0, 7).all()

    def test_adds_premium_services(self, sample_df):
        eng = FeatureEngineer()
        out = eng.fit_transform(sample_df.drop(columns=["churn", "customer_id"]))
        assert "has_premium_services" in out.columns
        assert set(out["has_premium_services"].unique()).issubset({0, 1})


class TestPreprocessor:
    def test_output_is_array(self, sample_df):
        from sklearn.pipeline import Pipeline

        X = sample_df.drop(columns=["churn", "customer_id"])
        pipe = Pipeline([
            ("fe", FeatureEngineer()),
            ("pre", build_preprocessor()),
        ])
        out = pipe.fit_transform(X)
        assert out.shape[0] == len(sample_df)
        assert out.shape[1] > 0

    def test_no_nan_after_preprocessing(self, sample_df):
        from sklearn.pipeline import Pipeline

        X = sample_df.drop(columns=["churn", "customer_id"])
        pipe = Pipeline([
            ("fe", FeatureEngineer()),
            ("pre", build_preprocessor()),
        ])
        out = pipe.fit_transform(X)
        assert not np.isnan(out).any(), "NaN values found after preprocessing"
