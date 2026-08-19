"""Preprocessing — missing value imputation, encoding, scaling via sklearn Pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

CATEGORICAL_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
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
]

NUMERICAL_FEATURES = ["tenure", "monthly_charges", "total_charges"]

ENGINEERED_FEATURES = [
    "tenure_group",
    "charges_per_month_ratio",
    "service_count",
    "has_premium_services",
]


NUMERICAL_MISSING_COLS = ["tenure", "monthly_charges", "total_charges"]
MISSING_FLAG_FEATURES = [f"{c}_missing" for c in NUMERICAL_MISSING_COLS]


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Adds binary missing-indicator columns then median-imputes numericals.

    Must run before FeatureEngineer so derived features (charges_per_month_ratio,
    tenure_group) operate on already-imputed values without ad-hoc fillna().
    """

    def fit(self, x: pd.DataFrame, y=None) -> "MissingValueHandler":
        self.medians_ = {c: x[c].median() for c in NUMERICAL_MISSING_COLS if c in x.columns}
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()
        for col in NUMERICAL_MISSING_COLS:
            if col in df.columns:
                df[f"{col}_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(self.medians_.get(col, 0))
        return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain-derived features before encoding."""

    def fit(self, x: pd.DataFrame, y=None) -> "FeatureEngineer":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = x.copy()

        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"],
        ).astype(str)

        safe_tenure = df["tenure"].replace(0, 1)
        df["charges_per_month_ratio"] = (
            df["total_charges"] / safe_tenure
        ).round(4)

        service_cols = [
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
            "multiple_lines",
        ]
        df["service_count"] = sum(
            (df[col] == "Yes").astype(int) for col in service_cols
        )

        df["has_premium_services"] = (
            (df["online_security"] == "Yes") | (df["tech_support"] == "Yes")
        ).astype(int)

        return df


def build_preprocessor() -> ColumnTransformer:
    all_cat = CATEGORICAL_FEATURES + ["tenure_group"]
    all_num = NUMERICAL_FEATURES + [
        "charges_per_month_ratio",
        "service_count",
        "has_premium_services",
    ] + MISSING_FLAG_FEATURES

    cat_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    num_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", MinMaxScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, all_cat),
            ("num", num_pipeline, all_num),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    cat_names = preprocessor.transformers_[0][2]
    num_names = preprocessor.transformers_[1][2]
    return list(cat_names) + list(num_names)
