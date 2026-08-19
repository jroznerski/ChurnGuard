"""Model training — XGBoost with cross-validation, SHAP explainability, and artifact saving."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.pipeline.preprocessing import (
    FeatureEngineer,
    MissingValueHandler,
    build_preprocessor,
    get_feature_names,
)


class ModelTrainer:
    def __init__(
        self,
        model_dir: str = "models",
        random_state: int = 42,
        test_size: float = 0.2,
        threshold: float = 0.45,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.random_state = random_state
        self.test_size = test_size
        self.threshold = threshold
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _build_pipeline(self) -> Pipeline:
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2.5,
            eval_metric="auc",
            use_label_encoder=False,
            random_state=self.random_state,
            verbosity=0,
        )
        return Pipeline(
            steps=[
                ("missing_handler", MissingValueHandler()),
                ("feature_engineer", FeatureEngineer()),
                ("preprocessor", build_preprocessor()),
                ("classifier", xgb),
            ]
        )

    def _evaluate(
        self, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, Any]:
        proba = pipeline.predict_proba(x_test)[:, 1]
        y_pred = (proba >= self.threshold).astype(int)
        auc = roc_auc_score(y_test, proba)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()
        logger.info(f"AUC: {auc:.4f} | F1: {f1:.4f} | Threshold: {self.threshold}")
        return {
            "auc_roc": round(auc, 4),
            "f1_score": round(f1, 4),
            "precision": round(report["1"]["precision"], 4),
            "recall": round(report["1"]["recall"], 4),
            "accuracy": round(report["accuracy"], 4),
            "confusion_matrix": cm,
        }

    def _cross_validate(
        self, pipeline: Pipeline, x: pd.DataFrame, y: pd.Series
    ) -> dict[str, float]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(pipeline, x, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        logger.info(f"CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
        return {"cv_auc_mean": round(scores.mean(), 4), "cv_auc_std": round(scores.std(), 4)}

    def _compute_shap(
        self, pipeline: Pipeline, x_test: pd.DataFrame, n_samples: int = 500
    ) -> dict[str, float]:
        feature_engineer = pipeline.named_steps["feature_engineer"]
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]

        x_eng = feature_engineer.transform(x_test.head(n_samples))
        x_proc = preprocessor.transform(x_eng)
        feature_names = get_feature_names(preprocessor)

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(x_proc)
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = dict(zip(feature_names, mean_abs.tolist()))
        top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
        return dict(top)

    def _tune_hyperparams(self, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """RandomizedSearchCV over XGBoost hyperparams; returns best fitted pipeline."""
        param_dist = {
            "classifier__n_estimators": [100, 200, 300, 400],
            "classifier__max_depth": [3, 4, 5, 6, 7],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.6, 0.7, 0.8, 0.9],
            "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            self._build_pipeline(),
            param_dist,
            n_iter=30,
            scoring="roc_auc",
            cv=cv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(x_train, y_train)
        logger.info(f"Tuning best CV AUC: {search.best_score_:.4f} | {search.best_params_}")
        return search.best_estimator_

    def _find_threshold(self, pipeline: Pipeline, x_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Find threshold maximising F1 on the validation split."""
        probas = pipeline.predict_proba(x_val)[:, 1]
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.1, 0.9, 81):
            f1 = f1_score(y_val, (probas >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        logger.info(f"Optimal threshold: {best_t:.2f} (val F1={best_f1:.4f})")
        return round(best_t, 2)


    def train(self, df: pd.DataFrame, tune: bool = False) -> dict[str, Any]:
        target = "churn"
        drop_cols = [target, "customer_id"]
        x = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df[target]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        x_train_sub, x_val, y_train_sub, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )

        if tune:
            logger.info("Running RandomizedSearchCV …")
            pipeline = self._tune_hyperparams(x_train, y_train)
        else:
            pipeline = self._build_pipeline()
            logger.info("Starting cross-validation …")
            cv_metrics = self._cross_validate(pipeline, x_train, y_train)
            logger.info("Training final model …")
            pipeline.fit(x_train_sub, y_train_sub)

        logger.info("Selecting threshold on validation set …")
        self.threshold = self._find_threshold(pipeline, x_val, y_val)
        pipeline.fit(x_train, y_train)

        eval_metrics = self._evaluate(pipeline, x_test, y_test)
        logger.info("Computing SHAP feature importances …")
        shap_importances = self._compute_shap(pipeline, x_test)

        metadata: dict[str, Any] = {
            "model_name": "XGBoostClassifier",
            "threshold": self.threshold,
            "train_size": len(x_train),
            "test_size": len(x_test),
            "churn_rate": round(float(y.mean()), 4),
            "metrics": {**(cv_metrics if not tune else {}), **eval_metrics},
            "shap_feature_importance": shap_importances,
        }

        self._save(pipeline, metadata)
        return metadata

    def _save(self, pipeline: Pipeline, metadata: dict) -> None:
        model_path = self.model_dir / "churn_model.joblib"
        meta_path = self.model_dir / "metadata.json"

        joblib.dump(pipeline, model_path)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.success(f"Model saved → {model_path}")
        logger.success(f"Metadata saved → {meta_path}")
