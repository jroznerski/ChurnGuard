"""Entry-point script: generate data (if needed) → run pipeline → train model."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse

import pandas as pd
import yaml
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline

from scripts.generate_data import generate_customers
from src.pipeline.data_ingestion import DataIngestionPipeline
from src.pipeline.model_trainer import ModelTrainer
from src.pipeline.preprocessing import FeatureEngineer, build_preprocessor


def _run_baseline(df: pd.DataFrame, threshold: float = 0.45, random_state: int = 42) -> None:
    """Logistic regression baseline — reference point for XGBoost comparison."""
    target = "churn"
    x = df.drop(columns=[c for c in [target, "customer_id"] if c in df.columns])
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )
    pipe = SkPipeline([
        ("fe", FeatureEngineer()),
        ("pre", build_preprocessor()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
    ])
    pipe.fit(x_train, y_train)
    proba = pipe.predict_proba(x_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, y_pred)
    logger.info(f"Baseline LR  — AUC: {auc:.4f} | F1: {f1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ChurnGuard model")
    _cfg_path = "config/config.yaml"
    _default_threshold = 0.45
    if os.path.exists(_cfg_path):
        with open(_cfg_path) as _f:
            _cfg = yaml.safe_load(_f)
        _default_threshold = _cfg.get("model", {}).get("threshold", 0.45)

    parser.add_argument("--data", default="data/raw/customers.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--threshold", type=float, default=_default_threshold)
    parser.add_argument("--tune", action="store_true", help="Run RandomizedSearchCV hyperparameter tuning")
    parser.add_argument("--generate", action="store_true", help="Regenerate synthetic data")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/training.log", rotation="10 MB", level="INFO")

    if args.generate or not os.path.exists(args.data):
        logger.info("Generating synthetic customer data …")
        os.makedirs(os.path.dirname(args.data), exist_ok=True)
        df = generate_customers()
        df.to_csv(args.data, index=False)
        logger.info(f"Data saved → {args.data}")

    ingestion = DataIngestionPipeline(args.data)
    df = ingestion.load()

    _run_baseline(df, threshold=args.threshold)
    trainer = ModelTrainer(model_dir=args.model_dir, threshold=args.threshold)
    metadata = trainer.train(df, tune=args.tune)

    logger.success("Training complete.")
    logger.info(f"AUC-ROC : {metadata['metrics']['auc_roc']}")
    logger.info(f"F1 Score: {metadata['metrics']['f1_score']}")
    logger.info(f"CV AUC  : {metadata['metrics']['cv_auc_mean']} ± {metadata['metrics']['cv_auc_std']}")


if __name__ == "__main__":
    main()
