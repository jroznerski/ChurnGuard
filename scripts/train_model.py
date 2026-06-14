"""Entry-point script: generate data (if needed) → run pipeline → train model."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse

from loguru import logger

from scripts.generate_data import generate_customers
from src.pipeline.data_ingestion import DataIngestionPipeline
from src.pipeline.model_trainer import ModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ChurnGuard model")
    parser.add_argument("--data", default="data/raw/customers.csv")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--threshold", type=float, default=0.45)
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

    trainer = ModelTrainer(model_dir=args.model_dir, threshold=args.threshold)
    metadata = trainer.train(df)

    logger.success("Training complete.")
    logger.info(f"AUC-ROC : {metadata['metrics']['auc_roc']}")
    logger.info(f"F1 Score: {metadata['metrics']['f1_score']}")
    logger.info(f"CV AUC  : {metadata['metrics']['cv_auc_mean']} ± {metadata['metrics']['cv_auc_std']}")


if __name__ == "__main__":
    main()
