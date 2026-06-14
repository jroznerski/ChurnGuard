.PHONY: install data train api dashboard test docker-build docker-up clean help

PYTHON := python
PIP    := pip

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install Python dependencies
	$(PIP) install -r requirements.txt

data:  ## Generate synthetic customer dataset
	$(PYTHON) scripts/generate_data.py

train:  ## Train the XGBoost churn model (generates data if missing)
	$(PYTHON) scripts/train_model.py --generate

api:  ## Start the FastAPI server (dev mode with auto-reload)
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:  ## Launch the Streamlit dashboard
	streamlit run app/dashboard.py

test:  ## Run pytest test suite
	pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=term-missing

docker-build:  ## Build Docker images
	docker compose build

docker-up:  ## Start all services (trainer → api → dashboard)
	docker compose up

docker-down:  ## Stop all services
	docker compose down

clean:  ## Remove generated data, models, logs
	rm -rf data/raw/*.csv data/processed/*.parquet data/processed/*.csv \
		models/*.joblib models/*.json logs/*.log
