.DEFAULT_GOAL := help
.PHONY: help install install-train run frontend test lint format \
        prepare train evaluate export pipeline docker-build docker-run clean

PYTHON ?= python
PORT   ?= 8000
IMAGE  ?= croplens-api

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

# ---------- Setup ----------
install: ## Install API + dev dependencies
	$(PYTHON) -m pip install -r api/requirements.txt pytest ruff httpx

install-train: ## Install training dependencies (heavy)
	$(PYTHON) -m pip install -r training/requirements.txt

# ---------- Run ----------
run: ## Run the API locally with auto-reload
	cd api && uvicorn app.main:app --reload --port $(PORT)

frontend: ## Serve the frontend at http://localhost:5500
	cd frontend && $(PYTHON) -m http.server 5500

# ---------- Quality ----------
test: ## Run tests
	$(PYTHON) -m pytest

lint: ## Check code style
	ruff check .
	ruff format --check .

format: ## Auto-format and fix lint issues
	ruff check --fix .
	ruff format .

# ---------- Training pipeline ----------
prepare: ## Clean data and create train/val/test split
	$(PYTHON) training/prepare_data.py

train: ## Train the model
	$(PYTHON) training/train.py --config training/config.yaml

evaluate: ## Evaluate on the test set and save plots
	$(PYTHON) training/evaluate.py

export: ## Export model, labels and metrics to models/
	$(PYTHON) training/export.py

pipeline: prepare train evaluate export ## Run the full training pipeline

# ---------- Docker ----------
# Build context is the repo root so the image can include models/ and api/.
docker-build: ## Build the API Docker image
	docker build -f api/Dockerfile -t $(IMAGE) .

docker-run: ## Run the API container on http://localhost:8000
	docker run --rm --env-file .env -p $(PORT):8000 $(IMAGE)

# ---------- Cleanup ----------
clean: ## Remove caches and build artifacts
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
