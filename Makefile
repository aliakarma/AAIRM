# AAIRM Makefile
# Run `make help` to list all available targets.

.PHONY: help install install-dev lint format typecheck \
        test test-fast test-integration smoke \
        download-data preprocess-data generate-synthetic \
        run-paper-experiment run-ablation run-realworld \
        docs serve-docs docker-build docker-run clean

# ── Python interpreter ──────────────────────────────────────────────────────
PYTHON ?= python3

# ── Colours ─────────────────────────────────────────────────────────────────
BLUE  := \033[36m
RESET := \033[0m

help:  ## Show all available make targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    sort | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-30s$(RESET) %s\n", $$1, $$2}'

# ── Installation ─────────────────────────────────────────────────────────────
install:  ## Install package in editable mode (runtime deps only)
	pip install -e .

install-dev:  ## Install package + all dev dependencies + pre-commit hooks
	pip install -e ".[dev]"
	pre-commit install
	@echo "✓ Development environment ready."

# ── Code quality ─────────────────────────────────────────────────────────────
lint:  ## Run ruff linter on all Python files
	ruff check aairm/ tests/ scripts/ experiments/

format:  ## Auto-format code with black and ruff --fix
	black aairm/ tests/ scripts/ experiments/
	ruff check --fix aairm/ tests/ scripts/ experiments/

typecheck:  ## Run mypy static type checker
	mypy aairm/ --ignore-missing-imports

# ── Testing ──────────────────────────────────────────────────────────────────
test:  ## Run full test suite with coverage (unit + smoke + integration)
	$(PYTHON) -m pytest tests/ -v \
	    --cov=aairm --cov-report=term-missing --cov-report=xml \
	    --timeout=300

test-fast:  ## Run unit tests only (excludes slow, integration, llm)
	$(PYTHON) -m pytest tests/unit/ -v \
	    -m "not slow and not llm" \
	    --cov=aairm --cov-report=term-missing \
	    --timeout=120

test-integration:  ## Run integration tests (requires full env)
	$(PYTHON) -m pytest tests/integration/ -v \
	    -m "not llm" \
	    --timeout=300

smoke:  ## Run quick smoke test (10 SKUs, 7 days, < 60 seconds)
	$(PYTHON) -m pytest tests/smoke/ -v --timeout=60

# ── Data ─────────────────────────────────────────────────────────────────────
download-data:  ## Download M5, Favorita, and Instacart datasets from Kaggle
	$(PYTHON) scripts/download_datasets.py

preprocess-data:  ## Run all data preprocessing pipelines
	$(PYTHON) scripts/preprocess_all.py

generate-synthetic:  ## Generate the 1,200-SKU synthetic simulation dataset
	$(PYTHON) scripts/generate_synthetic.py
	@echo "✓ Synthetic data written to data/synthetic/"

# ── Experiments ───────────────────────────────────────────────────────────────
run-paper-experiment:  ## Reproduce all paper results (Tables 2 & 3)
	$(PYTHON) experiments/run_paper_experiment.py \
	    --config configs/simulation_1200sku.yaml

run-paper-experiment-fast:  ## Fast smoke run (10 SKUs, 30 days, ~30s)
	$(PYTHON) experiments/run_paper_experiment.py --fast --no-assert

run-ablation:  ## Run all four ablation studies
	$(PYTHON) experiments/run_ablation.py

run-realworld:  ## Evaluate on M5 + Favorita (requires download-data first)
	$(PYTHON) experiments/run_realworld.py

# ── Documentation ─────────────────────────────────────────────────────────────
docs:  ## Build MkDocs documentation site (strict mode)
	mkdocs build --strict

serve-docs:  ## Serve documentation locally at http://localhost:8000
	mkdocs serve

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:  ## Build Docker image tagged aairm:latest
	docker build -t aairm:latest .

docker-run:  ## Run paper experiment in Docker
	docker-compose up aairm-experiment

docker-test:  ## Run test suite in Docker
	docker-compose up aairm-test

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:  ## Remove build artefacts, caches, and coverage reports
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ htmlcov/ site/
	rm -f coverage.xml .coverage
	@echo "✓ Clean."
