.PHONY: help install install-dev test test-cov lint format clean build publish pre-commit

.DEFAULT_GOAL := help

PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := pytest
BLACK := black
RUFF := ruff
ISORT := isort
MYPY := mypy

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in production mode
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install

test: ## Run tests
	$(PYTEST) tests/ -v

test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ --cov=deepfake_detector --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

test-fast: ## Run tests in parallel (fast)
	$(PYTEST) tests/ -n auto

lint: ## Run all linters
	@echo "Running ruff..."
	$(RUFF) check deepfake_detector scripts tests
	@echo "Running black check..."
	$(BLACK) --check deepfake_detector scripts tests
	@echo "Running isort check..."
	$(ISORT) --check-only deepfake_detector scripts tests
	@echo "Running mypy..."
	$(MYPY) deepfake_detector --ignore-missing-imports || true

format: ## Format code with black, ruff, and isort
	@echo "Running ruff fix..."
	$(RUFF) check --fix deepfake_detector scripts tests
	@echo "Running black..."
	$(BLACK) deepfake_detector scripts tests
	@echo "Running isort..."
	$(ISORT) deepfake_detector scripts tests

clean: ## Clean build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

build: clean ## Build distribution packages
	$(PYTHON) -m build

publish: build ## Publish to PyPI (requires credentials)
	$(PYTHON) -m twine upload dist/*

publish-test: build ## Publish to TestPyPI
	$(PYTHON) -m twine upload --repository testpypi dist/*

pre-commit: ## Run pre-commit on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

docs: ## Build documentation
	@echo "Documentation build not yet configured"

serve-docs: ## Serve documentation locally
	@echo "Documentation serve not yet configured"

# Development helpers
watch-test: ## Watch for changes and run tests
	$(PYTEST) tests/ --testmon --looponfail

check: lint test ## Run all checks (lint + test)

ci: lint test-cov ## Run CI checks locally

# Training helpers
train-example: ## Run example training
	python scripts/train.py --help

test-model: ## Run example testing
	python scripts/test.py --help

inference-example: ## Run example inference
	python scripts/inference.py --help

extract-faces: ## Run face extraction
	python scripts/extract_faces.py --help

# Version management
version: ## Show current version
	@$(PYTHON) -c "from deepfake_detector import __version__; print(__version__)"

bump-patch: ## Bump patch version
	@echo "Not implemented - manually edit deepfake_detector/__init__.py"

bump-minor: ## Bump minor version
	@echo "Not implemented - manually edit deepfake_detector/__init__.py"

bump-major: ## Bump major version
	@echo "Not implemented - manually edit deepfake_detector/__init__.py"
