# Makefile for mem0ai project
# Provides convenient commands for development, testing, and linting

.PHONY: help install install-dev lint format type-check security test clean docker-build docker-run docs pre-commit setup-dev

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements-dev.txt
	pip install -r requirements.txt
	pre-commit install

# Development setup
setup-dev: install-dev ## Complete development environment setup
	@echo "Setting up development environment..."
	pre-commit install --install-hooks
	@echo "Creating .secrets.baseline for detect-secrets..."
	detect-secrets scan --baseline .secrets.baseline
	@echo "Development environment setup complete!"

# Linting and formatting
lint: ## Run all linting checks
	@echo "Running Ruff linter..."
	ruff check .
	@echo "Running Ruff formatter check..."
	ruff format --check .
	@echo "Running Black formatter check..."
	black --check .
	@echo "Running isort import check..."
	isort --check-only .

lint-fix: ## Run linting with auto-fix
	@echo "Running Ruff with auto-fix..."
	ruff check . --fix
	@echo "Running Ruff formatter..."
	ruff format .
	@echo "Running Black formatter..."
	black .
	@echo "Running isort..."
	isort .

format: lint-fix ## Alias for lint-fix

# Type checking
type-check: ## Run MyPy type checking
	@echo "Running MyPy type checking..."
	mypy src/ auth/ security/ monitoring/ --ignore-missing-imports

# Security scanning
security: ## Run security scans
	@echo "Running Bandit security scan..."
	bandit -r . -f txt
	@echo "Running Safety dependency check..."
	safety check
	@echo "Detecting secrets..."
	detect-secrets scan --baseline .secrets.baseline

security-report: ## Generate detailed security reports
	@echo "Generating Bandit JSON report..."
	bandit -r . -f json -o reports/bandit-report.json
	@echo "Generating Safety JSON report..."
	safety check --json --output reports/safety-report.json
	@echo "Security reports generated in reports/ directory"

# Code quality
quality: ## Run code quality checks
	@echo "Running Vulture (dead code detection)..."
	vulture src/ auth/ security/ monitoring/ --min-confidence 80
	@echo "Running Radon (complexity analysis)..."
	radon cc src/ auth/ security/ monitoring/ -a
	radon mi src/ auth/ security/ monitoring/
	@echo "Running Xenon (complexity monitoring)..."
	xenon --max-absolute B --max-modules A --max-average A src/ auth/ security/ monitoring/

# Testing
test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov=auth --cov=security --cov=monitoring --cov-report=html --cov-report=term

test-security: ## Run security-specific tests
	pytest tests/ -v -m security

# Pre-commit
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

# Docker targets
docker-build: ## Build Docker image
	docker build -t mem0ai:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 mem0ai:latest

docker-lint: ## Lint Dockerfiles
	hadolint Dockerfile
	hadolint app/Dockerfile

# Database targets
db-init: ## Initialize database with schema
	psql -f setup_pgvector.sql
	psql -f supabase_schema.sql

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	@echo "Add documentation generation commands here"

# Cleaning
clean: ## Clean up generated files
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf reports/
	@echo "Cleanup complete!"

clean-logs: ## Clean log files
	rm -rf logs/*.log
	rm -rf logs/*.out

# Reports directory
reports-dir: ## Create reports directory
	mkdir -p reports

# All checks (CI pipeline)
ci: lint type-check security test ## Run all CI checks

# Development workflow
dev-check: lint-fix type-check security quality test ## Run all development checks with auto-fix

# Environment setup
env-check: ## Check environment and dependencies
	@echo "Python version:"
	python --version
	@echo "Pip version:"
	pip --version
	@echo "Installed packages:"
	pip list
	@echo "Pre-commit hooks:"
	pre-commit --version

# Dependency management
deps-update: ## Update dependencies
	pip-compile --upgrade requirements.in
	pip-compile --upgrade requirements-dev.in

deps-sync: ## Sync dependencies
	pip-sync requirements.txt requirements-dev.txt

# Git hooks
git-hooks: ## Install git hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

# Performance profiling
profile: ## Run performance profiling
	@echo "Add profiling commands here"

# Configuration validation
validate-config: ## Validate configuration files
	@echo "Validating YAML files..."
	yamllint .
	@echo "Validating JSON files..."
	find . -name "*.json" -exec python -m json.tool {} \; > /dev/null
	@echo "Validating TOML files..."
	python -c "import tomllib; [tomllib.load(open(f, 'rb')) for f in ['pyproject.toml']]"
	@echo "Configuration validation complete!"

# Version management
version: ## Show current version
	@python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

# Quick development commands
quick-lint: ## Quick lint (Ruff only)
	ruff check . --fix
	ruff format .

quick-test: ## Quick test (no coverage)
	pytest tests/ -x

# Help with colors
.DEFAULT_GOAL := help