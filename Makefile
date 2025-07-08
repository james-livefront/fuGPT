.PHONY: help setup install format lint test dev clean docs

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Set up development environment
	@./scripts/setup.sh

install:  ## Install dependencies
	@uv sync --extra dev

format:  ## Auto-format code
	@./scripts/format.sh

lint:  ## Run linting checks
	@./scripts/lint.sh

test:  ## Run tests
	@./scripts/test.sh

dev:  ## Run complete development workflow
	@./scripts/dev.sh

clean:  ## Clean up build artifacts
	@echo "🧹 Cleaning up..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .ruff_cache/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@rm -rf .tox/
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@echo "✅ Cleanup complete!"

docs:  ## Build documentation
	@echo "📚 Building documentation..."
	@mkdir -p docs
	@uv run sphinx-quickstart -q -p "fuGPT" -a "James Fishwick" -v "1.0.0" --ext-autodoc --ext-viewcode --makefile --no-batchfile docs
	@echo "✅ Documentation setup complete!"

run-vectorize:  ## Run vectorization script
	@uv run python vectorize-repo.py

run-analyze:  ## Run analysis script
	@uv run python analyze-repo.py

# Development shortcuts
pre-commit:  ## Run pre-commit hooks
	@uv run pre-commit run --all-files

update-deps:  ## Update dependencies
	@uv sync --upgrade

security-check:  ## Run security checks
	@uv run bandit -r . -f json
	@uv run safety check
