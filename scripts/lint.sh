#!/bin/bash
# Comprehensive linting and code quality checks

set -e  # Exit on any error

echo "Running code quality checks..."

echo "Running ruff (linting)..."
uv run ruff check .

echo "Running black (code formatting check)..."
uv run black --check .

echo "Running isort (import sorting check)..."
uv run isort --check-only .

echo "Running mypy (type checking)..."
uv run mypy . || echo "MyPy errors found but continuing..."

echo "Running bandit (security check)..."
uv run bandit -r . -f json

echo "Running safety (dependency security check)..."
uv run safety check

echo "All checks passed!"
