#!/bin/bash
# Setup development environment

set -e  # Exit on any error

echo "Setting up development environment..."

echo "Installing dependencies..."
uv sync --extra dev

echo "Installing pre-commit hooks..."
uv run pre-commit install

echo "Creating necessary directories..."
mkdir -p tests
mkdir -p docs

echo "Running initial code formatting..."
uv run black .
uv run isort .

echo "Development environment setup complete!"
echo ""
echo "Available scripts:"
echo "  ./scripts/format.sh  - Auto-format code"
echo "  ./scripts/lint.sh    - Run linting checks"
echo "  ./scripts/test.sh    - Run tests"
echo "  ./scripts/dev.sh     - Complete development workflow"
echo ""
echo "To run the tool:"
echo "  uv run python vectorize-repo.py"
echo "  uv run python analyze-repo.py"
