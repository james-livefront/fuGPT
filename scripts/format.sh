#!/bin/bash
# Auto-format code

set -e  # Exit on any error

echo "Auto-formatting code..."

echo "Running isort (import sorting)..."
uv run isort .

echo "Running black (code formatting)..."
uv run black .

echo "Running ruff (auto-fix)..."
uv run ruff check --fix .

echo "Code formatting complete!"
