#!/bin/bash
# Development workflow script

set -e  # Exit on any error

echo "Running complete development workflow..."

echo "Step 1: Auto-formatting code..."
./scripts/format.sh

echo "Step 2: Running linting and checks..."
./scripts/lint.sh

echo "Step 3: Running tests..."
./scripts/test.sh

echo "Step 4: Running pre-commit hooks..."
uv run pre-commit run --all-files

echo "Development workflow complete! Ready to commit."
