#!/bin/bash
# Run tests with coverage

set -e  # Exit on any error

echo "Running tests..."

# Create tests directory if it doesn't exist
mkdir -p tests

# Run tests with coverage
echo "Running pytest with coverage..."
uv run pytest

echo "Coverage report generated in htmlcov/"
echo "Tests completed!"
