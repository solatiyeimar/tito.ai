#!/bin/bash
# Code quality checks with uv

set -e

echo "🔍 Running code quality checks..."

echo "📝 Formatting with black..."
uv run black .

echo "📋 Sorting imports with isort..."
uv run isort .

echo "🔎 Linting with flake8..."
uv run flake8 .

echo "🏷️  Type checking with mypy..."
uv run mypy .

echo "✅ All checks passed!"
