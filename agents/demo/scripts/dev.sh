#!/bin/bash
# Development server with uv

set -e

echo "🚀 Starting development server with uv..."
uv run python main.py --bot-type flow
