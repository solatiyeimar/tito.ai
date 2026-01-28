#!/bin/bash
# Setup development environment with uv

set -e

echo "📦 Installing uv if not present..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo "🔧 Setting up project with uv..."
uv sync

echo "✅ Setup complete! Run './scripts/dev.sh' to start development server"
