# Build stage: install dependencies using uv
FROM python:3.11-slim-bullseye AS builder

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/
ENV PATH="/uv/bin:$PATH"

LABEL maintainer="Tito AI Team"
LABEL description="Tito AI Backend - Voice AI Assistant (Build Stage)"

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files only for caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev


# Runtime stage: final image
FROM python:3.11-slim-bullseye AS runtime

LABEL maintainer="Tito AI Team"
LABEL description="Tito AI Backend - Voice AI Assistant"

WORKDIR /app

# Set recommended Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HOST=0.0.0.0 \
    FAST_API_PORT=7860 \
    PYTHONPATH="/app"

# Create a non-root user for security with home directory
RUN groupadd -r tito && useradd -r -m -g tito tito && \
    chown tito:tito /app

# Copy the virtual environment and application code from the builder stage
COPY --from=builder --chown=tito:tito /app/.venv /app/.venv
COPY --chown=tito:tito . /app

# Switch to the non-root user
USER tito

# Expose the FastAPI server port
EXPOSE 7860

# Basic healthcheck to ensure the server is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the FastAPI server
CMD ["python", "main.py"]
 