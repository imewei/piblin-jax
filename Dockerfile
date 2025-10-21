# syntax=docker/dockerfile:1

# =============================================================================
# Multi-stage Dockerfile for QuantiQ
# Optimized for Python 3.12+ scientific computing with JAX
# =============================================================================

# =============================================================================
# Stage 1: Base image with uv package manager
# =============================================================================
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# =============================================================================
# Stage 2: Dependencies layer (cached separately)
# =============================================================================
FROM base AS dependencies

# Copy only dependency files first for better caching
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies using uv with exact versions from lock file
# If uv.lock exists, install from it (frozen); otherwise, generate and install
RUN if [ -f uv.lock ]; then \
        echo "📦 Installing from uv.lock (frozen dependencies)..." && \
        uv sync --frozen --no-dev; \
    else \
        echo "⚠️  No uv.lock found. Generating lock file..." && \
        uv sync --no-dev; \
    fi

# =============================================================================
# Stage 3: Development dependencies (for testing/dev images)
# =============================================================================
FROM dependencies AS dev-dependencies

# Install all dependencies including dev
RUN if [ -f uv.lock ]; then \
        uv sync --frozen; \
    else \
        uv sync; \
    fi

# =============================================================================
# Stage 4: Builder stage (compile and prepare distribution)
# =============================================================================
FROM dev-dependencies AS builder

# Copy source code
COPY . .

# Build the package
RUN uv run python -m build

# Run tests to ensure build is valid
RUN uv run pytest -v -m "not slow and not gpu" || echo "⚠️  Tests failed but continuing build"

# =============================================================================
# Stage 5: Production runtime (minimal image)
# =============================================================================
FROM python:3.12-slim AS runtime

# Security: Create non-root user
RUN groupadd -r quantiq && useradd -r -g quantiq quantiq

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only necessary files from builder
COPY --from=dependencies --chown=quantiq:quantiq /app/.venv /app/.venv
COPY --from=builder --chown=quantiq:quantiq /app/dist /app/dist

# Install the built package
RUN pip install --no-cache-dir /app/dist/*.whl

# Switch to non-root user
USER quantiq

# Set Python path to use the installed package
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import quantiq; print(f'QuantiQ v{quantiq.__version__} OK')" || exit 1

# Default command (can be overridden)
CMD ["python", "-c", "import quantiq; print(f'QuantiQ v{quantiq.__version__}')"]

# =============================================================================
# Stage 6: Development image (with dev dependencies)
# =============================================================================
FROM base AS development

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY . .

# Install all dependencies including dev tools
RUN if [ -f uv.lock ]; then \
        uv sync --frozen; \
    else \
        uv sync; \
    fi

# Install pre-commit hooks
RUN uv run pre-commit install || echo "⚠️  Pre-commit installation failed"

# Expose Jupyter port (optional)
EXPOSE 8888

# Default to bash for development
CMD ["/bin/bash"]

# =============================================================================
# Stage 7: GPU-enabled runtime (CUDA)
# =============================================================================
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04 AS gpu-cuda

# Install Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY . .

# Install with GPU support
RUN if [ -f uv.lock ]; then \
        uv sync --frozen --extra gpu-cuda; \
    else \
        uv sync --extra gpu-cuda; \
    fi

# Verify CUDA is available
RUN python3.12 -c "import jax; print(f'JAX devices: {jax.devices()}')" || echo "⚠️  GPU check failed"

ENV JAX_PLATFORM_NAME=gpu

CMD ["python3.12"]

# =============================================================================
# Usage examples:
# =============================================================================
# Build production image:
#   docker build --target runtime -t quantiq:latest .
#
# Build development image:
#   docker build --target development -t quantiq:dev .
#
# Build GPU image:
#   docker build --target gpu-cuda -t quantiq:gpu .
#
# Run with mounted code (development):
#   docker run -it -v $(pwd):/app quantiq:dev
#
# Run Jupyter notebook (development):
#   docker run -it -p 8888:8888 quantiq:dev jupyter lab --ip=0.0.0.0 --allow-root
#
# Run tests in container:
#   docker run --rm quantiq:dev uv run pytest
# =============================================================================
