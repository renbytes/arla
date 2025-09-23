# ==============================================================================
# ARLA Project Dockerfile - Optimized for Development Speed
# ==============================================================================

FROM python:3.11.9-slim

# Set environment variables for Poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_CACHE_DIR=/opt/poetry-cache \
    PATH="$POETRY_HOME/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    graphviz \
    graphviz-dev \
    libgraphviz-dev \
    pkg-config \
    curl \
    gifsicle \
    && curl -sSL https://install.python-poetry.org | python - \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create the full directory structure that Poetry expects
RUN mkdir -p \
    /app/agent-core/src/agent_core \
    /app/agent-engine/src/agent_engine \
    /app/agent-sim/src/agent_sim \
    /app/agent-concurrent/src/agent_concurrent \
    /app/agent-persist/src/agent_persist \
    /app/simulations \
    /app/data \
    /app/mlruns

# Copy all pyproject.toml files first for dependency resolution
COPY pyproject.toml poetry.lock* ./
COPY agent-core/pyproject.toml ./agent-core/pyproject.toml
COPY agent-engine/pyproject.toml ./agent-engine/pyproject.toml
COPY agent-sim/pyproject.toml ./agent-sim/pyproject.toml
COPY agent-concurrent/pyproject.toml ./agent-concurrent/pyproject.toml
COPY agent-persist/pyproject.toml ./agent-persist/pyproject.toml

# Create minimal __init__.py files so packages can be found
RUN touch /app/agent-core/src/agent_core/__init__.py && \
    touch /app/agent-engine/src/agent_engine/__init__.py && \
    touch /app/agent-sim/src/agent_sim/__init__.py && \
    touch /app/agent-concurrent/src/agent_concurrent/__init__.py && \
    touch /app/agent-persist/src/agent_persist/__init__.py

# Install all dependencies including local packages in editable mode
# This will work now because the directory structure exists
RUN $POETRY_HOME/bin/poetry install --without dev --no-root && \
    $POETRY_HOME/bin/poetry run pip install -e ./agent-core -e ./agent-engine -e ./agent-sim -e ./agent-concurrent -e ./agent-persist && \
    rm -rf $POETRY_CACHE_DIR

# Set Python path
ENV PYTHONPATH="/app/agent-core/src:/app/agent-engine/src:/app/agent-sim/src:/app/agent-concurrent/src:/app/agent-persist/src:/app/simulations"

# Default command - keep container running
CMD ["tail", "-f", "/dev/null"]
