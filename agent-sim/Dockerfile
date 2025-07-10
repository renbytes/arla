FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Copy requirements file AND all local package source code first.
# This is necessary because requirements.txt refers to the local paths,
# and pip needs them to exist before it can install them.
COPY requirements.txt .
COPY agent-concurrent/ ./agent-concurrent/
COPY agent-core/ ./agent-core/
COPY agent-engine/ ./agent-engine/
COPY agent-persist/ ./agent-persist/
COPY agent-sim/ ./agent-sim/

# 2. Now, install all dependencies in a single step.
# This layer will be rebuilt if requirements.txt OR any of the copied
# source code files change. This is a necessary trade-off for correctness
# in a monorepo setup with local editable installs.
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data logs

# Default command
CMD ["echo", "Container is running. Use docker-compose command to specify a task."]
