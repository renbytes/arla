FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. First, copy the requirements file.
COPY requirements.txt .

# 2. Next, copy all the source code for your local packages.
#    Pip needs these directories to exist to install from them.
COPY agent-concurrent/ ./agent-concurrent/
COPY agent-core/ ./agent-core/
COPY agent-engine/ ./agent-engine/
COPY agent-persist/ ./agent-persist/
COPY agent-sim/ ./agent-sim/

# 3. Now that all source files are present, run pip install.
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data logs

# This command is overridden by your docker-compose.yml `command` field,
# but it's good practice to have a default.
CMD ["echo", "Container is running. Use docker-compose command to specify a task."]
