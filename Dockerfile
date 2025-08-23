# Use an official Python base image
FROM python:3.11.9-slim

# Configure environment variables for Python and Poetry
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_CREATE=false

# Add Poetry's bin directory to the system PATH.
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set the working directory
WORKDIR /app

# Install system dependencies and Poetry itself
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gifsicle \
    && curl -sSL https://install.python-poetry.org | python - \
    && apt-get remove -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy ALL project files before installing dependencies.
# This ensures Poetry can find the local path dependencies (agent-core, etc.).
COPY . .

# Install build dependencies, install Python packages, then remove build dependencies.
# This ensures packages that need compilation (like ecos, osqp) can be built,
# while keeping the final image size smaller.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git graphviz libgraphviz-dev \
    && poetry install --without dev \
    && apt-get purge -y --auto-remove build-essential cmake git graphviz libgraphviz-dev \
    && rm -rf /var/lib/apt/lists/*

# The default command to run when the container starts
CMD ["tail", "-f", "/dev/null"]
