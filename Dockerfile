# Pin the Python version to match your poetry.lock file
FROM python:3.11.9-slim

# Set the working directory inside the container
WORKDIR /app

# Use modern ENV syntax
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1

# Install build tools, git, AND the graphviz development library
RUN apt-get update && apt-get install -y build-essential git graphviz-dev

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy dependency files first
COPY pyproject.toml poetry.lock ./
COPY agent-sim/pyproject.toml ./agent-sim/

# Copy the entire project source code BEFORE running install
COPY . .

# Install project dependencies using the lock file
RUN poetry install --no-root

# The command to run when the container starts
CMD ["tail", "-f", "/dev/null"]