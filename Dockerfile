# Use an official Python base image
FROM python:3.11.9-slim

# Configure environment variables for Python, Poetry, and Rust
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV CARGO_HOME="/opt/cargo"
ENV RUSTUP_HOME="/opt/rustup"
ENV PATH="$CARGO_HOME/bin:$POETRY_HOME/bin:$PATH"

# Set the working directory
WORKDIR /app

# Install all system dependencies, including the Rust toolchain and build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential cmake git graphviz libgraphviz-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain stable -y \
    && curl -sSL https://install.python-poetry.org | python - \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project context
COPY . .

# Install all Python dependencies, INCLUDING dev dependencies
# The --without dev flag has been removed to ensure maturin is installed.
RUN poetry install

# The default command to run when the container starts
CMD ["tail", "-f", "/dev/null"]