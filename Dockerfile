# Build stage
FROM python:3.11.9-slim as builder

ENV POETRY_HOME="/opt/poetry"
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       cmake \
       git \
       graphviz-dev \
       libgraphviz-dev \
       pkg-config \
       curl \
    && curl -sSL https://install.python-poetry.org | python -

WORKDIR /app
COPY . .
RUN poetry install --without dev

# Runtime stage
FROM python:3.11.9-slim

# Install Poetry in runtime stage
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       graphviz \
       gifsicle \
       curl \
    && curl -sSL https://install.python-poetry.org | python - \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages and binaries
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy the application code and pyproject.toml/poetry.lock
COPY --from=builder /app .

CMD ["tail", "-f", "/dev/null"]
