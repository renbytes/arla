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

# Step 1: Install ALL necessary system libraries and keep them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       graphviz \
       libgraphviz-dev \
       gifsicle \
       curl \
    && curl -sSL https://install.python-poetry.org | python - \
    && apt-get remove -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Copy project files
COPY . .

# Step 3: Now, run poetry install. The system libraries from Step 1 are still present.
RUN poetry install --without dev

# The default command to run when the container starts
CMD ["tail", "-f", "/dev/null"]
