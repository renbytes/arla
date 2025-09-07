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

# Step 1: Install system dependencies and Poetry
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       graphviz \
       graphviz-dev \
       libgraphviz-dev \
       pkg-config \
       gifsicle \
       curl \
    && curl -sSL https://install.python-poetry.org | python - \
    && rm -rf /var/lib/apt/lists/*

# Step 2: Copy project files
COPY . .

# Step 3: Install Python dependencies
RUN poetry install --without dev

# Clean up build dependencies (optional - only if you want to reduce image size)
# RUN apt-get remove -y build-essential curl \
#     && apt-get autoremove -y

# The default command to run when the container starts
CMD ["tail", "-f", "/dev/null"]
