# Contributor's Guide: Development Setup

Welcome, and thank you for your interest in contributing to the ARLA framework! This guide will walk you through setting up a complete local development environment.

The entire ARLA ecosystem, including the database, MLflow server, and application workers, is managed via Docker Compose to ensure a consistent and reproducible setup.

## Prerequisites

Before you start, make sure you have the following tools installed on your system:

- **Git**: For cloning the repository and managing versions.
- **Docker & Docker Compose**: For running the containerized development environment.
- **An OpenAI API Key**: Required for running the cognitive systems that rely on LLMs.

## Step 1: Fork and Clone the Repository

First, fork the official ARLA repository on GitHub, and then clone your fork to your local machine.

```bash
git clone https://github.com/renbytes/arla.git
cd arla
```

## Step 2: Configure Your Local Environment

The project uses a `.env` file to manage environment variables for API keys and service connections.

1. **Create the `.env` file**: The `Makefile` includes a command to do this for you.

```bash
make setup
```

This copies the `.env.example` file to a new `.env` file.

2. **Add your OpenAI API Key**: Open the newly created `.env` file and add your secret key from OpenAI.

```bash
# in .env
OPENAI_API_KEY=sk-YourSecretKeyGoesHere
```

## Step 3: Build and Start the Services

The `Makefile` provides a simple command to build the Docker images and start all the services.

```bash
make up
```

This command does the following:

- Builds the Docker images for the `app` and `worker` services, installing all Python dependencies with Poetry.
- Starts the PostgreSQL database, Redis message broker, and MLflow tracking server.
- Creates a shared volume to cache the Python dependencies, making subsequent builds much faster.

The first time you run this command, it may take several minutes.

## Step 4: Verify Your Setup

Once the `make up` command has finished, you can verify that all services are running correctly.

1. **Check Container Status**:

```bash
docker compose ps
```

You should see all services (`app`, `worker`, `db`, `redis`, `mlflow`) running and in a `healthy` state.

2. **Run the Test Suite**: The best way to ensure your environment is set up correctly for development is to run the full test suite.

```bash
make test
```

If all tests pass, your development environment is ready to go!

You are now fully set up to start contributing to the ARLA framework. Thank you for helping to build the future of agent-based simulation!