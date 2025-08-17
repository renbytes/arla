# ARLA: Agent-based Reinforcement Learning Architecture

This repository contains the complete source code for the ARLA project, a multi-agent simulation platform designed for studying emergent behavior and complex cognitive architectures.

## Quickstart (Clone → Run in ~60s)

This guide helps you set up the development environment and run a sample simulation in minutes.

### 1. Prerequisites
- **Python 3.11**
- **[Poetry](https://python-poetry.org/docs/#installation)**: A modern dependency management tool for Python.

### 2. Installation
Clone the repository and use `poetry install` to create a virtual environment and install all dependencies from `pyproject.toml`.

```bash
git clone git@github.com:renbytes/arla.git
cd arla
poetry install
```

This command handles everything: it creates a `.venv`, installs all dependencies, and links the local `agent-*` subpackages in editable mode.

### 3. Run a Simulation

Activate the environment with `poetry shell` or use `poetry run` to execute commands.

```bash
# Activate the virtual environment
poetry shell

# Smoke test the CLI
arla-sim --help

# Run an example simulation for 50 steps
arla-sim --scenario simulations/soul_sim/scenarios/default.json --steps 50
```

### Optional: Run with Docker Compose

If you prefer a containerized environment, a docker-compose setup is provided for a headless run.

1. Ensure Docker is running.
2. Build and run the service:

```bash
docker compose -f docker/compose.yaml up --build
```

This command will build a Python image, mount the project directory, install dependencies using Poetry, and execute the example simulation.