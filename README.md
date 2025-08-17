# ARLA: Agent-based Reinforcement Learning Architecture

This repository contains the complete source code for the ARLA project, a multi-agent simulation platform designed for studying emergent behavior and complex cognitive architectures.

## Quickstart (Clone → Run in \~60s)

This guide helps you set up the development environment and run a sample simulation in minutes.

### 1\. Prerequisites

  - **Python 3.11**
  - **[Poetry](https://python-poetry.org/docs/#installation)**: A modern dependency management tool for Python.
  - **[Make](https://www.gnu.org/software/make/)**: For running helper commands from the `Makefile`.
  - **[Docker](https://www.docker.com/products/docker-desktop/)**: For the containerized workflow.

### 2\. Installation

Clone the repository and use `poetry install` to create a virtual environment and install all dependencies.

```bash
git clone git@github.com:renbytes/arla.git
cd arla
poetry install
```

This command handles everything: it creates a `.venv`, installs all dependencies, and links the local `agent-*` subpackages in editable mode.

### 3\. Run a Simulation Locally

The main entrypoint for local simulations is the `agent_sim.main` module. You can run commands within the Poetry virtual environment by activating it first with `poetry env activate`, or by prefixing each command with `poetry run`.

```bash
# Activate the virtual environment (do this once per session)
poetry env activate

# Install the packages
poetry install

# Smoke test the local runner to see available options
poetry run arla --help

# Run an example simulation for 50 steps
poetry run arla --scenario simulations/soul_sim/scenarios/default.json --steps 50
```

### 4\. Run with Docker Compose (Recommended)

The provided `Makefile` contains the simplest way to use the containerized environment.

1.  **Start Services**: Build the Docker images and start the application, database, and other services in the background.

    ```bash
    make up
    ```

2.  **Run Simulation**: Execute the example simulation inside the running `app` container.

    ```bash
    make run-example
    ```

3.  **View Logs**: You can tail the logs from all running services using:

    ```bash
    make logs
    ```

4.  **Stop Services**: When you're finished, stop and remove all containers.

    ```bash
    make down
    ```