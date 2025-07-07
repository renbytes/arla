# ARLA: A Modular Multi-Agent Simulation Platform

Affective Reinforcement Learning Architecture (ARLA) is a Python-based monorepo for creating sophisticated multi-agent simulations, with a focus on cognitive architectures. It uses a clean, decoupled Entity-Component-System (ECS) architecture, making it highly modular and extensible for research and development.

## Core Concepts

The project is organized as a monorepo containing several distinct but interconnected Python packages. Understanding the role of each is key to working with ARLA.

**[agent-core](./agent-core)**: Contains the foundational data structures, component interfaces, and abstract base classes that all other parts of the system rely on.

**[agent-engine](./agent-engine)**: The world-agnostic simulation engine. It orchestrates the main simulation loop, manages systems, and is responsible for the core cognitive processing, but knows nothing about specific game rules.

**[agent-concurrent](./agent-concurrent)**: A small, powerful library providing runners for executing simulation systems in parallel.

**[agent-persist](./agent-persist)**: A library for serializing and deserializing the simulation state, allowing you to save and load snapshots.

**[agent-sim](./agent-sim)**: A concrete implementation of a simulation. This package defines the world-specific rules, components (e.g., PositionComponent), and systems (e.g., CombatSystem) to create a runnable simulation.

## Getting Started

Follow these steps to get the ARLA project running on your local machine.

### 1. Prerequisites

- Python 3.11+
- pip and venv (usually included with Python)
- Git

### 2. Clone the Repository

First, clone the ARLA repository to your local machine:

```bash
git clone https://github.com/renbytes/arla.git
cd arla
```

### 3. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

Run this:
```bash
conda create --name arla python=3.11 -y
conda activate arla
```

### 4. Install All Packages

The monorepo contains multiple installable packages. The following command uses pip's "editable" (-e) mode to install them all. This creates links to your source code, so any changes you make are immediately available without reinstalling.

Run this command from the root `arla/` directory:

```bash
pip install -e ./agent-core -e ./agent-concurrent -e ./agent-engine -e ./agent-persist -e ./agent-sim
```

This command automatically finds the `pyproject.toml` in each subdirectory and installs it, along with all of its dependencies.

## Running the Simulation

The main entry point for running a simulation is located in the agent-sim package. The project uses Hydra for configuration management, allowing you to easily override settings from the command line.

To run the default simulation scenario:

```bash
python agent-sim/src/main.py scenario_path=agent-sim/src/simulations/soul_sim/scenarios/default.json
```

You can override any parameter defined in the config.yaml files. For example, to run the simulation for only 50 steps:

```bash
python agent-sim/src/main.py scenario_path=agent-sim/src/simulations/soul_sim/scenarios/default.json simulation.steps=50
```

## Running Tests

The project uses pytest for testing. After setting up your environment, you can run the entire test suite from the root arla/ directory:

```bash
pytest
```

This command will automatically discover and run all tests across all sub-repos.

## Code Quality (Pre-Commit)

The repository is configured with pre-commit hooks to automatically lint and format code, ensuring consistency.

To enable these hooks for your own commits, install pre-commit and set it up:

```bash
pre-commit install
```

> Note: This only needs to be run once after cloning the repo

Now, the ruff and mypy checks will run automatically on your staged files every time you git commit.

## Project Structure

```
arla/
├── .github/                 # GitHub Actions CI workflows
├── .pre-commit-config.yaml  # Pre-commit hook definitions
├── agent-concurrent/        # Concurrent execution library
│   ├── pyproject.toml
│   └── src/
├── agent-core/              # Core interfaces and components
│   ├── pyproject.toml
│   └── src/
├── agent-engine/            # World-agnostic simulation engine
│   ├── pyproject.toml
│   └── src/
├── agent-persist/           # State persistence library
│   ├── pyproject.toml
│   └── src/
├── agent-sim/               # Example simulation implementation
│   ├── pyproject.toml
│   └── src/
├── pyproject.toml           # Root mypy configuration
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request. (A formal CONTRIBUTING.md guide will be added soon)
