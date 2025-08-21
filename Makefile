# ==============================================================================
# ARLA Project Makefile
#
# Provides a simplified interface for managing the Docker-based development
# environment and running simulations.
# ==============================================================================

.PHONY: help setup up down logs init-db run run-local cli test lint make-gif
.DEFAULT_GOAL := help

# --- Variable Definitions ---
# These variables can be overridden from the command line.
# Example: make run-local PACKAGE=simulations.new_sim ...

# The Python package of the simulation to run.
PACKAGE ?= simulations.schelling_sim
# Path to the base YAML config file for the simulation.
CONFIG ?= simulations/schelling_sim/config/config.yml
# Path to the scenario JSON file or experiment YAML file.
FILE ?= simulations/schelling_sim/scenarios/default.json
# Number of simulation steps for local runs.
STEPS ?= 200
# Generic arguments for the CLI.
ARGS ?=
# Default directory for rendering output.
RENDER_DIR ?= data/gif_renders
# Default frames per second for rendering GIFs.
FPS ?= 15


# --- Core Docker Commands ---

## up: Build images and start all services in the background.
up:
	@echo "🚀 Building images and starting all services..."
	@docker compose up -d --build

## down: Stop and remove all containers, networks, and volumes.
down:
	@echo "🛑 Stopping and removing all services..."
	@docker compose down -v

## logs: View and follow live logs from all running services.
logs:
	@echo "📜 Tailing logs..."
	@docker compose logs -f

## init-db: Connects to the DB and creates all necessary tables.
init-db:
	@echo "Initializing database and creating tables..."
	@docker compose exec app poetry run python -m agent_sim.infrastructure.database.init_db

## setup: Create the .env file from the example template.
setup:
	@echo "📋 Copying .env.example to .env..."
	@cp .env.example .env
	@echo "✅ Done. Please add your OPENAI_API_KEY to the .env file."


# --- Simulation & Development Commands ---

## run: Run a full experiment, submitting jobs to the Celery queue.
run:
	@echo "▶️ Running experiment from: $(FILE)"
	@docker compose exec app poetry run agentsim run-experiment $(FILE)

## run-local: Run a single, local simulation for quick testing and debugging.
run-local:
	@echo "▶️ Running Local Simulation"
	@echo "   - Package:  $(PACKAGE)"
	@echo "   - Config:   $(CONFIG)"
	@echo "   - Scenario: $(FILE)"
	@echo "   - Steps:    $(STEPS)"
	@docker compose exec app poetry run python -m agent_sim.main \
	  --package $(PACKAGE) \
	  --config $(CONFIG) \
	  --scenario $(FILE) \
	  --steps $(STEPS)

## run-example: Run the Schelling simulation with default parameters.
run-example:
	@echo "▶️ Running Schelling Simulation Example (150 steps)..."
	@docker compose exec app poetry run python -m agent_sim.main \
	  --package "simulations.schelling_sim" \
	  --config "simulations/schelling_sim/config/config.yml" \
	  --scenario "simulations/schelling_sim/scenarios/default.json" \
	  --steps 150

## make-gif: Creates a GIF from the most recent simulation render.
make-gif:
	@echo "🎬 Creating GIF from frames in $(RENDER_DIR)..."
	@poetry run python scripts/create_gif.py $(RENDER_DIR) simulation.gif --fps $(FPS)

## cli: Run any 'agentsim' command inside the container.
cli:
	@echo "Running CLI command: agentsim $(ARGS)"
	@docker compose exec app poetry run agentsim $(ARGS)

## test: Run the full pytest suite inside the container.
test:
	@echo "🧪 Running pytest test suite..."
	@docker compose exec app pytest

## lint: Run the Ruff linter to check for code style issues.
lint:
	@echo "🎨 Linting with Ruff..."
	@docker compose exec app ruff check .

## help: Display this help message.
help:
	@echo "ARLA Project Makefile"
	@echo "---------------------"
	@echo "Usage: make <target> [VARIABLE=value]"
	@echo ""
	@echo "Available Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Example Local Run:"
	@echo "  make run-local FILE=path/to/your.json STEPS=100"
	@echo ""

