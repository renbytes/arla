# ==============================================================================
# ARLA Project Makefile
#
# Provides a simplified interface for managing the Docker-based development
# environment and running simulations.
# ==============================================================================

.PHONY: help setup up down logs init-db run run-local cli test lint make-gif
.DEFAULT_GOAL := help

# --- Variable Definitions ---
# These variables MUST be provided from the command line.
# Example: make run-local PACKAGE=simulations.berry_sim ...
PACKAGE :=
CONFIG :=
FILE :=

# --- Optional Variable Definitions ---
# These have sensible defaults but can be overridden.
STEPS ?= 200
ARGS ?=
RENDER_DIR ?= data/gif_renders
FPS ?= 15
WORKERS ?= 4


# --- Core Docker Commands ---

## up: Build images, start services, and initialize the database.
up:
	@echo "🚀 Building images and starting all services..."
	@docker compose up -d --build
	@echo "⏳ Waiting 10 seconds for services to stabilize..."
	@sleep 10
	@echo "🔑 Initializing database tables..."
	@make init-db
	@echo "✅ All services are up and the database is initialized."

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

## run: Start Celery workers and run a full experiment.
run:
	@echo "👷 Starting services and $(WORKERS) Celery workers..."
	@docker compose up -d app worker --scale worker=$(WORKERS)
	@echo "⏳ Waiting 5 seconds for services to initialize..."
	@sleep 5
	@echo "▶️ Submitting experiment from: $(FILE)"
	# This command will now succeed because the 'app' container is running.
	@docker compose exec app poetry run agentsim run-experiment $(FILE)

## run-local: Run a single, local simulation for quick testing and debugging.
run-local:
	# These checks ensure that required variables are defined.
	@if [ -z "$(PACKAGE)" ]; then \
		echo "❌ Error: PACKAGE variable is not set."; \
		echo "   Please specify the simulation package to run."; \
		echo "   Example: make run-local PACKAGE=simulations.berry_sim ..."; \
		exit 1; \
	fi
	@if [ -z "$(CONFIG)" ]; then \
		echo "❌ Error: CONFIG variable is not set."; \
		echo "   Please specify the path to the configuration file."; \
		echo "   Example: make run-local CONFIG=simulations/berry_sim/config/config.yml ..."; \
		exit 1; \
	fi
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Error: FILE variable is not set."; \
		echo "   Please specify the path to the scenario file."; \
		echo "   Example: make run-local FILE=simulations/berry_sim/scenarios/default.json ..."; \
		exit 1; \
	fi
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

## make-gif: Creates a GIF from a specific simulation run's render.
make-gif:
	@if [ -z "$(RUN_ID)" ]; then \
		echo "❌ Error: RUN_ID variable is not set."; \
		echo "   Please specify the ID of the simulation run to render."; \
		echo "   Example: make make-gif RUN_ID=71bcad3e64a346618715e3b8be195e16 ..."; \
		exit 1; \
	fi
	@echo "🎬 Creating GIF from frames in $(RENDER_DIR)/$(RUN_ID)..."
	@docker compose exec app poetry run python scripts/create_gif.py $(RENDER_DIR)/$(RUN_ID) simulation-$(RUN_ID).gif --fps $(FPS)
	@echo "🔁 Ensuring GIF loops indefinitely..."
	@docker compose exec app gifsicle --loopcount=0 --batch simulation-$(RUN_ID).gif

## cli: Run any 'agentsim' command inside the container.
cli:
	@echo "Running CLI command: agentsim $(ARGS)"
	@docker compose exec app poetry run agentsim $(ARGS)

## test: Run the full pytest suite inside the container.
test:
	@echo "🧪 Running pytest test suite..."
	@poetry run pytest

## lint: Run the Ruff linter to check for code style issues.
lint:
	@echo "🎨 Linting with Ruff..."
	@poetry run ruff check .

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
	@echo "  make run-local PACKAGE=simulations.berry_sim CONFIG=simulations/berry_sim/config/config.yml FILE=simulations/berry_sim/scenarios/default.json"
	@echo ""
