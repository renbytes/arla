.PHONY: setup up down run-example logs test

# Default target
all: up

# Set up the environment by copying the example .env file
setup:
	@if [ ! -f .env ]; then \
		echo "--- Setup: Creating .env file from .env.example..."; \
		cp .env.example .env; \
	else \
		echo "--- Setup: .env file already exists."; \
	fi

# Build and start all services in the background
up: setup
	@echo "--- Starting all services with Docker Compose..."
	docker compose up --build -d

# Stop and remove all services
down:
	@echo "--- Stopping all services..."
	docker compose down

# Use the direct path to the venv's python executable
run-example:
	@echo "--- Running: Executing example scenario inside Docker..."
	docker compose exec app /app/.venv/bin/python -m agent_sim.main --scenario simulations/soul_sim/scenarios/default.json --steps 50

# View the logs from all running services
logs:
	@echo "--- Tailing logs from all services..."
	docker compose logs -f

# Run the full test suite inside the Docker container using the full path
test:
	@echo "--- Testing: Running pytest test suite..."
	docker compose exec app /app/.venv/bin/pytest