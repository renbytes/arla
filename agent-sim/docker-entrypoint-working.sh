#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Wait for the database to be ready
# This requires installing netcat (e.g., `apt-get install -y netcat`) in your Dockerfile
while ! nc -z postgres 5432; do
  echo "Waiting for postgres..."
  sleep 1
done
echo "PostgreSQL started"

# Run the database initialization script
echo "--- Running database initialization"
python -m agent_sim.infrastructure.database.init_db

# The database is now ready.
# `exec "$@"` runs the command passed to the script. In a Dockerfile,
# this will be the CMD instruction (e.g., starting the celery worker).
echo "--- DB initialization complete. Starting worker..."
exec "$@"
