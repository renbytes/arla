#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Install local packages now that the volume is mounted
echo "--- Installing local packages ---"
pip install --no-cache-dir -r requirements-local.txt

# Wait for the database to be ready
while ! nc -z postgres 5432; do
    echo "Waiting for postgres..."
    sleep 1
done
echo "PostgreSQL started"

# Run the database initialization script
echo "--- Running database initialization ---"
python -m agent_sim.infrastructure.database.init_db

# The database is now ready.
echo "--- DB initialization complete. Starting worker... ---"
exec "$@"