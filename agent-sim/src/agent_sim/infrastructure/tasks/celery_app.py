# src/agent_sim/infrastructure/tasks/celery_app.py
import os
from pathlib import Path

from celery import Celery
from dotenv import load_dotenv

# Build an absolute path to the .env file in the project's root directory.
# This script is in .../src/agent_sim/infrastructure/tasks/, so the root is 4 levels up.
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
env_path = project_root / ".env"

if env_path.exists():
    print(f"Loading environment variables from: {env_path}")
    load_dotenv(dotenv_path=env_path)
else:
    # This warning helps in debugging if the .env file is ever misplaced.
    print(
        f"WARNING: .env file not found at {env_path}. Using default environment variables."
    )

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
app = Celery("agent_sim", include=["agent_sim.infrastructure.tasks.simulation_tasks"])

app.conf.worker_pool_restarts = True
# The 'prefork' pool is the default and recommended for CPU-bound tasks.
# It creates multiple worker processes, which is more robust for parallel execution.
# The previous 'solo' setting is for single-threaded debugging and was likely
# contributing to the event loop conflict.
app.conf.worker_pool = "prefork"

# Configure Celery
app.conf.update(
    broker_url=REDIS_URL,
    result_backend=REDIS_URL,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes max per task
    task_soft_time_limit=25 * 60,  # 25 minutes soft limit
    worker_prefetch_multiplier=1,  # One task per worker at a time
    database_pool_recycle=3600,
)
