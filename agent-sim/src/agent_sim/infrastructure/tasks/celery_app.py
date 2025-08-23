# FILE: agent-sim/src/agent_sim/infrastructure/tasks/celery_app.py

import os
from pathlib import Path

from celery import Celery
from celery.signals import worker_process_init
from dotenv import load_dotenv

from agent_sim.infrastructure.data.async_runner import get_async_runner


project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
env_path = project_root / ".env"

if env_path.exists():
    print(f"Loading environment variables from: {env_path}")
    load_dotenv(dotenv_path=env_path)
else:
    print(
        f"WARNING: .env file not found at {env_path}. Using default environment variables."
    )

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
app = Celery("agent_sim", include=["agent_sim.infrastructure.tasks.simulation_tasks"])

app.conf.worker_pool_restarts = True
app.conf.worker_pool = "prefork"

app.conf.update(
    broker_url=REDIS_URL,
    result_backend=REDIS_URL,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=None,
    task_soft_time_limit=None,
    worker_prefetch_multiplier=1,
    database_pool_recycle=3600,
)


@worker_process_init.connect
def on_worker_init(**kwargs):
    """
    Handler called after a worker process forks.
    This is the safe place to initialize our async runner.
    """
    print(f"Celery worker process {os.getpid()} initialized. Creating async runner.")
    get_async_runner()
