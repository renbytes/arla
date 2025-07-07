import os

from celery import Celery  # type: ignore

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
app = Celery("agent_soul_sim", include=["src.infrastructure.tasks.simulation_tasks"])

app.conf.worker_pool_restarts = True
# Comment when running in cloud/production. Will throw error if running locally.
app.conf.worker_pool = "solo"

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

# Auto-discover tasks
app.autodiscover_tasks(["src.infrastructure"])
