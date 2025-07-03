import os
from celery import Celery

broker_url = os.getenv("CELERY_BROKER", "redis://redis:6379/0")
backend_url = os.getenv("CELERY_BACKEND", "redis://redis:6379/1")

celery_app = Celery("agent_sim", broker=broker_url, backend=backend_url)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)
