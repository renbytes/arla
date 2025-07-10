# FILE: agent-sim/src/agent_sim/infrastructure/tasks/simulation_tasks.py

import importlib
import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, NoReturn

import mlflow
from agent_sim.infrastructure.data.async_runner import async_runner
from agent_sim.infrastructure.database.async_database_manager import AsyncDatabaseManager
from agent_sim.infrastructure.tasks.celery_app import app
from celery import Task

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _handle_simulation_exception(exc: Exception, task: Task, task_id: str, run_id_str: str) -> NoReturn:
    """Logs failure to MLflow and DB, and decides whether to retry the task."""
    error_msg = f"Simulation failed for run {run_id_str}: {str(exc)}\n{traceback.format_exc()}"
    print(f"[{task_id}] ERROR: {error_msg}")

    if mlflow.active_run():
        mlflow.set_tag("status", "failed")
        mlflow.log_param("error", error_msg)

    db_manager = AsyncDatabaseManager()
    # Convert string ID to UUID object for the database call
    async_runner.run_async(db_manager.update_simulation_run_status(uuid.UUID(run_id_str), "failed", error_msg))

    if any(keyword in str(exc).lower() for keyword in ["database is locked", "timeout", "connection"]):
        print(f"[{task_id}] Retrying due to transient error...")
        raise task.retry(exc=exc, countdown=60, max_retries=3)

    raise exc


@app.task(bind=True, name="tasks.run_simulation", queue="simulations")
def run_simulation_task(
    self: Task,
    config_overrides: Dict[str, Any],
    simulation_package: str,
    run_id: str,  # This will now be received as a string
    experiment_id: str,
    experiment_name: str,
) -> Dict[str, Any]:
    """Worker task that runs a single simulation and logs to a pre-existing run."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    task_id = self.request.id or "local-task"
    db_manager = AsyncDatabaseManager()

    print(f"[{task_id}] Received job for run '{run_id}'")

    # Convert string ID to UUID object for the database call
    async_runner.run_async(db_manager.update_simulation_run_status(uuid.UUID(run_id), "running"))

    # This call now works because run_id is a string
    with mlflow.start_run(run_id=run_id):
        try:
            mlflow.set_tag("status", "running")
            mlflow.set_tag("celery_task_id", task_id)

            self.update_state(state="PROGRESS", meta={"status": "Running simulation", "progress": 10})

            # Pass the string run_id to the simulation
            sim_module = importlib.import_module(f"{simulation_package}.run")
            sim_module.start_simulation(run_id, task_id, experiment_id, config_overrides)

            self.update_state(state="SUCCESS", meta={"status": "Completed", "progress": 100})
            mlflow.set_tag("status", "completed")
            # Convert string ID to UUID object for the database call
            async_runner.run_async(db_manager.update_simulation_run_status(uuid.UUID(run_id), "completed"))
            print(f"[{task_id}] Successfully completed simulation run: {run_id}")
            return {"run_id": run_id, "status": "completed"}

        except Exception as exc:
            _handle_simulation_exception(exc, self, task_id, run_id)


@app.task(bind=True, name="tasks.run_experiment", queue="experiments")
def run_experiment_task(
    self: Task,
    scenario_paths: list[str],
    runs_per_scenario: int,
    base_config: Dict[str, Any],
    simulation_package: str,
    experiment_name: str,
) -> Dict[str, Any]:
    """Orchestrator task that creates DB/MLflow records and submits worker tasks."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    db_manager = AsyncDatabaseManager()

    try:
        mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
        mlflow_exp_id = (
            mlflow_experiment.experiment_id if mlflow_experiment else mlflow.create_experiment(name=experiment_name)
        )

        db_experiment_uuid = async_runner.run_async(
            db_manager.create_experiment(
                name=experiment_name,
                config=base_config,
                total_runs=len(scenario_paths) * runs_per_scenario,
                simulation_package=simulation_package,
                mlflow_experiment_id=mlflow_exp_id,
            )
        )
    except Exception as e:
        print(f"[bold red]FATAL: Could not create/get MLflow/DB experiment. Error: {e}[/bold red]")
        raise

    job_ids = []
    for scenario_path in scenario_paths:
        for run_num in range(runs_per_scenario):
            run_name = f"{os.path.basename(scenario_path).replace('.json', '')}-{run_num:03d}-{uuid.uuid4().hex[:4]}"
            config_overrides = {**base_config, "scenario_path": scenario_path}
            config_overrides.setdefault("simulation", {})["random_seed"] = (
                config_overrides["simulation"].get("random_seed", 1) + run_num
            )

            # 1. Let MLflow create the run and give us its string UUID
            with mlflow.start_run(experiment_id=mlflow_exp_id, run_name=run_name) as run:
                mlflow_run_id_str = run.info.run_id
                mlflow.set_tag("status", "queued")

            # 2. Submit the Celery task, passing the string ID
            job = (
                run_simulation_task.s(
                    config_overrides, simulation_package, mlflow_run_id_str, mlflow_exp_id, experiment_name
                )
                .set(queue="simulations")
                .delay()
            )

            # 3. Create our DB record, converting the string to a UUID object for the database
            async_runner.run_async(
                db_manager.create_simulation_run(
                    run_id=uuid.UUID(mlflow_run_id_str),
                    experiment_id=db_experiment_uuid,
                    task_id=job.id,
                    scenario_name=os.path.basename(scenario_path).replace(".json", ""),
                    config=config_overrides,
                )
            )

            job_ids.append(job.id)
            print(f"  Submitted job {job.id} for run '{mlflow_run_id_str}'")

    return {
        "mlflow_experiment_id": mlflow_exp_id,
        "total_jobs_submitted": len(job_ids),
        "celery_task_ids": job_ids,
    }
