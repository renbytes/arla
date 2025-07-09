import importlib
import json
import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, NoReturn

import mlflow
from agent_sim.infrastructure.tasks.celery_app import app
from celery import Task

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _handle_simulation_exception(exc: Exception, task: Task, task_id: str) -> NoReturn:
    """Logs failure to MLflow and decides whether to retry the task."""
    error_msg = f"Simulation failed: {str(exc)}\n{traceback.format_exc()}"
    print(f"[{task_id}] ERROR: {error_msg}")
    mlflow.set_tag("status", "failed")
    mlflow.log_param("error", error_msg)

    # Retry on transient errors like DB connection issues
    if any(keyword in str(exc).lower() for keyword in ["database is locked", "timeout", "connection"]):
        print(f"[{task_id}] Retrying due to transient error...")
        raise task.retry(exc=exc, countdown=60, max_retries=3)

    raise exc


@app.task(bind=True, name="tasks.run_simulation", queue="simulations")
def run_simulation_task(
    self: Task,
    config_overrides: Dict[str, Any],
    simulation_package: str,
    run_id: str,
    experiment_id: str,
    experiment_name: str,
) -> Dict[str, Any]:
    """
    Generic task that dynamically loads and runs a specific simulation package.
    This task is AGNOSTIC to the simulation's internal logic or schemas.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    task_id = self.request.id or "local-task"

    print(f"[{task_id}] Received job for run '{run_id}'")
    print(f"[{task_id}] Target simulation package: '{simulation_package}'")

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_id, experiment_id=experiment_id):
        try:
            # Log parameters and tags to MLflow
            scenario_path = config_overrides.get("scenario_path", "N/A")
            mlflow.log_params(config_overrides)
            mlflow.set_tag("celery_task_id", task_id)
            mlflow.set_tag("simulation_package", simulation_package)
            mlflow.set_tag("scenario", os.path.basename(scenario_path).replace(".json", ""))

            # Update Celery task state
            self.update_state(state="PROGRESS", meta={"status": "Loading simulation package", "progress": 5})

            print("--- FINAL CONFIGURATION BEING PASSED TO SIMULATION ---")
            print(json.dumps(config_overrides, indent=2))

            # Dynamically import and run the specific simulation's entry point
            sim_module = importlib.import_module(f"{simulation_package}.run")
            sim_module.start_simulation(run_id, task_id, experiment_id, config_overrides)

            self.update_state(state="SUCCESS", meta={"status": "Completed", "progress": 100})
            mlflow.set_tag("status", "completed")
            print(f"[{task_id}] Successfully completed simulation run: {run_id}")
            return {"run_id": run_id, "status": "completed"}

        except Exception as exc:
            # The helper function will log to MLflow and re-raise the exception
            _handle_simulation_exception(exc, self, task_id)


@app.task(bind=True, name="tasks.run_experiment", queue="experiments")
def run_experiment_task(
    self: Task,
    scenario_paths: list[str],
    runs_per_scenario: int,
    base_config: Dict[str, Any],
    simulation_package: str,
    experiment_name: str,
) -> Dict[str, Any]:
    """Submits multiple simulation tasks for a complete experiment."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # Create an MLflow experiment to group the runs
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(name=experiment_name)
        print(f"Using MLflow experiment: '{experiment_name}' (ID: {experiment_id})")
    except Exception as e:
        print(f"[bold red]FATAL: Could not create or get MLflow experiment. Error: {e}[/bold red]")
        raise

    job_ids = []
    for scenario_path in scenario_paths:
        for run_num in range(runs_per_scenario):
            # Create a unique run ID and apply run-specific config overrides
            run_id = f"{os.path.basename(scenario_path).replace('.json', '')}-{run_num:03d}-{uuid.uuid4().hex[:4]}"
            config_overrides = base_config.copy()
            config_overrides["scenario_path"] = scenario_path

            # Ensure each run in a set has a different seed for statistical variance
            if "simulation" not in config_overrides:
                config_overrides["simulation"] = {}
            base_seed = config_overrides["simulation"].get("random_seed", 1)
            config_overrides["simulation"]["random_seed"] = base_seed + run_num

            # Submit the specific simulation task
            job = (
                run_simulation_task.s(
                    config_overrides=config_overrides,
                    simulation_package=simulation_package,
                    run_id=run_id,
                    experiment_id=experiment_id,
                    experiment_name=experiment_name,
                )
                .set(queue="simulations")
                .delay()
            )

            job_ids.append(job.id)
            print(f"  Submitted job {job.id} for run '{run_id}'")

    return {
        "mlflow_experiment_id": experiment_id,
        "mlflow_experiment_name": experiment_name,
        "total_jobs_submitted": len(job_ids),
        "celery_task_ids": job_ids,
    }
