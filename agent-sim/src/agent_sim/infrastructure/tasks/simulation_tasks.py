# FILE: agent-sim/src/agent_sim/infrastructure/tasks/simulation_tasks.py

import importlib
import os
import traceback
import uuid
from typing import Any, Dict, NoReturn, Tuple

import mlflow
from agent_sim.infrastructure.data.async_runner import get_async_runner
from agent_sim.infrastructure.database.async_database_manager import (
    AsyncDatabaseManager,
)
from agent_sim.infrastructure.tasks.celery_app import app
from celery import Task
from mlflow.tracking import MlflowClient


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """Flattens a nested dictionary for MLflow parameter logging."""
    # CORRECTED: Added a type hint for the 'items' list.
    items: list[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _handle_simulation_exception(
    exc: Exception, task: Task, task_id: str, run_id_str: str
) -> NoReturn:
    """Logs failure to MLflow and DB, and decides whether to retry the task."""
    error_msg = (
        f"Simulation failed for run {run_id_str}: {str(exc)}\n{traceback.format_exc()}"
    )
    print(f"[{task_id}] ERROR: {error_msg}")

    try:
        if mlflow.active_run():
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error", error_msg)
            mlflow.end_run(status="FAILED")
    except Exception as mlflow_exc:
        print(f"[{task_id}] Could not log error to MLflow: {mlflow_exc}")

    db_manager = AsyncDatabaseManager()
    async_runner = get_async_runner()
    async_runner.run_async(
        db_manager.update_simulation_run_status(
            uuid.UUID(run_id_str), "failed", error_msg
        )
    )

    if any(
        keyword in str(exc).lower()
        for keyword in ["database is locked", "timeout", "connection"]
    ):
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
    variation_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Worker task that runs a single simulation and logs to a pre-existing run."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable must be set.")
    mlflow.set_tracking_uri(tracking_uri)
    task_id = self.request.id or "local-task"
    db_manager = AsyncDatabaseManager()
    async_runner = get_async_runner()

    print(f"[{task_id}] Received job for run '{run_id}'")

    async_runner.run_async(
        db_manager.update_simulation_run_status(uuid.UUID(run_id), "running")
    )

    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("status", "running")
            mlflow.set_tag("celery_task_id", task_id)

            if variation_overrides:
                mlflow.log_params(_flatten_dict(variation_overrides))

            self.update_state(
                state="PROGRESS", meta={"status": "Running simulation", "progress": 10}
            )

            sim_module = importlib.import_module(f"{simulation_package}.run")
            sim_module.start_simulation(
                run_id, task_id, experiment_name, config_overrides
            )

            self.update_state(
                state="SUCCESS", meta={"status": "Completed", "progress": 100}
            )
            mlflow.set_tag("status", "completed")
            async_runner.run_async(
                db_manager.update_simulation_run_status(uuid.UUID(run_id), "completed")
            )
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
    variation_name: str,
    variation_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Orchestrator task that creates DB/MLflow records and submits worker tasks."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable must be set.")
    mlflow.set_tracking_uri(tracking_uri)
    db_manager = AsyncDatabaseManager()
    async_runner = get_async_runner()
    mlflow_client = MlflowClient()

    try:
        mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
        mlflow_exp_id = (
            mlflow_experiment.experiment_id
            if mlflow_experiment
            else mlflow.create_experiment(name=experiment_name)
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
        print(
            f"[bold red]FATAL: Could not create/get MLflow/DB experiment. Error: {e}[/bold red]"
        )
        raise

    job_ids = []
    for scenario_path in scenario_paths:
        for run_num in range(runs_per_scenario):
            scenario_name = os.path.basename(scenario_path).replace(".json", "")
            run_name = f"{variation_name}-{scenario_name}-{run_num:03d}"

            config_overrides = {**base_config, "scenario_path": scenario_path}

            sim_config = base_config.get("simulation", {})
            base_seed = sim_config.get("random_seed", 1)
            new_seed = base_seed + run_num

            if "simulation" not in config_overrides:
                config_overrides["simulation"] = {}
            config_overrides["simulation"]["random_seed"] = new_seed

            run = mlflow_client.create_run(
                experiment_id=mlflow_exp_id,
                tags={"mlflow.runName": run_name, "status": "queued"},
            )
            mlflow_run_id_str = run.info.run_id

            job = (
                run_simulation_task.s(
                    config_overrides,
                    simulation_package,
                    mlflow_run_id_str,
                    mlflow_exp_id,
                    experiment_name,
                    variation_overrides,
                )
                .set(queue="simulations")
                .delay()
            )

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
