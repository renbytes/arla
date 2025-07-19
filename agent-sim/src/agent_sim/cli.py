import subprocess
import sys
from pathlib import Path

import typer
from agent_sim.infrastructure.tasks.celery_app import app as celery_app
from agent_sim.infrastructure.tasks.simulation_tasks import run_experiment_task
from omegaconf import OmegaConf
from rich import print
from rich.console import Console
from rich.table import Table

# ------------------------------------------------------------------------
# Global constants that are **objects**, not *calls* in the function default
# positions – this satisfies flake8‑bugbear B008.
# ------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent

EXPERIMENT_FILE_ARGUMENT = typer.Argument(
    ...,  # required
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Path to the experiment definition YAML file.",
)

QUEUE_OPTION = typer.Option(
    "simulations",
    "--queue",
    "-Q",
    help="The Celery queue to consume from.",
)

CONCURRENCY_OPTION = typer.Option(
    4,
    "--concurrency",
    "-c",
    help="Number of concurrent worker processes.",
)

EXPERIMENTS_DIR_OPTION = typer.Option(
    str(PROJECT_ROOT / "experiments"),
    "--experiments-dir",
    help="The directory containing experiment definition files.",
    exists=True,
    file_okay=False,
)

# CLI App Initialization----------------------------------------------
# Create a Typer app for a clean command‑line interface.
app = typer.Typer(
    name="agentsim",
    help="A unified CLI for managing Agent‑based Simulations.",
    add_completion=False,
)
console = Console()


# ------------------------------------------------------------------------
# Command: run‑experiment
# ------------------------------------------------------------------------
@app.command(name="run-experiment")
def run_experiment(
    experiment_file: Path = EXPERIMENT_FILE_ARGUMENT,
):
    """Parse an experiment definition file and submit Celery tasks for each
    variation defined inside it."""

    console.rule(f"[bold green]🚀 Launching Experiment: {experiment_file.name}")

    try:
        exp_def = OmegaConf.load(experiment_file)
    except Exception as e:  # noqa: BLE001 – broad except needed for user feedback
        print(f"[bold red]Error: Could not parse experiment file {experiment_file}.[/bold red]")
        # B904 – explicitly chain the original exception.
        raise typer.Exit(code=1) from e

    # Load and merge configurations----------------------------------
    base_config_path = PROJECT_ROOT / exp_def.get("base_config_path", "")  # type: ignore
    if not base_config_path.exists():
        print(f"[bold red]Error: Base config file not found at {base_config_path}[/bold red]")
        raise typer.Exit(code=1)

    base_config = OmegaConf.load(base_config_path)
    variations = exp_def.get("variations", [{"name": "default", "overrides": {}}])  # type: ignore

    # Initialize the counter before the loop
    total_jobs = 0

    for variation in variations:
        variation_name = variation.get("name", "unnamed_variation")
        print(f"\n[cyan]Submitting tasks for variation: [bold]{variation_name}[/bold]")

        # Merge base config with variation‑specific overrides
        final_config = OmegaConf.merge(base_config, variation.get("overrides", {}))

        # Convert to a plain dict for Celery serialisation
        config_dict = OmegaConf.to_container(final_config, resolve=True)

        # Give each variation its own MLflow experiment name
        experiment_name = f"{exp_def.get('experiment_name', 'UnnamedExp')} - {variation_name}"  # type: ignore

        # Submit the master experiment task to Celery
        run_experiment_task.delay(
            scenario_paths=list(exp_def.get("scenarios", [])),  # type: ignore
            runs_per_scenario=exp_def.get("runs_per_scenario", 1),  # type: ignore
            base_config=config_dict,
            simulation_package=exp_def.get("simulation_package"),  # type: ignore
            experiment_name=experiment_name,
        )
        jobs_in_variation = len(exp_def.get("scenarios", [])) * exp_def.get("runs_per_scenario", 1)  # type: ignore
        total_jobs += jobs_in_variation
        print(f"[green]✔ Submitted {jobs_in_variation} simulation runs for this variation.[/green]")

    console.rule(f"[bold green]✅ Experiment '{exp_def.get('experiment_name')}' fully submitted.")  # type: ignore
    print(f"Total simulation runs queued: [bold cyan]{total_jobs}[/bold cyan]")
    print("Monitor your Celery workers and MLflow UI for progress.")


# ------------------------------------------------------------------------
# Command: start‑worker
# ------------------------------------------------------------------------
@app.command(name="start-worker")
def start_worker(
    # B008 – the Option objects are defined at module level.
    queue: str = QUEUE_OPTION,
    concurrency: int = CONCURRENCY_OPTION,
):
    """Start a Celery worker with a simplified command that hides the verbose
    `celery -A ... worker` incantation."""

    console.rule(f"[bold blue]👷 Starting Celery Worker for Queue: {queue}")

    celery_app_path = "agent_sim.infrastructure.tasks.celery_app"
    command: list[str] = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        celery_app_path,
        "worker",
        "--loglevel=INFO",
        "-Q",
        queue,
        "-c",
        str(concurrency),
        "--pool=prefork",  # a more robust pool for CPU‑bound tasks
    ]

    print(f"Running command: [dim]{' '.join(command)}[/dim]")
    try:
        subprocess.run(command, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[bold red]Error starting Celery worker: {e}[/bold red]")
        print("Please ensure Celery is installed in your environment and Redis is running.")
        # B904 – chain the caught exception.
        raise typer.Exit(code=1) from e


# ------------------------------------------------------------------------
# Command: list‑experiments
# ------------------------------------------------------------------------
@app.command(name="list-experiments")
def list_experiments(
    experiments_dir: Path = EXPERIMENTS_DIR_OPTION,
):
    """Scan the experiments directory and list all available experiment files."""

    console.rule("[bold yellow]🔎 Available Experiments")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Experiment File", style="dim", width=50)
    table.add_column("Full Path")

    exp_files = sorted(experiments_dir.glob("*.yaml")) + sorted(experiments_dir.glob("*.yml"))

    if not exp_files:
        print(
            "[yellow]No experiment definition files (.yaml) found in:[/yellow]",
            str(experiments_dir),
        )
        raise typer.Exit()

    for file_path in exp_files:
        table.add_row(file_path.name, str(file_path))

    console.print(table)


# ------------------------------------------------------------------------
# Command: health‑check
# ------------------------------------------------------------------------
@app.command(name="health-check")
def health_check():
    """Ping the Celery workers to check if they are operational."""

    console.rule("[bold purple]🩺 Health Check")
    try:
        inspector = celery_app.control.inspect()
        active_workers = inspector.ping()

        if not active_workers:
            print("[bold red]Error: No active Celery workers found.[/bold red]")
            print("Make sure your workers are running and connected to the same broker.")
            raise typer.Exit(code=1)

        print("[green]✔ Successfully pinged active workers:[/green]")
        for worker, reply in active_workers.items():
            print(f"  - [cyan]{worker}[/cyan]: {reply}")

    except Exception as e:  # noqa: BLE001 – broad except here is pragmatic
        print(f"[bold red]Error connecting to Celery broker: {e}[/bold red]")
        print("Please ensure Redis (or your broker) is running and accessible.")
        # B904 – chain the caught exception so we keep its traceback.
        raise typer.Exit(code=1) from e


# ------------------------------------------------------------------------
# Script entry‑point
# ------------------------------------------------------------------------
if __name__ == "__main__":
    app()
