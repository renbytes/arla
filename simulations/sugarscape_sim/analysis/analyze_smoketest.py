"""
Analyzes the results of the Sugarscape smoke test to validate survival pressure.

This script connects to the PostgreSQL database, fetches the final active agent
count for all runs in a specified experiment, and determines if the environment
is harsh enough by checking if the average survival rate is below a defined threshold.

Example Usage:
poetry run python simulations/sugarscape_sim/analysis/analyze_smoketest.py "Sugarscape - Smoke Test"
"""

import os

import pandas as pd
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy import create_engine, text

# Initialize Typer for CLI argument parsing
app = typer.Typer()


def get_db_connection():
    """
    Creates and returns a SQLAlchemy engine connected to the database.

    Loads database credentials from the .env file located at the project root.

    Returns:
        A SQLAlchemy Engine instance for database connectivity.
    """
    project_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    load_dotenv(os.path.join(project_root, ".env"))

    db_url = os.getenv("DATABASE_URL", "").replace("postgresql+asyncpg", "postgresql")
    if not db_url:
        raise ValueError("DATABASE_URL not found in .env file.")
    return create_engine(db_url)


def fetch_smoketest_data(engine, experiment_name: str) -> pd.DataFrame:
    """
    Fetches the final active agent count for each completed run in the smoke test.

    Args:
        engine: The SQLAlchemy engine for the database connection.
        experiment_name: The name of the smoke test experiment to analyze.

    Returns:
        A pandas DataFrame containing the final active agent count for each run.
    """
    query = text(
        """
        WITH LatestExperiment AS (
            -- Find the most recent experiment matching the smoke test name
            SELECT id FROM experiments
            WHERE name = :exp_name
            ORDER BY created_at DESC
            LIMIT 1
        ),
        FinalMetrics AS (
            -- Get the metric value from the last tick for each simulation run
            SELECT DISTINCT ON (simulation_id)
                simulation_id,
                (data ->> 'active_agents')::int AS final_active_agents
            FROM metrics
            WHERE simulation_id IN (
                SELECT id FROM simulation_runs WHERE experiment_id = (SELECT id FROM LatestExperiment)
            )
            ORDER BY simulation_id, tick DESC
        )
        -- Join the results to get the final agent count for completed runs
        SELECT
            sr.variation_name as agent_group,
            fm.final_active_agents
        FROM
            simulation_runs sr
        JOIN
            FinalMetrics fm ON sr.id = fm.simulation_id
        WHERE
            sr.experiment_id = (SELECT id FROM LatestExperiment)
            AND sr.completed_at IS NOT NULL; -- Ensure we only analyze completed runs
        """
    )
    with engine.connect() as connection:
        df = pd.read_sql(query, connection, params={"exp_name": experiment_name})
    return df


def print_smoketest_summary(
    df: pd.DataFrame, initial_population: int, survival_threshold: float
):
    """
    Analyzes the smoke test data and prints a formatted summary and verdict.

    Args:
        df: DataFrame with the smoke test results.
        initial_population: The starting number of agents in the simulation.
        survival_threshold: The survival rate below which the test is considered a "PASS".
    """
    console = Console()
    console.rule("[bold yellow]Sugarscape Smoke Test Analysis[/bold yellow]")

    if df.empty:
        console.print(
            Panel(
                "[bold red]No completed runs found for the specified experiment.[/bold red]\nPlease check the experiment name and if the simulations ran successfully.",
                title="Error",
                border_style="red",
            )
        )
        return

    # --- Analysis ---
    mean_final_agents = df["final_active_agents"].mean()
    std_final_agents = df["final_active_agents"].std()
    survival_rate = mean_final_agents / initial_population
    num_runs = len(df)
    test_passed = survival_rate < survival_threshold

    # --- Summary Table ---
    summary_table = Table(title="Smoke Test Results")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="magenta")

    summary_table.add_row("Initial Agent Population", str(initial_population))
    summary_table.add_row("Number of Completed Runs", str(num_runs))
    summary_table.add_row("Mean Final Agent Count", f"{mean_final_agents:.2f}")
    summary_table.add_row("Std Dev of Final Count", f"{std_final_agents:.2f}")
    summary_table.add_row("Average Survival Rate", f"{survival_rate:.2%}")
    console.print(summary_table)

    # --- Verdict ---
    verdict_title = "Verdict"
    if test_passed:
        verdict_style = "green"
        verdict_text = (
            f"✅ [bold]PASS[/bold]\n\nThe average survival rate ({survival_rate:.2%}) "
            f"is below the threshold of {survival_threshold:.0%}.\n"
            "The environment is sufficiently harsh. You can proceed with the main experiment."
        )
    else:
        verdict_style = "red"
        verdict_text = (
            f"❌ [bold]FAIL[/bold]\n\nThe average survival rate ({survival_rate:.2%}) "
            f"is above the threshold of {survival_threshold:.0%}.\n"
            "The environment is still too forgiving. Consider making it harsher."
        )

    console.print(Panel(verdict_text, title=verdict_title, border_style=verdict_style))
    console.rule()


@app.command()
def main(
    experiment_name: str = typer.Argument(
        "Sugarscape - Smoke Test",
        help="The exact name of the experiment to analyze from MLflow/database.",
    ),
    initial_population: int = typer.Option(
        50, "--population", "-p", help="Initial number of agents in the simulation."
    ),
    survival_threshold: float = typer.Option(
        0.50,
        "--threshold",
        "-t",
        help="Survival rate below which the test is considered a PASS.",
    ),
):
    """Main function to orchestrate the smoke test analysis."""
    try:
        engine = get_db_connection()
        data = fetch_smoketest_data(engine, experiment_name)
        print_smoketest_summary(data, initial_population, survival_threshold)

    except Exception as e:
        console = Console()
        console.print(f"\n[bold red]An error occurred during analysis:[/bold red] {e}")


if __name__ == "__main__":
    app()
