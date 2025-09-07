"""
Performs a statistical analysis of the Sugarscape experiment results.

This script connects to the PostgreSQL database, fetches key performance
and social metrics for each simulation run from a specified experiment,
and conducts an ANOVA test to determine if there are statistically
significant differences between the experimental groups.

Example Usage:
poetry run python simulations/sugarscape_sim/analysis/analyze_sugarscape.py "Sugarscape - Full Cognitive Comparison"
"""

import os

import pandas as pd
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from scipy.stats import f_oneway
from sqlalchemy import create_engine, text
from statsmodels.stats.multicomp import pairwise_tukeyhsd

app = typer.Typer()


def get_db_connection():
    """Creates and returns a SQLAlchemy engine connected to the database."""
    project_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    load_dotenv(os.path.join(project_root, ".env"))

    db_url = os.getenv("DATABASE_URL", "").replace("postgresql+asyncpg", "postgresql")
    if not db_url:
        raise ValueError("DATABASE_URL not found in .env file.")
    return create_engine(db_url)


def fetch_experiment_data(engine, experiment_name: str) -> pd.DataFrame:
    """
    Fetches final survival and aggregated social metrics for each run in the experiment.
    """
    query = text(
        """
        WITH LatestExperiment AS (
            -- Step 1: Find the ID of the most recent experiment with the given name.
            SELECT id FROM experiments
            WHERE name = :exp_name
            ORDER BY created_at DESC
            LIMIT 1
        ),
        AggregatedMetrics AS (
            -- Step 2: For each simulation, get the final active agent count and
            -- sum the total social actions over the entire run.
            SELECT
                simulation_id,
                -- Get the last recorded value for active_agents for each simulation
                (array_agg((data->>'active_agents')::int ORDER BY tick DESC))[1] AS final_active_agents,
                -- Sum the counts for each social action, defaulting to 0 if the key is missing
                SUM(COALESCE((data->>'attack_count')::int, 0)) AS total_attacks,
                SUM(COALESCE((data->>'share_count')::int, 0)) AS total_shares,
                SUM(COALESCE((data->>'reproduce_count')::int, 0)) AS total_reproductions
            FROM metrics
            WHERE simulation_id IN (
                SELECT id FROM simulation_runs WHERE experiment_id = (SELECT id FROM LatestExperiment)
            )
            GROUP BY simulation_id
        )
        -- Step 3: Join the results with the simulation run details to get the group name.
        SELECT
            sr.variation_name AS agent_group,
            am.final_active_agents,
            am.total_attacks,
            am.total_shares,
            am.total_reproductions
        FROM
            simulation_runs sr
        JOIN
            AggregatedMetrics am ON sr.id = am.simulation_id
        WHERE
            sr.experiment_id = (SELECT id FROM LatestExperiment)
            AND sr.completed_at IS NOT NULL;
        """
    )
    with engine.connect() as connection:
        df = pd.read_sql(query, connection, params={"exp_name": experiment_name})
    return df


def print_analysis_summary(
    df: pd.DataFrame, metric_column: str, metric_title: str, alpha: float = 0.05
):
    """
    Performs ANOVA and Tukey's HSD test for a given metric and prints a summary.
    """
    console = Console()
    console.rule(f"[bold cyan]Analysis for: {metric_title}[/bold cyan]")

    # --- Group Summary Table ---
    summary_table = Table(title=f"Group Summary Statistics ({metric_title})")
    summary_table.add_column("Agent Group", style="cyan")
    summary_table.add_column("Mean", style="magenta")
    summary_table.add_column("Std Dev", style="green")
    summary_table.add_column("N", style="yellow")

    for group_name in sorted(df["agent_group"].unique()):
        group_data = df[df["agent_group"] == group_name][metric_column]
        summary_table.add_row(
            group_name,
            f"{group_data.mean():.2f}",
            f"{group_data.std():.2f}",
            str(group_data.count()),
        )
    console.print(summary_table)

    # --- Statistical Tests ---
    groups = [
        df[df["agent_group"] == g][metric_column] for g in df["agent_group"].unique()
    ]

    # Check for variance before running tests
    if all(g.var() == 0 for g in groups):
        console.print(
            "\n[yellow]⚠️  Skipping statistical tests: No variance found in any group for this metric.[/yellow]"
        )
        console.rule()
        return

    f_stat, p_value = f_oneway(*groups)

    console.print("\n[bold]🔬 ANOVA Test Results:[/bold]")
    console.print(f"  - F-Statistic: {f_stat:.4f}")
    console.print(f"  - P-Value: {p_value:.4f}")

    if p_value < alpha:
        console.print(
            f"  [green]✅ The p-value is less than {alpha}, indicating a statistically significant difference exists somewhere among the groups.[/green]"
        )

        tukey_result = pairwise_tukeyhsd(
            endog=df[metric_column], groups=df["agent_group"], alpha=alpha
        )
        console.print(
            "\n[bold]Pairwise Tukey HSD Test (shows which groups differ):[/bold]"
        )
        console.print(str(tukey_result))

    else:
        console.print(
            f"  [yellow]❌ The p-value is greater than {alpha}, indicating no statistically significant difference was found.[/yellow]"
        )
    console.rule()


@app.command()
def main(
    experiment_name: str = typer.Argument(
        ..., help="The exact name of the experiment to analyze from the database."
    ),
):
    """Main function to orchestrate the analysis for multiple metrics."""
    try:
        engine = get_db_connection()
        data = fetch_experiment_data(engine, experiment_name)

        if data.empty or len(data["agent_group"].unique()) < 2:
            print(
                f"Could not find sufficient data for all experimental groups in '{experiment_name}'. Analysis cannot proceed."
            )
            print("Found data:")
            print(data)
            return

        metrics_to_analyze = [
            ("final_active_agents", "Final Active Agents (Survival)"),
            ("total_attacks", "Total Attacks"),
            ("total_shares", "Total Shares"),
            ("total_reproductions", "Total Reproductions"),
        ]

        for metric_col, metric_title in metrics_to_analyze:
            if metric_col in data.columns:
                print_analysis_summary(data, metric_col, metric_title)
            else:
                print(f"Metric column '{metric_col}' not found in data. Skipping.")

    except Exception as e:
        print(f"\nAn error occurred during analysis: {e}")


if __name__ == "__main__":
    app()
