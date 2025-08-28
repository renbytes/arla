# FILE: simulations/sugarscape_sim/analysis/analyze_sugarscape.py
"""
Performs a statistical analysis of the Sugarscape experiment results.

This script connects to the PostgreSQL database, fetches the final
active agent count for each simulation run, and conducts an ANOVA test
to determine if there are statistically significant performance differences
between the experimental groups.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from scipy.stats import f_oneway
from sqlalchemy import create_engine, text
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def get_db_connection():
    """Creates and returns a SQLAlchemy engine connected to the database."""
    project_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    load_dotenv(os.path.join(project_root, ".env"))

    db_url = os.getenv("DATABASE_URL", "").replace("postgresql+asyncpg", "postgresql")
    if not db_url:
        raise ValueError("DATABASE_URL not found in .env file.")
    return create_engine(db_url)


def fetch_experiment_data(engine, experiment_name: str) -> pd.DataFrame:
    """Fetches the final active agent count for each run in the experiment."""
    query = text(
        """
        WITH LatestExperiment AS (
            SELECT id FROM experiments
            WHERE name = :exp_name
            ORDER BY created_at DESC
            LIMIT 1
        ),
        FinalMetrics AS (
            SELECT DISTINCT ON (simulation_id)
                simulation_id,
                (data ->> 'active_agents')::int AS final_active_agents
            FROM metrics
            WHERE simulation_id IN (
                SELECT id FROM simulation_runs WHERE experiment_id = (SELECT id FROM LatestExperiment)
            )
            ORDER BY simulation_id, tick DESC
        )
        SELECT
            sr.variation_name as agent_group, -- CHANGED: Select the new column
            fm.final_active_agents
        FROM
            simulation_runs sr
        JOIN
            FinalMetrics fm ON sr.id = fm.simulation_id
        WHERE
            sr.experiment_id = (SELECT id FROM LatestExperiment);
        """
    )
    with engine.connect() as connection:
        df = pd.read_sql(query, connection, params={"exp_name": experiment_name})
    return df


def print_analysis_summary(df: pd.DataFrame, alpha: float = 0.05):
    """Performs ANOVA and Tukey's HSD test and prints a summary."""
    console = Console()
    console.rule("[bold cyan]Sugarscape Experiment: Statistical Analysis[/bold cyan]")

    groups = [
        df[df["agent_group"] == g]["final_active_agents"]
        for g in df["agent_group"].unique()
    ]
    f_stat, p_value = f_oneway(*groups)

    # Print Group Summaries
    summary_table = Table(title="Group Summary Statistics (Final Active Agents)")
    summary_table.add_column("Agent Group", style="cyan")
    summary_table.add_column("Mean", style="magenta")
    summary_table.add_column("Std Dev", style="green")
    summary_table.add_column("N", style="yellow")

    for group_name in sorted(df["agent_group"].unique()):
        group_data = df[df["agent_group"] == group_name]["final_active_agents"]
        summary_table.add_row(
            group_name,
            f"{group_data.mean():.2f}",
            f"{group_data.std():.2f}",
            str(group_data.count()),
        )
    console.print(summary_table)

    # Print ANOVA Results
    console.print("\n[bold]🔬 ANOVA Test Results:[/bold]")
    console.print(f"  - F-Statistic: {f_stat:.4f}")
    console.print(f"  - P-Value: {p_value:.4f}")

    if p_value < alpha:
        console.print(
            f"  [green]✅ The p-value is less than {alpha}, indicating a statistically significant difference exists somewhere among the groups.[/green]"
        )

        # Perform and print Tukey's HSD post-hoc test
        tukey_result = pairwise_tukeyhsd(
            endog=df["final_active_agents"], groups=df["agent_group"], alpha=alpha
        )

        console.print(
            "\n[bold]Pairwise Tukey HSD Test (shows which groups differ):[/bold]"
        )
        console.print(str(tukey_result))

    else:
        console.print(
            f"  [yellow]❌ The p-value is greater than {alpha}, indicating no statistically significant difference was found among the groups.[/yellow]"
        )

    console.rule()


def main():
    """Main function to orchestrate the analysis."""
    try:
        engine = get_db_connection()
        data = fetch_experiment_data(engine, "Sugarscape - Cognitive Strategy Analysis")

        if data.empty or len(data["agent_group"].unique()) < 4:
            print(
                "Could not find sufficient data for all four experimental groups. Analysis cannot proceed."
            )
            print("Found data:")
            print(data)
            return

        print_analysis_summary(data)

    except Exception as e:
        print(f"\nAn error occurred during analysis: {e}")


if __name__ == "__main__":
    main()
