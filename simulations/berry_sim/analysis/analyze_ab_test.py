# FILE: scripts/analyze_ab_test.py
"""
Performs a statistical analysis of the Berry Sim A/B test results.

This script connects to the project's PostgreSQL database, fetches the final
average health for each simulation run from the most recent experiment,
and conducts an independent samples t-test to determine if the performance
difference between the Causal Agent and the Baseline Agent is statistically
significant.

Sample Usage:
    # Run from within the Docker container to ensure DB access
    docker compose exec app poetry run python scripts/analyze_ab_test.py
"""

import os
from typing import Dict, Tuple

import pandas as pd
from dotenv import load_dotenv
from scipy.stats import ttest_ind
from sqlalchemy import create_engine, text


def get_db_connection():
    """
    Creates and returns a SQLAlchemy engine connected to the database.

    Loads database credentials from the .env file.

    Returns:
        A SQLAlchemy Engine instance.
    """
    # Load environment variables from the project root .env file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(project_root, ".env"))

    db_url = os.getenv("DATABASE_URL", "").replace("postgresql+asyncpg", "postgresql")
    if not db_url:
        raise ValueError("DATABASE_URL not found in .env file.")

    return create_engine(db_url)


def fetch_experiment_data(engine, experiment_name: str) -> pd.DataFrame:
    """
    Fetches the final average health for each run in a given experiment.

    Args:
        engine: The SQLAlchemy engine for the database connection.
        experiment_name: The name of the experiment to analyze.

    Returns:
        A pandas DataFrame containing the agent type and final health for each run.
    """
    # This query selects data from the two most recent experiments
    # matching the name to correctly capture both halves of the A/B test.
    query = text(
        """
        WITH LatestExperimentIDs AS (
            -- Step 1: Find the IDs of the two most recent experiments for the A/B test.
            SELECT id
            FROM experiments
            WHERE name = :exp_name
            ORDER BY created_at DESC
            LIMIT 2
        ),
        FinalMetrics AS (
            -- Step 2: Get the metric value from the very last tick for each simulation run
            -- belonging to those experiments.
            SELECT DISTINCT ON (simulation_id)
                simulation_id,
                (data ->> 'average_agent_health')::float AS final_avg_health
            FROM metrics
            WHERE simulation_id IN (
                SELECT id FROM simulation_runs WHERE experiment_id IN (SELECT id FROM LatestExperimentIDs)
            )
            ORDER BY simulation_id, tick DESC
        )
        -- Step 3: Join the results to get the agent type and final health.
        SELECT
            CASE
                WHEN sr.config -> 'simulation' ->> 'enable_causal_system' = 'true'
                THEN 'Causal Agent'
                ELSE 'Baseline Agent'
            END AS agent_type,
            fm.final_avg_health
        FROM
            simulation_runs sr
        JOIN
            FinalMetrics fm ON sr.id = fm.simulation_id
        WHERE
            sr.experiment_id IN (SELECT id FROM LatestExperimentIDs);
        """
    )
    with engine.connect() as connection:
        df = pd.read_sql(query, connection, params={"exp_name": experiment_name})
    return df


def perform_t_test(
    df: pd.DataFrame, group_col: str, value_col: str
) -> Tuple[Dict[str, float], float, float]:
    """
    Performs an independent samples t-test between two groups in a DataFrame.

    Args:
        df: The DataFrame containing the data.
        group_col: The name of the column that defines the two groups.
        value_col: The name of the column with the values to compare.

    Returns:
        A tuple containing:
        - A dictionary of group means.
        - The calculated t-statistic.
        - The calculated p-value.
    """
    groups = df[group_col].unique()
    if len(groups) != 2:
        # Provide more debugging info in the error message
        error_msg = (
            f"Expected 2 groups for t-test, but found {len(groups)}. "
            f"Groups found: {list(groups)}. "
            "This may happen if one variation of the A/B test failed to produce metrics."
        )
        raise ValueError(error_msg)

    group1_data = df[df[group_col] == groups[0]][value_col]
    group2_data = df[df[group_col] == groups[1]][value_col]

    # Calculate means for reporting
    group_means = {
        groups[0]: group1_data.mean(),
        groups[1]: group2_data.mean(),
    }

    # Perform the t-test
    t_stat, p_value = ttest_ind(group1_data, group2_data, equal_var=False)

    return group_means, t_stat, p_value


def print_analysis_summary(
    group_means: Dict[str, float], t_stat: float, p_value: float, alpha: float = 0.05
):
    """
    Prints a formatted, human-readable summary of the t-test results.

    Args:
        group_means: A dictionary of the mean value for each group.
        t_stat: The t-statistic from the test.
        p_value: The p-value from the test.
        alpha: The significance level to test against.
    """
    print("\n--- A/B Test Statistical Analysis ---")
    print("\n📋 Group Averages (Final Health):")
    for group, mean in group_means.items():
        print(f"  - {group}: {mean:.2f}")

    print("\n🔬 T-Test Results:")
    print(f"  - T-Statistic: {t_stat:.4f}")
    print(f"  - P-Value: {p_value:.4f}")

    print("\n💡 Conclusion:")
    if p_value < alpha:
        print(
            f"  The p-value ({p_value:.4f}) is less than our significance level ({alpha})."
        )
        print(
            "  ✅ We can conclude that there is a **statistically significant** difference"
        )
        print("     in the average final health between the two agent types.")
    else:
        print(
            f"  The p-value ({p_value:.4f}) is greater than our significance level ({alpha})."
        )
        print(
            "  ❌ We **cannot** conclude that there is a statistically significant difference"
        )
        print("     in the average final health between the two agent types.")
    print("-------------------------------------\n")


def main():
    """
    Main function to orchestrate the analysis.
    """
    try:
        engine = get_db_connection()
        data = fetch_experiment_data(engine, "Berry Sim - Causal Agent A/B Test")

        if data.empty or len(data["agent_type"].unique()) < 2:
            print("Could not find data for both agent types. Found data:")
            print(data)
            return

        means, t_stat, p_value = perform_t_test(
            data, group_col="agent_type", value_col="final_avg_health"
        )
        print_analysis_summary(means, t_stat, p_value)

    except Exception as e:
        print(f"\nAn error occurred during analysis: {e}")


if __name__ == "__main__":
    main()
