"""
Analyzes the output of the Cognitive Voter Model simulation from the database.

This script connects to the simulation's database (SQLite or PostgreSQL),
extracts agent state data, and generates a series of visualizations to analyze
the emergent social dynamics, focusing on opinion polarization, spatial
clustering, and the relationship between identity and belief stability.
"""

import os
import random
import traceback
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


class SimulationAnalyzer:
    """
    A class to encapsulate the analysis of a single simulation run.
    """

    def __init__(self, db_url: str, output_dir: str = "analysis/output"):
        """
        Initializes the analyzer with the database connection URL and output path.

        Args:
            db_url (str): The full SQLAlchemy connection URL for the database.
            output_dir (str): Directory to save the generated plots.
        """
        self.db_url = db_url
        self.output_dir = output_dir
        try:
            self.engine = create_engine(db_url)
            # Test the connection
            with self.engine.connect() as _connection:
                pass
        except Exception as e:
            print(f"Error connecting to the database at {db_url}")
            raise e

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Analyzer initialized. Output will be saved to: {self.output_dir}")

    def run_full_analysis(self, run_id: Optional[str] = None):
        """
        Executes all planned analyses for a specific simulation run.

        Args:
            run_id (Optional[str]): The UUID of the simulation run to analyze.
                                    If None, the most recent run is used.
        """
        if run_id is None:
            run_id = self._get_most_recent_run_id()
            if run_id is None:
                print("No simulation runs found in the database.")
                return
        print(f"\n--- Analyzing Simulation Run: {run_id} ---")

        # 1. Fetch and process data
        df = self._fetch_and_process_data(run_id)
        if df.empty:
            print(f"No data found for run_id: {run_id}")
            return

        # 2. Run individual analyses with detailed error handling
        plot_functions = [
            self.plot_opinion_polarization,
            self.plot_spatial_clustering,
            self.plot_identity_vs_opinion_change,
            self.plot_individual_trajectories,
        ]
        for plot_func in plot_functions:
            try:
                plot_func(df.copy(), run_id)  # Pass a copy to prevent side effects
            except Exception as e:
                print(f"\n--- ERROR DURING PLOTTING: {plot_func.__name__} ---")
                print(f"An error occurred: {e}")
                print(traceback.format_exc())
                print("---------------------------------------------------\n")
                # Continue to the next plot instead of stopping
                continue

        print(f"\n--- Analysis complete for run: {run_id} ---")

    def _get_most_recent_run_id(self) -> Optional[str]:
        """Fetches the UUID of the most recently created simulation run."""
        query = "SELECT id FROM simulation_runs ORDER BY created_at DESC LIMIT 1"
        with self.engine.connect() as connection:
            result = connection.execute(text(query)).scalar_one_or_none()
        return str(result) if result else None

    def _fetch_and_process_data(self, run_id: str) -> pd.DataFrame:
        """
        Fetches agent state data and processes it into a structured DataFrame.

        Args:
            run_id (str): The UUID of the simulation run.

        Returns:
            pd.DataFrame: A DataFrame with columns for tick, agent_id, opinion,
                          position, and identity coherence.
        """
        print("Fetching and processing data from database...")
        # CORRECTED QUERY: Cast directly to ::int and ::float in the SQL query.
        query = f"""
            SELECT tick, agent_id,
                   components_data -> 'OpinionComponent' ->> 'opinion' AS opinion,
                   (components_data -> 'PositionComponent' ->> 'position_x')::int AS pos_x,
                   (components_data -> 'PositionComponent' ->> 'position_y')::int AS pos_y,
                   (components_data -> 'IdentityComponent' ->> 'identity_coherence')::float AS identity_coherence
            FROM agent_states
            WHERE simulation_id = '{run_id}'
        """
        if "sqlite" in self.db_url:
            query = f"""
                SELECT tick, agent_id, json_extract(components_data, '$.OpinionComponent.opinion') AS opinion,
                       json_extract(components_data, '$.PositionComponent.position_x') AS pos_x,
                       json_extract(components_data, '$.PositionComponent.position_y') AS pos_y,
                       json_extract(components_data, '$.IdentityComponent.identity_coherence') AS identity_coherence
                FROM agent_states
                WHERE simulation_id = '{run_id}'
            """
        df = pd.read_sql(query, self.engine)

        if df.empty:
            return df

        # --- Data Cleaning and Debugging ---
        print("\n--- Data Info Before Cleaning ---")
        df.info()
        print("---------------------------------\n")

        # Explicitly convert columns to numeric, coercing errors to NaN
        df["pos_x"] = pd.to_numeric(df["pos_x"], errors="coerce")
        df["pos_y"] = pd.to_numeric(df["pos_y"], errors="coerce")
        df["identity_coherence"] = pd.to_numeric(df["identity_coherence"], errors="coerce")

        # Report and drop rows with conversion errors
        invalid_rows = df[df.isnull().any(axis=1)]
        if not invalid_rows.empty:
            print(f"Warning: Found {len(invalid_rows)} rows with invalid data that will be dropped.")
        df.dropna(subset=["pos_x", "pos_y", "identity_coherence"], inplace=True)

        # Ensure position columns are integers for grid plotting
        if not df.empty:
            df["pos_x"] = df["pos_x"].astype(int)
            df["pos_y"] = df["pos_y"].astype(int)

        print("\n--- Data Info AFTER Cleaning ---")
        df.info()
        print("--------------------------------\n")

        # FIX: Calculate opinion changes properly for string opinions
        # First, sort by agent_id and tick to ensure proper chronological order
        df_sorted = df.sort_values(["agent_id", "tick"])

        # Create a shifted opinion column to compare with previous value
        df_sorted["prev_opinion"] = df_sorted.groupby("agent_id")["opinion"].shift(1)

        # Calculate opinion changes: 1 if opinion changed from previous tick, 0 otherwise
        # For the first tick of each agent, prev_opinion will be NaN, so we mark as 0 (no change)
        df_sorted["opinion_changed"] = (
            (df_sorted["opinion"] != df_sorted["prev_opinion"]) & (df_sorted["prev_opinion"].notna())
        ).astype(int)

        # Drop the temporary column and restore original order
        df = df_sorted.drop("prev_opinion", axis=1).sort_index()

        print(f"Data processed. Found {len(df)} valid state records.")
        return df

    def plot_opinion_polarization(self, df: pd.DataFrame, run_id: str):
        """
        Plots the percentage of agents holding each opinion over time.
        """
        print("Generating opinion polarization plot...")
        opinion_counts = df.groupby(["tick", "opinion"]).size().unstack(fill_value=0)
        opinion_percentages = opinion_counts.divide(opinion_counts.sum(axis=1), axis=0)

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))
        opinion_percentages.plot(ax=ax, color=["#3498db", "#e67e22"])

        ax.set_title(f"Opinion Polarization Over Time\n(Run: {run_id[:8]})", fontsize=16)
        ax.set_xlabel("Simulation Tick", fontsize=12)
        ax.set_ylabel("Percentage of Population", fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(title="Opinion")

        filename = os.path.join(self.output_dir, f"{run_id}_polarization.png")
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"Saved: {filename}")

    def plot_spatial_clustering(self, df: pd.DataFrame, run_id: str):
        """
        Generates grid plots showing opinion clusters at different ticks.
        """
        print("Generating spatial clustering plots...")
        ticks_to_plot = [
            df["tick"].min(),
            df["tick"].median(),
            df["tick"].max(),
        ]
        grid_size = (int(df["pos_x"].max() + 1), int(df["pos_y"].max() + 1))
        color_map = {"Blue": "#3498db", "Orange": "#e67e22"}

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        fig.suptitle(f"Spatial Clustering of Opinions\n(Run: {run_id[:8]})", fontsize=18)

        for i, tick in enumerate(ticks_to_plot):
            ax = axes[i]
            tick_data = df[df["tick"] == tick]
            ax.set_title(f"Tick: {int(tick)}", fontsize=14)
            ax.set_xticks(range(grid_size[1]))
            ax.set_yticks(range(grid_size[0]))
            ax.grid(True, which="both", color="gray", linestyle="-", linewidth=0.5)

            # ADDED: Specific try-except block for debugging the exact failing line.
            for _index, row in tick_data.iterrows():
                try:
                    # Ensure types are correct right before use
                    pos_y = int(row["pos_y"])
                    pos_x = int(row["pos_x"])
                    opinion = row["opinion"]

                    rect = plt.Rectangle(
                        (pos_y - 0.5, pos_x - 0.5),
                        1,
                        1,
                        color=color_map.get(opinion, "gray"),
                    )
                    ax.add_patch(rect)
                except (TypeError, ValueError) as e:
                    print("\n--- DEBUG: Error caught in plot_spatial_clustering ---")
                    print(f"Error: {e} on tick {tick}")
                    print("Problematic row data:")
                    print(row)
                    print(f"Type of pos_y: {type(row['pos_y'])}")
                    print(f"Type of pos_x: {type(row['pos_x'])}")
                    print("----------------------------------------------------------\n")
                    # Continue to the next row to see if there are more errors
                    continue
            ax.set_aspect("equal")
            ax.invert_yaxis()

        filename = os.path.join(self.output_dir, f"{run_id}_clustering.png")
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"Saved: {filename}")

    def plot_identity_vs_opinion_change(self, df: pd.DataFrame, run_id: str):
        """
        Plots average identity coherence vs. the rate of opinion changes.
        """
        print("Generating identity coherence vs. opinion change plot...")
        analysis_df = (
            df.groupby("tick")
            .agg(
                avg_coherence=("identity_coherence", "mean"),
                total_changes=("opinion_changed", "sum"),
            )
            .reset_index()
        )

        # Use a rolling average to smooth the opinion change data
        analysis_df["changes_rolling_avg"] = analysis_df["total_changes"].rolling(window=25, min_periods=1).mean()

        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax1.set_title(
            f"Identity Coherence vs. Rate of Opinion Change\n(Run: {run_id[:8]})",
            fontsize=16,
        )
        ax1.set_xlabel("Simulation Tick", fontsize=12)

        # Plot Identity Coherence on the left y-axis
        color1 = "tab:red"
        ax1.set_ylabel("Average Identity Coherence", color=color1, fontsize=12)
        ax1.plot(
            analysis_df["tick"],
            analysis_df["avg_coherence"],
            color=color1,
            label="Avg. Identity Coherence",
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_ylim(0, 1)

        # Plot Opinion Changes on the right y-axis
        ax2 = ax1.twinx()
        color2 = "tab:blue"
        ax2.set_ylabel("Opinion Changes (25-tick Rolling Avg)", color=color2, fontsize=12)
        ax2.plot(
            analysis_df["tick"],
            analysis_df["changes_rolling_avg"],
            color=color2,
            label="Opinion Changes",
        )
        ax2.tick_params(axis="y", labelcolor=color2)

        fig.tight_layout()
        filename = os.path.join(self.output_dir, f"{run_id}_identity_vs_change.png")
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"Saved: {filename}")

    def plot_individual_trajectories(self, df: pd.DataFrame, run_id: str):
        """
        Plots the opinion history for a small, random sample of agents.
        """
        print("Generating individual agent trajectories plot...")
        agent_ids = df["agent_id"].unique()
        sample_size = min(len(agent_ids), 4)
        sample_agents = random.sample(list(agent_ids), sample_size)

        sample_df = df[df["agent_id"].isin(sample_agents)]
        opinion_map = {"Blue": 1, "Orange": 0}
        sample_df["opinion_numeric"] = sample_df["opinion"].map(opinion_map)

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(
            data=sample_df,
            x="tick",
            y="opinion_numeric",
            hue="agent_id",
            ax=ax,
            marker="o",
            markersize=4,
            linestyle="",
        )

        ax.set_title(
            f"Opinion Trajectories for Sample Agents\n(Run: {run_id[:8]})",
            fontsize=16,
        )
        ax.set_xlabel("Simulation Tick", fontsize=12)
        ax.set_ylabel("Opinion", fontsize=12)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Orange", "Blue"])
        ax.legend(title="Agent ID")

        filename = os.path.join(self.output_dir, f"{run_id}_agent_trajectories.png")
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"Saved: {filename}")


if __name__ == "__main__":
    # --- Configuration ---
    # Automatically load environment variables from a .env file
    load_dotenv()

    # NOTE: For PostgreSQL, you may need to install the driver: pip install psycopg2-binary
    # The script now reads the database URL from an environment variable.
    DATABASE_URL = os.getenv("LOCAL_DATABASE_URL", "sqlite:///data/db/agent_sim.db").replace(
        "postgresql+asyncpg", "postgresql"
    )

    # Optional: Specify a run ID. If None, the latest run will be analyzed.
    SPECIFIC_RUN_ID = "99c8f92b-371c-4a3b-8cc6-ae9a70495e41"

    # --- Execution ---
    try:
        analyzer = SimulationAnalyzer(db_url=DATABASE_URL)
        analyzer.run_full_analysis(run_id=SPECIFIC_RUN_ID)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please ensure your database is running and the connection URL is correct.")
        print("For PostgreSQL, you may need to run: pip install psycopg2-binary")
