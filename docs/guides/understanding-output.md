# Understanding the Simulation Output

After running a simulation, the ARLA framework generates a set of files and data designed to make your results transparent, reproducible, and easy to analyze. This guide explains the key outputs and where to find them.

## 1. The Run Directory

Every simulation run creates a unique, timestamped directory inside `data/logs/`. This directory is the central location for all artifacts related to that specific run.

A typical run directory looks like this:

```
data/logs/
└── 20250818_041805_d5c8f0ec-b0bd-4f0f-91f6-b35941f6ea51/
    ├── manifest.json
    ├── resolved_config.yml
    └── snapshots/
        └── snapshot_tick_50.json
```

- **`manifest.json`**: A critical file for reproducibility. It contains essential metadata about the run, including the unique run ID, the Git commit hash, and the random seed used.
- **`resolved_config.yml`**: An exact copy of the configuration used for the run, including all base settings and any overrides from an experiment file.
- **`snapshots/`**: This directory contains full snapshots of the simulation state, saved periodically. These can be used to resume a simulation or to analyze the state at a specific point in time.

## 2. The Database

The most detailed, granular data from your simulation is stored in a PostgreSQL database. This includes:

- **Agent States**: A snapshot of every agent's components at every single tick.
- **Events**: A log of every action taken by every agent, including the outcome and reward.
- **LLM Interactions**: A complete record of every prompt sent to the LLM and the corresponding response.
- **Metrics**: Aggregated, simulation-wide metrics for each tick.

You can connect to this database using any standard SQL client to perform detailed analysis.

## 3. MLflow Tracking

For high-level analysis and comparison between runs, ARLA is integrated with **MLflow**. The MLflow UI provides a powerful way to visualize and compare the results of your experiments.

### Accessing the MLflow UI

While your Docker services are running, you can access the MLflow UI in your web browser at:

[**http://localhost:5001**](http://localhost:5001)

### What to Look For

- **Experiments**: Your runs will be grouped by the `experiment_name` defined in your experiment file.
- **Parameters**: MLflow logs all the configuration parameters for each run, making it easy to see what changed between different variations.
- **Metrics**: Key performance indicators, like average agent reward or total resources, are plotted over time. This is the best way to visualize the high-level dynamics of your simulation.
- **Tags**: Important metadata, like the run ID and status, are stored as tags.

Using these three outputs—the run directory, the database, and the MLflow UI—you have a complete and auditable record of every simulation you run.
