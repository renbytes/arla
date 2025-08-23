# Running Simulations & Experiments

The ARLA framework provides two distinct workflows for running simulations, each tailored to a specific need: quick, iterative development and large-scale, rigorous experimentation. Understanding when to use each is key to an efficient research process.

## 1. Local Development: `make run-local`

For day-to-day development, debugging, or visualizing a single simulation run, the `run-local` command is your primary tool. It executes a single simulation directly inside the main `app` container, bypassing the distributed Celery workers.

### When to Use It

- Testing a new feature or system
- Debugging an agent's behavior
- Quickly generating a GIF for a specific scenario
- Running a one-off simulation without the overhead of the experiment queue

### How to Use It

The `run-local` command requires you to specify the simulation's package, configuration file, and scenario file.

```bash
make run-local \
  PACKAGE="simulations.berry_sim" \
  CONFIG="simulations/berry_sim/config/config.yml" \
  FILE="simulations/berry_sim/scenarios/default.json"
```

## 2. Research & A/B Tests: `make run`

For formal research, A/B tests, or any large-scale data collection, the `run` command provides a powerful, distributed workflow. It uses Celery to manage a queue of simulation jobs, allowing you to run many simulations in parallel across multiple worker processes.

### When to Use It

- Running ablation studies with multiple variations
- Executing a scenario multiple times with different random seeds for statistical significance
- Long-running simulations that you want to execute in the background

### How to Use It

This command requires a single `FILE` argument pointing to an experiment definition YAML file. This file orchestrates the entire experiment.

```bash
# This will automatically start 4 workers and submit the experiment
make run FILE=simulations/berry_sim/experiments/causal_ab_test.yml

# To run with more parallel processes, specify the number of workers
make run FILE=simulations/berry_sim/experiments/causal_ab_test.yml WORKERS=8
```

## 3. Analyzing Your Results

Whether you run a local simulation or a full experiment, ARLA produces a consistent set of outputs designed for comprehensive analysis.

- **MLflow UI (http://localhost:5001)**: Your first stop for high-level analysis. Use it to compare metrics between different runs in an experiment, view parameters, and get a quick overview of the results.
- **PostgreSQL Database**: For deep, granular analysis. The database contains a complete, tick-by-tick record of every agent's state and every action taken. Use a SQL client to connect to the database on port `5432` and run custom queries.
- **Run Directory (`data/logs/`)**: Each simulation creates a unique directory containing artifacts for reproducibility, including the exact configuration file used and periodic snapshots of the simulation state.

## 4. Visualization: Creating a GIF

To create a visual representation of a simulation, you first need to enable rendering in its configuration file.

1. **Enable Rendering**: In your simulation's `config.yml`, set `rendering.enabled` to `true`.
2. **Run the Simulation**: Run either a `local-run` or a full `run`. The image frames will be saved to a unique subdirectory named after the `Run ID`.
3. **Find the Run ID**: Go to the MLflow UI to find the `Run ID` of the specific simulation you want to visualize.
4. **Generate the GIF**: Use the `make-gif` command, providing the base render directory and the specific `RUN_ID`.

```bash
# Example for a berry_sim run
make make-gif RENDER_DIR=data/gif_renders/berry_sim RUN_ID=d6e572a855844e21a13e0e85b254fa96
```

This will create a uniquely named GIF (e.g., `simulation-71bcad3e64a346618715e3b8be195e16.gif`) in the project root.
