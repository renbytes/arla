# Running Simulations

The ARLA framework provides two primary methods for running simulations: a simple command for local testing and a powerful experiment runner for large-scale, reproducible research.

## 1. Running a Single, Local Simulation

For quick testing, debugging, or development, the `make run-example` command is the easiest way to start a simulation.

```bash
make run-example
```

This command executes the `agent_sim.main` script inside the Docker container, using a default scenario and configuration file. It's the perfect way to see your changes in action immediately.

### Customizing the Local Run

You can also run the simulation manually to specify a different scenario, configuration file, or number of steps.

```bash
docker compose exec app python -m agent_sim.main \
  --scenario simulations/soul_sim/scenarios/your_scenario.json \
  --config simulations/soul_sim/config/your_config.yml \
  --steps 100
```

## 2. Running a Full Experiment

For rigorous research, you'll want to use the experiment runner. This system uses **Celery** to manage a queue of simulation jobs, allowing you to run many simulations in parallel across multiple workers.

### How It Works

You define an experiment in a `.yml` file, specifying:

- A base configuration file.
- One or more scenarios to run.
- A list of "variations" that override the base configuration.
- The number of times to repeat each run.

The experiment runner then creates a unique job for every combination of scenario and variation and sends it to the Celery workers for execution.

### Step 1: Define Your Experiment

Create a new `.yml` file in the `experiments/` directory (e.g., `experiments/my_study.yml`).

Here is an example that tests the effect of agent lifespan on the simulation outcome:

```yaml
# experiments/my_study.yml

experiment_name: "Lifespan Ablation Study"
base_config_path: "simulations/soul_sim/config/base_config.yml"
simulation_package: "simulations.soul_sim"
scenarios:
  - "simulations/soul_sim/scenarios/default.json"
runs_per_scenario: 5

variations:
  - name: "Short Lifespan"
    overrides:
      agent:
        foundational:
          lifespan_std_dev_percent: 0.5
  - name: "Long Lifespan"
    overrides:
      agent:
        foundational:
          lifespan_std_dev_percent: 0.1
```

### Step 2: Start the Workers

Before you can submit the experiment, you need to start the Celery workers that will process the jobs.

```bash
# This starts the worker service defined in your docker-compose.yml
docker compose up -d worker
```

You can scale up the number of workers to run more simulations in parallel:

```bash
docker compose up -d --scale worker=4
```

### Step 3: Submit the Experiment

Once the workers are running, use the `run-experiment` command to submit your experiment file to the queue.

```bash
docker compose exec app poetry run agentsim run-experiment experiments/my_study.yml
```

The system will then distribute the simulation jobs to the available workers. All results, including logs, manifests, and MLflow data, will be automatically tracked and saved.
