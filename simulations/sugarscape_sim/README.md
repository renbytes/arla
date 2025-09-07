# Sugarscape Simulation

This directory contains a complete implementation of the classic [Sugarscape model](https://en.wikipedia.org/wiki/Sugarscape) using the ARLA framework.

## What is the Sugarscape Model?

The Sugarscape model, originally developed by Joshua M. Epstein and Robert Axtell, is a foundational agent-based model used to explore how complex social phenomena like trade, conflict, and wealth distribution can emerge from simple agent rules in a resource-constrained environment.

The core idea is that agents live on a 2D grid where a resource ("sugar") is unevenly distributed in two nutrient-rich peaks. Each agent has a vision range and a metabolic rate, causing them to burn energy each turn. To survive, they must move across the landscape to find and harvest sugar. If an agent's energy drops to zero, it "dies" and is removed from the simulation. This simple setup leads to the emergent behavior of agents migrating and clustering on the resource peaks.

## Run Book

Follow these steps to run the simulation and visualize the results.

### Step 1: Start the Environment

First, build the Docker images, start all services (PostgreSQL, MLflow, etc.), and initialize the database tables.

```bash
# Start all services in the background
make up

# Create the necessary tables in the database (only needs to be run once)
make init-db
```

### Step 2: Run the Simulation

Execute the run-local command. This will run the Sugarscape simulation for 1000 steps with the default "forager" agents. Frames for the visualization will be saved automatically.

```bash
make run-local \
  PACKAGE="simulations.sugarscape_sim" \
  CONFIG="simulations/sugarscape_sim/config/config.yml" \
  FILE="simulations/sugarscape_sim/scenarios/default.json" \
  STEPS=1000
```

Or use the command below to run the full A/B test:
```bash
make run FILE=simulations/sugarscape_sim/experiments/sugarscape_ab_test.yml WORKERS=6
```

### Step 3: Generate the Visualization

After the simulation is complete, create the animated GIF from the saved frames. You will need to get the RUN_ID from the MLflow UI for the simulation you just ran.

```bash
make make-gif RENDER_DIR=data/gif_renders/sugarscape_sim RUN_ID=<your_run_id_here>
```

This will create a GIF file named after the run ID in the project's root directory.

### Step 3. Run the A/B test analysis

Run the following the run the A/B test analysis:
```bash
docker compose exec app poetry run python simulations/sugarscape_sim/analysis/analyze_sugarscape.py
```

## Evaluating the Results

There are two primary ways to analyze the outcome of your simulation run.

### 1. MLflow Dashboard

For quantitative analysis, the MLflow UI provides detailed graphs of the simulation's metrics over time.

**How to Access:** Open your browser and go to http://localhost:5001.

**What to Look For:** Navigate to the simulations.sugarscape_sim-local experiment. You should see the following patterns in the "Model metrics" tab:

- **active_agents:** This graph will likely show an initial drop as agents with poor starting positions or strategies fail to find sugar and die off, eventually stabilizing as the surviving population finds the resource peaks.

- **average_agent_energy:** This will fluctuate as agents burn energy moving, find sugar, and consume it.

- **total_sugar_in_env:** This metric shows the dynamic between agent consumption and the environment's regeneration of sugar.

### 2. The Animated GIF

The animated GIF provides a powerful qualitative view of the emergent behavior.

**What the Colors Mean:**

- **Shades of Yellow:** The sugar distribution. The brightest yellow areas are the resource peaks with the most sugar.
- **White Squares:** The agents.
- **Dark Grey:** Empty cells with no sugar.

**What to Look For:** The animation should start with agents scattered randomly. As it plays, you will see them migrate from the resource-poor areas toward the two bright yellow "sugar peaks." Over time, a stable population will form, clustered on these peaks, consuming the sugar as it regenerates.

## Experiment Further

Now that you have a working baseline, try changing the parameters to see how they affect the outcome!

**Change the Agent Archetype:** Edit `simulations/sugarscape_sim/scenarios/default.json`. Change the `agent_distribution` to use 50 "rusher" agents instead of "forager" agents. Rushers have low vision but a lower metabolic rate. How does this different strategy affect the population's survival rate?

**Change the Environment:** Edit `simulations/sugarscape_sim/config/config.yml` and change the `sugar_regeneration_rate` from 1 to 0 to create a world with finite resources. How long can the population survive?
