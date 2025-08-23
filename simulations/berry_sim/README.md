# Berry Toxicity Experiment

This directory contains a complete implementation of the Berry Toxicity Experiment using the ARLA framework. The simulation is designed not just to observe what agents learn, but to probe how they learn by testing their ability to distinguish true causation from simple correlation.

## What is the Berry Toxicity Experiment?

This experiment serves as a benchmark for testing and validating an agent's cognitive architecture, particularly its capacity for causal reasoning. Agents exist in a grid world where they must eat berries to survive, but the berries have complex, contextual rules:

- **Red Berries**: Always safe and provide a positive health boost.
- **Yellow Berries**: Their effect is truly random (sometimes good, sometimes bad) and they spawn near rocks, creating a potential confounding variable for the agent to incorrectly learn from.
- **Blue Berries**: These are the core of the causal test. For the first 1,000 simulation steps, they are always safe. Afterwards, they become toxic, but only when near a water source. A simple agent might only learn "blue berry = good," while a more advanced agent should learn the new contextual rule.

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

You can run the simulation in two ways: a single local run for quick testing, or a full-scale experiment with multiple variations.

#### Option A: Run a Single Local Simulation

This command runs the baseline "heuristic" agent directly, bypassing the Celery workers. It's ideal for quick debugging and development.

```bash
make run-local \
  PACKAGE="simulations.berry_sim" \
  CONFIG="simulations/berry_sim/config/config.yml" \
  FILE="simulations/berry_sim/scenarios/default.json"
```

#### Option B: Run the Full A/B Test Experiment

This command uses the updated run target to automatically start the Celery workers and submits the A/B test comparing the baseline agent to the advanced causal agent.

```bash
make run FILE=simulations/berry_sim/experiments/causal_ab_test.yml
```

### Step 3: Generate the Visualization

After a local or experimental run is complete, create the animated GIF from the saved frames.

```bash
make make-gif RENDER_DIR=data/gif_renders/berry_sim
```

This will create a `simulation.gif` file in the root of the project.

## Evaluating the Results

### 1. MLflow Dashboard

For quantitative analysis, the MLflow UI provides detailed graphs of the simulation's metrics.

**How to Access**: Open your browser and go to http://localhost:5001.

**What to Look For**: Navigate to the "Berry Sim - Causal Agent A/B Test" experiment. Compare the runs from the "Baseline-Heuristic-Agent" and "Causal-QLearning-Agent" variations.

- **average_agent_health**: The baseline agent's health will dip sharply after tick 1000 when the blue berries become toxic. The causal agent should adapt and maintain a higher average health.
- **causal_understanding_score**: This is the key metric. The baseline agent's score should be low (near random guessing), while the causal agent's score should be high, indicating it correctly learned the new rule about water.
- **correlation_confusion_index**: This metric, which ranges from 0.0 (perfect knowledge) to 1.0 (max confusion), measures if agents are fooled by the random yellow berries.

### 2. The Animated GIF

The `simulation.gif` provides a qualitative view of the emergent behavior.

**What the Colors Mean**:
- White: Healthy Agents
- Red: Low-Health Agents
- Red/Blue/Yellow Circles: Berries
- Dark Blue: Water tiles
- Gray: Rock tiles

**What to Look For**: Watch the agents' behavior after tick 1000. Do the baseline agents continue to eat blue berries near water and turn red (low health)? Do the causal agents learn to avoid them?

## Experiment Further

- **Modify the Rules**: Edit `simulations/berry_sim/environment.py` to add more complex environmental rules or confounding variables and see if the causal agent can still figure them out.
- **A/B Test Other Systems**: Create a new experiment file in `simulations/berry_sim/experiments/` to test the impact of other cognitive systems from agent-engine, like the `AffectSystem` or `IdentitySystem`, on the agents' learning and survival.
