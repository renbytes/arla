# Schelling Segregation Model Simulation

This directory contains a complete implementation of the Schelling Segregation Model using the ARLA framework.

## What is the Schelling Model?

The Schelling Model is a classic agent-based model that demonstrates how large-scale segregation can emerge from the simple, individual preferences of agents.

The core idea is simple: agents of different groups (e.g., blue and red) are placed on a grid. Each agent is "satisfied" only if a certain percentage of its neighbors are from the same group. If an agent is unsatisfied, it moves to a random empty location. The simulation shows that even with a low satisfaction threshold (e.g., "I'm happy if only 40% of my neighbors are like me"), the grid will quickly evolve from a random mix into highly segregated clusters.

## 🏃‍♀️ Run Book

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

Execute the local run command. This will run the Schelling simulation for 75 steps, which is enough to see the pattern emerge. The frames for the visualization will be saved automatically.

```bash
make run-local \
  PACKAGE="simulations.schelling_sim" \
  CONFIG="simulations/schelling_sim/config/config.yml" \
  FILE="simulations/schelling_sim/scenarios/default.json" \
  STEPS=150
```

### Step 3: Generate the Visualization

After the simulation is complete, create the animated GIF from the saved frames.

```bash
make make-gif RENDER_DIR=data/gif_renders/schelling_sim
```

This will create a `simulation.gif` file in the root of the project.

## 📊 Evaluating the Results

There are two primary ways to analyze the outcome of your simulation run.

### 1. MLflow Dashboard

For quantitative analysis, the MLflow UI provides detailed graphs of the simulation's metrics over time.

* **How to Access**: Open your browser and go to **http://localhost:5001**.
* **What to Look For**: Navigate to the `simulations.schelling_sim-local` experiment. You should see the following patterns in the "Model metrics" tab:
   * `satisfaction_rate`: This graph should start very low and then shoot up rapidly, stabilizing around 70-80% within the first 30 steps. This shows the system reaching equilibrium.
   * `segregation_index`: This graph should do the opposite. It will start high (near 1.0) and drop quickly as agents move, eventually stabilizing. This visually confirms the model is working.

### 2. The Animated GIF

The `simulation.gif` provides a powerful qualitative view of the emergent behavior.

* **What the Colors Mean**:
   * **Blue / Red**: Agents belonging to one of the two groups.
   * **Dark Gray**: Empty cells on the grid.
* **What to Look For**: The animation should start with a random, "salt-and-pepper" distribution of red and blue agents. As it plays, you will see the agents quickly self-organize, moving away from dissimilar neighbors until distinct, segregated clusters of blue and red have formed. The animation will loop, allowing you to see this emergent pattern repeat.

## 🔬 Experiment Further

Now that you have a working baseline, try changing the parameters to see how they affect the outcome!

* **Change the Satisfaction Threshold**: Edit `simulations/schelling_sim/config/config.yml` and change `satisfaction_threshold` from `0.6` to `0.4`. Does the grid become more or less segregated?
* **Change the Agent Count**: Edit `simulations/schelling_sim/scenarios/default.json` and change `num_agents` to a lower value. How does density affect the final pattern?
