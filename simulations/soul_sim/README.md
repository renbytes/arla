# Soul-Sim

Soul-Sim is a detailed agent-based simulation built on the ARLA framework. It explores emergent social dynamics, learning, and identity formation in a resource-constrained world.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd soul-sim
    ```

2.  **Install in editable mode:**
    This command installs the project and its development dependencies. Using editable mode (`-e`) means that changes to the source code will be immediately available without reinstalling.
    ```bash
    pip install -e ".[dev]"
    ```

## Running the Simulation

The simulation is launched using Hydra for configuration management.

**Default Run:**
```bash
python src/main.py --config-path=src/config --config-name=default
```

**Overriding Parameters:**
You can override any parameter from the configuration file directly from the command line.
```bash
python src/main.py --config-path=src/config --config-name=default simulation.steps=50 agent.vitals.initial_health=150
```
