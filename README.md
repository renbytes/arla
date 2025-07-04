# Agent Persist

**A robust, type-safe library for persisting and restoring agent-based simulation states.**

`agent-persist` provides the critical infrastructure for saving and loading the complete state of a simulation. By leveraging Pydantic for data validation and serialization, it ensures that saved data is structured, correct, and can be reliably restored for analysis, debugging, or resuming a simulation.

---

### Key Features

- **Robust Data Validation:** Built on `pydantic`, ensuring that all saved and loaded data conforms to a strict, well-defined schema. This prevents data corruption and runtime errors.
- **Clear Schema:** Defines a clear, hierarchical data model (`SimulationSnapshot`, `AgentSnapshot`, `ComponentSnapshot`) that is self-documenting and easy to understand.
- **Extensible Store Interface:** Provides an abstract `StateStore` base class, making it easy to implement new storage backends (e.g., a database, cloud storage) in the future.
- **File-Based Implementation:** Includes a ready-to-use `FileStateStore` that handles saving snapshots to and loading from local JSON files.

---

### Installation

You can install the library directly from PyPI using pip:

```bash
pip install agent-persist
```

## Development Setup

To set up a local development environment, clone the repository and install it in editable mode with its development dependencies.

1. **Clone the repository:**

```bash
git clone https://github.com/renbytes/agent-persist.git
cd agent-persist
```

2. **Create and activate a virtual environment (recommended):**

```bash
# Using venv
python -m venv venv
source venv/bin/activate

# Or using Conda
conda create --name agent-persist-dev python=3.9
conda activate agent-persist-dev
```

3. **Install the package in editable mode with dev dependencies:** This command installs the package and includes tools like `pytest` and `mypy`.

```bash
pip install -e .[dev]
```

## Basic Usage

The library is designed to be straightforward. You create a `SimulationSnapshot` model with your simulation's data and use a `StateStore` to save or load it.

```python
from pathlib import Path
from agent_persist import FileStateStore, SimulationSnapshot

# 1. Create a snapshot of your simulation state
snapshot_data = SimulationSnapshot(
    simulation_id="sim_alpha_42",
    current_tick=250,
    agents=[
        {
            "agent_id": "agent_x",
            "components": [
                {
                    "component_type": "IdentityComponent",
                    "data": {"stability": 0.85}
                },
                {
                    "component_type": "TimeBudgetComponent",
                    "data": {"current_time_budget": 750}
                }
            ]
        }
    ],
    environment_state={"world_time": "night"}
)

# 2. Initialize a store with a file path
snapshot_path = Path("./simulation_saves/snapshot_tick_250.json")
store = FileStateStore(file_path=snapshot_path)

# 3. Save the snapshot
store.save(snapshot_data)

# 4. Load the snapshot back at a later time
loaded_snapshot = store.load()

print(f"Loaded simulation '{loaded_snapshot.simulation_id}' from tick {loaded_snapshot.current_tick}.")
```

## Running Tests and Type Checking

To ensure code quality and correctness, you can run the included unit tests and static type checker.

1. **Run Unit Tests:** From the root of the project directory, run `pytest`:

```bash
pytest
```

2. **Run Static Type Checker:** To check for type errors with `mypy`, run:

```bash
mypy src
```

3. **Install Pre-Commit Hooks:** To check errors prior to pushing code, run:

```bash
pre-commit install
```
