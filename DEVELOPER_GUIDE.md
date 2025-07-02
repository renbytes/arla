# AGENT Engine - Developer Guide

Welcome to the AGENT Engine! This guide provides all the necessary steps to set up your development environment, run tests, manage dependencies, and extend the engine with new cognitive systems.

## 1. Getting Started: Environment Setup

Follow these steps to get a clean, reproducible development environment.

### Step 1: Clone the Repository

First, clone the agent-engine repository to your local machine. You will also need to clone agent-core, as the engine depends on it.

```bash
# Clone the engine
git clone https://github.com/renbytes/agent-engine.git
cd agent-engine

# Clone the core library alongside it
git clone https://github.com/renbytes/agent-core.git ../agent-core
```

### Step 2: Create and Activate the Conda Environment

We use Conda to manage Python versions and environments, ensuring consistency across different machines.

Create the environment (you only need to do this once):

```bash
conda create --name agent-dev python=3.11
```

Activate the environment (do this every time you start a new terminal session):

```bash
conda activate agent-dev
```

Your terminal prompt should now start with `(agent-dev)`.

### Step 3: Install Dependencies

This project uses `pyproject.toml` to define its dependencies. We will install agent-core in "editable" mode so that changes you make there are immediately available to agent-engine.

**Install agent-core in editable mode:**
Navigate to the agent-core directory and run:

```bash
cd ../agent-core
pip install -e .
cd ../agent-engine
```

**Install agent-engine's dependencies:**
This command installs the main dependencies plus the development tools (pytest, ruff, mypy, pip-tools).

```bash
pip install ".[dev]"
```

## 2. Dependency Management

To ensure reproducible builds, we use pip-tools to generate pinned requirements.txt files from pyproject.toml.

### Workflow for Adding a New Dependency

1. Add the new package name to the dependencies list in `pyproject.toml`.

2. Run the pip-compile command to regenerate the requirements files with the new package and its sub-dependencies pinned to specific versions.

```bash
# For main dependencies
python -m piptools compile --output-file=requirements.txt pyproject.toml

# For development dependencies
python -m piptools compile --extra dev --output-file=requirements-dev.txt pyproject.toml
```

3. Commit the updated `pyproject.toml` and the newly generated `requirements.txt` / `requirements-dev.txt` files to version control.

## 3. Code Quality and Testing

### Pre-commit Hooks

This repository is configured with pre-commit hooks to automatically check and format your code every time you make a commit.

Install the hooks (only needs to be run once per clone):

```bash
pre-commit install
```

Commit your code:

```bash
git add .
git commit -m "feat: Add new awesome feature"
```

The hooks will run automatically. If they fail, some files may have been auto-formatted. Simply `git add` the modified files and commit again.

### Running Unit Tests

The test suite uses pytest. To run all tests, simply execute the following command from the root of the agent-engine directory:

```bash
pytest
```

This will automatically discover and run all files named `test_*.py`.

## 4. Extending the Engine: Adding a New System

The primary way to extend the engine is by adding new CognitiveSystems. These systems are world-agnostic and operate on the components defined in agent-core.

Here is a basic example of how to create a new CuriositySystem.

### Step 1: Create the System File

Create a new file at `src/agent_engine/systems/curiosity_system.py`.

### Step 2: Define the Component (in agent-core)

First, define the data component that this system will operate on. This belongs in agent-core because it's part of the agent's core "soul".

```python
# In agent-core/src/agent_core/core/ecs/component.py

class CuriosityComponent(Component):
    """Stores the agent's desire to explore unknown states."""
    def __init__(self, exploration_drive: float = 0.5):
        self.drive = exploration_drive # A value from 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {"exploration_drive": self.drive}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        if not 0.0 <= self.drive <= 1.0:
            return False, ["Exploration drive is out of bounds [0, 1]."]
        return True, []
```

### Step 3: Implement the System Logic (in agent-engine)

Now, implement the system logic in your new file. This system will listen for an action and, if the outcome was neutral or boring, it will slightly increase the agent's curiosity.

```python
# In agent-engine/src/agent_engine/systems/curiosity_system.py

from typing import Any, Dict, List, Type
from agent_core.core.ecs.component import Component, AffectComponent
from agent_core.core.ecs.system import System
# Import your new component
from agent_core.core.ecs.component import CuriosityComponent

class CuriositySystem(System):
    """Increases an agent's curiosity after boring events."""

    REQUIRED_COMPONENTS: List[Type[Component]] = [CuriosityComponent, AffectComponent]

    def __init__(self, simulation_state: Any, config: Dict[str, Any], cognitive_scaffold: Any):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.event_bus.subscribe("action_executed", self.on_action_executed)

    def on_action_executed(self, event_data: Dict[str, Any]):
        entity_id = event_data["entity_id"]
        action_outcome = event_data["action_outcome"]

        components = self.simulation_state.get_entities_with_components(
            self.REQUIRED_COMPONENTS
        ).get(entity_id)

        if not components:
            return

        curiosity_comp = components.get(CuriosityComponent)

        # If the action had a very low impact (low reward magnitude)
        if abs(action_outcome.reward) < 0.01:
            curiosity_comp.drive = min(1.0, curiosity_comp.drive + 0.05)
            print(f"DEBUG: {entity_id}'s curiosity increased to {curiosity_comp.drive:.2f}")

    def update(self, current_tick: int):
        # This system is purely event-driven, but could have decay logic here.
        pass
```

### Step 4: Register the New System

Finally, in your main application (`agent-soul-sim/src/main.py`), you would register this new system with the SystemManager.

```python
# In agent-soul-sim/src/main.py

from agent_engine.systems import CuriositySystem # Import the new system

# ... inside your main function ...
manager = SimulationManager(...)
manager.register_system(CuriositySystem) # Register it like any other system
# ...
```
