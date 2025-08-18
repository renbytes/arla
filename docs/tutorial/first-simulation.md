# Tutorial: Building Your First Simulation

Welcome to the ARLA framework! This tutorial will guide you through the process of creating a simple simulation from scratch. By the end, you will have a single agent that moves randomly in a 2D grid world.

This will teach you the fundamental concepts of ARLA: Components, Actions, and Systems.

## Step 1: Create the Simulation Directory

First, let's create a new directory for our simulation inside the existing simulations folder. We'll call it `simple_sim`.

```bash
mkdir -p simulations/simple_sim/actions
```

Your new directory structure should look like this:

```
simulations/
└── simple_sim/
    └── actions/
```

## Step 2: Define a Custom Component

Every object in an ARLA simulation is an Entity defined by its Components. Components are pure data containers. Let's create a component to store an agent's position.

Create a new file: `simulations/simple_sim/components.py`

```python
# simulations/simple_sim/components.py

from agent_core.core.ecs.component import Component

class PositionComponent(Component):
    """A simple component to store an entity's x, y coordinates."""
    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y

    def to_dict(self):
        return {"x": self.x, "y": self.y}

    def validate(self, entity_id: str):
        return True, []
```

## Step 3: Define a Custom Action

Actions define what an agent can do. They contain the logic for how an action is executed and what its effects are. Let's create a simple `MoveAction`.

Create a new file: `simulations/simple_sim/actions/move_action.py`

```python
# simulations/simple_sim/actions/move_action.py

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import ActionOutcome

@action_registry.register
class MoveAction(ActionInterface):
    """A simple action that allows an agent to move to a new position."""
    @property
    def action_id(self) -> str:
        return "move"

    @property
    def name(self) -> str:
        return "Move"

    def get_base_cost(self, simulation_state) -> float:
        return 1.0

    def generate_possible_params(self, entity_id, simulation_state, current_tick):
        # For simplicity, we'll just generate one possible move.
        # A real action would check for valid neighboring tiles.
        return [{"dx": 1, "dy": 0}]

    def execute(self, entity_id, simulation_state, params, current_tick):
        # This is a placeholder. The actual movement logic will be in a System.
        return ActionOutcome(success=True, message="Move planned.", base_reward=0.1)

    def get_feature_vector(self, entity_id, simulation_state, params):
        return [1.0]
```

## Step 4: Define a Custom System

Systems contain all the logic. They operate on entities that have a specific set of components. Our `MovementSystem` will be responsible for actually changing the `PositionComponent` of agents that have chosen to move.

Create a new file: `simulations/simple_sim/systems.py`

```python
# simulations/simple_sim/systems.py

from agent_engine.simulation.system import System
from simulations.simple_sim.components import PositionComponent

class MovementSystem(System):
    """This system updates the position of agents that have executed a move action."""
    def __init__(self, simulation_state, config, cognitive_scaffold):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.event_bus.subscribe("execute_move_action", self.on_move_execute)

    def on_move_execute(self, event_data):
        entity_id = event_data["entity_id"]
        params = event_data["action_plan_component"].params
        pos_comp = self.simulation_state.get_component(entity_id, PositionComponent)

        if pos_comp:
            pos_comp.x += params.get("dx", 0)
            pos_comp.y += params.get("dy", 0)
            print(f"Agent {entity_id} moved to ({pos_comp.x}, {pos_comp.y})")

        # Important: publish the outcome so other systems know the action is done.
        self.event_bus.publish("action_outcome_ready", event_data)

    async def update(self, current_tick: int):
        # This system is purely event-driven.
        pass
```

## Step 5: Create the Simulation Entrypoint

Now we need a `run.py` file to tie everything together. This script will be responsible for setting up the `SimulationManager`, registering our custom system, and starting the simulation.

Create a new file: `simulations/simple_sim/run.py`

```python
# simulations/simple_sim/run.py

import asyncio
from agent_engine.simulation.engine import SimulationManager
from simulations.simple_sim.systems import MovementSystem
# ... other necessary imports for your specific setup ...

async def setup_and_run(run_id, task_id, experiment_id, config_overrides):
    # This is a simplified setup. A real implementation would have a full
    # world environment, scenario loader, etc.
    print("--- [simple_sim] Initializing Simple Simulation ---")

    # 1. Create a mock environment and other dependencies
    mock_env = MagicMock()
    mock_scenario_loader = MagicMock()
    # ... create other mock dependencies ...

    # 2. Instantiate the SimulationManager
    manager = SimulationManager(
        config=config_overrides,
        environment=mock_env,
        scenario_loader=mock_scenario_loader,
        # ... other dependencies ...
    )

    # 3. Register your custom system
    manager.register_system(MovementSystem)

    # 4. Run the simulation
    print(f"--- [simple_sim] Starting simulation loop for run: {run_id}")
    await manager.run()

def start_simulation(run_id, task_id, experiment_id, config_overrides):
    asyncio.run(setup_and_run(run_id, task_id, experiment_id, config_overrides))
```

> **Note**: The `run.py` is simplified for this tutorial. You would need to adapt it to your project's specific requirements and dependency injection patterns.