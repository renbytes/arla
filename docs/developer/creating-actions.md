# How-to: Creating a New Action

Actions are the fundamental building blocks of agent behavior in the ARLA framework. They define what an agent can do, from moving in the world to communicating with other agents. This guide will walk you through the process of creating a new, custom action.

## 1. The Action Interface

Every action in ARLA must implement the `ActionInterface`. This is a strict contract that ensures the simulation engine can interact with any action in a predictable way.

The interface requires you to implement several key methods and properties:

- **`action_id`**: A unique, lowercase string identifier (e.g., "move", "extract_resources").
- **`name`**: A human-readable, capitalized name (e.g., "Move").
- **`get_base_cost()`**: The default cost (e.g., in time or energy) of performing the action.
- **`generate_possible_params()`**: This crucial method generates all valid variations of the action for a given agent at a specific moment (e.g., all valid adjacent tiles to move to).
- **`execute()`**: A placeholder for the action's logic. The actual state change is typically handled by a world-specific system that listens for an event fired by this action.
- **`get_feature_vector()`**: A method to represent the action and its parameters as a numerical vector for machine learning models.

## 2. Example: Creating a CommunicateAction

Let's create a new action that allows an agent to communicate a message to a nearby agent.

### Step 1: Create the Action File

First, create a new Python file in your simulation's actions directory. For example: `simulations/my_sim/actions/communicate_action.py`.

### Step 2: Implement the Interface

Now, add the following code to your new file. This is a complete, well-documented implementation of a `CommunicateAction`.

```python
# simulations/my_sim/actions/communicate_action.py

from typing import Any, Dict, List

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.abstractions import SimulationState

# The @action_registry.register decorator is crucial!
# It makes the action discoverable by the simulation engine.
@action_registry.register
class CommunicateAction(ActionInterface):
    """
    An action that allows an agent to send a message to another agent.
    """

    @property
    def action_id(self) -> str:
        return "communicate"

    @property
    def name(self) -> str:
        return "Communicate"

    def get_base_cost(self, simulation_state: SimulationState) -> float:
        """The base cost in the time budget to perform this action."""
        return 2.0

    def generate_possible_params(
        self, entity_id: str, simulation_state: SimulationState, current_tick: int
    ) -> List[Dict[str, Any]]:
        """
        Generates a list of all possible communication actions.
        In this case, it finds all other agents within a certain radius.
        """
        params_list = []
        my_pos = simulation_state.get_component(entity_id, "PositionComponent")
        if not my_pos or not simulation_state.environment:
            return []

        # Find nearby agents to communicate with
        nearby_agents = simulation_state.environment.get_entities_in_radius(
            center=(my_pos.x, my_pos.y), radius=5
        )

        for agent_id, _ in nearby_agents:
            if agent_id != entity_id:
                params_list.append(
                    {
                        "target_agent_id": agent_id,
                        "message": "Hello!", # A real implementation might generate messages
                    }
                )
        return params_list

    def execute(
        self,
        entity_id: str,
        simulation_state: SimulationState,
        params: Dict[str, Any],
        current_tick: int,
    ) -> ActionOutcome:
        """
        This method doesn't change the state directly. Instead, it returns an
        ActionOutcome. A world-specific 'CommunicationSystem' would listen for the
        'execute_communicate_action' event and handle the logic.
        """
        target_id = params.get("target_agent_id")
        message = params.get("message")
        return ActionOutcome(
            success=True,
            message=f"Agent {entity_id} communicated '{message}' to {target_id}.",
            base_reward=0.5,
        )

    def get_feature_vector(
        self,
        entity_id: str,
        simulation_state: SimulationState,
        params: Dict[str, Any],
    ) -> List[float]:
        """
        Generates a simple feature vector for this action variant.
        A real implementation would be more sophisticated.
        """
        # [is_communicate_action, has_target]
        return [1.0, 1.0 if params.get("target_agent_id") else 0.0]
```

## 3. How It Works

1. **Registration**: The `@action_registry.register` decorator automatically makes your new action available to the entire simulation.

2. **Generation**: During an agent's turn, the `ActionGenerator` will call `generate_possible_params()` on your `CommunicateAction`. This method will find all nearby agents and create a distinct action variant for each one.

3. **Selection**: The list of possible actions is passed to the `DecisionSelector`, which might be a machine learning model or a simple heuristic.

4. **Execution**: If the `CommunicateAction` is chosen, the `SimulationManager` will eventually fire an `execute_communicate_action` event. A custom system (that you would write) would listen for this event and handle the actual logic of the communication.

You have now successfully created a new, custom action. You can follow this pattern to add any behavior you can imagine to your ARLA simulation.