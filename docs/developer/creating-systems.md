# How-to: Creating a New System

In the ARLA framework, Systems are where all the logic lives. While Actions define what an agent can do, Systems define how those actions (and other game rules) affect the simulation world. They are the engines of change in the ECS pattern.

This guide will show you how to create a custom system that responds to the CommunicateAction we built in the previous guide.

## 1. The Role of a System

A System is a class that typically performs one specific job. For example, you might have a MovementSystem, a CombatSystem, or a WeatherSystem.

Key characteristics of a System:

- It contains logic, not data.
- It operates on a set of entities that have a specific combination of Components.
- It communicates with other systems indirectly via the Event Bus.

## 2. Example: Creating a CommunicationSystem

Let's build a system that handles the logic for our CommunicateAction. This system will listen for communication events and, for now, simply log that the communication occurred.

### Step 1: Create the System File

First, create a new Python file in your simulation's main directory. For example: `simulations/my_sim/systems.py`.

### Step 2: Implement the System

Add the following code to your new file. This is a complete implementation of a CommunicationSystem.

```python
# simulations/my_sim/systems.py

from typing import Any, Dict

from agent_engine.simulation.system import System
from agent_core.core.ecs.component import SocialMemoryComponent # Assuming this exists

class CommunicationSystem(System):
    """
    Handles the logic for communication between agents. This system listens for
    communication events and updates the social memory of the recipient.
    """

    def __init__(
        self,
        simulation_state: "SimulationState",
        config: Any,
        cognitive_scaffold: Any,
    ):
        # It's crucial to call the parent constructor
        super().__init__(simulation_state, config, cognitive_scaffold)

        # Subscribe this system to the event fired when a CommunicateAction is chosen.
        # The event name is always 'execute_{action_id}_action'.
        if self.event_bus:
            self.event_bus.subscribe("execute_communicate_action", self.on_communicate)

    def on_communicate(self, event_data: Dict[str, Any]) -> None:
        """
        This is the event handler that contains the core logic.
        It's called by the Event Bus when a 'execute_communicate_action' event occurs.
        """
        entity_id = event_data["entity_id"]
        params = event_data["action_plan_component"].params
        target_id = params.get("target_agent_id")
        message = params.get("message")

        if not target_id:
            return

        print(
            f"INFO: CommunicationSystem handled event: "
            f"Agent {entity_id} said '{message}' to Agent {target_id}."
        )

        # In a real simulation, you would update the target agent's state.
        # For example, you might update their SocialMemoryComponent.
        social_mem = self.simulation_state.get_component(
            target_id, SocialMemoryComponent
        )
        if social_mem:
            # Update the social schema for the communicating agent
            pass # Your logic here

        # CRITICAL: After handling the action's logic, you must publish the
        # 'action_outcome_ready' event. This tells the ActionSystem that the
        # action is fully resolved and the final reward can be calculated.
        if self.event_bus:
            self.event_bus.publish("action_outcome_ready", event_data)

    async def update(self, current_tick: int) -> None:
        """
        The main update loop for the system. Since this system is purely
        event-driven, this method does nothing. However, it is still
        required by the System interface.
        """
        pass
```

## 3. Registering Your New System

The final step is to tell the SimulationManager to include your new system in the simulation. You do this in your simulation's `run.py` entrypoint.

Simply import your new system and register it with the manager:

```python
# simulations/my_sim/run.py

# ... other imports
from agent_engine.simulation.engine import SimulationManager
from simulations.my_sim.systems import CommunicationSystem # <-- Import your system

async def setup_and_run(run_id, task_id, experiment_id, config_overrides):
    # ... (setup code for manager and dependencies)

    manager = SimulationManager(...)

    # ... (register other systems)

    # Register your new system
    manager.register_system(CommunicationSystem)

    # ... (run the simulation)
    await manager.run()

# ...
```

## 4. How It Works

1. **Initialization**: When the simulation starts, the SimulationManager creates an instance of your CommunicationSystem.

2. **Subscription**: During its `__init__`, the system subscribes its `on_communicate` method to the `execute_communicate_action` event.

3. **Event Firing**: When an agent chooses to perform a CommunicateAction, the engine eventually fires the `execute_communicate_action` event.

4. **Handling**: The Event Bus sees the event and calls the subscribed `on_communicate` method, triggering your custom logic.

5. **Resolution**: Your system finishes its logic and fires the `action_outcome_ready` event, completing the action's lifecycle.

You have now successfully created a custom, event-driven system. This decoupled pattern is the key to building complex and maintainable simulations in ARLA.
