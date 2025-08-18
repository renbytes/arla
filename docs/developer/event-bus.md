# Developer Guide: The Event Bus

In the ARLA framework, systems are designed to be completely isolated from one another. A MovementSystem has no direct knowledge of a CombatSystem, and neither knows about the ReflectionSystem. This decoupling is achieved through a central communication channel: the Event Bus.

Understanding how to use the Event Bus is essential for creating new systems and extending the A-life simulation.

## 1. What is the Event Bus?

The Event Bus is a simple but powerful implementation of the publish-subscribe design pattern.

**Publishing**: When something significant happens in a system (e.g., an action is completed, an agent's state changes), that system can "publish" an event to the bus. An event is just a string name (e.g., "action_executed") and a dictionary of data.

**Subscribing**: Other systems can "subscribe" to specific event names. When an event is published, the Event Bus automatically calls the handler methods of all subscribed systems, passing them the event data.

This pattern ensures that systems don't need to know about each other, making the entire architecture highly modular and easy to test.

## 2. How to Subscribe to an Event

You subscribe to an event within a system's `__init__` method. The System base class provides access to the event bus via `self.event_bus`.

### Example: A System Listening for Agent Deaths

```python
# simulations/my_sim/systems/graveyard_system.py

from agent_engine.simulation.system import System

class GraveyardSystem(System):
    """A system that listens for agent deactivation and logs it."""

    def __init__(self, simulation_state, config, cognitive_scaffold):
        super().__init__(simulation_state, config, cognitive_scaffold)

        # Subscribe the 'on_agent_deactivated' method to the 'agent_deactivated' event
        if self.event_bus:
            self.event_bus.subscribe("agent_deactivated", self.on_agent_deactivated)

    def on_agent_deactivated(self, event_data: dict) -> None:
        """This method is called whenever an 'agent_deactivated' event is published."""
        agent_id = event_data.get("entity_id")
        final_tick = event_data.get("current_tick")
        print(f"GraveyardSystem: Agent {agent_id} was deactivated at tick {final_tick}.")

    async def update(self, current_tick: int):
        # This system is purely event-driven
        pass
```

## 3. How to Publish an Event

You can publish an event from any system by calling `self.event_bus.publish()`.

### Example: A System that Triggers a Heatwave

```python
# simulations/my_sim/systems/weather_system.py

from agent_engine.simulation.system import System

class WeatherSystem(System):
    """A system that manages the weather and can trigger heatwaves."""

    async def update(self, current_tick: int):
        # Every 100 ticks, trigger a heatwave
        if current_tick > 0 and current_tick % 100 == 0:
            print("WeatherSystem: A heatwave has begun!")

            event_data = {
                "temperature_increase": 20.0,
                "duration_ticks": 50,
                "current_tick": current_tick,
            }

            # Publish the 'heatwave_started' event for other systems to react to
            if self.event_bus:
                self.event_bus.publish("heatwave_started", event_data)
```

## 4. Core Engine Events

The ARLA agent-engine publishes several core events that your custom systems can subscribe to. These are the primary hooks for integrating your world's logic with the main simulation loop.

### `action_chosen`
Published when an agent's DecisionSelector has chosen an action for the current tick.

**event_data**: `{ "entity_id": str, "action_plan_component": ActionPlanComponent, "current_tick": int }`

### `execute_{action_id}_action`
A dynamic event fired by the ActionSystem. Your world-specific systems should subscribe to these to implement the logic for your custom actions.

**event_data**: Same as `action_chosen`.

### `action_outcome_ready`
Your world-specific system must publish this event after it has fully handled an `execute_*` event. This signals that the action is resolved.

**event_data**: Same as `action_chosen`, but now includes the ActionOutcome.

### `action_executed`
Published by the ActionSystem after the final reward has been calculated. This is the main event for cognitive systems to listen to.

**event_data**: `{ "entity_id": str, "action_plan": ActionPlanComponent, "action_outcome": ActionOutcome, "current_tick": int }`

### `reflection_completed`
Published by the ReflectionSystem after an agent has completed a full metacognitive cycle.

**event_data**: `{ "tick": int, "entity_id": str, "context": dict }`