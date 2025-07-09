# src/simulations/soul_sim/systems/failed_states_system.py
"""
Tracks and decays "failed states" - locations where agents receive negative rewards.
"""

from typing import Any, Dict, List, Type, cast

from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import Component
from agent_engine.simulation.system import System

from ..components import FailedStatesComponent, PositionComponent


class FailedStatesSystem(System):
    """
    Tracks locations where agents experience negative outcomes to help them
    learn to avoid those areas.
    """

    # Defines the components this system operates on.
    REQUIRED_COMPONENTS: List[Type[Component]] = [FailedStatesComponent]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.event_bus:
            self.event_bus.subscribe("action_executed", self.on_action_executed)

    def on_action_executed(self, event_data: Dict[str, Any]):
        """
        Updates the failed states tracker for an agent if the action's reward
        was below a configured threshold.
        """
        entity_id = event_data["entity_id"]
        action_outcome = cast(ActionOutcome, event_data["action_outcome"])

        # Retrieve the necessary components for the acting entity
        failed_states_comp = self.simulation_state.get_component(entity_id, FailedStatesComponent)
        pos_comp = self.simulation_state.get_component(entity_id, PositionComponent)

        if not isinstance(failed_states_comp, FailedStatesComponent) or not isinstance(pos_comp, PositionComponent):
            return

        # Get the reward threshold from the config
        failed_state_config = self.config.get("learning", {}).get("failed_state", {})
        threshold = failed_state_config.get("threshold", -0.01)

        # If the action was a failure, increment the counter for that location
        if action_outcome.reward <= threshold:
            agent_pos = pos_comp.position
            failed_states_comp.tracker[agent_pos] = failed_states_comp.tracker.get(agent_pos, 0) + 1

    async def update(self, current_tick: int) -> None:
        """
        Periodically decays the failure count for all states across all entities,
        allowing agents to "forget" and potentially retry actions in those locations.
        """
        # Run decay logic only periodically to save computation
        if (current_tick + 1) % 20 != 0:
            return

        failed_state_config = self.config.get("learning", {}).get("failed_state", {})
        decay_rate = failed_state_config.get("decay_rate", 0.95)

        # Get all entities that have a FailedStatesComponent
        target_entities = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)

        for _, components in target_entities.items():
            failed_states_comp = cast(FailedStatesComponent, components.get(FailedStatesComponent))

            # Use a copy of keys for safe iteration while modifying the dictionary
            for state in list(failed_states_comp.tracker.keys()):
                failed_states_comp.tracker[state] *= decay_rate
                # Remove the state from the tracker if its failure count is negligible
                if failed_states_comp.tracker[state] < 0.01:
                    del failed_states_comp.tracker[state]
