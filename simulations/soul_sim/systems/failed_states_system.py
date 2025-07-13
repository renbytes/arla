# src/agent_engine/systems/failed_states_system.py
"""
Tracks and decays "failed states" - locations where agents receive negative rewards.
"""

from typing import Any, Dict, List, Type, cast

from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import Component
from agent_engine.simulation.system import System

from simulations.soul_sim.components import FailedStatesComponent, PositionComponent


class FailedStatesSystem(System):
    """
    Tracks locations where agents experience negative outcomes to help them
    learn to avoid those areas.
    """

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

        failed_states_comp = self.simulation_state.get_component(entity_id, FailedStatesComponent)
        pos_comp = self.simulation_state.get_component(entity_id, PositionComponent)

        if not isinstance(failed_states_comp, FailedStatesComponent) or not isinstance(pos_comp, PositionComponent):
            return

        # Use direct attribute access on the validated config object
        threshold = self.config.learning.failed_state.threshold

        if action_outcome.reward <= threshold:
            agent_pos = pos_comp.position
            failed_states_comp.tracker[agent_pos] = failed_states_comp.tracker.get(agent_pos, 0) + 1

    async def update(self, current_tick: int) -> None:
        """
        Periodically decays the failure count for all states across all entities,
        allowing agents to "forget" and potentially retry actions in those locations.
        """
        if (current_tick + 1) % 20 != 0:
            return

        # Use direct attribute access on the validated config object
        decay_rate = self.config.learning.failed_state.decay_rate

        target_entities = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)

        for _, components in target_entities.items():
            failed_states_comp = cast(FailedStatesComponent, components.get(FailedStatesComponent))

            for state in list(failed_states_comp.tracker.keys()):
                failed_states_comp.tracker[state] *= decay_rate
                if failed_states_comp.tracker[state] < 0.01:
                    del failed_states_comp.tracker[state]
