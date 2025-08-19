# simulations/schelling_sim/systems.py

from typing import Any, Dict, List, Type

from agent_core.core.ecs.component import Component
from agent_engine.simulation.system import System

from .components import PositionComponent, SchellingAgentComponent


class SatisfactionSystem(System):
    """
    Calculates and updates the satisfaction state of each agent based on its
    neighbors.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [
        PositionComponent,
        SchellingAgentComponent,
    ]

    async def update(self, current_tick: int) -> None:
        """
        Iterates through all Schelling agents and updates their `is_satisfied`
        status based on the types of their neighbors.
        """
        env = self.simulation_state.environment

        # Check if environment has the necessary method instead of strict type checking
        if not hasattr(env, "get_neighbors_of_position"):
            return

        all_agents = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)

        # Fixed B007 - renamed unused loop variable to _
        for _, components in all_agents.items():
            pos_comp = components.get(PositionComponent)
            agent_comp = components.get(SchellingAgentComponent)

            if not all([pos_comp, agent_comp]):
                continue

            neighbors = env.get_neighbors_of_position(pos_comp.position)
            if not neighbors:
                agent_comp.is_satisfied = True
                continue

            same_type_neighbors = 0
            for neighbor_id in neighbors.values():
                neighbor_agent_comp = self.simulation_state.get_component(neighbor_id, SchellingAgentComponent)
                if neighbor_agent_comp and neighbor_agent_comp.agent_type == agent_comp.agent_type:
                    same_type_neighbors += 1

            satisfaction_ratio = float(same_type_neighbors) / float(len(neighbors))
            agent_comp.is_satisfied = satisfaction_ratio >= agent_comp.satisfaction_threshold


class MovementSystem(System):
    """
    Listens for move events and executes them by updating the environment
    and the agent's PositionComponent.
    """

    def __init__(
        self,
        simulation_state: Any,
        config: Dict[str, Any],
        cognitive_scaffold: Any,
    ) -> None:
        super().__init__(simulation_state, config, cognitive_scaffold)
        if self.event_bus:
            self.event_bus.subscribe("execute_move_to_empty_cell_action", self.on_move_execute)

    def on_move_execute(self, event_data: Dict[str, Any]) -> None:
        """Handles the execution of a move action."""
        entity_id = event_data["entity_id"]
        params = event_data["action_plan_component"].params
        pos_comp = self.simulation_state.get_component(entity_id, PositionComponent)
        env = self.simulation_state.environment

        # Check for necessary method instead of strict type checking
        if not all([pos_comp, hasattr(env, "move_entity")]):
            self._publish_outcome(event_data, success=False)
            return

        from_pos = pos_comp.position
        to_pos = (params["target_x"], params["target_y"])

        if env.move_entity(entity_id, from_pos, to_pos):
            pos_comp.move_to(to_pos[0], to_pos[1])
            self._publish_outcome(event_data, success=True)
        else:
            self._publish_outcome(event_data, success=False)

    def _publish_outcome(self, event_data: Dict[str, Any], success: bool) -> None:
        """Publishes the result of the action execution to the event bus."""
        if hasattr(event_data.get("action_outcome"), "success"):
            event_data["action_outcome"].success = success
        if self.event_bus:
            self.event_bus.publish("action_outcome_ready", event_data)

    async def update(self, current_tick: int) -> None:
        """This system is purely event-driven."""
        pass
