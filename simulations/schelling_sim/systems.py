# simulations/schelling_sim/systems.py

import os
from typing import Any, Dict, List, Type

from agent_core.core.ecs.component import Component
from agent_core.agents.actions.base_action import ActionOutcome
from agent_engine.simulation.system import System
from simulations.schelling_sim.renderer import SchellingRenderer

from .components import GroupComponent, PositionComponent, SatisfactionComponent
from .environment import SchellingGridEnvironment


class SatisfactionSystem(System):
    """
    Calculates and updates the satisfaction state of each agent based on its
    neighbors.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [
        PositionComponent,
        GroupComponent,
        SatisfactionComponent,
    ]

    async def update(self, current_tick: int) -> None:
        """
        Iterates through all Schelling agents and updates their `is_satisfied`
        status based on the types of their neighbors.
        """
        env = self.simulation_state.environment
        if not isinstance(env, SchellingGridEnvironment):
            return

        all_agents = self.simulation_state.get_entities_with_components(
            self.REQUIRED_COMPONENTS
        )

        for _, components in all_agents.items():
            pos_comp = components.get(PositionComponent)
            group_comp = components.get(GroupComponent)
            satisfaction_comp = components.get(SatisfactionComponent)

            if not all([pos_comp, group_comp, satisfaction_comp]):
                continue

            neighbors = env.get_neighbors_of_position(pos_comp.position)
            num_neighbors = len(neighbors)

            # An agent with no neighbors is considered satisfied.
            if num_neighbors == 0:
                satisfaction_comp.is_satisfied = True
                continue

            # Count neighbors of the same group type.
            same_type_neighbors = 0
            for neighbor_id in neighbors.values():
                neighbor_group_comp = self.simulation_state.get_component(
                    neighbor_id, GroupComponent
                )
                if (
                    neighbor_group_comp
                    and neighbor_group_comp.agent_type == group_comp.agent_type
                ):
                    same_type_neighbors += 1

            # The agent is satisfied if the ratio of same-type neighbors
            # to total neighbors meets or exceeds its personal threshold.
            satisfaction_ratio = same_type_neighbors / num_neighbors
            is_now_satisfied = (
                satisfaction_ratio >= satisfaction_comp.satisfaction_threshold
            )
            satisfaction_comp.is_satisfied = is_now_satisfied


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
            self.event_bus.subscribe(
                "execute_move_to_empty_cell_action", self.on_move_execute
            )

    def on_move_execute(self, event_data: Dict[str, Any]) -> None:
        """Handles the execution of a move action."""
        entity_id = event_data["entity_id"]
        action_plan = event_data["action_plan_component"]
        params = action_plan.params

        pos_comp = self.simulation_state.get_component(entity_id, PositionComponent)
        env = self.simulation_state.environment

        outcome: ActionOutcome

        if not all([pos_comp, isinstance(env, SchellingGridEnvironment)]):
            outcome = ActionOutcome(
                success=False,
                message="Missing component or wrong env.",
                base_reward=-0.1,
            )
        else:
            from_pos = pos_comp.position
            to_pos = (params["target_x"], params["target_y"])

            if env.move_entity(entity_id, from_pos, to_pos):
                pos_comp.move_to(to_pos[0], to_pos[1])
                outcome = ActionOutcome(
                    success=True, message="Move successful.", base_reward=1.0
                )
            else:
                outcome = ActionOutcome(
                    success=False, message="Target cell was occupied.", base_reward=-0.1
                )

        # Add the created outcome to the event data dictionary
        event_data["action_outcome"] = outcome
        event_data["original_action_plan"] = event_data.pop("action_plan_component")

        if self.event_bus:
            self.event_bus.publish("action_outcome_ready", event_data)

    async def update(self, current_tick: int) -> None:
        """This system is purely event-driven."""
        pass


class RenderingSystem(System):
    """A system that renders the simulation state to an image at each tick."""

    REQUIRED_COMPONENTS: List[Type[Component]] = [PositionComponent, GroupComponent]

    def __init__(
        self,
        simulation_state: Any,
        config: Dict[str, Any],
        cognitive_scaffold: Any,
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)

        env_params = config.get("environment", {}).get("params", {})
        width = env_params.get("width", 50)
        height = env_params.get("height", 50)

        render_config = config.get("rendering", {})
        base_output_dir = render_config.get("output_directory", "data/renders/default")
        pixel_scale = render_config.get("pixel_scale", 1)

        run_id = self.simulation_state.simulation_id
        self.unique_output_dir = os.path.join(base_output_dir, run_id)

        self.renderer = SchellingRenderer(
            width, height, self.unique_output_dir, pixel_scale
        )
        print(
            f"🎨 RenderingSystem initialized. Frames will be saved to '{self.unique_output_dir}'."
        )

    async def update(self, current_tick: int) -> None:
        """On each tick, render a new frame."""
        self.renderer.render_frame(self.simulation_state, current_tick)
