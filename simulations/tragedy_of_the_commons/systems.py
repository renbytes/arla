"""
Defines the logic systems for the Tragedy of the Commons simulation.

Systems contain the "verb" logic of the simulation. They operate on entities
that have a specific set of components and communicate with each other
indirectly through an event bus.
"""

from typing import Any, Dict, List, Type, cast

from agent_core.agents.actions.action_outcome import ActionOutcome
from agent_core.core.ecs.component import Component, TimeBudgetComponent
from agent_engine.simulation.system import System

from .components import EnergyComponent, PositionComponent, ResourceComponent
from .environment import CommonsEnvironment
from .renderer import CommonsRenderer


class MetabolismSystem(System):
    """Handles agent metabolism (passive energy decay)."""

    REQUIRED_COMPONENTS: List[Type[Component]] = [EnergyComponent, TimeBudgetComponent]

    async def update(self, current_tick: int) -> None:
        """Processes passive energy decay for all active agents."""
        agents = self.simulation_state.get_entities_with_components(
            self.REQUIRED_COMPONENTS
        )
        metabolic_cost = self.config.agent.metabolic_cost_per_tick

        for _, components in agents.items():
            energy_comp = cast(EnergyComponent, components.get(EnergyComponent))
            time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))

            if time_comp.is_active:
                energy_comp.current_energy -= metabolic_cost


class VitalsSystem(System):
    """Checks agent vitals and handles deactivation (death) when energy is depleted."""

    REQUIRED_COMPONENTS: List[Type[Component]] = [EnergyComponent, TimeBudgetComponent]

    async def update(self, current_tick: int) -> None:
        """Checks for agents with zero or less energy and deactivates them."""
        env = self.simulation_state.environment
        if not isinstance(env, CommonsEnvironment):
            return

        agents = self.simulation_state.get_entities_with_components(
            self.REQUIRED_COMPONENTS
        )

        for agent_id, components in list(agents.items()):
            energy_comp = cast(EnergyComponent, components.get(EnergyComponent))
            time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))

            if time_comp.is_active and energy_comp.current_energy <= 0:
                time_comp.is_active = False
                env.remove_entity(agent_id)
                if self.event_bus:
                    self.event_bus.publish(
                        "agent_deactivated",
                        {"entity_id": agent_id, "current_tick": current_tick},
                    )


class MovementSystem(System):
    """Handles the execution of agent movement actions."""

    def __init__(self, sim_state: Any, config: Any, scaffold: Any):
        super().__init__(sim_state, config, scaffold)
        if self.event_bus:
            self.event_bus.subscribe("execute_move_action", self.on_move)

    def on_move(self, event_data: Dict[str, Any]):
        """Event handler for when a move action is chosen."""
        entity_id = event_data["entity_id"]
        params = event_data["action_plan_component"].params
        pos_comp = self.simulation_state.get_component(entity_id, PositionComponent)
        env = self.simulation_state.environment

        if not pos_comp or not isinstance(env, CommonsEnvironment):
            self._publish_outcome(event_data, False, -1.0, "Missing components.")
            return

        pos_comp = cast(PositionComponent, pos_comp)

        target_pos = params["target_pos"]
        old_pos = pos_comp.position
        pos_comp.x, pos_comp.y = target_pos
        env.update_entity_position(entity_id, old_pos, target_pos)

        self._publish_outcome(event_data, True, 0.0, "Move successful.")

    def _publish_outcome(
        self, event_data: Dict[str, Any], success: bool, reward: float, message: str
    ):
        """Publishes the final outcome of the action."""
        event_data["action_outcome"] = ActionOutcome(
            success, message, base_reward=reward
        )
        event_data["original_action_plan"] = event_data.pop("action_plan_component")
        if self.event_bus:
            self.event_bus.publish("action_outcome_ready", event_data)

    async def update(self, current_tick: int):
        """This system is purely event-driven."""
        pass


class GrazingSystem(System):
    """Handles the execution of resource grazing actions."""

    def __init__(self, sim_state: Any, config: Any, scaffold: Any):
        super().__init__(sim_state, config, scaffold)
        if self.event_bus:
            self.event_bus.subscribe("execute_graze_action", self.on_graze)

    def on_graze(self, event_data: Dict[str, Any]):
        """Event handler for when a graze action is chosen."""
        entity_id = event_data["entity_id"]
        pos_comp = self.simulation_state.get_component(entity_id, PositionComponent)
        energy_comp = self.simulation_state.get_component(entity_id, EnergyComponent)
        env = self.simulation_state.environment

        if not pos_comp or not energy_comp or not isinstance(env, CommonsEnvironment):
            self._publish_outcome(event_data, False, -1.0, "Missing components.")
            return

        pos_comp = cast(PositionComponent, pos_comp)
        energy_comp = cast(EnergyComponent, energy_comp)

        graze_amount = self.config.agent.graze_amount
        consumed_amount = env.consume_resource(pos_comp.position, graze_amount)
        energy_comp.current_energy += consumed_amount

        self._publish_outcome(
            event_data,
            True,
            float(consumed_amount),
            f"Grazed {consumed_amount:.2f} resources.",
        )

    def _publish_outcome(
        self, event_data: Dict[str, Any], success: bool, reward: float, message: str
    ):
        """Publishes the final outcome of the action."""
        event_data["action_outcome"] = ActionOutcome(
            success, message, base_reward=reward
        )
        event_data["original_action_plan"] = event_data.pop("action_plan_component")
        if self.event_bus:
            self.event_bus.publish("action_outcome_ready", event_data)

    async def update(self, current_tick: int):
        """This system is purely event-driven."""
        pass


class ResourceRegenerationSystem(System):
    """Handles the regeneration of grass resources across the environment."""

    REQUIRED_COMPONENTS: List[Type[Component]] = [ResourceComponent]

    async def update(self, current_tick: int) -> None:
        """Processes resource regeneration for all grass patches."""
        resource_patches = self.simulation_state.get_entities_with_components(
            self.REQUIRED_COMPONENTS
        )
        for _, components in resource_patches.items():
            res_comp = cast(ResourceComponent, components.get(ResourceComponent))
            res_comp.regenerate()


class RenderingSystem(System):
    """A system that renders the simulation state to an image at each tick."""

    def __init__(self, simulation_state: Any, config: Any, cognitive_scaffold: Any):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.renderer = CommonsRenderer(
            width=config.environment.params.width,
            height=config.environment.params.height,
            output_dir=f"{config.rendering.output_directory}/{simulation_state.simulation_id}",
            pixel_scale=config.rendering.pixel_scale,
        )

    async def update(self, current_tick: int) -> None:
        """On each tick, render a new frame."""
        self.renderer.render_frame(self.simulation_state, current_tick)
