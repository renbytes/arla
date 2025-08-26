"""
Defines the logic systems for the Sugarscape simulation.
Systems contain all the business logic of the simulation. They operate on
entities that have a specific set of components and communicate with each other
indirectly through an event bus. This file implements the core mechanics
of the Sugarscape model as described in the paper.
"""

import random
import uuid
from typing import Any, Dict, List, Type, cast

from agent_core.agents.actions.action_outcome import ActionOutcome
from agent_core.core.ecs.component import Component, TimeBudgetComponent
from agent_engine.simulation.system import System

from .components import (
    EnergyComponent,
    MetabolismComponent,
    PositionComponent,
)
from .environment import SugarscapeEnvironment


class MetabolismSystem(System):
    """
    Handles agent metabolism, energy decay, death, and sugar regeneration.
    This system runs every tick for all agents with energy and metabolism.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [
        EnergyComponent,
        MetabolismComponent,
        TimeBudgetComponent,
    ]

    async def update(self, current_tick: int) -> None:
        """
        Processes passive energy decay for all agents and handles death.
        Also regenerates sugar in the environment.
        """
        env = self.simulation_state.environment
        if not isinstance(env, SugarscapeEnvironment):
            return

        # Regenerate sugar across the entire map
        env.regenerate_sugar()

        # Process agent metabolism
        agents = self.simulation_state.get_entities_with_components(
            self.REQUIRED_COMPONENTS
        )
        for agent_id, components in agents.items():
            energy_comp = cast(EnergyComponent, components.get(EnergyComponent))
            metabolism_comp = cast(
                MetabolismComponent, components.get(MetabolismComponent)
            )
            time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))

            if not time_comp.is_active:
                continue

            # Apply metabolic cost
            energy_comp.current_energy -= metabolism_comp.metabolic_rate

            # Check for death
            if energy_comp.current_energy <= 0:
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

        if not pos_comp or not isinstance(env, SugarscapeEnvironment):
            self._publish_outcome(
                event_data, success=False, reward=-1.0, message="Missing components."
            )
            return

        pos_comp = cast(PositionComponent, pos_comp)

        target_pos = params["target_pos"]
        old_pos = pos_comp.position
        pos_comp.x, pos_comp.y = target_pos
        env.update_entity_position(entity_id, old_pos, target_pos)

        self._publish_outcome(
            event_data, success=True, reward=0.0, message="Move successful."
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


class HarvestSystem(System):
    """Handles the execution of sugar harvesting actions."""

    def __init__(self, sim_state: Any, config: Any, scaffold: Any):
        super().__init__(sim_state, config, scaffold)
        if self.event_bus:
            self.event_bus.subscribe("execute_harvest_action", self.on_harvest)

    def on_harvest(self, event_data: Dict[str, Any]):
        """Event handler for when a harvest action is chosen."""
        entity_id = event_data["entity_id"]
        pos_comp = self.simulation_state.get_component(entity_id, PositionComponent)
        energy_comp = self.simulation_state.get_component(entity_id, EnergyComponent)
        env = self.simulation_state.environment

        if (
            not pos_comp
            or not energy_comp
            or not isinstance(env, SugarscapeEnvironment)
        ):
            self._publish_outcome(
                event_data, success=False, reward=-1.0, message="Missing components."
            )
            return

        pos_comp = cast(PositionComponent, pos_comp)
        energy_comp = cast(EnergyComponent, energy_comp)

        harvested_amount = env.consume_sugar(pos_comp.position)
        energy_comp.current_energy += harvested_amount

        self._publish_outcome(
            event_data,
            success=True,
            reward=float(harvested_amount),
            message=f"Harvested {harvested_amount} sugar.",
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


class SocialSystem(System):
    """Handles all social interactions: sharing, attacking, and reproducing."""

    def __init__(self, sim_state: Any, config: Any, scaffold: Any):
        super().__init__(sim_state, config, scaffold)
        if self.event_bus:
            self.event_bus.subscribe("execute_share_action", self.on_share)
            self.event_bus.subscribe("execute_attack_action", self.on_attack)
            self.event_bus.subscribe("execute_reproduce_action", self.on_reproduce)

    def on_share(self, event_data: Dict[str, Any]):
        """Handles energy sharing between agents."""
        sender_id = event_data["entity_id"]
        params = event_data["action_plan_component"].params
        target_id = params["target_id"]
        amount = params["amount"]

        sender_energy = self.simulation_state.get_component(sender_id, EnergyComponent)
        target_energy = self.simulation_state.get_component(target_id, EnergyComponent)

        if not sender_energy or not target_energy:
            self._publish_outcome(event_data, False, -0.1, "Share failed.")
            return

        sender_energy = cast(EnergyComponent, sender_energy)
        target_energy = cast(EnergyComponent, target_energy)

        if sender_energy.current_energy < amount:
            self._publish_outcome(event_data, False, -0.1, "Share failed.")
            return

        sender_energy.current_energy -= amount
        target_energy.current_energy += amount
        self._publish_outcome(event_data, True, float(amount), "Shared energy.")

    def on_attack(self, event_data: Dict[str, Any]):
        """Handles one agent attacking another to steal energy."""
        attacker_id = event_data["entity_id"]
        params = event_data["action_plan_component"].params
        target_id = params["target_id"]

        attacker_energy = self.simulation_state.get_component(
            attacker_id, EnergyComponent
        )
        target_energy = self.simulation_state.get_component(target_id, EnergyComponent)
        target_time = self.simulation_state.get_component(
            target_id, TimeBudgetComponent
        )

        if not attacker_energy or not target_energy or not target_time:
            self._publish_outcome(event_data, False, -1.0, "Attack failed.")
            return

        attacker_energy = cast(EnergyComponent, attacker_energy)
        target_energy = cast(EnergyComponent, target_energy)
        target_time = cast(TimeBudgetComponent, target_time)

        stolen_energy = target_energy.current_energy
        attacker_energy.current_energy += stolen_energy
        target_energy.current_energy = 0
        target_time.is_active = False  # The target is eliminated

        env = self.simulation_state.environment
        if isinstance(env, SugarscapeEnvironment):
            env.remove_entity(target_id)

        self._publish_outcome(
            event_data, True, float(stolen_energy), "Attack successful."
        )

    def on_reproduce(self, event_data: Dict[str, Any]):
        """Handles agent reproduction."""
        parent_id = event_data["entity_id"]
        parent_energy = self.simulation_state.get_component(parent_id, EnergyComponent)
        parent_pos = self.simulation_state.get_component(parent_id, PositionComponent)
        parent_metabolism = self.simulation_state.get_component(
            parent_id, MetabolismComponent
        )
        env = self.simulation_state.environment

        if not all(
            [
                parent_energy,
                parent_pos,
                parent_metabolism,
                isinstance(env, SugarscapeEnvironment),
            ]
        ):
            self._publish_outcome(event_data, False, -1.0, "Reproduction failed.")
            return

        parent_energy = cast(EnergyComponent, parent_energy)
        parent_metabolism = cast(MetabolismComponent, parent_metabolism)
        env = cast(
            SugarscapeEnvironment, env
        )  # Add explicit cast after isinstance check

        # Find an empty cell for the offspring
        empty_cell = env.get_random_empty_cell()
        if not empty_cell:
            self._publish_outcome(event_data, False, -0.5, "No space to reproduce.")
            return

        # Split energy
        repro_cost = event_data["action_plan_component"].action_type.get_base_cost(
            self.simulation_state
        )
        parent_energy.current_energy -= repro_cost
        child_energy = repro_cost / 2.0

        # Create the new agent (child)
        child_id = f"agent_{uuid.uuid4().hex[:6]}"
        self.simulation_state.add_entity(child_id)

        # Inherit traits with slight mutation (a classic Sugarscape feature)
        child_metabolism_rate = max(
            1, parent_metabolism.metabolic_rate + random.randint(-1, 1)
        )
        child_vision = max(1, parent_metabolism.vision_range + random.randint(-1, 1))

        # Add all necessary components for the new agent
        self.simulation_state.add_component(
            child_id, PositionComponent(x=empty_cell[0], y=empty_cell[1])
        )
        self.simulation_state.add_component(
            child_id,
            EnergyComponent(current_energy=child_energy, initial_energy=child_energy),
        )
        self.simulation_state.add_component(
            child_id,
            MetabolismComponent(
                metabolic_rate=child_metabolism_rate, vision_range=child_vision
            ),
        )
        self.simulation_state.add_component(
            child_id, TimeBudgetComponent(initial_time_budget=9999)
        )  # Effectively infinite lifespan

        env.update_entity_position(child_id, None, empty_cell)
        self._publish_outcome(event_data, True, 50.0, "Reproduction successful.")

    def _publish_outcome(
        self, event_data: Dict[str, Any], success: bool, reward: float, message: str
    ):
        """Helper to publish the outcome of a social action."""
        event_data["action_outcome"] = ActionOutcome(
            success, message, base_reward=reward
        )
        event_data["original_action_plan"] = event_data.pop("action_plan_component")
        if self.event_bus:
            self.event_bus.publish("action_outcome_ready", event_data)

    async def update(self, current_tick: int):
        """This system is purely event-driven."""
        pass
