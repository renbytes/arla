# simulations/emergence_sim/systems/opinion_dynamics_system.py
"""Contains the core logic for the Cognitive Voter Model."""

import random
import uuid
from typing import Dict, List, Type, cast

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import (
    ActionPlanComponent,
    Component,
    IdentityComponent,
    TimeBudgetComponent,
)
from agent_engine.simulation.system import System

from simulations.emergence_sim.components import OpinionComponent, PositionComponent


class OpinionDynamicsSystem(System):
    """
    Manages the spread of opinions based on the Voter Model, but with a
    cognitive twist: an agent's identity coherence makes it resistant to
    changing its opinion. This system proactively creates interactions each tick.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [
        PositionComponent,
        OpinionComponent,
        IdentityComponent,
        TimeBudgetComponent,
    ]

    def __init__(self, *args, **kwargs):
        """
        Initializes the system.
        """
        super().__init__(*args, **kwargs)
        # This system is no longer event-driven, so it does not subscribe to events.
        # It directly creates and finalizes the outcome.
        self.influence_action_instance: ActionInterface = action_registry.get_action("influence")()
        # NEW: Add a small baseline chance for opinion change to prevent gridlock.
        self.open_mindedness_factor = 0.05  # 5% chance to change regardless of identity

    async def update(self, current_tick: int):
        """
        On each tick, iterates through all active agents and forces an
        'influence' interaction with a random neighbor.
        """
        all_agents = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)

        # Shuffle agent order to ensure fairness
        agent_ids = list(all_agents.keys())
        if self.simulation_state.main_rng:
            self.simulation_state.main_rng.shuffle(agent_ids)

        for agent_id in agent_ids:
            components = all_agents[agent_id]
            time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))

            # Only process active agents
            if not time_comp or not time_comp.is_active:
                continue

            self._process_influence_interaction(agent_id, components, current_tick)

    def _process_influence_interaction(
        self, agent_id: str, components: Dict[Type[Component], Component], current_tick: int
    ):
        """
        Handles the core logic for a single agent's influence interaction.
        """
        pos_comp = cast(PositionComponent, components.get(PositionComponent))
        opinion_comp = cast(OpinionComponent, components.get(OpinionComponent))
        identity_comp = cast(IdentityComponent, components.get(IdentityComponent))

        # 1. Find a random neighbor to be influenced by
        nearby_agents = self.simulation_state.environment.get_entities_in_radius(pos_comp.position, radius=1.5)
        valid_neighbors = [(nid, npos) for nid, npos in nearby_agents if nid != agent_id]

        if not valid_neighbors:
            return  # No one nearby to interact with

        neighbor_id, _ = random.choice(valid_neighbors)
        neighbor_opinion_comp = self.simulation_state.get_component(neighbor_id, OpinionComponent)

        if not neighbor_opinion_comp:
            return

        # 2. Apply the cognitive logic
        if opinion_comp.opinion == neighbor_opinion_comp.opinion:
            # Opinions align, positive reinforcement
            outcome = ActionOutcome(True, "Maintained opinion; neighbor agrees.", 1.0, {})
        else:
            # Opinions conflict, check identity for resistance
            coherence = identity_comp.multi_domain_identity.get_identity_coherence()

            # CORRECTED: Calculate change probability with the new factor
            identity_based_prob = 1.0 - coherence
            change_probability = self.open_mindedness_factor + (
                (1.0 - self.open_mindedness_factor) * identity_based_prob
            )

            if random.random() < change_probability:
                # Opinion changes
                original_opinion = opinion_comp.opinion
                opinion_comp.opinion = neighbor_opinion_comp.opinion
                outcome = ActionOutcome(
                    True,
                    f"Opinion changed from {original_opinion} to {neighbor_opinion_comp.opinion}.",
                    1.0,
                    {},
                )
            else:
                # Agent resists the change due to strong identity
                outcome = ActionOutcome(False, "Resisted opinion change due to strong identity.", -1.0, {})

        # 3. Publish the final, loggable outcome directly
        action_plan = ActionPlanComponent(action_type=self.influence_action_instance)
        self._publish_final_outcome(agent_id, action_plan, outcome, current_tick)

    def _publish_final_outcome(self, entity_id: str, plan: ActionPlanComponent, outcome: ActionOutcome, tick: int):
        """
        Helper to publish the final 'action_executed' event that the
        LoggingSystem and QLearningSystem listen for.
        """
        if self.event_bus:
            # Finalize the outcome details, as the ActionSystem would have
            outcome.details["event_id"] = uuid.uuid4().hex
            outcome.details["reward_breakdown"] = {"base_reward": outcome.base_reward}

            self.event_bus.publish(
                "action_executed",
                {
                    "entity_id": entity_id,
                    "action_plan": plan,
                    "action_outcome": outcome,
                    "current_tick": tick,
                },
            )
