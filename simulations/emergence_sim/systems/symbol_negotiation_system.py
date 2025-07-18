# simulations/emergence_sim/systems/symbol_negotiation_system.py
"""
This system orchestrates the "Naming Game," a turn-based interaction designed
to facilitate the emergence of a shared, grounded lexicon among agents.
"""

from typing import Dict, List, Optional, Set, Type, cast

from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import (
    ActionPlanComponent,
    Component,
    TimeBudgetComponent,
)
from agent_engine.simulation.system import System
from agent_engine.systems.components import QLearningComponent

from simulations.emergence_sim.components import (
    ConceptualSpaceComponent,
    PositionComponent,
)


class SymbolNegotiationSystem(System):
    """
    Manages the 'Naming Game' interaction between agents to ground symbols.
    This system pairs up nearby agents, has one act as a "speaker" to name an
    object, and the other as a "listener" to guess the object.
    Based on the
    success of the interaction, it provides a reward signal to both agents,
    driving the learning of a shared vocabulary via the QLearningSystem.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [
        PositionComponent,
        ConceptualSpaceComponent,
        TimeBudgetComponent,
        QLearningComponent,
    ]

    async def update(self, current_tick: int) -> None:
        """
        On each tick, identifies pairs of nearby, active agents and initiates
        a round of the Naming Game between them.
        """
        all_agents = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)

        # Filter for active agents and shuffle for random pairing
        active_agent_ids = [
            eid
            for eid, comps in all_agents.items()
            if comps.get(TimeBudgetComponent) and cast(TimeBudgetComponent, comps.get(TimeBudgetComponent)).is_active
        ]
        if self.simulation_state.main_rng:
            self.simulation_state.main_rng.shuffle(active_agent_ids)

        processed_agents: Set[str] = set()
        for speaker_id in active_agent_ids:
            if speaker_id in processed_agents:
                continue

            speaker_pos_comp = cast(PositionComponent, all_agents[speaker_id].get(PositionComponent))
            if not speaker_pos_comp:
                continue

            # Find a nearby, available listener
            listener_id = self._find_nearby_listener(speaker_id, speaker_pos_comp, all_agents, processed_agents)
            if not listener_id:
                continue

            # Mark both as processed for this tick to avoid re-pairing
            processed_agents.add(speaker_id)
            processed_agents.add(listener_id)

            # Orchestrate the game between the pair
            await self._play_naming_game(speaker_id, listener_id, current_tick)

    def _find_nearby_listener(
        self,
        speaker_id: str,
        speaker_pos_comp: PositionComponent,
        all_agents: Dict[str, Dict[Type[Component], Component]],
        processed_agents: Set[str],
    ) -> Optional[str]:
        """Finds a single, random, nearby, and available listener."""
        if not self.simulation_state.environment:
            return None

        # In a full implementation, the radius would be a config parameter
        nearby_entities = self.simulation_state.environment.get_entities_in_radius(speaker_pos_comp.position, radius=3)

        available_listeners = [
            eid for eid, _ in nearby_entities if eid != speaker_id and eid in all_agents and eid not in processed_agents
        ]

        if not available_listeners:
            return None
        if not self.simulation_state.main_rng:
            return None

        return self.simulation_state.main_rng.choice(available_listeners)

    async def _play_naming_game(self, speaker_id: str, listener_id: str, current_tick: int) -> None:
        """Orchestrates a single round of the Naming Game."""
        # 1. Speaker selects a target object from its perception
        speaker_pos = cast(
            PositionComponent,
            self.simulation_state.get_component(speaker_id, PositionComponent),
        ).position
        perceivable_objects = (
            self.simulation_state.environment.get_objects_in_radius(speaker_pos, radius=5)
            if self.simulation_state.environment
            else []
        )

        if not perceivable_objects or not self.simulation_state.main_rng:
            return  # No object to talk about

        target_object_id, _ = self.simulation_state.main_rng.choice(perceivable_objects)

        # 2. Speaker proposes a symbol for the object.
        # A more advanced agent would learn to select a symbol. Here, we invent one.
        proposed_symbol = f"token_{self.simulation_state.main_rng.integers(100, 999)}"

        # 3. Listener attempts to guess the object based on the symbol.
        # A more advanced agent would use its ConceptualSpaceComponent to inform the guess.
        listener_pos = cast(
            PositionComponent,
            self.simulation_state.get_component(listener_id, PositionComponent),
        ).position
        listener_perceived_objects = (
            self.simulation_state.environment.get_objects_in_radius(listener_pos, radius=5)
            if self.simulation_state.environment
            else []
        )

        if not listener_perceived_objects:
            return  # Listener cannot see any objects to guess from

        guessed_object_id, _ = self.simulation_state.main_rng.choice(listener_perceived_objects)

        # 4. Evaluate success and determine reward
        success = guessed_object_id == target_object_id
        if success:
            reward = 10.0  # High reward for successful communication
            message = f"Successfully communicated about {target_object_id}."
            self._update_conceptual_space(speaker_id, proposed_symbol, target_object_id)
            self._update_conceptual_space(listener_id, proposed_symbol, target_object_id)
        else:
            reward = -2.0  # Penalty for miscommunication
            message = f"Failed to communicate. Speaker meant {target_object_id}, Listener guessed {guessed_object_id}."
        # 5. Publish outcomes for both agents to learn from the interaction
        speaker_outcome = ActionOutcome(success, message, reward, {"status": "communication_judged"})
        listener_outcome = ActionOutcome(success, message, reward, {"status": "communication_judged"})

        # Create dummy ActionPlanComponents to fit the event system's contract
        propose_action = action_registry.get_action("propose_symbol")
        guess_action = action_registry.get_action("guess_object")

        if not propose_action or not guess_action:
            return

        speaker_action_plan = ActionPlanComponent(
            action_type=propose_action(),
            params={"target_object_id": target_object_id, "symbol": proposed_symbol},
        )
        listener_action_plan = ActionPlanComponent(
            action_type=guess_action(),
            params={
                "guessed_object_id": guessed_object_id,
                "received_symbol": proposed_symbol,
            },
        )

        self._publish_outcome(speaker_id, speaker_action_plan, speaker_outcome, current_tick)
        self._publish_outcome(listener_id, listener_action_plan, listener_outcome, current_tick)

    def _update_conceptual_space(self, entity_id: str, symbol: str, object_id: str) -> None:
        """
        Updates the agent's conceptual space, reinforcing the link between
        a symbol and the perceptual properties of an object.
        (This is a placeholder for a more complex implementation).
        """
        concept_comp = self.simulation_state.get_component(entity_id, ConceptualSpaceComponent)
        if isinstance(concept_comp, ConceptualSpaceComponent):
            # In a full implementation, this would involve:
            # 1. Getting the perceptual properties of the object (e.g., color, size).
            # 2. Updating the 'centroid' and 'boundary' for the given 'symbol'
            #    in the component's 'concepts' dictionary.
            if symbol not in concept_comp.concepts:
                concept_comp.concepts[symbol] = {
                    "object_ids": [],
                    "successful_associations": 0,
                }
            concept_comp.concepts[symbol]["object_ids"].append(object_id)
            concept_comp.concepts[symbol]["successful_associations"] += 1

    def _publish_outcome(
        self,
        entity_id: str,
        action_plan: ActionPlanComponent,
        outcome: ActionOutcome,
        tick: int,
    ) -> None:
        """Helper to publish the action outcome to the event bus."""
        if self.event_bus:
            self.event_bus.publish(
                "action_outcome_ready",
                {
                    "entity_id": entity_id,
                    "action_outcome": outcome,
                    "original_action_plan": action_plan,
                    "current_tick": tick,
                },
            )
