# simulations/emergence_sim/actions/communication_actions.py
"""
Defines the primitive communication actions for the 'emergence_sim'.
These actions form the foundation for the emergence of language and culture,
enabling agents to propose symbols, guess meanings, and share narratives.
"""

import random
from typing import TYPE_CHECKING, Any, Dict, List

# Imports from the core library, defining the base class and interfaces
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent
from agent_core.core.ecs.component import MemoryComponent

# World-specific components needed for action generation
# These would be defined in simulations/emergence_sim/components.py
from ..components import ConceptualSpaceComponent, PositionComponent

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState


def _create_emergence_feature_vector(
    action_id: str,
    intent: Intent,
    base_cost: float,
    params: Dict[str, Any],
    param_feature_map: Dict[str, Any],
) -> List[float]:
    """
    Helper function to create a standardized action feature vector for this simulation.
    This mirrors the utility function found in other simulation packages.
    """
    action_ids = action_registry.action_ids
    action_one_hot = [1.0 if aid == action_id else 0.0 for aid in action_ids]

    intents = list(Intent)
    intent_one_hot = [1.0 if i == intent else 0.0 for i in intents]

    time_cost_feature = [base_cost / 10.0]  # Normalize cost

    # Placeholder for parameter features, can be expanded
    param_features = [0.0] * 5
    for param_name, mapping in param_feature_map.items():
        if param_name in params:
            idx, value, normalizer = mapping
            # Ensure value is numeric before division
            numeric_value = value if isinstance(value, (int, float)) else 0.0
            param_features[idx] = float(numeric_value) / float(normalizer) if normalizer != 0 else 0.0

    return action_one_hot + intent_one_hot + time_cost_feature + param_features


@action_registry.register
class ProposeSymbolAction(ActionInterface):
    """An agent proposes a symbol for a perceived object to another agent."""

    action_id = "propose_symbol"
    name = "Propose Symbol"

    def get_base_cost(self, simulation_state: "SimulationState") -> float:
        """Returns the base time budget cost for this action."""
        # This could be loaded from a config file in a full implementation
        return 1.0

    def generate_possible_params(
        self, entity_id: str, simulation_state: "SimulationState", current_tick: int
    ) -> List[Dict[str, Any]]:
        """
        Generates parameters for proposing symbols for nearby objects.
        This is a key part of the "Naming Game".
        """
        params_list: List[Dict[str, Any]] = []
        pos_comp = simulation_state.get_component(entity_id, PositionComponent)
        concept_comp = simulation_state.get_component(entity_id, ConceptualSpaceComponent)

        if not isinstance(pos_comp, PositionComponent) or not hasattr(pos_comp, "environment"):
            return []

        # In a full implementation, this would find perceivable objects.
        # For now, we'll assume a placeholder method exists.
        # perceivable_objects = pos_comp.environment.get_objects_in_radius(pos_comp.position, 5)
        perceivable_objects = [("object_1", (1, 1)), ("object_2", (2, 2))]  # Placeholder

        for obj_id, _obj_pos in perceivable_objects:
            # Agent can either use a known symbol or invent a new one
            symbol_to_propose = f"token_{random.randint(100, 999)}"
            if isinstance(concept_comp, ConceptualSpaceComponent):
                # A more advanced agent might look up a learned concept
                pass

            params_list.append(
                {
                    "target_object_id": obj_id,
                    "symbol": symbol_to_propose,
                    "intent": Intent.COOPERATE,
                }
            )
        return params_list

    def execute(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
        current_tick: int,
    ) -> Dict[str, Any]:
        """
        The core logic for this action is handled by the SymbolNegotiationSystem,
        which listens for the event triggered by this action's execution.
        This method simply confirms the action took place.
        """
        return {"status": "symbol_proposed", "symbol": params.get("symbol")}

    def get_feature_vector(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
    ) -> List[float]:
        """Generates the feature vector for this action variant."""
        return _create_emergence_feature_vector(
            self.action_id,
            params.get("intent", Intent.COOPERATE),
            self.get_base_cost(simulation_state),
            params,
            {},  # No specific parameter features for now
        )


@action_registry.register
class GuessObjectAction(ActionInterface):
    """An agent guesses which object a received symbol refers to."""

    action_id = "guess_object"
    name = "Guess Object"

    def get_base_cost(self, simulation_state: "SimulationState") -> float:
        """Returns the base time budget cost for this action."""
        return 1.0

    def generate_possible_params(
        self, entity_id: str, simulation_state: "SimulationState", current_tick: int
    ) -> List[Dict[str, Any]]:
        """
        This is a reactive action. The SymbolNegotiationSystem will provide the
        context (the symbol to be guessed). For the agent's decision-making,
        it needs to generate possible guesses from its perception.
        """
        params_list: List[Dict[str, Any]] = []
        pos_comp = simulation_state.get_component(entity_id, PositionComponent)

        if not isinstance(pos_comp, PositionComponent) or not hasattr(pos_comp, "environment"):
            return []

        # Agent generates a guess for each object it can perceive.
        # perceivable_objects = pos_comp.environment.get_objects_in_radius(pos_comp.position, 5)
        perceivable_objects = [("object_1", (1, 1)), ("object_2", (2, 2))]  # Placeholder

        for obj_id, _obj_pos in perceivable_objects:
            params_list.append({"guessed_object_id": obj_id, "intent": Intent.COOPERATE})

        return params_list

    def execute(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
        current_tick: int,
    ) -> Dict[str, Any]:
        """
        The core logic is handled by the SymbolNegotiationSystem, which evaluates
        the guess against the speaker's original target.
        """
        return {"status": "object_guessed", "guess": params.get("guessed_object_id")}

    def get_feature_vector(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
    ) -> List[float]:
        """Generates the feature vector for this action variant."""
        return _create_emergence_feature_vector(
            self.action_id,
            params.get("intent", Intent.COOPERATE),
            self.get_base_cost(simulation_state),
            params,
            {},
        )


@action_registry.register
class ShareNarrativeAction(ActionInterface):
    """An agent broadcasts its latest reflection summary to nearby agents."""

    action_id = "share_narrative"
    name = "Share Narrative"

    def get_base_cost(self, simulation_state: "SimulationState") -> float:
        """Returns the base time budget cost for this action."""
        return 2.0  # Slightly more costly as it's a higher-level social act

    def generate_possible_params(
        self, entity_id: str, simulation_state: "SimulationState", current_tick: int
    ) -> List[Dict[str, Any]]:
        """
        An agent can choose to share its narrative if it has a recent
        reflection summary in its memory.
        """
        mem_comp = simulation_state.get_component(entity_id, MemoryComponent)
        if isinstance(mem_comp, MemoryComponent) and mem_comp.last_llm_reflection_summary:
            return [{"intent": Intent.COOPERATE}]
        return []

    def execute(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
        current_tick: int,
    ) -> Dict[str, Any]:
        """
        Publishes a 'narrative_shared' event on the EventBus. The
        NarrativeConsensusSystem will listen for this event and handle the logic
        of how other agents process the shared story.
        """
        mem_comp = simulation_state.get_component(entity_id, MemoryComponent)
        narrative_text = mem_comp.last_llm_reflection_summary if isinstance(mem_comp, MemoryComponent) else ""

        if narrative_text and simulation_state.event_bus:
            simulation_state.event_bus.publish(
                "narrative_shared",
                {
                    "speaker_id": entity_id,
                    "narrative": narrative_text,
                    "tick": current_tick,
                },
            )
            return {"status": "narrative_shared"}

        return {"status": "narrative_share_failed", "reason": "No narrative to share"}

    def get_feature_vector(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
    ) -> List[float]:
        """Generates the feature vector for this action variant."""
        return _create_emergence_feature_vector(
            self.action_id,
            params.get("intent", Intent.COOPERATE),
            self.get_base_cost(simulation_state),
            params,
            {},
        )
