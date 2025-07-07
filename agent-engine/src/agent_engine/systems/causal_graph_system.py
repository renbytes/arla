# src/agent_engine/systems/causal_graph_system.py
"""
Constructs and maintains a symbolic causal graph for each agent's memory.
"""

from typing import Any, Dict, List, Tuple, Type, cast

# Imports from agent-core
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import (
    ActionPlanComponent,
    Component,
    EmotionComponent,
    GoalComponent,
    MemoryComponent,
)
from agent_core.core.ecs.event_bus import EventBus
from agent_core.environment.state_node_encoder_interface import (
    StateNodeEncoderInterface,
)

# Imports from agent_engine
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.simulation.system import System


def _create_action_outcome_node(
    action_type: str, intent: str, outcome_details: Dict[str, Any], reward_value: float
) -> Tuple[str, str, str, str]:
    """Creates a tuple representation of an action and its outcome."""
    outcome_status = outcome_details.get("status", "UNKNOWN")
    return ("ACTION_OUTCOME", action_type, intent, outcome_status)


class CausalGraphSystem(System):
    """
    Constructs and maintains a symbolic causal graph for each agent's memory,
    representing learned cause-and-effect relationships.
    This system is now decoupled from world-specific components for state representation.
    It relies on an injected StateNodeEncoderInterface to provide generalized state nodes.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [
        MemoryComponent,
        EmotionComponent,
        GoalComponent,
    ]

    def __init__(
        self,
        simulation_state: SimulationState,
        config: Dict[str, Any],
        cognitive_scaffold: Any,
        state_node_encoder: StateNodeEncoderInterface,
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)

        event_bus = simulation_state.event_bus
        if not event_bus:
            raise ValueError("EventBus not initialized in SimulationState")
        self.event_bus: EventBus = event_bus

        self.state_node_encoder = state_node_encoder
        self.event_bus.subscribe("action_executed", self.on_action_executed)

    def on_action_executed(self, event_data: Dict[str, Any]) -> None:
        """Event handler that fetches dependencies and updates the causal graph."""
        entity_id = event_data["entity_id"]
        components = self.simulation_state.entities.get(entity_id, {})

        required_types: List[Type[Component]] = [
            MemoryComponent,
            EmotionComponent,
            GoalComponent,
        ]
        if not all(isinstance(components.get(t), t) for t in required_types):
            missing = [t.__name__ for t in required_types if not isinstance(components.get(t), t)]
            print(f"WARNING: CausalGraphSystem for {entity_id} missing components: {missing}")
            return

        self._update_causal_link(entity_id, components, event_data)

    def _update_causal_link(
        self,
        entity_id: str,
        components: Dict[Type[Component], Component],
        event_data: Dict[str, Any],
    ) -> None:
        """Pure logic for creating and linking nodes in an agent's causal graph."""
        mem_comp = cast(MemoryComponent, components.get(MemoryComponent))
        emotion_comp = cast(EmotionComponent, components.get(EmotionComponent))

        if not all([mem_comp, emotion_comp]):
            print(f"ERROR: CausalGraphSystem missing core components for {entity_id}.")
            return

        action_plan = cast(ActionPlanComponent, event_data["action_plan"])
        action_outcome = cast(ActionOutcome, event_data["action_outcome"])

        current_state_node = self.state_node_encoder.encode_state_for_causal_graph(
            entity_id=entity_id,
            components=components,
            current_tick=event_data["current_tick"],
            config=self.config,
        )

        action_type_name = getattr(action_plan.action_type, "name", "unknown")
        intent = getattr(action_plan, "intent", None)
        intent_name = getattr(intent, "name", "UNKNOWN")

        action_outcome_node = _create_action_outcome_node(
            action_type_name, intent_name, action_outcome.details, action_outcome.reward
        )

        link_weight = (abs(action_outcome.reward) * 0.1) + (emotion_comp.arousal * 0.5) + 0.1

        if not hasattr(mem_comp, "causal_graph") or mem_comp.causal_graph is None:
            mem_comp.causal_graph = {}

        # Safely access the previous state node, which is a dynamic attribute.
        previous_node = getattr(mem_comp, "previous_state_node", None)

        if previous_node is not None:
            if previous_node not in mem_comp.causal_graph:
                mem_comp.causal_graph[previous_node] = {}
            mem_comp.causal_graph[previous_node][action_outcome_node] = (
                mem_comp.causal_graph[previous_node].get(action_outcome_node, 0.0) + link_weight
            )

        if action_outcome_node not in mem_comp.causal_graph:
            mem_comp.causal_graph[action_outcome_node] = {}
        mem_comp.causal_graph[action_outcome_node][current_state_node] = (
            mem_comp.causal_graph[action_outcome_node].get(current_state_node, 0.0) + link_weight
        )

        # Cache the current state for the next tick. Mypy is ignored because
        # this attribute is not formally declared on the MemoryComponent class.
        mem_comp.previous_state_node = current_state_node

    async def update(self, current_tick: int) -> None:
        """Periodically decays the strength of all causal links in every agent's memory."""
        if (current_tick + 1) % 10 != 0:
            return

        target_entities = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)
        decay_rate = self.config.get("learning", {}).get("causal_decay_rate", 0.95)

        for components_dict in target_entities.values():
            mem_comp = cast(MemoryComponent, components_dict.get(MemoryComponent))
            if not mem_comp or not hasattr(mem_comp, "causal_graph") or mem_comp.causal_graph is None:
                continue

            # Iterate over copies to allow for safe modification during iteration
            for cause_node, effects in list(mem_comp.causal_graph.items()):
                for effect_node, weight in list(effects.items()):
                    new_weight = weight * decay_rate
                    if new_weight < 0.01:
                        del effects[effect_node]
                    else:
                        effects[effect_node] = new_weight
                if not effects:
                    del mem_comp.causal_graph[cause_node]
