# src/agent_engine/simulation/simulation_state.py
"""
Consolidates all simulation data, acting as the central state container.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import numpy as np
from numpy.typing import NDArray

# Imports from agent_core
from agent_core.cognition.identity.domain_identity import IdentityDomain
from agent_engine.simulation.abstractions import (
    SimulationState as AbstractSimulationState,
)
from agent_core.core.ecs.component import (
    AffectComponent,
    Component,
    EmotionComponent,
    GoalComponent,
    IdentityComponent,
)
from agent_engine.utils.config_utils import get_config_value

# Forward references for type hinting to avoid circular imports
if TYPE_CHECKING:
    from agent_core.cognition.scaffolding import CognitiveScaffold
    from agent_core.core.ecs.event_bus import EventBus
    from agent_core.environment.interface import EnvironmentInterface
    from agent_engine.simulation.system import SystemManager


class SimulationState(AbstractSimulationState):
    """
    Consolidates all simulation data into a single, world-agnostic container.
    It holds references to the core services and the ECS data.
    """

    def __init__(self, config: Dict[str, Any], device: Any) -> None:
        self.config = config
        self.device = device
        # FIX: Changed the value type from 'Any' to 'Component'.
        # This correctly reflects that the dictionary stores component instances,
        # fixing the [arg-type] errors in multiple downstream systems.
        self.entities: Dict[str, Dict[Type[Component], Component]] = {}

        # --- Core Service and State Attributes ---
        self.simulation_id: str = ""
        self.environment: Optional["EnvironmentInterface"] = None
        self.event_bus: Optional["EventBus"] = None
        self.system_manager: Optional["SystemManager"] = None
        self.db_logger: Optional[Any] = None
        self.llm_client: Optional[Any] = None
        self.cognitive_scaffold: Optional["CognitiveScaffold"] = None
        self.main_rng: Optional[np.random.Generator] = None
        # --- End of declarations ---

    def add_entity(self, entity_id: str) -> None:
        """Adds a new entity to the simulation."""
        if entity_id in self.entities:
            raise ValueError(f"Entity with ID {entity_id} already exists.")
        self.entities[entity_id] = {}

    def add_component(self, entity_id: str, component: Component) -> None:
        """Adds a component to an existing entity."""
        if entity_id not in self.entities:
            raise ValueError(f"Cannot add component. Entity with ID {entity_id} does not exist.")
        self.entities[entity_id][type(component)] = component

    def get_component(self, entity_id: str, component_type: Type[Component]) -> Optional[Component]:
        """Retrieves a component of a specific type from an entity."""
        return self.entities.get(entity_id, {}).get(component_type)

    def remove_entity(self, entity_id: str) -> None:
        """Removes an entity and all its components from the simulation."""
        if entity_id in self.entities:
            del self.entities[entity_id]

    def get_entities_with_components(
        self, component_types: List[Type[Component]]
    ) -> Dict[str, Dict[Type[Component], Component]]:
        """
        Returns a dictionary of entities that possess all the specified component types.
        """
        matching_entities = {}
        for entity_id, components in self.entities.items():
            if all(comp_type in components for comp_type in component_types):
                matching_entities[entity_id] = components

        return matching_entities

    def get_internal_state_features_for_entity(
        self,
        id_comp: Optional[IdentityComponent],
        aff_comp: Optional[AffectComponent],
        goal_comp: Optional[GoalComponent],
        emo_comp: Optional[EmotionComponent],
    ) -> NDArray[np.float32]:
        """
        Constructs the internal state feature vector by concatenating the full,
        multi-domain identity representation and other cognitive states.
        """
        features: List[NDArray[np.float32]] = []
        flags: List[float] = []
        main_embedding_dim = get_config_value(
            self.config, "agent.cognitive.embeddings.main_embedding_dim", default=1536
        )

        # Emotion & Affect features
        if emo_comp and aff_comp:
            features.append(
                np.array(
                    [
                        emo_comp.valence,
                        emo_comp.arousal,
                        aff_comp.prediction_delta_magnitude,
                        aff_comp.predictive_delta_smooth,
                    ],
                    dtype=np.float32,
                )
            )
            flags.append(1.0)
        else:
            features.append(np.zeros(4, dtype=np.float32))
            flags.append(0.0)

        # Current Goal features
        if (
            goal_comp
            and goal_comp.current_symbolic_goal
            and goal_comp.current_symbolic_goal in goal_comp.symbolic_goals_data
        ):
            embedding = goal_comp.symbolic_goals_data[goal_comp.current_symbolic_goal]["embedding"]
            features.append(embedding.astype(np.float32))
            flags.append(1.0)
        else:
            features.append(np.zeros(main_embedding_dim, dtype=np.float32))
            flags.append(0.0)

        # Multi-Domain Identity features
        if id_comp and hasattr(id_comp, "multi_domain_identity"):
            flags.append(1.0)
            domain_embeddings = [
                id_comp.multi_domain_identity.get_domain_embedding(domain).astype(np.float32)
                for domain in IdentityDomain
            ]
            features.append(np.concatenate(domain_embeddings))
        else:
            flags.append(0.0)
            num_domains = len(IdentityDomain)
            features.append(np.zeros(main_embedding_dim * num_domains, dtype=np.float32))

        features.append(np.array(flags, dtype=np.float32))
        return np.concatenate(features).astype(np.float32)
