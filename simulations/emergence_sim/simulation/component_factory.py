# FILE: simulations/emergence_sim/simulation/component_factory.py

from typing import Any, Dict

from agent_core.core.ecs.component import (
    ActionOutcomeComponent,
    ActionPlanComponent,
    AffectComponent,
    BeliefSystemComponent,
    CompetenceComponent,
    Component,
    EmotionComponent,
    EpisodeComponent,
    GoalComponent,
    IdentityComponent,
    MemoryComponent,
    SocialMemoryComponent,
    TimeBudgetComponent,
    ValidationComponent,
    ValueSystemComponent,
)
from agent_core.core.ecs.component_factory_interface import ComponentFactoryInterface
from agent_engine.cognition.identity.domain_identity import MultiDomainIdentity
from agent_engine.systems.components import QLearningComponent

from simulations.emergence_sim.components import (
    ConceptualSpaceComponent,
    DebtLedgerComponent,
    InventoryComponent,
    PositionComponent,
    RitualComponent,
    SocialCreditComponent,
)


class EmergenceComponentFactory(ComponentFactoryInterface):
    """Creates component instances from saved data for the emergence simulation."""

    def __init__(self, environment: Any, config: Any):
        self.environment = environment
        self.config = config
        # The dispatch dictionary is created once during initialization.
        self._component_map = self._initialize_component_map()

    def _create_conceptual_space_component(self, data: Dict[str, Any]) -> Component:
        """Helper for multi-line component creation."""
        comp = ConceptualSpaceComponent(quality_dimensions={"color": 3, "shape": 3})
        comp.concepts = data.get("concepts_known", {})
        return comp

    def _initialize_component_map(self) -> Dict[str, Any]:
        """
        Initializes a mapping from component type names to the functions
        that create them. This replaces the long if-elif-else chain.
        """
        # Pre-load configs for cleaner lambda functions
        q_conf = self.config.learning.q_learning
        mem_conf = self.config.learning.memory
        emb_conf = self.config.agent.cognitive.embeddings

        return {
            # Core Engine Components
            "TimeBudgetComponent": lambda data: TimeBudgetComponent(**data),
            "QLearningComponent": lambda data: QLearningComponent(
                state_feature_dim=q_conf.state_feature_dim,
                internal_state_dim=q_conf.internal_state_dim,
                action_feature_dim=q_conf.action_feature_dim,
                q_learning_alpha=q_conf.alpha,
                initial_epsilon=q_conf.initial_epsilon,
                device="cpu",
            ),
            "MemoryComponent": lambda data: MemoryComponent(),
            "IdentityComponent": lambda data: IdentityComponent(multi_domain_identity=MultiDomainIdentity()),
            "GoalComponent": lambda data: GoalComponent(embedding_dim=emb_conf.main_embedding_dim),
            "EmotionComponent": lambda data: EmotionComponent(**data),
            "AffectComponent": lambda data: AffectComponent(affective_buffer_maxlen=mem_conf.affective_buffer_maxlen),
            "ActionPlanComponent": lambda data: ActionPlanComponent(),
            "ActionOutcomeComponent": lambda data: ActionOutcomeComponent(),
            "CompetenceComponent": lambda data: CompetenceComponent(),
            "EpisodeComponent": lambda data: EpisodeComponent(),
            "BeliefSystemComponent": lambda data: BeliefSystemComponent(),
            "SocialMemoryComponent": lambda data: SocialMemoryComponent(
                schema_embedding_dim=emb_conf.schema_embedding_dim, device="cpu"
            ),
            "ValidationComponent": lambda data: ValidationComponent(),
            "ValueSystemComponent": lambda data: ValueSystemComponent(),
            # Emergence Sim Components
            "PositionComponent": lambda data: PositionComponent(
                position=(data["position_x"], data["position_y"]), environment=self.environment
            ),
            "InventoryComponent": lambda data: InventoryComponent(initial_resources=data["current_resources"]),
            "ConceptualSpaceComponent": self._create_conceptual_space_component,
            "RitualComponent": lambda data: RitualComponent(),
            "DebtLedgerComponent": lambda data: DebtLedgerComponent(),
            "SocialCreditComponent": lambda data: SocialCreditComponent(initial_credit=data["social_credit_score"]),
        }

    def create_component(self, component_type: str, data: Dict[str, Any]) -> Component:
        """
        Maps a component type string to its constructor using a dispatch dictionary.
        This method is now simple and has a low complexity score.
        """
        # Find the correct factory function by checking if its key is in the full component path
        for name, factory_func in self._component_map.items():
            if name in component_type:
                return factory_func(data)

        raise ValueError(f"Unknown component type in factory: {component_type}")
