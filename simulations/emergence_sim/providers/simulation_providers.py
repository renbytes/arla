# FILE: simulations/emergence_sim/providers/simulation_providers.py

from typing import Any, Dict, Optional, Tuple, Type, cast

import numpy as np
from agent_core.core.ecs.abstractions import SimulationState
from agent_core.core.ecs.component import (
    Component,
    MemoryComponent,
    TimeBudgetComponent,
)
from agent_core.environment.controllability_provider_interface import (
    ControllabilityProviderInterface,
)
from agent_core.environment.state_node_encoder_interface import (
    StateNodeEncoderInterface,
)
from agent_core.environment.vitality_metrics_provider_interface import (
    VitalityMetricsProviderInterface,
)
from agent_core.policy.reward_calculator_interface import RewardCalculatorInterface
from agent_core.policy.state_encoder_interface import StateEncoderInterface

from simulations.emergence_sim.components import (
    ConceptualSpaceComponent,
    DebtLedgerComponent,
    InventoryComponent,
    OpinionComponent,  # New Import
    PositionComponent,
    SocialCreditComponent,
)


class EmergenceRewardCalculator(RewardCalculatorInterface):
    """
    Calculates a subjective reward based on the agent's values and social credit.
    """

    def calculate_final_reward(
        self,
        base_reward: float,
        action_intent: str,
        entity_components: Dict[Type[Component], "Component"],
        **kwargs,
    ) -> Tuple[float, Dict[str, Any]]:
        final_reward = base_reward
        breakdown = {"base_reward": base_reward}
        return final_reward, breakdown


class EmergenceNarrativeContextProvider(StateNodeEncoderInterface):
    """
    Creates narrative context from an agent's social state and also encodes
    that state for the causal graph.
    """

    def get_narrative_context(
        self,
        components: Dict[Type["Component"], "Component"],
        current_tick: int,
        config: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """Creates a detailed text summary of the agent's social situation for the LLM."""

        credit_comp = cast(SocialCreditComponent, components.get(SocialCreditComponent))
        debt_comp = cast(DebtLedgerComponent, components.get(DebtLedgerComponent))
        _concept_comp = cast(ConceptualSpaceComponent, components.get(ConceptualSpaceComponent))
        mem_comp = cast(MemoryComponent, components.get(MemoryComponent))
        opinion_comp = cast(OpinionComponent, components.get(OpinionComponent))  # New

        # Basic Social Standing
        credit_score = credit_comp.score if credit_comp else 0.0
        _num_debts = len(debt_comp.obligations) if debt_comp else 0
        opinion = opinion_comp.opinion if opinion_comp else "undecided"  # New

        narrative_parts = [
            f"My current social credit is {credit_score:.2f}. I am part of the '{opinion}' faction."  # Modified
        ]

        if mem_comp:
            recent_gives = 0
            reflection_interval = config.learning.memory.reflection_interval
            start_tick = max(0, current_tick - reflection_interval)

            for event in mem_comp.episodic_memory:
                if event.get("tick", 0) >= start_tick:
                    action_plan = event.get("action_plan")
                    if action_plan and hasattr(action_plan, "action_type"):
                        if getattr(action_plan.action_type, "action_id", "") == "give_resource":
                            recent_gives += 1

            if recent_gives > 0:
                narrative_parts.append(f"Recently, I have been generous, giving resources away {recent_gives} time(s).")

        # Combine into final narrative
        narrative = " ".join(narrative_parts)
        return {"narrative": narrative, "llm_final_account": narrative}

    def encode_state_for_causal_graph(
        self, components: Dict[Type["Component"], "Component"], **kwargs
    ) -> Tuple[Any, ...]:
        """Encodes the agent's social state into a simple tuple for causal analysis."""
        credit_comp = cast(SocialCreditComponent, components.get(SocialCreditComponent))
        debt_comp = cast(DebtLedgerComponent, components.get(DebtLedgerComponent))

        credit_level = "low"
        if credit_comp and credit_comp.score > 0.66:
            credit_level = "high"
        elif credit_comp and credit_comp.score > 0.33:
            credit_level = "medium"

        debt_level = "none"
        if debt_comp and len(debt_comp.obligations) > 5:
            debt_level = "high"
        elif debt_comp and len(debt_comp.obligations) > 0:
            debt_level = "some"

        return ("STATE", f"credit_{credit_level}", f"debt_{debt_level}")


class EmergenceStateEncoder(StateEncoderInterface):
    """Encodes the simulation state into a feature vector for the Q-learning agent."""

    def encode_state(
        self,
        simulation_state: SimulationState,
        entity_id: str,
        config: Dict[str, Any],
        target_entity_id: Optional[str] = None,
    ) -> np.ndarray:
        """Creates a feature vector based on the agent's social and physical state."""
        components = simulation_state.entities.get(entity_id, {})

        pos_comp = cast(PositionComponent, components.get(PositionComponent))
        inv_comp = cast(InventoryComponent, components.get(InventoryComponent))
        credit_comp = cast(SocialCreditComponent, components.get(SocialCreditComponent))
        debt_comp = cast(DebtLedgerComponent, components.get(DebtLedgerComponent))

        # Physical State
        pos_x_norm = (
            pos_comp.position[0] / simulation_state.environment.width
            if pos_comp and simulation_state.environment
            else 0
        )
        pos_y_norm = (
            pos_comp.position[1] / simulation_state.environment.height
            if pos_comp and simulation_state.environment
            else 0
        )
        resources_norm = min(inv_comp.current_resources, 50.0) / 50.0 if inv_comp else 0

        # Social state is now included in the feature vector
        credit_score = credit_comp.score if credit_comp else 0.5
        debt_count_norm = min(len(debt_comp.obligations), 10) / 10.0 if debt_comp else 0.0

        features = np.array(
            [pos_x_norm, pos_y_norm, resources_norm, credit_score, debt_count_norm],
            dtype=np.float32,
        )

        # Pad with zeros to match the dimension specified in the config
        padded_features = np.zeros(config.learning.q_learning.state_feature_dim, dtype=np.float32)
        padded_features[: len(features)] = features

        return padded_features

    def encode_internal_state(self, components: Dict[Type[Component], Component], config: Any) -> np.ndarray:
        """Encodes the agent's internal, social state into a feature vector."""
        credit_comp = cast(SocialCreditComponent, components.get(SocialCreditComponent))
        debt_comp = cast(DebtLedgerComponent, components.get(DebtLedgerComponent))
        concept_comp = cast(ConceptualSpaceComponent, components.get(ConceptualSpaceComponent))

        credit_score = credit_comp.score if credit_comp else 0
        num_debts = len(debt_comp.obligations) if debt_comp else 0
        num_concepts = len(concept_comp.concepts) if concept_comp else 0

        features = np.array([credit_score, num_debts, num_concepts], dtype=np.float32)

        padded_features = np.zeros(config.learning.q_learning.internal_state_dim, dtype=np.float32)
        padded_features[: len(features)] = features

        return padded_features


class EmergenceControllabilityProvider(ControllabilityProviderInterface):
    """
    Estimates agent controllability based on its social standing.
    An agent with high social credit is deemed to have more control over social outcomes.
    """

    def get_controllability_score(self, components: Dict[Type["Component"], "Component"], **kwargs) -> float:
        credit_comp = cast(SocialCreditComponent, components.get(SocialCreditComponent))
        return credit_comp.score if credit_comp else 0.5


class EmergenceVitalityMetricsProvider(VitalityMetricsProviderInterface):
    """
    Provides normalized metrics about the agent's "vitality", which in this
    simulation are its time budget and resources.
    """

    def get_normalized_vitality_metrics(
        self, components: Dict[Type["Component"], "Component"], **kwargs
    ) -> Dict[str, float]:
        time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))
        inv_comp = cast(InventoryComponent, components.get(InventoryComponent))

        time_norm = 0.0
        if time_comp and time_comp.max_time_budget > 0:
            time_norm = time_comp.current_time_budget / time_comp.max_time_budget

        resources_norm = 0.0
        if inv_comp:
            resources_norm = min(inv_comp.current_resources, 50.0) / 50.0

        return {
            "health_norm": 1.0,
            "time_norm": np.clip(time_norm, 0.0, 1.0),
            "resources_norm": np.clip(resources_norm, 0.0, 1.0),
        }
