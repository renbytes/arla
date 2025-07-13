# from typing import Dict, List

# from pydantic import BaseModel


# # --- Sub-configs for Agent Foundational ---
# class VitalsConfig(BaseModel):
#     initial_time_budget: float
#     initial_health: float
#     initial_resources: float


# class AttributesConfig(BaseModel):
#     initial_attack_power: float
#     initial_speed: int


# class FoundationalConfig(BaseModel):
#     num_agents: int
#     lifespan_std_dev_percent: float
#     vitals: VitalsConfig
#     attributes: AttributesConfig


# # --- Sub-configs for Agent Cognitive ---
# class EmbeddingsConfig(BaseModel):
#     identity_dim: int
#     main_embedding_dim: int
#     schema_embedding_dim: int


# class ArchitectureFlags(BaseModel):
#     enable_reflection: bool = True
#     enable_emotion_clustering: bool = True
#     enable_goal_modulation: bool = True
#     enable_identity_inference: bool = True
#     enable_social_memory: bool = True


# class ArchetypeConfig(BaseModel):
#     components: List[str]


# class CognitiveConfig(BaseModel):
#     embeddings: EmbeddingsConfig
#     architecture_flags: ArchitectureFlags


# # --- Sub-configs for Agent Costs ---
# class ActionCostsConfig(BaseModel):
#     base: float
#     communicate: float
#     extract: float
#     combat: float
#     flee: float
#     reflect: float
#     nest: float


# class CostsConfig(BaseModel):
#     actions: ActionCostsConfig


# # --- Sub-configs for Agent Dynamics ---
# class DynamicsDecay(BaseModel):
#     time_budget_per_step: float
#     health_per_step: float
#     resources_per_step: float


# class DynamicsRegen(BaseModel):
#     health_from_mining: float
#     time_from_mining: float
#     health_from_farming: float
#     resource_depletion_yield_bonus: float


# class DynamicsGrowth(BaseModel):
#     time_bonus: float
#     attack_bonus: float


# class DynamicsConfig(BaseModel):
#     decay: DynamicsDecay
#     regeneration: DynamicsRegen
#     growth: DynamicsGrowth


# # --- Sub-configs for Emotional/Identity Dynamics ---
# class EmotionalAppraisalWeights(BaseModel):
#     goal_relevance: float
#     agency: float
#     social_feedback: float


# class EmotionalDynamicsConfig(BaseModel):
#     temporal: Dict[str, float]
#     noise_std: float
#     appraisal_weights: EmotionalAppraisalWeights


# class IdentityDynamicsConfig(BaseModel):
#     domain_learning_rates: Dict[str, float]
#     domain_validation_weights: Dict[str, float]
#     update_strength_cap: float
#     identity_coherence_minimum: float


# # --- Top-level Agent Config ---
# class AgentConfig(BaseModel):
#     foundational: FoundationalConfig
#     cognitive: CognitiveConfig
#     costs: CostsConfig
#     dynamics: DynamicsConfig
#     emotional_dynamics: EmotionalDynamicsConfig
#     identity_dynamics: IdentityDynamicsConfig
