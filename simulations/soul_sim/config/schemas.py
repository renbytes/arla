# simulations/soul_sim/config/schemas.py
"""
This module defines the single, hierarchical Pydantic model for all
soul-sim configuration, providing a centralized source of truth and validation.
"""

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# LLM
class LLMConfig(BaseModel):
    provider: str
    completion_model: str
    embedding_model: str
    temperature: float = Field(..., ge=0.0, le=1.0)
    max_tokens: int = Field(..., gt=0)
    reflection_prompt_prefix: str


# Learning Sub-configs
class QLearningConfig(BaseModel):
    initial_epsilon: float
    epsilon_decay_rate: float
    min_epsilon: float
    alpha: float
    gamma: float
    state_feature_dim: int
    action_feature_dim: int


class RewardsConfig(BaseModel):
    base_reward: float
    move_reward_base: float
    communicate_reward_base: float
    combat_reward_hit: float
    flee_reward_success: float
    action_bonus_for_active_engagement: float
    combat_reward_defeat: float
    exploration_bonus: float
    collaboration_bonus_per_agent: float
    combat_penalty_defeated: float
    flee_penalty_blocked: float
    proximity_reward_weight: float
    nest_resource_cost: float
    decay_reduction_per_nest: float
    max_decay_reduction: float


class FailedStateConfig(BaseModel):
    threshold: float
    penalty_weight: float
    decay_rate: float


class MemoryConfig(BaseModel):
    short_term_memory_maxlen: int
    affective_buffer_maxlen: int
    emotion_cluster_min_data: int
    reflection_interval: int
    cognitive_dissonance_threshold: float


class IdentityLearningConfig(BaseModel):
    update_factor: float
    affect_mod_positive_action_arousal: float
    affect_mod_positive_action_surprise: float
    self_schema_update_periods: int
    competence_bonus_multiplier: float


class CriticalStateConfig(BaseModel):
    health_threshold_percent: float
    time_threshold_percent: float
    resource_threshold: float


# Top-level Learning Config
class LearningConfig(BaseModel):
    q_learning: QLearningConfig
    rewards: RewardsConfig
    failed_state: FailedStateConfig
    memory: MemoryConfig
    identity: IdentityLearningConfig
    critical_state: CriticalStateConfig


# Environment
class EnvironmentConfig(BaseModel):
    grid_world_size: Tuple[int, int]
    num_single_resources: int
    num_double_resources: int
    num_triple_resources: int
    resource_respawn_single: int
    resource_respawn_double: int
    resource_respawn_triple: int
    farm_threshold_resources: float
    farm_yield_per_step: float
    farm_creation_cost_resources: float
    farm_maintenance_cost_per_step: float
    farm_protection_bonus_attack: float
    combat_time_stolen_percent: float
    combat_time_loss_percent: float


# Agent Sub-configs
class VitalsConfig(BaseModel):
    initial_time_budget: float
    initial_health: float
    initial_resources: float


class AttributesConfig(BaseModel):
    initial_attack_power: float
    initial_speed: int


class FoundationalConfig(BaseModel):
    num_agents: int
    lifespan_std_dev_percent: float
    vitals: VitalsConfig
    attributes: AttributesConfig


class EmbeddingsConfig(BaseModel):
    identity_dim: int
    main_embedding_dim: int
    schema_embedding_dim: int


class ArchitectureFlags(BaseModel):
    enable_reflection: bool
    enable_emotion_clustering: bool
    enable_goal_modulation: bool
    enable_identity_inference: bool
    enable_social_memory: bool


class CognitiveConfig(BaseModel):
    embeddings: EmbeddingsConfig
    architecture_flags: ArchitectureFlags


class ActionCostsConfig(BaseModel):
    base: float
    communicate: float
    extract: float
    combat: float
    flee: float
    reflect: float
    nest: float


class CostsConfig(BaseModel):
    actions: ActionCostsConfig


class DynamicsDecay(BaseModel):
    time_budget_per_step: float
    health_per_step: float
    resources_per_step: float


class DynamicsRegen(BaseModel):
    health_from_mining: float
    time_from_mining: float
    health_from_farming: float
    resource_depletion_yield_bonus: float


class DynamicsGrowth(BaseModel):
    time_bonus: float
    attack_bonus: float


class DynamicsConfig(BaseModel):
    decay: DynamicsDecay
    regeneration: DynamicsRegen
    growth: DynamicsGrowth


class EmotionalAppraisalWeights(BaseModel):
    goal_relevance: float
    agency: float
    social_feedback: float


class EmotionalDynamicsConfig(BaseModel):
    temporal: Dict[str, float]
    noise_std: float
    appraisal_weights: EmotionalAppraisalWeights


class IdentityDynamicsConfig(BaseModel):
    domain_learning_rates: Dict[str, float]
    domain_validation_weights: Dict[str, float]
    update_strength_cap: float
    identity_coherence_minimum: float


class AgentConfig(BaseModel):
    foundational: FoundationalConfig
    cognitive: CognitiveConfig
    costs: CostsConfig
    dynamics: DynamicsConfig
    emotional_dynamics: EmotionalDynamicsConfig
    identity_dynamics: IdentityDynamicsConfig


# Simulation
class SimulationConfig(BaseModel):
    steps: int
    log_directory: str
    database_directory: str
    database_file: str
    enable_debug_logging: bool
    random_seed: Optional[int] = None
    enable_rendering: bool = False


# The Main Application Config Schema
class SoulSimAppConfig(BaseModel):
    """The single, validated, hierarchical configuration model for soul-sim."""

    action_modules: List[str]
    logging: Dict[str, List[str]]
    simulation: SimulationConfig
    llm: LLMConfig
    agent: AgentConfig
    environment: EnvironmentConfig
    learning: LearningConfig
    scenario_path: Optional[str] = None
    simulation_package: str = "unknown"
