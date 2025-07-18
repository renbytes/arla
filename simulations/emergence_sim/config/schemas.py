# FILE: simulations/emergence_sim/config/schemas.py

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str
    completion_model: str
    embedding_model: str
    temperature: float = Field(..., ge=0.0, le=1.0)
    max_tokens: int = Field(..., gt=0)
    reflection_prompt_prefix: str


# --- (The rest of the models are unchanged) ---


class SocialCreditSystemConfig(BaseModel):
    initial_credit: float = 0.5
    cooperation_bonus: float = 0.05
    defection_penalty: float = 0.1
    decay_rate: float = 0.001


class SymbolNegotiationSystemConfig(BaseModel):
    interaction_success_threshold: float = 0.85
    meaning_drift_factor: float = 0.02


class NarrativeConsensusSystemConfig(BaseModel):
    consensus_threshold: int = 3
    narrative_influence_factor: float = 0.1


class SystemsConfig(BaseModel):
    enable_symbol_negotiation: bool
    enable_narrative_consensus: bool
    enable_ritualization: bool
    enable_social_credit: bool
    enable_normative_abstraction: bool
    social_credit: SocialCreditSystemConfig = Field(default_factory=SocialCreditSystemConfig)
    symbol_negotiation: SymbolNegotiationSystemConfig = Field(default_factory=SymbolNegotiationSystemConfig)
    narrative_consensus: NarrativeConsensusSystemConfig = Field(default_factory=NarrativeConsensusSystemConfig)


class SimulationConfig(BaseModel):
    steps: int
    log_directory: str
    database_directory: str
    database_file: str
    enable_debug_logging: bool
    random_seed: Optional[int] = None
    systems: SystemsConfig


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
    main_embedding_dim: int
    schema_embedding_dim: int


class CognitiveConfig(BaseModel):
    embeddings: EmbeddingsConfig


class ActionCostsConfig(BaseModel):
    propose_symbol: float
    guess_object: float
    share_narrative: float
    give_resource: float
    request_resource: float


class CostsConfig(BaseModel):
    actions: ActionCostsConfig


class EmotionalAppraisalWeights(BaseModel):
    goal_relevance: float
    agency: float
    social_feedback: float


class EmotionalDynamicsConfig(BaseModel):
    temporal: Dict[str, float]
    noise_std: float
    appraisal_weights: EmotionalAppraisalWeights


class AgentConfig(BaseModel):
    foundational: FoundationalConfig
    cognitive: CognitiveConfig
    costs: CostsConfig
    emotional_dynamics: EmotionalDynamicsConfig


class QLearningConfig(BaseModel):
    initial_epsilon: float
    epsilon_decay_rate: float
    min_epsilon: float
    alpha: float
    gamma: float
    state_feature_dim: int
    internal_state_dim: int
    action_feature_dim: int


class RewardsConfig(BaseModel):
    base_reward: float
    move_reward_base: float
    give_resource_bonus: float


class MemoryConfig(BaseModel):
    short_term_memory_maxlen: int
    affective_buffer_maxlen: int
    emotion_cluster_min_data: int
    reflection_interval: int
    cognitive_dissonance_threshold: float


class LearningConfig(BaseModel):
    q_learning: QLearningConfig
    rewards: RewardsConfig
    memory: MemoryConfig


class EnvironmentConfig(BaseModel):
    grid_world_size: Tuple[int, int]
    num_objects: int


# --- Top-Level Application Configuration (Defined Last) ---


class EmergenceSimAppConfig(BaseModel):
    """The root configuration model for an Emergence-Sim run."""

    simulation: SimulationConfig
    agent: AgentConfig
    environment: EnvironmentConfig
    learning: LearningConfig
    llm: LLMConfig
    action_modules: List[str]
    logging: Dict[str, List[str]]
    scenario_path: Optional[str] = None
