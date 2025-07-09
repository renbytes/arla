from pydantic import BaseModel

# --- Learning Sub-configs ---
class QLearningConfig(BaseModel):
    initial_epsilon: float
    epsilon_decay_rate: float
    min_epsilon: float
    alpha: float
    gamma: float

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

# --- Top-level Learning Config ---
class LearningConfig(BaseModel):
    q_learning: QLearningConfig
    rewards: RewardsConfig
    failed_state: FailedStateConfig
    memory: MemoryConfig
    identity: IdentityLearningConfig
    critical_state: CriticalStateConfig
