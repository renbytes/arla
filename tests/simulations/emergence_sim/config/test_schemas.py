# FILE: tests/config/test_schemas.py

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

# Assume this is the path to your main Pydantic model
from simulations.emergence_sim.config.schemas import EmergenceSimAppConfig

# --- Test Fixtures ---


@pytest.fixture
def full_config_yaml(tmp_path: Path) -> Path:
    """
    Creates a temporary, complete emergence_config.yml file that mirrors
    the structure of a real configuration file.
    """
    config_content = """
simulation:
  steps: 200
  log_directory: "data/logs/emergence_sim"
  database_directory: "data/db"
  database_file: "agent_sim.db"
  enable_debug_logging: false
  random_seed: 123
  systems:
    enable_symbol_negotiation: true
    enable_narrative_consensus: false
    enable_ritualization: false
    enable_social_credit: false
    enable_normative_abstraction: false
    social_credit:
      initial_credit: 0.6
      cooperation_bonus: 0.06
      defection_penalty: 0.11
      decay_rate: 0.002
    symbol_negotiation:
      interaction_success_threshold: 0.9
      meaning_drift_factor: 0.03
    narrative_consensus:
      consensus_threshold: 4
      narrative_influence_factor: 0.2

action_modules:
  - "simulations.emergence_sim.actions.communication_actions"
  - "simulations.emergence_sim.actions.economic_actions"

logging:
  components_to_log:
    - "simulations.emergence_sim.components.SocialCreditComponent"

# CORRECTED: Added the missing top-level 'llm' block to the test data.
llm:
  provider: "openai"
  completion_model: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"
  temperature: 0.2
  max_tokens: 500
  reflection_prompt_prefix: "You are an agent. Reflect."

agent:
  foundational:
    num_agents: 15
    lifespan_std_dev_percent: 0.0
    vitals:
      initial_time_budget: 500.0
      initial_health: 100.0
      initial_resources: 5.0
    attributes:
      initial_attack_power: 0.0
      initial_speed: 1
  cognitive:
    embeddings:
      main_embedding_dim: 1536
      schema_embedding_dim: 128
  costs:
    actions:
      propose_symbol: 1.0
      guess_object: 1.0
      share_narrative: 2.0
      give_resource: 1.0
      request_resource: 1.0
  emotional_dynamics:
    temporal:
      valence_decay_rate: 0.95
      arousal_decay_rate: 0.90
    noise_std: 0.02
    appraisal_weights:
      goal_relevance: 0.7
      agency: 0.5
      social_feedback: 0.3

learning:
  q_learning:
    initial_epsilon: 0.5
    epsilon_decay_rate: 0.999
    min_epsilon: 0.05
    alpha: 0.001
    gamma: 0.98
    state_feature_dim: 16
    internal_state_dim: 8
    action_feature_dim: 15
  rewards:
    base_reward: 0.0
    move_reward_base: -0.1
    give_resource_bonus: 1.0
  memory:
    short_term_memory_maxlen: 10
    affective_buffer_maxlen: 200
    emotion_cluster_min_data: 50
    reflection_interval: 25
    cognitive_dissonance_threshold: 0.7

environment:
  grid_world_size: [15, 15]
  num_objects: 50

scenario_path: "simulations/emergence_sim/scenarios/default.json"
"""
    config_file = tmp_path / "test_emergence_config.yml"
    config_file.write_text(config_content)
    return config_file


# --- Unit Tests ---


def test_load_and_validate_full_config(full_config_yaml: Path):
    """
    Tests that a full, valid YAML config file can be loaded and parsed
    by the EmergenceSimAppConfig model without raising a validation error.
    """
    with open(full_config_yaml, "r") as f:
        config_data = yaml.safe_load(f)
    try:
        validated_config = EmergenceSimAppConfig(**config_data)
    except ValidationError as e:
        pytest.fail(f"Configuration validation failed: {e}")

    assert validated_config.simulation.steps == 200
    assert validated_config.simulation.systems.social_credit.initial_credit == 0.6
    assert validated_config.agent.foundational.num_agents == 15
    assert validated_config.learning.rewards.give_resource_bonus == 1.0
    assert validated_config.llm.provider == "openai"


def test_full_config_structure_is_validated(full_config_yaml: Path):
    """
    Tests that all nested Pydantic models are correctly instantiated when
    loading a valid YAML file.
    """
    with open(full_config_yaml, "r") as f:
        config_data = yaml.safe_load(f)
    try:
        validated_config = EmergenceSimAppConfig(**config_data)
    except ValidationError as e:
        pytest.fail(f"Configuration validation failed unexpectedly: {e}")

    from simulations.emergence_sim.config.schemas import (
        AgentConfig,
        EmotionalDynamicsConfig,
        LearningConfig,
        LLMConfig,
        SimulationConfig,
    )

    assert isinstance(validated_config.simulation, SimulationConfig)
    assert isinstance(validated_config.agent, AgentConfig)
    assert isinstance(validated_config.agent.emotional_dynamics, EmotionalDynamicsConfig)
    assert isinstance(validated_config.learning, LearningConfig)
    assert isinstance(validated_config.llm, LLMConfig)
    assert validated_config.agent.emotional_dynamics.noise_std == 0.02
