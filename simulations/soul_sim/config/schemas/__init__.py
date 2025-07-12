from typing import Optional

from pydantic import BaseModel

from .agent import AgentConfig
from .environment import EnvironmentConfig
from .learning import LearningConfig
from .llm import LLMConfig
from .simulation import SimulationConfig


# This is now specific to soul-sim
class SoulSimAppConfig(BaseModel):
    agent: AgentConfig
    environment: EnvironmentConfig
    learning: LearningConfig
    llm: LLMConfig
    simulation: SimulationConfig
    scenario_path: Optional[str] = None


__all__ = ["SoulSimAppConfig"]
