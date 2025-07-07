# src/agent_persist/models.py

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ComponentSnapshot(BaseModel):
    """
    A snapshot of a single component's state.

    Uses a flexible dictionary to store the component's data, allowing it
    to capture the state of any component type.
    """

    component_type: str = Field(..., description="The full class name of the component.")
    data: Dict[str, Any] = Field(..., description="The serialized data from the component's to_dict() method.")


class AgentSnapshot(BaseModel):
    """
    A snapshot of a single agent's state, including all of its components.
    """

    agent_id: str = Field(..., description="The unique identifier for the agent.")
    components: List[ComponentSnapshot] = Field(..., description="A list of all components attached to the agent.")


class SimulationSnapshot(BaseModel):
    """
    The top-level data model representing a complete snapshot of the simulation state.
    """

    simulation_id: str = Field(..., description="The unique identifier for the simulation run.")
    current_tick: int = Field(..., description="The simulation tick at which the snapshot was taken.")
    agents: List[AgentSnapshot] = Field(..., description="A list of all agents and their states in the simulation.")

    # Optional field for storing environment-specific data
    environment_state: Optional[Dict[str, Any]] = Field(None, description="A dictionary for any world-specific state.")

    class Config:
        """Pydantic configuration options."""

        # Allows for pretty-printing and easier debugging of saved files.
        json_schema_extra = {
            "example": {
                "simulation_id": "sim_1672531200",
                "current_tick": 150,
                "agents": [
                    {
                        "agent_id": "agent_x",
                        "components": [
                            {
                                "component_type": "TimeBudgetComponent",
                                "data": {"current_time_budget": 850.5},
                            },
                            {
                                "component_type": "EmotionComponent",
                                "data": {"valence": 0.6, "arousal": 0.7},
                            },
                        ],
                    }
                ],
                "environment_state": {"weather": "sunny"},
            }
        }
