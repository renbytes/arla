# simulations/disease_sim/config/schemas.py
"""
Pydantic schemas for the Disease Simulation configuration.

This file defines the data structure for all simulation parameters, ensuring
that configurations are type-safe, validated, and self-documenting.
"""

from typing import Dict, Optional, List, Any  # FIX: Added 'Any' to the import list
from pydantic import BaseModel, Field


class DiseaseConfig(BaseModel):
    """Parameters defining the characteristics of the disease."""

    infection_probability_i: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of transmission from an Infectious agent (β).",
    )
    infection_probability_e: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of transmission from an Exposed agent (θ).",
    )
    incubation_period_mean: int = Field(
        ..., gt=0, description="Average ticks an agent stays in the Exposed state."
    )
    infection_period_mean: int = Field(
        ..., gt=0, description="Average ticks an agent stays in the Infectious state."
    )


class NetworkConfig(BaseModel):
    """Parameters for generating the social contact network."""

    avg_degree: int = Field(
        ..., gt=0, description="The average number of connections per agent."
    )
    rewiring_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of rewiring an edge to create a long-range link.",
    )


class InterventionConfig(BaseModel):
    """Parameters for modeling policy interventions."""

    lockdown_tick: Optional[int] = Field(
        None, description="The tick at which lockdown measures are implemented."
    )
    lockdown_effectiveness: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="A multiplier (0-1) reducing transmission probabilities.",
    )


class AppConfig(BaseModel):
    """The root configuration schema for the disease simulation."""

    simulation_package: str
    simulation: Dict[str, Any]
    environment: Dict[str, Any]
    agent: Dict[str, Any]
    learning: Dict[str, Any]
    scenario_path: str
    scenario_loader: Dict[str, Any]
    action_generator: Dict[str, Any]
    decision_selector: Dict[str, Any]
    component_factory: Dict[str, Any]
    actions: List[str]
    logging: Dict[str, Any]
    rendering: Dict[str, Any]

    # Disease-specific configuration sections
    disease: DiseaseConfig
    network: NetworkConfig
    interventions: InterventionConfig
