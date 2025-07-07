"""Concrete implementations of the provider interfaces needed by ARLA."""

from agent_core.environment.controllability_provider_interface import (
    ControllabilityProviderInterface,
)
from agent_core.environment.state_node_encoder_interface import StateNodeEncoderInterface
from agent_core.environment.vitality_metrics_provider_interface import (
    VitalityMetricsProviderInterface,
)


class GridStateNodeEncoder(StateNodeEncoderInterface):
    def encode_state_for_causal_graph(self, *args, **kwargs):
        # Return a placeholder symbolic node
        return ("STATE", "placeholder")


class GridVitalityProvider(VitalityMetricsProviderInterface):
    def get_normalized_vitality_metrics(self, *args, **kwargs):
        return {"health_norm": 0.5, "time_norm": 0.5, "resources_norm": 0.5}


class GridControllabilityProvider(ControllabilityProviderInterface):
    def get_controllability_score(self, *args, **kwargs) -> float:
        return 0.5
