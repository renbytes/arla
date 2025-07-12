# src/agent_core/environment/vitality_metrics_provider_interface.py
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Type

# Forward reference for Component from agent_core
if TYPE_CHECKING:
    from agent_core.core.ecs.component import Component


class VitalityMetricsProviderInterface(ABC):
    """
    Abstract Base Class for vitality metrics providers.

    This interface defines the contract for how a specific simulation's
    world state components are transformed into generalized, normalized
    metrics relevant for affective appraisal.

    The concrete implementation will reside in the final simulation application
    (e.g., agent-soul-sim) and will extract data from world-specific components
    (like HealthComponent, InventoryComponent, TimeBudgetComponent) to provide
    world-agnostic vitality scores.
    """

    @abstractmethod
    def get_normalized_vitality_metrics(
        self,
        entity_id: str,
        components: Dict[Type["Component"], "Component"],
        config: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Extracts and normalizes world-specific vitality metrics from an entity's
        components into a dictionary of generalized floats (0.0 to 1.0).

        Args:
            entity_id: The ID of the entity.
            components: A dictionary of the entity's components.
            config: The simulation configuration.

        Returns:
            A dictionary of normalized vitality metrics.
            Example: {"health_norm": 0.8, "time_norm": 0.5, "resources_norm": 0.7}
        """
        raise NotImplementedError
