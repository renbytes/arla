# simulations/soul_sim/simulation/component_factory.py

from typing import Any, Dict

from agent_core.core.ecs.component import Component
from agent_core.core.ecs.component_factory_interface import ComponentFactoryInterface
from agent_core.environment.interface import EnvironmentInterface
from agent_engine.cognition.identity.domain_identity import MultiDomainIdentity
from agent_engine.utils.class_importer import import_class


class SoulSimComponentFactory(ComponentFactoryInterface):
    """
    A soul-sim specific factory for creating components during snapshot restoration.
    It handles components that require special constructor arguments, like the environment.
    """

    def __init__(self, environment: EnvironmentInterface, config: Dict[str, Any]):
        self.environment = environment
        self.config = config

    def create_component(self, component_type: str, data: Dict[str, Any]) -> Component:
        """Creates a component, injecting dependencies where necessary."""
        component_class = import_class(component_type)

        # Handle special cases for components needing live objects
        if component_type.endswith("PositionComponent"):
            # PositionComponent needs a reference to the live environment object
            return component_class(environment=self.environment, **data)

        if component_type.endswith("IdentityComponent"):
            # IdentityComponent needs a MultiDomainIdentity object
            embedding_dim = (
                self.config.get("agent", {}).get("cognitive", {}).get("embeddings", {}).get("identity_dim", 1536)
            )
            mdi = MultiDomainIdentity(embedding_dim=embedding_dim)
            return component_class(multi_domain_identity=mdi, **data)

        # Generic case: most components can be created directly from their data
        return component_class(**data)
