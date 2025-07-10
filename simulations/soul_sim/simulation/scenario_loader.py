import importlib
import json
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, cast

from omegaconf import DictConfig, OmegaConf
from rich import print

# Core imports
from agent_core.core.ecs.component import Component
from agent_core.simulation.scenario_loader_interface import ScenarioLoaderInterface
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.utils.config_utils import get_config_value
from agent_engine.cognition.identity.domain_identity import IdentityDomain

# Simulation-specific component imports
from simulations.soul_sim.components import PositionComponent, ResourceComponent

logger = logging.getLogger(__name__)

def _import_class(class_path: str) -> Type[Component]:
    """Helper to dynamically import a component class from its full path string."""
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        print(f"[DEBUG] Importing module: {module_path}, class: {class_name}")
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        print(f"[DEBUG] Successfully imported: {cls}")
        return cls
    except Exception as e:
        print(f"[ERROR] Failed to import {class_path}: {e}")
        raise


class ScenarioLoader(ScenarioLoaderInterface):
    """
    Debug version of ScenarioLoader with extensive logging.
    """

    def __init__(self, config: Any):
        # Convert OmegaConf to a standard dict for easier processing
        if isinstance(config, DictConfig):
            self.config = cast(Dict[str, Any], OmegaConf.to_container(config, resolve=True))
        else:
            self.config = config

        print(f"[DEBUG] ScenarioLoader initialized with config keys: {list(self.config.keys())}")
        print(f"[DEBUG] Agent config structure: {self.config.get('agent', {}).keys()}")

        # The SimulationState will be injected by the main run script before `load` is called
        self.simulation_state: Optional[SimulationState] = None
        # This will hold the parsed data from the scenario JSON file
        self.scenario_data: Optional[Dict[str, Any]] = None

    def load(self) -> None:
        """
        Loads the scenario file, parses its data, and then populates the
        simulation state with agents and resources.
        """
        print("[DEBUG] ScenarioLoader.load() called")

        if not self.simulation_state:
            raise RuntimeError("SimulationState has not been set on the ScenarioLoader before calling load().")

        scenario_path = self.config.get("scenario_path")
        print(f"[DEBUG] Scenario path: {scenario_path}")

        if not scenario_path:
            logger.error("ScenarioLoader: 'scenario_path' not found in configuration. Cannot load scenario.")
            return

        try:
            print(f"--- Attempting to load scenario file from: {scenario_path} ---")
            with open(scenario_path, "r") as f:
                self.scenario_data = json.load(f)
            print(f"[DEBUG] Loaded scenario data: {self.scenario_data}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load or parse scenario file at {scenario_path}: {e}")
            return

        print(f"--- Loading scenario: {self.scenario_data.get('name', 'Untitled')} ---")
        self._create_resources(self.scenario_data.get("resources", {}))
        self._create_agents(self.scenario_data.get("groups", []))
        print("--- Scenario loading complete. ---")

    def _create_resources(self, resource_config: Dict[str, Any]):
        """Initializes resource entities based on scenario config."""
        print(f"[DEBUG] Creating resources with config: {resource_config}")

        if not self.simulation_state or not self.simulation_state.environment:
            print("[DEBUG] No simulation state or environment - skipping resource creation")
            return

        for i, res_data in enumerate(resource_config.get("resource_list", [])):
            res_id = f"resource_{i}"
            print(f"[DEBUG] Creating resource {res_id} with data: {res_data}")

            self.simulation_state.add_entity(res_id)

            pos = tuple(res_data["pos"])
            self.simulation_state.add_component(
                res_id, PositionComponent(position=pos, environment=self.simulation_state.environment)
            )
            self.simulation_state.add_component(res_id, ResourceComponent(**res_data["params"]))
            print(f"[DEBUG] Successfully created resource {res_id}")

    def _create_agents(self, agent_groups: List[Dict[str, Any]]):
        """
        Creates agents by iterating through the 'groups' defined in the scenario.
        """
        print(f"[DEBUG] Creating agents from groups: {agent_groups}")

        if not self.simulation_state or not self.simulation_state.environment:
            print("[DEBUG] No simulation state or environment - skipping agent creation")
            return

        if not agent_groups:
            logger.warning("Scenario file contains no agent groups. The simulation will end immediately.")
            return

        valid_positions = self.simulation_state.environment.get_valid_positions()
        print(f"[DEBUG] Valid positions count: {len(valid_positions)}")
        agent_counter = 0

        for group in agent_groups:
            num_agents_in_group = group.get("count", 0)
            archetype_name = group.get("type", "default")

            print(f"Creating {num_agents_in_group} agents of type '{archetype_name}'...")

            for agent_idx in range(num_agents_in_group):
                entity_id = f"agent_{agent_counter}"
                initial_pos = tuple(valid_positions[agent_counter % len(valid_positions)])
                print(f"[DEBUG] Creating agent {entity_id} at position {initial_pos}")

                try:
                    self._create_agent_with_components(entity_id, initial_pos, archetype_name)
                    print(f"[DEBUG] Successfully created agent {entity_id}")
                except Exception as e:
                    print(f"[ERROR] Failed to create agent {entity_id}: {e}")
                    import traceback
                    traceback.print_exc()

                agent_counter += 1

        print(f"[DEBUG] Total agents created: {agent_counter}")

    def _create_agent_with_components(self, entity_id: str, initial_pos: Tuple[int, int], archetype_name: str):
        """Dynamically instantiates and adds all components listed in the config for a specific archetype."""
        print(f"[DEBUG] Creating agent {entity_id} with archetype {archetype_name}")

        if not self.simulation_state:
            print("[DEBUG] No simulation state - returning")
            return

        self.simulation_state.add_entity(entity_id)

        component_config_path = f"agent.cognitive.archetypes.{archetype_name}.components"
        component_paths = get_config_value(self.config, component_config_path, [])

        print(f"[DEBUG] Component paths for {archetype_name}: {component_paths}")

        if not component_paths:
            logger.warning(f"Warning: No components listed for archetype '{archetype_name}'. Agent '{entity_id}' will be empty.")
            return

        all_possible_kwargs = self._prepare_component_kwargs(initial_pos)
        print(f"[DEBUG] Available kwargs keys: {list(all_possible_kwargs.keys())}")

        component_count = 0
        for path_str in component_paths:
            try:
                print(f"[DEBUG] Processing component: {path_str}")
                ComponentClass = _import_class(path_str)

                constructor_params = inspect.signature(ComponentClass.__init__).parameters
                print(f"[DEBUG] Constructor params for {ComponentClass.__name__}: {list(constructor_params.keys())}")

                valid_kwargs = {
                    name: all_possible_kwargs[name]
                    for name in constructor_params
                    if name in all_possible_kwargs and name != 'self'
                }
                print(f"[DEBUG] Valid kwargs for {ComponentClass.__name__}: {list(valid_kwargs.keys())}")

                component_instance = ComponentClass(**valid_kwargs)
                self.simulation_state.add_component(entity_id, component_instance)
                component_count += 1
                print(f"[DEBUG] Successfully added component {ComponentClass.__name__} to {entity_id}")

            except Exception as e:
                print(f"[ERROR] Failed to load component '{path_str}' for '{entity_id}': {e}")
                import traceback
                traceback.print_exc()

        print(f"[DEBUG] Agent {entity_id} created with {component_count} components")

    def _prepare_component_kwargs(self, initial_pos: tuple) -> Dict[str, Any]:
        """Gathers all possible constructor arguments for any component."""
        if not self.simulation_state: return {}

        main_embedding_dim = get_config_value(self.config, "agent.cognitive.embeddings.main_embedding_dim", 1536)
        num_domains = len(IdentityDomain)

        # This calculation MUST match the vector construction in SimulationState.get_internal_state_features_for_entity
        # It's composed of: (affect/emotion vector) + (goal embedding) + (N * identity domain embeddings) + (flags)
        calculated_internal_dim = 4 + main_embedding_dim + (num_domains * main_embedding_dim) + 3

        kwargs = {
            "config": self.config,
            "device": self.simulation_state.device,
            "environment": self.simulation_state.environment,
            "position": initial_pos,
            "initial_time_budget": get_config_value(self.config, "agent.foundational.vitals.initial_time_budget", 1000.0),
            "initial_health": get_config_value(self.config, "agent.foundational.vitals.initial_health", 100.0),
            "initial_resources": get_config_value(self.config, "agent.foundational.vitals.initial_resources", 10.0),
            "attack_power": get_config_value(self.config, "agent.foundational.attributes.initial_attack_power", 10.0),
            "affective_buffer_maxlen": get_config_value(self.config, "learning.memory.affective_buffer_maxlen", 100),
            "lifespan_std_dev_percent": get_config_value(self.config, "agent.foundational.lifespan_std_dev_percent", 0.1),
            "schema_embedding_dim": get_config_value(self.config, "agent.cognitive.embeddings.schema_embedding_dim", 128),
            "embedding_dim": get_config_value(self.config, "agent.cognitive.embeddings.main_embedding_dim", 1536),
            "state_feature_dim": 20,
            "internal_state_dim": 20,
            "action_feature_dim": 5,
            "state_feature_dim": get_config_value(self.config, "learning.q_learning.state_feature_dim", 16),
            "internal_state_dim": calculated_internal_dim,
            "action_feature_dim": get_config_value(self.config, "learning.q_learning.action_feature_dim", 5),
            "q_learning_alpha": get_config_value(self.config, "learning.q_learning.alpha", 0.001),
        }

        # Add a multi-domain identity instance, which IdentityComponent requires.
        try:
            from agent_engine.cognition.identity.domain_identity import MultiDomainIdentity
            # This creates the object that will be injected into the IdentityComponent constructor.
            kwargs["multi_domain_identity"] = MultiDomainIdentity(embedding_dim=kwargs["embedding_dim"])
            print("[DEBUG] Created MultiDomainIdentity instance")
        except Exception as e:
            print(f"[DEBUG] Could not create MultiDomainIdentity: {e}")

        print(f"[DEBUG] Prepared kwargs: {list(kwargs.keys())}")
        return kwargs
