# environment/resources.py
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import Component classes for type hinting and manipulation
from agent_core.environment.interface import EnvironmentInterface

from simulations.soul_sim.components import PositionComponent, ResourceComponent

# Resource Constants
RESOURCE_TYPES: Dict[str, Dict[str, Any]] = {
    "SINGLE_NODE": {
        "min_agents": 1,
        "max_agents": 1,
        "initial_health": 10,
        "mining_rate": 2,  # How much 'health' is reduced per mine
        "reward_per_mine": 0.5,  # Reward for each successful mine hit
        "resource_yield": 1,  # Additional reward/resource gain when depleted
        "resource_respawn_time": 5,  # Number of steps until respawn for resources
    },
    "DOUBLE_NODE": {
        "min_agents": 2,
        "max_agents": 2,
        "initial_health": 30,
        "mining_rate": 3,
        "reward_per_mine": 1.0,  # Increased for collaboration incentive
        "resource_yield": 5,  # Increased for collaboration incentive
        "resource_respawn_time": 10,  # Number of steps until respawn
    },
    "TRIPLE_NODE": {
        "min_agents": 3,
        "max_agents": 3,
        "initial_health": 60,
        "mining_rate": 4,
        "reward_per_mine": 1.5,  # Increased for collaboration incentive
        "resource_yield": 10,  # Increased for collaboration incentive
        "resource_respawn_time": 25,  # Number of steps until respawn
    },
}

# Resource Management Functions


def init_resources(
    environment: EnvironmentInterface,
    num_single: int = 2,
    num_double: int = 1,
    num_triple: int = 1,
    seed: Optional[int] = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    Initializes resource nodes, returning their initial data dictionaries.
    These data dictionaries will then be used by SimulationManager to create actual ECS entities
    with ResourceComponent and PositionComponent.
    """
    initial_resources_data: Dict[str, Dict[str, Any]] = {}

    rng = np.random.default_rng(seed)

    # Get valid positions from the environment object
    all_grid_positions = environment.get_valid_positions()
    rng.shuffle(all_grid_positions)

    def prepare_resource_data(res_type_key: str, count: int):
        for _ in range(count):
            if not all_grid_positions:
                print(f"Warning: Not enough space to place all {res_type_key} resources.")
                break
            pos = all_grid_positions.pop(0)
            res_id = f"resource_{str(uuid.uuid4())[:8]}"  # Unique ID for each resource entity
            res_info = RESOURCE_TYPES[res_type_key]
            initial_resources_data[res_id] = {
                "id": res_id,
                "pos": list(pos),
                "type": res_type_key,
                "min_agents_needed": res_info["min_agents"],
                "max_agents_allowed": res_info["max_agents"],
                "initial_health": res_info["initial_health"],
                "mining_rate": res_info["mining_rate"],
                "reward_per_mine_action": res_info["reward_per_mine"],
                "resource_yield": res_info["resource_yield"],
                "resource_respawn_time": res_info["resource_respawn_time"],
                "depleted_timer": 0,  # Counter for steps since depletion
                "mined_by_agents": [],  # Agents who contributed to mining (cleared on respawn)
                "is_depleted": False,
            }

    prepare_resource_data("TRIPLE_NODE", num_triple)
    prepare_resource_data("DOUBLE_NODE", num_double)
    prepare_resource_data("SINGLE_NODE", num_single)

    return initial_resources_data


def get_resource_at_pos(simulation_state_or_resources, pos: Tuple[int, int]) -> Optional[Tuple[str, Any]]:
    """
    Returns the entity ID and ResourceComponent at a given position by checking ECS components.

    Args:
        simulation_state_or_resources: Either a SimulationState object or resources dict (for backwards compatibility)
        pos: Position tuple (x, y)

    Returns:
        Tuple of (entity_id, resource_component) if found, None otherwise
    """
    # Handle both old dict-based approach and new ECS approach
    if hasattr(simulation_state_or_resources, "entities"):
        # New ECS approach
        simulation_state = simulation_state_or_resources
        for entity_id, components in simulation_state.entities.items():
            res_comp = components.get(ResourceComponent)
            pos_comp = components.get(PositionComponent)

            if res_comp and pos_comp and not res_comp.is_depleted:
                if pos_comp.position == pos:
                    return entity_id, res_comp
    else:
        # Old dict-based approach (backwards compatibility)
        resources_dict = simulation_state_or_resources
        for resource_id, resource_data in resources_dict.items():
            if resource_data.get("pos") == list(pos) and not resource_data.get("is_depleted", False):
                return resource_id, resource_data

    return None


# The `mine_resource` function should now operate on ResourceComponent instances.
# It also needs to report 'num_agents_effectively_mining' which is logic
# that belongs in the `ResourceSystem`.
def mine_resource(
    resource_comp: ResourceComponent,
    resource_id: str,
    mining_entity_id: str,
    current_entities_at_location: List[str],
) -> int:
    """
    Mines a resource component. Modifies the ResourceComponent directly.
    Returns the number of agents effectively mining (for reward calculation).
    This logic has been moved to the ResourceSystem for execution.
    This function should only contain the direct modification logic for the component.
    """
    if resource_comp.is_depleted:
        return 0

    num_entities_at_node = len(current_entities_at_location)
    num_entities_actually_mining_effectively = 0

    if num_entities_at_node < resource_comp.min_agents_needed:
        return 0

    if num_entities_at_node > resource_comp.max_agents_allowed:
        num_entities_actually_mining_effectively = resource_comp.max_agents_allowed
    else:
        num_entities_actually_mining_effectively = num_entities_at_node

    resource_comp.current_health -= resource_comp.mining_rate

    # Track who contributed to mining (for depletion bonus distribution)
    if mining_entity_id not in resource_comp.mined_by_agents:
        resource_comp.mined_by_agents.append(mining_entity_id)

    if resource_comp.current_health <= 0:
        resource_comp.current_health = 0
        resource_comp.is_depleted = True
        resource_comp.depleted_timer = 0
        print(f"      Resource {resource_comp.type} has been depleted!")

    return num_entities_actually_mining_effectively


def check_and_respawn_resources(resource_components_map: Dict[str, ResourceComponent]):
    """
    Iterates through ResourceComponent instances and respawns depleted ones if their timer is up.
    This logic has been moved to the ResourceSystem's update method.
    """
    resource_lock = threading.Lock()

    with resource_lock:
        for res_id, res_comp in resource_components_map.items():
            if res_comp.is_depleted:
                res_comp.depleted_timer += 1
                if res_comp.depleted_timer >= res_comp.resource_respawn_time:
                    res_comp.is_depleted = False
                    res_comp.current_health = res_comp.initial_health
                    res_comp.depleted_timer = 0
                    res_comp.mined_by_agents = []
                    print(f"    Resource {res_comp.type} at {res_id[:8]} has respawned!")
