# FILE: simulations/sugarscape_sim/actions.py
"""
Defines the agent actions for the Sugarscape simulation.
Each class implements the ActionInterface, defining what an agent can do.
The logic for how these actions affect the world is handled by Systems,
which listen for events published when an action's 'execute' method is called.
"""

from typing import Any, Dict, List, cast

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.action_outcome import ActionOutcome
from agent_core.core.ecs.abstractions import SimulationState as AbstractSimulationState

from .components import EnergyComponent, PositionComponent
from .environment import SugarscapeEnvironment


@action_registry.register
class MoveAction(ActionInterface):
    """Allows an agent to move to an adjacent, unoccupied cell."""

    @property
    def action_id(self) -> str:
        return "move"

    @property
    def name(self) -> str:
        return "Move"

    def get_base_cost(self, simulation_state: AbstractSimulationState) -> float:
        """The paper specifies a movement cost of 2 energy units."""
        return 2.0

    def generate_possible_params(
        self, entity_id: str, sim_state: AbstractSimulationState, tick: int
    ) -> List[Dict[str, Any]]:
        """Generates parameters for all valid moves (N, S, E, W)."""
        # FIX: Check for attribute existence instead of strict type for mock-friendliness.
        if not hasattr(sim_state, "environment"):
            return []

        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        env = sim_state.environment
        if not pos_comp or not isinstance(env, SugarscapeEnvironment):
            return []

        pos_comp = cast(PositionComponent, pos_comp)

        valid_moves = []
        # N, S, E, W directions
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            new_pos = (pos_comp.position[0] + dx, pos_comp.position[1] + dy)
            if env.is_valid_position(new_pos) and not env.get_entities_at_position(
                new_pos
            ):
                valid_moves.append({"target_pos": new_pos})
        return valid_moves

    def execute(
        self,
        entity_id: str,
        sim_state: AbstractSimulationState,
        params: Dict[str, Any],
        tick: int,
    ) -> ActionOutcome:
        """Signals the intent to move; logic is handled by MovementSystem."""
        return ActionOutcome(success=True, message="Move initiated.", base_reward=0.0)

    def get_feature_vector(
        self, entity_id: str, sim_state: AbstractSimulationState, params: Dict[str, Any]
    ) -> List[float]:
        """One-hot encoding for the action space."""
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@action_registry.register
class HarvestAction(ActionInterface):
    """Allows an agent to harvest sugar from its current cell."""

    @property
    def action_id(self) -> str:
        return "harvest"

    @property
    def name(self) -> str:
        return "Harvest Sugar"

    def get_base_cost(self, simulation_state: AbstractSimulationState) -> float:
        return 0.0  # Harvesting itself costs no energy

    def generate_possible_params(
        self, entity_id: str, sim_state: AbstractSimulationState, tick: int
    ) -> List[Dict[str, Any]]:
        """Action is possible if there is sugar at the agent's location."""
        # FIX: Check for attribute existence instead of strict type for mock-friendliness.
        if not hasattr(sim_state, "environment"):
            return []

        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        env = sim_state.environment
        if not pos_comp or not isinstance(env, SugarscapeEnvironment):
            return []

        pos_comp = cast(PositionComponent, pos_comp)

        if env.get_sugar_at(pos_comp.position) > 0:
            return [{}]  # No parameters needed
        return []

    def execute(
        self,
        entity_id: str,
        sim_state: AbstractSimulationState,
        params: Dict[str, Any],
        tick: int,
    ) -> ActionOutcome:
        """Signals intent to harvest; logic is handled by HarvestSystem."""
        return ActionOutcome(
            success=True, message="Harvest initiated.", base_reward=0.0
        )

    def get_feature_vector(
        self, entity_id: str, sim_state: AbstractSimulationState, params: Dict[str, Any]
    ) -> List[float]:
        """One-hot encoding for the action space."""
        return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


@action_registry.register
class ShareAction(ActionInterface):
    """Allows an agent to share energy with an adjacent agent."""

    @property
    def action_id(self) -> str:
        return "share"

    @property
    def name(self) -> str:
        return "Share Energy"

    def get_base_cost(self, simulation_state: AbstractSimulationState) -> float:
        return 0.0

    def generate_possible_params(
        self, entity_id: str, sim_state: AbstractSimulationState, tick: int
    ) -> List[Dict[str, Any]]:
        """Generates share options for all adjacent agents."""
        # FIX: Check for attribute existence instead of strict type for mock-friendliness.
        if not hasattr(sim_state, "environment"):
            return []

        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        energy_comp = sim_state.get_component(entity_id, EnergyComponent)
        env = sim_state.environment

        if (
            not pos_comp
            or not energy_comp
            or not isinstance(env, SugarscapeEnvironment)
        ):
            return []

        energy_comp = cast(EnergyComponent, energy_comp)
        pos_comp = cast(PositionComponent, pos_comp)

        if energy_comp.current_energy <= 1:
            return []

        possible_shares = []
        neighbors = env.get_neighbors(pos_comp.position)
        for neighbor_pos in neighbors:
            target_agents = env.get_entities_at_position(neighbor_pos)
            if target_agents:
                target_id = target_agents.pop()
                # Share a portion of current energy
                share_amount = int(energy_comp.current_energy * 0.25)
                if share_amount > 0:
                    possible_shares.append(
                        {"target_id": target_id, "amount": share_amount}
                    )
        return possible_shares

    def execute(
        self,
        entity_id: str,
        sim_state: AbstractSimulationState,
        params: Dict[str, Any],
        tick: int,
    ) -> ActionOutcome:
        """Signals intent to share; logic is handled by SocialSystem."""
        return ActionOutcome(success=True, message="Share initiated.", base_reward=0.0)

    def get_feature_vector(
        self, entity_id: str, sim_state: AbstractSimulationState, params: Dict[str, Any]
    ) -> List[float]:
        """One-hot encoding for the action space."""
        return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


@action_registry.register
class AttackAction(ActionInterface):
    """Allows an agent to attack an adjacent agent to steal energy."""

    @property
    def action_id(self) -> str:
        return "attack"

    @property
    def name(self) -> str:
        return "Attack"

    def get_base_cost(self, simulation_state: AbstractSimulationState) -> float:
        return 1.0  # Attacking has a small energy cost

    def generate_possible_params(
        self, entity_id: str, sim_state: AbstractSimulationState, tick: int
    ) -> List[Dict[str, Any]]:
        """Generates attack options for all adjacent agents."""
        # FIX: Check for attribute existence instead of strict type for mock-friendliness.
        if not hasattr(sim_state, "environment"):
            return []

        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        env = sim_state.environment
        if not pos_comp or not isinstance(env, SugarscapeEnvironment):
            return []

        pos_comp = cast(PositionComponent, pos_comp)

        possible_attacks = []
        neighbors = env.get_neighbors(pos_comp.position)
        for neighbor_pos in neighbors:
            target_agents = env.get_entities_at_position(neighbor_pos)
            if target_agents:
                possible_attacks.append({"target_id": target_agents.pop()})
        return possible_attacks

    def execute(
        self,
        entity_id: str,
        sim_state: AbstractSimulationState,
        params: Dict[str, Any],
        tick: int,
    ) -> ActionOutcome:
        """Signals intent to attack; logic is handled by SocialSystem."""
        return ActionOutcome(success=True, message="Attack initiated.", base_reward=0.0)

    def get_feature_vector(
        self, entity_id: str, sim_state: AbstractSimulationState, params: Dict[str, Any]
    ) -> List[float]:
        """One-hot encoding for the action space."""
        return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


@action_registry.register
class ReproduceAction(ActionInterface):
    """Allows an agent to reproduce if it has sufficient energy."""

    @property
    def action_id(self) -> str:
        return "reproduce"

    @property
    def name(self) -> str:
        return "Reproduce"

    def get_base_cost(self, simulation_state: AbstractSimulationState) -> float:
        """The paper specifies a reproduction cost of 150 energy units."""
        return 150.0

    def generate_possible_params(
        self, entity_id: str, sim_state: AbstractSimulationState, tick: int
    ) -> List[Dict[str, Any]]:
        """Action is possible if agent has enough energy."""
        energy_comp = sim_state.get_component(entity_id, EnergyComponent)
        if not energy_comp:
            return []

        energy_comp = cast(EnergyComponent, energy_comp)
        if energy_comp.current_energy > self.get_base_cost(sim_state):
            return [{}]
        return []

    def execute(
        self,
        entity_id: str,
        sim_state: AbstractSimulationState,
        params: Dict[str, Any],
        tick: int,
    ) -> ActionOutcome:
        """Signals intent to reproduce; logic is handled by SocialSystem."""
        return ActionOutcome(
            success=True, message="Reproduction initiated.", base_reward=0.0
        )

    def get_feature_vector(
        self, entity_id: str, sim_state: AbstractSimulationState, params: Dict[str, Any]
    ) -> List[float]:
        """One-hot encoding for the action space."""
        return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


@action_registry.register
class StayAction(ActionInterface):
    """Allows an agent to stay in its current position."""

    @property
    def action_id(self) -> str:
        return "stay"

    @property
    def name(self) -> str:
        return "Stay"

    def get_base_cost(self, simulation_state: AbstractSimulationState) -> float:
        """The paper specifies a cost of 1 for staying put."""
        return 1.0

    def generate_possible_params(
        self, entity_id: str, sim_state: AbstractSimulationState, tick: int
    ) -> List[Dict[str, Any]]:
        """This action is always possible."""
        return [{}]

    def execute(
        self,
        entity_id: str,
        sim_state: AbstractSimulationState,
        params: Dict[str, Any],
        tick: int,
    ) -> ActionOutcome:
        """Logic is handled by MetabolismSystem (passive energy decay)."""
        return ActionOutcome(success=True, message="Agent stays put.", base_reward=0.0)

    def get_feature_vector(
        self, entity_id: str, sim_state: AbstractSimulationState, params: Dict[str, Any]
    ) -> List[float]:
        """One-hot encoding for the action space."""
        return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
