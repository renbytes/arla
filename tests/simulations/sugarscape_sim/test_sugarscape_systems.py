"""
Unit tests for the systems in the Sugarscape simulation.

These tests verify the core logic of the simulation, ensuring that each
system correctly processes events and updates the state of the components
and the environment in a predictable manner. Pytest fixtures and mocks are
used to create isolated testing environments for each system.
"""

import pytest
import unittest
from unittest.mock import MagicMock, ANY

from agent_core.core.ecs.component import TimeBudgetComponent
from simulations.sugarscape_sim.components import (
    EnergyComponent,
    MetabolismComponent,
    PositionComponent,
)
from simulations.sugarscape_sim.environment import SugarscapeEnvironment
from simulations.sugarscape_sim.systems import (
    HarvestSystem,
    MetabolismSystem,
    MovementSystem,
    SocialSystem,
    VitalsSystem,
)


@pytest.fixture
def mock_sim_state():
    """Provides a mock simulation state with a mock environment and event bus."""
    state = MagicMock()
    # Ensure the mock environment has the necessary methods defined by the spec.
    state.environment = MagicMock(spec=SugarscapeEnvironment)
    state.event_bus = MagicMock()
    return state


class TestMetabolismSystem:
    """Tests for the MetabolismSystem."""

    @pytest.mark.asyncio
    async def test_energy_decay_and_death(self, mock_sim_state):
        """
        Verify that agents lose energy each tick via MetabolismSystem and are
        correctly deactivated by VitalsSystem when energy reaches zero.
        """
        # System Instantiation
        metabolism_system = MetabolismSystem(mock_sim_state, {}, MagicMock())
        vitals_system = VitalsSystem(mock_sim_state, {}, MagicMock())

        # Component Setup
        agent_id = "agent_1"
        energy_comp = EnergyComponent(current_energy=10, initial_energy=100)
        metabolism_comp = MetabolismComponent(metabolic_rate=2, vision_range=5)
        time_comp = TimeBudgetComponent(initial_time_budget=100)

        mock_sim_state.get_entities_with_components.return_value = {
            agent_id: {
                EnergyComponent: energy_comp,
                MetabolismComponent: metabolism_comp,
                TimeBudgetComponent: time_comp,
            }
        }

        # --- Tick 1: Agent should lose energy but survive ---
        await metabolism_system.update(1)
        await vitals_system.update(1)

        assert energy_comp.current_energy == 8
        assert time_comp.is_active

        # --- Ticks 2-5: Agent should run out of energy and be deactivated ---
        for i in range(2, 6):
            await metabolism_system.update(i)
            await vitals_system.update(i)

        # Energy is depleted
        assert energy_comp.current_energy == 0
        # VitalsSystem should have deactivated the agent
        assert not time_comp.is_active
        # VitalsSystem should have removed the agent from the environment
        mock_sim_state.environment.remove_entity.assert_called_with(agent_id)
        # VitalsSystem should have published the deactivation event
        mock_sim_state.event_bus.publish.assert_called_with(
            "agent_deactivated", {"entity_id": agent_id, "current_tick": 5}
        )


class TestMovementSystem:
    """Tests for the MovementSystem."""

    def test_on_move(self, mock_sim_state):
        """Verify that a move event correctly updates an agent's position."""
        system = MovementSystem(mock_sim_state, {}, MagicMock())

        agent_id = "agent_1"
        pos_comp = PositionComponent(x=5, y=5)
        mock_sim_state.get_component.return_value = pos_comp

        # Configure the correct method on the mock.
        mock_sim_state.environment.update_entity_position.return_value = None

        event_data = {
            "entity_id": agent_id,
            "action_plan_component": MagicMock(params={"target_pos": (6, 5)}),
        }

        system.on_move(event_data)

        assert pos_comp.position == (6, 5)
        mock_sim_state.environment.update_entity_position.assert_called_with(
            agent_id, (5, 5), (6, 5)
        )
        mock_sim_state.event_bus.publish.assert_called_with("action_outcome_ready", ANY)


class TestHarvestSystem:
    """Tests for the HarvestSystem."""

    def test_on_harvest(self, mock_sim_state):
        """Verify that a harvest event increases agent energy."""
        system = HarvestSystem(mock_sim_state, {}, MagicMock())

        agent_id = "agent_1"
        pos_comp = PositionComponent(x=10, y=10)
        energy_comp = EnergyComponent(current_energy=50, initial_energy=100)

        def get_comp_side_effect(entity_id, comp_type):
            if comp_type == PositionComponent:
                return pos_comp
            if comp_type == EnergyComponent:
                return energy_comp
            return None

        mock_sim_state.get_component.side_effect = get_comp_side_effect
        mock_sim_state.environment.consume_sugar.return_value = 4

        event_data = {
            "entity_id": agent_id,
            "action_plan_component": MagicMock(),
        }

        system.on_harvest(event_data)

        assert energy_comp.current_energy == 54
        mock_sim_state.environment.consume_sugar.assert_called_with((10, 10))
        mock_sim_state.event_bus.publish.assert_called_with(
            "action_outcome_ready", unittest.mock.ANY
        )


class TestSocialSystem:
    """Tests for the SocialSystem's event handlers."""

    def test_on_share(self, mock_sim_state):
        """Verify that sharing correctly transfers energy."""
        system = SocialSystem(mock_sim_state, {}, MagicMock())

        sender_energy = EnergyComponent(100, 100)
        receiver_energy = EnergyComponent(20, 100)

        def get_comp(entity_id, _):
            return sender_energy if entity_id == "sender" else receiver_energy

        mock_sim_state.get_component.side_effect = get_comp

        event_data = {
            "entity_id": "sender",
            "action_plan_component": MagicMock(
                params={"target_id": "receiver", "amount": 25}
            ),
        }
        system.on_share(event_data)

        assert sender_energy.current_energy == 75
        assert receiver_energy.current_energy == 45
        mock_sim_state.event_bus.publish.assert_called_once()

    def test_on_attack(self, mock_sim_state):
        """Verify an attack transfers all energy and deactivates the target."""
        system = SocialSystem(mock_sim_state, {}, MagicMock())

        attacker_energy = EnergyComponent(50, 100)
        target_energy = EnergyComponent(30, 100)
        target_time = TimeBudgetComponent(100)

        def get_comp(entity_id, comp_type):
            if entity_id == "attacker":
                return attacker_energy
            if entity_id == "target" and comp_type == EnergyComponent:
                return target_energy
            if entity_id == "target" and comp_type == TimeBudgetComponent:
                return target_time
            return None

        mock_sim_state.get_component.side_effect = get_comp

        event_data = {
            "entity_id": "attacker",
            "action_plan_component": MagicMock(params={"target_id": "target"}),
        }
        system.on_attack(event_data)

        assert attacker_energy.current_energy == 80
        assert target_energy.current_energy == 0
        assert not target_time.is_active
        mock_sim_state.environment.remove_entity.assert_called_with("target")
        mock_sim_state.event_bus.publish.assert_called_once()

    def test_on_reproduce(self, mock_sim_state):
        """Verify reproduction creates a new agent and splits energy."""
        system = SocialSystem(mock_sim_state, {}, MagicMock())

        parent_energy = EnergyComponent(200, 200)
        parent_pos = PositionComponent(10, 10)
        parent_metabolism = MetabolismComponent(2, 5)

        def get_comp(entity_id, comp_type):
            if comp_type == EnergyComponent:
                return parent_energy
            if comp_type == PositionComponent:
                return parent_pos
            if comp_type == MetabolismComponent:
                return parent_metabolism
            return None

        mock_sim_state.get_component.side_effect = get_comp
        mock_sim_state.environment.get_random_empty_cell.return_value = (11, 11)

        mock_action = MagicMock()
        mock_action.get_base_cost.return_value = 150.0
        event_data = {
            "entity_id": "parent",
            "action_plan_component": MagicMock(action_type=mock_action),
        }
        system.on_reproduce(event_data)

        assert parent_energy.current_energy == 50

        assert mock_sim_state.add_entity.call_count == 1
        assert mock_sim_state.add_component.call_count > 3

        energy_call = [
            c
            for c in mock_sim_state.add_component.call_args_list
            if isinstance(c.args[1], EnergyComponent)
        ]
        assert energy_call[0].args[1].current_energy == 75.0

        mock_sim_state.event_bus.publish.assert_called_once()
