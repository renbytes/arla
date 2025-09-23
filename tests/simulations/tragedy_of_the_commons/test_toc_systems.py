"""
Unit tests for the system classes in the Tragedy of the Commons simulation.

This file tests the core logic of each system (Metabolism, Vitals, Movement, etc.)
to ensure they correctly modify component and environment state based on
the simulation rules.
"""

import pytest
from unittest.mock import MagicMock

# FIX: Use absolute imports to resolve the ImportError
from simulations.tragedy_of_the_commons.systems import (
    MetabolismSystem,
    VitalsSystem,
    MovementSystem,
    GrazingSystem,
    ResourceRegenerationSystem,
)
from simulations.tragedy_of_the_commons.components import (
    EnergyComponent,
    PositionComponent,
    ResourceComponent,
)
from simulations.tragedy_of_the_commons.environment import CommonsEnvironment
from agent_core.core.ecs.component import TimeBudgetComponent
from agent_core.agents.actions.action_outcome import ActionOutcome

# --- Mocks and Fixtures ---


@pytest.fixture
def mock_sim_state():
    """Provides a mock SimulationState for system tests."""
    state = MagicMock()
    state.environment = MagicMock(spec=CommonsEnvironment)
    state.config = MagicMock()
    state.event_bus = MagicMock()
    # Mock entities as a dictionary
    state.entities = {}

    def get_component(entity_id, comp_type):
        return state.entities.get(entity_id, {}).get(comp_type)

    def get_entities_with_components(comp_types):
        result = {}
        for entity_id, components in state.entities.items():
            if all(comp_type in components for comp_type in comp_types):
                result[entity_id] = components
        return result

    state.get_component.side_effect = get_component
    state.get_entities_with_components.side_effect = get_entities_with_components

    return state


# --- System Tests ---


@pytest.mark.asyncio
class TestMetabolismSystem:
    """Tests for the MetabolismSystem."""

    async def test_update_decreases_energy(self, mock_sim_state):
        """Verify that the system correctly applies metabolic cost."""
        # Arrange
        mock_sim_state.config.agent.metabolic_cost_per_tick = 0.5
        agent_id = "herder_1"
        energy_comp = EnergyComponent(current_energy=100.0, initial_energy=100.0)
        time_comp = TimeBudgetComponent(initial_time_budget=100)
        mock_sim_state.entities[agent_id] = {
            EnergyComponent: energy_comp,
            TimeBudgetComponent: time_comp,
        }

        system = MetabolismSystem(mock_sim_state, mock_sim_state.config, None)

        # Act
        await system.update(current_tick=1)

        # Assert
        assert energy_comp.current_energy == 99.5

    async def test_update_does_not_affect_inactive_agents(self, mock_sim_state):
        """Verify that inactive agents do not consume energy."""
        # Arrange
        mock_sim_state.config.agent.metabolic_cost_per_tick = 0.5
        agent_id = "herder_1"
        energy_comp = EnergyComponent(current_energy=100.0, initial_energy=100.0)
        time_comp = TimeBudgetComponent(initial_time_budget=100)
        time_comp.is_active = False  # Agent is inactive
        mock_sim_state.entities[agent_id] = {
            EnergyComponent: energy_comp,
            TimeBudgetComponent: time_comp,
        }

        system = MetabolismSystem(mock_sim_state, mock_sim_state.config, None)

        # Act
        await system.update(current_tick=1)

        # Assert
        assert energy_comp.current_energy == 100.0


@pytest.mark.asyncio
class TestVitalsSystem:
    """Tests for the VitalsSystem."""

    async def test_deactivates_agent_at_zero_energy(self, mock_sim_state):
        """Verify agents are deactivated when their energy reaches zero."""
        # Arrange
        agent_id = "herder_1"
        energy_comp = EnergyComponent(current_energy=0.0, initial_energy=100.0)
        time_comp = TimeBudgetComponent(initial_time_budget=100)
        mock_sim_state.entities[agent_id] = {
            EnergyComponent: energy_comp,
            TimeBudgetComponent: time_comp,
        }

        system = VitalsSystem(mock_sim_state, mock_sim_state.config, None)

        # Act
        await system.update(current_tick=1)

        # Assert
        assert not time_comp.is_active
        mock_sim_state.environment.remove_entity.assert_called_with(agent_id)
        mock_sim_state.event_bus.publish.assert_called_with(
            "agent_deactivated",
            {"entity_id": agent_id, "current_tick": 1},
        )


class TestMovementSystem:
    """Tests for the MovementSystem."""

    def test_on_move_updates_position(self, mock_sim_state):
        """Verify a successful move updates the PositionComponent and environment."""
        # Arrange
        agent_id = "herder_1"
        pos_comp = PositionComponent(x=5, y=5)
        mock_sim_state.entities[agent_id] = {PositionComponent: pos_comp}
        mock_sim_state.environment.is_valid_position.return_value = True
        mock_sim_state.environment.is_occupied_by_agent.return_value = False

        event_data = {
            "entity_id": agent_id,
            "action_plan_component": MagicMock(params={"target_pos": (6, 5)}),
        }

        system = MovementSystem(mock_sim_state, mock_sim_state.config, None)

        # Act
        system.on_move(event_data)

        # Assert
        assert pos_comp.position == (6, 5)
        mock_sim_state.environment.update_entity_position.assert_called_with(
            agent_id, (5, 5), (6, 5)
        )
        mock_sim_state.event_bus.publish.assert_called_with(
            "action_outcome_ready", event_data
        )


class TestGrazingSystem:
    """Tests for the GrazingSystem."""

    def test_on_graze_updates_energy_and_resource(self, mock_sim_state):
        """Verify grazing increases agent energy and decreases patch resources."""
        # Arrange
        agent_id = "herder_1"
        pos_comp = PositionComponent(x=5, y=5)
        energy_comp = EnergyComponent(current_energy=50.0, initial_energy=100.0)
        mock_sim_state.entities[agent_id] = {
            PositionComponent: pos_comp,
            EnergyComponent: energy_comp,
        }
        mock_sim_state.config.agent.graze_amount = 5.0
        # Mock environment returns 5.0 resources consumed
        mock_sim_state.environment.consume_resource.return_value = 5.0

        event_data = {"entity_id": agent_id, "action_plan_component": MagicMock()}

        system = GrazingSystem(mock_sim_state, mock_sim_state.config, None)

        # Act
        system.on_graze(event_data)

        # Assert
        assert energy_comp.current_energy == 55.0
        mock_sim_state.environment.consume_resource.assert_called_with((5, 5), 5.0)
        # Check that the reward in the outcome matches the consumed amount
        outcome = event_data["action_outcome"]
        assert isinstance(outcome, ActionOutcome)
        assert outcome.base_reward == 5.0


@pytest.mark.asyncio
class TestResourceRegenerationSystem:
    """Tests for the ResourceRegenerationSystem."""

    async def test_regenerates_resources(self, mock_sim_state):
        """Verify that the system calls regenerate on resource components."""
        # Arrange
        res_comp1 = MagicMock(spec=ResourceComponent)
        res_comp2 = MagicMock(spec=ResourceComponent)
        mock_sim_state.entities = {
            "grass_1": {ResourceComponent: res_comp1},
            "grass_2": {ResourceComponent: res_comp2},
        }

        system = ResourceRegenerationSystem(mock_sim_state, mock_sim_state.config, None)

        # Act
        await system.update(current_tick=1)

        # Assert
        res_comp1.regenerate.assert_called_once()
        res_comp2.regenerate.assert_called_once()
