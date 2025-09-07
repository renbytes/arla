# FILE: tests/simulations/berry_sim/test_systems.py
"""
Unit tests for the systems in the berry_sim simulation.
"""

import pytest
from unittest.mock import MagicMock, patch
from agent_core.core.ecs.component import TimeBudgetComponent
from simulations.berry_sim.systems import (
    BerrySpawningSystem,
    ConsumptionSystem,
    VitalsSystem,
)
from simulations.berry_sim.components import HealthComponent, PositionComponent
from simulations.berry_sim.environment import BerryWorldEnvironment


@pytest.fixture
def mock_sim_state_systems():
    """Provides a mock SimulationState for system tests."""
    state = MagicMock()
    state.environment = BerryWorldEnvironment(width=10, height=10)
    state.event_bus = MagicMock()
    # CORRECTED: Configure the mock to handle nested attribute access for config
    state.config = MagicMock()
    state.config.environment.spawning.red_rate = 0.1
    state.config.environment.spawning.blue_rate = 0.1
    state.config.environment.spawning.yellow_rate = 0.1
    return state


class TestBerrySpawningSystem:
    """Tests for the BerrySpawningSystem."""

    @pytest.mark.asyncio
    @patch("random.random")
    async def test_update_spawns_berries(self, mock_random, mock_sim_state_systems):
        """Verify berries are spawned when the random check passes."""
        system = BerrySpawningSystem(
            mock_sim_state_systems, mock_sim_state_systems.config, MagicMock()
        )

        mock_random.return_value = 0.01  # This will pass the < 0.1 check
        mock_sim_state_systems.environment.rock_locations.add((5, 5))

        await system.update(current_tick=500)

        berry_types = list(mock_sim_state_systems.environment.berry_locations.values())
        assert "red" in berry_types
        assert "blue" in berry_types
        assert "yellow" in berry_types


class TestConsumptionSystem:
    """Tests for the ConsumptionSystem."""

    def test_on_eat_berry_success(self, mock_sim_state_systems):
        """Test a successful berry consumption event."""
        system = ConsumptionSystem(mock_sim_state_systems, MagicMock(), MagicMock())
        agent_id = "agent_1"
        berry_pos = (2, 2)

        health_comp = HealthComponent(current_health=50, initial_health=100)
        pos_comp = PositionComponent(x=berry_pos[0], y=berry_pos[1])
        mock_sim_state_systems.get_component.side_effect = lambda eid, comp_type: {
            PositionComponent: pos_comp,
            HealthComponent: health_comp,
        }.get(comp_type)
        mock_sim_state_systems.environment.berry_locations[berry_pos] = "red"

        event_data = {
            "entity_id": agent_id,
            "action_plan_component": MagicMock(params={"berry_type": "red"}),
            "current_tick": 10,
        }

        system.on_eat_berry(event_data)

        assert health_comp.current_health == 60
        assert berry_pos not in mock_sim_state_systems.environment.berry_locations

        mock_sim_state_systems.event_bus.publish.assert_called_once()
        call_args = mock_sim_state_systems.event_bus.publish.call_args[0]
        assert call_args[0] == "action_outcome_ready"
        assert call_args[1]["action_outcome"].success is True


class TestVitalsSystem:
    """Tests for the VitalsSystem."""

    @pytest.mark.asyncio
    async def test_update_deactivates_agent_at_zero_health(
        self, mock_sim_state_systems
    ):
        """Verify an agent is deactivated when its health reaches zero."""
        system = VitalsSystem(mock_sim_state_systems, MagicMock(), MagicMock())
        agent_id = "agent_1"

        health_comp = HealthComponent(current_health=0, initial_health=100)
        time_comp = TimeBudgetComponent(initial_time_budget=100)
        time_comp.is_active = True
        components = {HealthComponent: health_comp, TimeBudgetComponent: time_comp}
        mock_sim_state_systems.get_entities_with_components.return_value = {
            agent_id: components
        }

        await system.update(current_tick=100)

        assert time_comp.is_active is False

        mock_sim_state_systems.event_bus.publish.assert_called_with(
            "agent_deactivated", {"entity_id": agent_id, "current_tick": 100}
        )
