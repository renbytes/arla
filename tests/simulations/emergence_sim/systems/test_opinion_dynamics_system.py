# tests/systems/test_opinion_dynamics_system.py
"""
Provides a comprehensive suite of unit tests for the OpinionDynamicsSystem,
ensuring its logic for opinion change and resistance is correct under various
cognitive states.
"""

import unittest
from unittest.mock import MagicMock, patch

from agent_core.agents.actions.base_action import ActionOutcome

# Import the IdentityComponent to use it as a dictionary key
from agent_core.core.ecs.component import IdentityComponent

from simulations.emergence_sim.components import OpinionComponent, PositionComponent
from simulations.emergence_sim.systems.opinion_dynamics_system import (
    OpinionDynamicsSystem,
)


class TestOpinionDynamicsSystem(unittest.TestCase):
    """
    Test suite for the OpinionDynamicsSystem.

    This class uses mock objects to isolate the system's logic from the broader
    simulation state, allowing for precise testing of its core functionality:
    - Opinion alignment reinforcement.
    - Opinion change under low identity coherence.
    - Opinion resistance under high identity coherence.
    """

    def setUp(self):
        """
        Set up a fresh testing environment before each test case.

        This method initializes mock objects for the simulation state, event bus,
        environment, and components required by the OpinionDynamicsSystem.
        """
        self.mock_sim_state = MagicMock()
        self.mock_event_bus = MagicMock()
        self.mock_environment = MagicMock()

        self.mock_sim_state.event_bus = self.mock_event_bus
        self.mock_sim_state.environment = self.mock_environment
        self.mock_sim_state.entities = {}

        # The system under test
        self.system = OpinionDynamicsSystem(
            simulation_state=self.mock_sim_state,
            config=MagicMock(),
            cognitive_scaffold=MagicMock(),
        )

    def _setup_agents(
        self,
        agent_id,
        agent_pos,
        agent_opinion,
        agent_identity_coherence,
        neighbor_id,
        neighbor_pos,
        neighbor_opinion,
    ):
        """
        Helper method to configure agents and their components for a test scenario.

        Args:
            agent_id (str): The ID of the primary agent.
            agent_pos (tuple): The position of the primary agent.
            agent_opinion (str): The initial opinion of the primary agent.
            agent_identity_coherence (float): The identity coherence score for the agent.
            neighbor_id (str): The ID of the neighboring agent.
            neighbor_pos (tuple): The position of the neighboring agent.
            neighbor_opinion (str): The opinion of the neighboring agent.
        """
        # Mock the main agent's components
        mock_identity_component = MagicMock()
        mock_identity_component.multi_domain_identity.get_identity_coherence.return_value = agent_identity_coherence

        self.mock_sim_state.entities[agent_id] = {
            PositionComponent: PositionComponent(agent_pos, self.mock_environment),
            OpinionComponent: OpinionComponent(agent_opinion),
            # FIX: Use the IdentityComponent class as the key, not a string.
            IdentityComponent: mock_identity_component,
        }

        # Mock the neighbor's components
        self.mock_sim_state.entities[neighbor_id] = {
            PositionComponent: PositionComponent(neighbor_pos, self.mock_environment),
            OpinionComponent: OpinionComponent(neighbor_opinion),
        }
        # A simplified way to get a component by type for the mock
        self.mock_sim_state.get_component.side_effect = lambda eid, ctype: self.mock_sim_state.entities[eid].get(ctype)

        # Mock environment responses
        self.mock_environment.get_neighbors.return_value = [neighbor_pos]
        self.mock_environment.get_entities_at_position.return_value = {neighbor_id}

    def test_opinion_aligns_with_neighbor(self):
        """
        Verify that an agent receives positive reinforcement if its opinion
        already matches its neighbor's.
        """
        # Arrange
        self._setup_agents(
            agent_id="agent_A",
            agent_pos=(0, 0),
            agent_opinion="Blue",
            agent_identity_coherence=0.5,
            neighbor_id="agent_B",
            neighbor_pos=(0, 1),
            neighbor_opinion="Blue",
        )
        event_data = {
            "entity_id": "agent_A",
            "action_plan_component": MagicMock(),
            "current_tick": 10,
        }

        # Act
        self.system.on_execute_influence(event_data)

        # Assert
        # Check that an outcome was published
        self.mock_event_bus.publish.assert_called_once()
        # Get the arguments passed to the publish call
        _event_name, published_data = self.mock_event_bus.publish.call_args[0]
        outcome: ActionOutcome = published_data["action_outcome"]

        self.assertTrue(outcome.success)
        self.assertEqual(outcome.base_reward, 1.0)
        self.assertIn("agrees", outcome.message)

        # Ensure the agent's opinion did not change
        agent_opinion_comp = self.mock_sim_state.entities["agent_A"][OpinionComponent]
        self.assertEqual(agent_opinion_comp.opinion, "Blue")

    def test_opinion_changes_with_low_identity_coherence(self):
        """
        Test that an agent with low identity coherence adopts the opinion of a
        dissenting neighbor.
        """
        # Arrange
        self._setup_agents(
            agent_id="agent_A",
            agent_pos=(0, 0),
            agent_opinion="Blue",
            agent_identity_coherence=0.2,  # Low coherence = high chance to change
            neighbor_id="agent_B",
            neighbor_pos=(0, 1),
            neighbor_opinion="Orange",
        )
        event_data = {
            "entity_id": "agent_A",
            "action_plan_component": MagicMock(),
            "current_tick": 10,
        }

        # Act
        # We patch random.random to ensure the change happens (returns value < probability)
        with patch("random.random", return_value=0.5):  # 0.5 < (1.0 - 0.2)
            self.system.on_execute_influence(event_data)

        # Assert
        self.mock_event_bus.publish.assert_called_once()
        _event_name, published_data = self.mock_event_bus.publish.call_args[0]
        outcome: ActionOutcome = published_data["action_outcome"]

        self.assertTrue(outcome.success)
        self.assertEqual(outcome.base_reward, 1.0)
        self.assertIn("changed to Orange", outcome.message)

        # Verify the agent's opinion was updated
        agent_opinion_comp = self.mock_sim_state.entities["agent_A"][OpinionComponent]
        self.assertEqual(agent_opinion_comp.opinion, "Orange")

    def test_opinion_resists_with_high_identity_coherence(self):
        """
        Test that an agent with high identity coherence resists adopting the
        opinion of a dissenting neighbor.
        """
        # Arrange
        self._setup_agents(
            agent_id="agent_A",
            agent_pos=(0, 0),
            agent_opinion="Blue",
            agent_identity_coherence=0.9,  # High coherence = low chance to change
            neighbor_id="agent_B",
            neighbor_pos=(0, 1),
            neighbor_opinion="Orange",
        )
        event_data = {
            "entity_id": "agent_A",
            "action_plan_component": MagicMock(),
            "current_tick": 10,
        }

        # Act
        # We patch random.random to ensure the change is resisted (returns value > probability)
        with patch("random.random", return_value=0.5):  # 0.5 > (1.0 - 0.9)
            self.system.on_execute_influence(event_data)

        # Assert
        self.mock_event_bus.publish.assert_called_once()
        _event_name, published_data = self.mock_event_bus.publish.call_args[0]
        outcome: ActionOutcome = published_data["action_outcome"]

        self.assertFalse(outcome.success)
        self.assertEqual(outcome.base_reward, -1.0)
        self.assertIn("Resisted opinion change", outcome.message)

        # Verify the agent's opinion did NOT change
        agent_opinion_comp = self.mock_sim_state.entities["agent_A"][OpinionComponent]
        self.assertEqual(agent_opinion_comp.opinion, "Blue")

    def test_no_neighbors_found(self):
        """
        Test that the system handles the case where an agent has no neighbors
        and does not publish an outcome.
        """
        # Arrange
        self._setup_agents("agent_A", (0, 0), "Blue", 0.5, "agent_B", (5, 5), "Orange")
        # Mock the environment to return no neighbors
        self.mock_environment.get_neighbors.return_value = []
        event_data = {
            "entity_id": "agent_A",
            "action_plan_component": MagicMock(),
            "current_tick": 10,
        }

        # Act
        self.system.on_execute_influence(event_data)

        # Assert
        # No outcome should be published if no interaction occurs
        self.mock_event_bus.publish.assert_not_called()
