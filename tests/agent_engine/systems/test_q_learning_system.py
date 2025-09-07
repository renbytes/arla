import pytest
from unittest.mock import MagicMock, create_autospec, patch
import numpy as np

from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_outcome import ActionOutcome
from agent_core.core.ecs.component import ActionPlanComponent, TimeBudgetComponent
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.systems.causal_graph_system import CausalGraphSystem
from agent_engine.systems.components import QLearningComponent
from agent_engine.systems.q_learning_system import QLearningSystem


@pytest.fixture
def system_setup():
    """A comprehensive fixture to set up the QLearningSystem and its dependencies."""
    mock_state = create_autospec(SimulationState, instance=True)
    mock_state.entities = MagicMock()
    mock_state.environment = MagicMock()
    mock_state.device = "cpu"  # Ensure device is set for tensor creation

    mock_bus = MagicMock()
    mock_encoder = MagicMock()
    mock_causal_system = create_autospec(CausalGraphSystem, instance=True)
    mock_causal_system.estimate_causal_effect.return_value = 2.0

    # Mock the config object to have the nested structure the system expects
    mock_config = MagicMock()
    mock_config.learning.q_learning.gamma = 0.95

    system = QLearningSystem(
        simulation_state=mock_state,
        config=mock_config,
        cognitive_scaffold=MagicMock(),
        state_encoder=mock_encoder,
        causal_graph_system=mock_causal_system,
    )
    system.event_bus = mock_bus

    agent_id = "agent_1"

    # Ensure the QLearningComponent is properly initialized for tests
    q_comp = QLearningComponent(
        state_feature_dim=16,
        internal_state_dim=1,
        action_feature_dim=13,
        q_learning_alpha=0.001,
        device="cpu",
    )

    mock_state.get_component.return_value = q_comp
    system.previous_states[agent_id] = np.ones(16)
    mock_encoder.encode_state.return_value = np.ones(16)
    mock_encoder.encode_internal_state.return_value = np.ones(1)

    return (
        system,
        mock_state,
        mock_bus,
        mock_encoder,
        mock_causal_system,
        agent_id,
        q_comp,
    )


class TestQLearningSystem:
    """Unit tests for the causally-enhanced QLearningSystem."""

    @patch("agent_engine.systems.q_learning_system.action_registry")
    @patch(
        "agent_engine.systems.q_learning_system.QLearningSystem._perform_learning_step"
    )
    def test_on_action_executed_uses_causal_reward(
        self, mock_learning_step, mock_registry, system_setup
    ):
        """
        Tests that the event handler correctly blends the observed reward with the
        causal estimate before calling the learning step.
        """
        system, mock_state, _, mock_encoder, mock_causal_system, agent_id, _ = (
            system_setup
        )
        mock_registry.get_all_actions.return_value = []
        mock_state.entities.get.return_value = {"some_component": MagicMock()}

        mock_action_type = create_autospec(ActionInterface, instance=True)
        mock_action_type.action_id = "move"
        mock_action_type.get_feature_vector.return_value = [0.0] * 13

        event_data = {
            "entity_id": agent_id,
            "action_plan": MagicMock(action_type=mock_action_type, params={}),
            "action_outcome": ActionOutcome(True, "m", 10.0, {}),
            "current_tick": 1,
        }

        system.on_action_executed(event_data)

        mock_causal_system.estimate_causal_effect.assert_called_once_with(
            agent_id=agent_id, treatment_value="move"
        )
        mock_encoder.encode_internal_state.assert_called_once()
        mock_learning_step.assert_called_once()
        blended_reward = mock_learning_step.call_args[0][6]
        assert blended_reward == pytest.approx(6.0)

    def test_on_action_executed_skips_if_no_previous_state(self, system_setup):
        """
        Tests that the system gracefully skips the learning step if no previous
        state is cached for the agent.
        """
        system, _, _, _, _, agent_id, _ = system_setup
        del system.previous_states[agent_id]

        event_data = {
            "entity_id": agent_id,
            "action_plan": MagicMock(action_type=create_autospec(ActionInterface)),
            "action_outcome": ActionOutcome(True, "m", 1.0, {}),
            "current_tick": 1,
        }

        with patch.object(system, "_perform_learning_step") as mock_learning_step:
            system.on_action_executed(event_data)
            mock_learning_step.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_skips_inactive_agents(self, system_setup):
        """
        Tests that the update method does not cache states for inactive agents.
        """
        system, mock_state, _, mock_encoder, _, _, _ = system_setup

        inactive_agent_id = "inactive_agent"
        inactive_time_comp = TimeBudgetComponent(100)
        inactive_time_comp.is_active = False

        mock_state.get_entities_with_components.return_value = {
            inactive_agent_id: {TimeBudgetComponent: inactive_time_comp}
        }

        await system.update(current_tick=1)

        mock_encoder.encode_state.assert_not_called()

    # --- NEW TESTS TO INCREASE COVERAGE ---

    def test_on_action_executed_handles_no_q_learning_component(self, system_setup):
        """
        Covers the case where an agent without a QLearningComponent performs an
        action, which should be gracefully ignored by the system.
        """
        system, mock_state, _, _, _, agent_id, _ = system_setup
        mock_state.get_component.return_value = None  # Agent has no Q-comp

        event_data = {
            "entity_id": agent_id,
            "action_plan": MagicMock(action_type=create_autospec(ActionInterface)),
            "action_outcome": ActionOutcome(True, "m", 1.0, {}),
            "current_tick": 1,
        }

        with patch.object(system, "_perform_learning_step") as mock_learning_step:
            system.on_action_executed(event_data)
            mock_learning_step.assert_not_called()

    @patch("agent_engine.systems.q_learning_system.action_registry")
    @patch(
        "agent_engine.systems.q_learning_system.QLearningSystem._perform_learning_step"
    )
    def test_on_action_executed_falls_back_without_causal_estimate(
        self, mock_learning_step, mock_registry, system_setup
    ):
        """
        Tests that the system uses the raw reward if the causal system fails
        to provide an estimate.
        """
        system, mock_state, _, _, mock_causal_system, agent_id, _ = system_setup
        mock_causal_system.estimate_causal_effect.return_value = (
            None  # Causal estimate fails
        )
        mock_registry.get_all_actions.return_value = []
        mock_state.entities.get.return_value = {"some_component": MagicMock()}
        mock_action_type = create_autospec(ActionInterface, instance=True)
        mock_action_type.action_id = "move"

        event_data = {
            "entity_id": agent_id,
            "action_plan": MagicMock(action_type=mock_action_type, params={}),
            "action_outcome": ActionOutcome(
                True, "m", 10.0, {}
            ),  # Observed reward is 10.0
            "current_tick": 1,
        }

        system.on_action_executed(event_data)

        mock_learning_step.assert_called_once()
        raw_reward = mock_learning_step.call_args[0][6]
        assert raw_reward == 10.0  # Should use the original, unblended reward

    def test_perform_learning_step_calculates_loss_correctly(self, system_setup):
        """
        Directly tests the _perform_learning_step method to ensure the Bellman
        equation and backpropagation are working as expected. This covers the
        core logic that was previously mocked.
        """
        system, mock_state, mock_bus, _, _, agent_id, q_comp = system_setup

        # Mock possible next actions to test the max_next_q calculation
        mock_next_action1 = MagicMock(spec=ActionInterface)
        mock_next_action1.get_feature_vector.return_value = [0.1] * 13
        mock_next_action2 = MagicMock(spec=ActionInterface)
        mock_next_action2.get_feature_vector.return_value = [0.9] * 13
        possible_next_actions = [
            ActionPlanComponent(mock_next_action1, {}),
            ActionPlanComponent(mock_next_action2, {}),
        ]

        # Use patch to spy on the optimizer and utility network
        with patch.object(
            q_comp.optimizer, "step"
        ) as mock_optimizer_step, patch.object(
            q_comp.utility_network, "forward", wraps=q_comp.utility_network.forward
        ) as mock_forward:
            system._perform_learning_step(
                entity_id=agent_id,
                q_comp=q_comp,
                old_state=np.array([0.5] * 16),
                new_state=np.array([0.6] * 16),
                action_features=[0.2] * 13,
                internal_features=np.array([0.3]),
                reward=10.0,
                possible_next_actions=possible_next_actions,
                current_tick=5,
            )

            # Assert that the model was called to calculate Q-values
            assert mock_forward.call_count > 0
            # Assert that the optimizer was called to update weights
            mock_optimizer_step.assert_called_once()
            # Assert that the event was published with a valid loss
            mock_bus.publish.assert_called_once()
            publish_args = mock_bus.publish.call_args[0]
            assert publish_args[0] == "q_learning_update"
            assert isinstance(publish_args[1]["q_loss"], float)
            assert not np.isnan(publish_args[1]["q_loss"])
