# agent-engine/tests/systems/test_reflection_system.py

from unittest.mock import ANY, MagicMock, call, patch

import pytest
from agent_core.core.ecs.component import (
    AffectComponent,
    EmotionComponent,
    EpisodeComponent,
    GoalComponent,
    IdentityComponent,
    MemoryComponent,
    SocialMemoryComponent,
    TimeBudgetComponent,
    ValidationComponent,
    ValueSystemComponent,
)
from agent_engine.cognition.identity.domain_identity import MultiDomainIdentity
from agent_engine.cognition.reflection.episode import Episode

# Subject under test
from agent_engine.systems.reflection_system import ReflectionSystem

# Fixtures


@pytest.fixture
def mock_simulation_state():
    """Mocks the SimulationState and its contained components."""
    state = MagicMock()

    # Use real instances for all required components to ensure the
    # `_all_required_components_present` check passes reliably.
    state.entities = {
        "agent1": {
            TimeBudgetComponent: TimeBudgetComponent(
                initial_time_budget=100.0, lifespan_std_dev_percent=0.0
            ),
            MemoryComponent: MemoryComponent(),
            EpisodeComponent: EpisodeComponent(),
            AffectComponent: AffectComponent(affective_buffer_maxlen=100),
            IdentityComponent: IdentityComponent(
                multi_domain_identity=MultiDomainIdentity(embedding_dim=4)
            ),
            GoalComponent: GoalComponent(embedding_dim=4),
            EmotionComponent: EmotionComponent(),
            SocialMemoryComponent: SocialMemoryComponent(
                schema_embedding_dim=4, device="cpu"
            ),
            ValidationComponent: ValidationComponent(),
            ValueSystemComponent: ValueSystemComponent(),
        }
    }

    # Ensure get_entities_with_components returns our mock entity
    state.get_entities_with_components.return_value = {
        "agent1": state.entities["agent1"]
    }
    return state


@pytest.fixture
def mock_narrative_provider():
    """Mocks the NarrativeContextProviderInterface."""
    provider = MagicMock()
    # Return a predictable context dictionary
    provider.get_narrative_context.return_value = {
        "narrative": "I was at (5,5) and felt neutral.",
        "llm_final_account": "",  # This is filled in by the system
        "social_feedback": {},
    }
    return provider


@pytest.fixture
def mock_cognitive_scaffold():
    """Mocks the CognitiveScaffold to return predictable LLM responses."""
    scaffold = MagicMock()

    # Set up a side effect to handle different query purposes
    def query_side_effect(agent_id, purpose, prompt, current_tick):
        if purpose == "episode_theming":
            return "A Quiet Day"
        if purpose == "reflection_synthesis":
            return "I have learned that I am a pensive agent."
        return "Unknown Purpose"

    scaffold.query.side_effect = query_side_effect
    return scaffold


@pytest.fixture
def mock_event_bus():
    """Mocks the EventBus."""
    return MagicMock()


@pytest.fixture
def reflection_system(
    mock_simulation_state,
    mock_narrative_provider,
    mock_cognitive_scaffold,
    mock_event_bus,
):
    """Provides an initialized ReflectionSystem with all dependencies mocked."""
    mock_simulation_state.event_bus = mock_event_bus

    system = ReflectionSystem(
        simulation_state=mock_simulation_state,
        config={"learning": {"memory": {"reflection_interval": 50}}},
        cognitive_scaffold=mock_cognitive_scaffold,
        narrative_context_provider=mock_narrative_provider,
    )
    return system


# Test Cases


class TestReflectionSystem:
    @pytest.mark.asyncio
    async def test_run_reflection_cycle_orchestration(
        self,
        reflection_system,
        mock_simulation_state,
        mock_narrative_provider,
        mock_cognitive_scaffold,
        mock_event_bus,
    ):
        """
        Tests that _run_reflection_cycle correctly orchestrates the full process:
        chunking, synthesizing, and publishing.
        """
        # Arrange
        # Use the correct key "action_plan" to match what ActionSystem publishes.
        reflection_system.event_buffer["agent1"] = [
            {
                "current_tick": 1,
                "action_plan": MagicMock(action_type=MagicMock(name="Test Action")),
            }
        ]
        components = mock_simulation_state.entities["agent1"]

        # Act
        # The component check is validated in a separate test, so we patch it here
        # to isolate the orchestration logic.
        with patch.object(
            reflection_system, "_all_required_components_present", return_value=True
        ):
            await reflection_system._run_reflection_cycle(
                entity_id="agent1",
                components=components,
                current_tick=50,
                is_final_reflection=False,
            )

        # Assert
        # 1. Verify episode chunking and processing was attempted
        # It should call the scaffold to get a theme for the new episode
        # Use keyword arguments to match the actual implementation
        mock_cognitive_scaffold.query.assert_any_call(
            agent_id="agent1", purpose="episode_theming", prompt=ANY, current_tick=50
        )

        # 2. Verify narrative synthesis was called
        mock_narrative_provider.get_narrative_context.assert_called_once()
        mock_cognitive_scaffold.query.assert_any_call(
            agent_id="agent1",
            purpose="reflection_synthesis",
            prompt=ANY,
            current_tick=50,
        )

        # 3. Verify the final events were published
        expected_calls = [
            call("reflection_validated", ANY),
            call("update_goals_event", ANY),
            call("reflection_completed", ANY),
        ]
        mock_event_bus.publish.assert_has_calls(expected_calls, any_order=True)

        # Check the content of the 'reflection_completed' event
        completed_call = next(
            c
            for c in mock_event_bus.publish.call_args_list
            if c[0][0] == "reflection_completed"
        )
        completed_data = completed_call[0][1]
        assert completed_data["entity_id"] == "agent1"
        assert (
            completed_data["context"]["llm_final_account"]
            == "I have learned that I am a pensive agent."
        )

    def test_on_action_executed_for_chunking(self, reflection_system):
        """
        Tests that events are correctly buffered for later processing.
        """
        # Arrange
        event_data_1 = {"entity_id": "agent1", "data": "event1"}
        event_data_2 = {"entity_id": "agent2", "data": "event2"}
        event_data_3 = {"entity_id": "agent1", "data": "event3"}

        # Act
        reflection_system.on_action_executed_for_chunking(event_data_1)
        reflection_system.on_action_executed_for_chunking(event_data_2)
        reflection_system.on_action_executed_for_chunking(event_data_3)

        # Assert
        assert len(reflection_system.event_buffer["agent1"]) == 2
        assert len(reflection_system.event_buffer["agent2"]) == 1
        assert reflection_system.event_buffer["agent1"][1]["data"] == "event3"

    def test_chunk_and_process_episodes(self, reflection_system, mock_simulation_state):
        """
        Tests that buffered events are correctly turned into an Episode object
        and that the buffer for the agent is cleared.
        """
        # Arrange
        entity_id = "agent1"
        components = mock_simulation_state.entities[entity_id]
        episode_comp = components[EpisodeComponent]

        # Add events to the buffer using the correct key
        reflection_system.event_buffer[entity_id] = [
            {
                "current_tick": 10,
                "action_plan": MagicMock(action_type=MagicMock(name="MOVE")),
            },
            {"current_tick": 12, "action_outcome": MagicMock(details={})},
        ]

        # Act
        reflection_system._chunk_and_process_episodes(entity_id, components, 20)

        # Assert
        assert len(episode_comp.episodes) == 1
        new_episode = episode_comp.episodes[0]
        assert isinstance(new_episode, Episode)
        assert new_episode.theme == "A Quiet Day"
        assert new_episode.start_tick == 10
        assert new_episode.end_tick == 20
        # Buffer for this agent should now be empty
        assert not reflection_system.event_buffer[entity_id]

    @pytest.mark.asyncio
    async def test_reflection_cycle_skips_for_missing_components(
        self, reflection_system, mock_narrative_provider
    ):
        """
        Tests that the reflection cycle gracefully exits if an agent is missing
        the required components for reflection, preventing errors.
        """
        # Arrange
        # Create a component dictionary that is missing a required component
        incomplete_components = {TimeBudgetComponent: TimeBudgetComponent(100, 0.0)}

        # Act
        await reflection_system._run_reflection_cycle(
            entity_id="agent_incomplete",
            components=incomplete_components,
            current_tick=50,
            is_final_reflection=False,
        )

        # Assert
        # The narrative provider should not have been called because the system exited early.
        mock_narrative_provider.get_narrative_context.assert_not_called()
