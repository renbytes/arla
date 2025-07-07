# src/agent_core/tests/agents/actions/test_base_action.py

# Subject under test
from agent_core.agents.actions.base_action import ActionOutcome, Intent

# --- Test Cases for Intent Enum ---


def test_intent_enum_values():
    """
    Tests that the Intent enum has the correct string values.
    """
    assert Intent.SOLITARY.value == "SOLITARY"
    assert Intent.COOPERATE.value == "COOPERATE"
    assert Intent.COMPETE.value == "COMPETE"


# --- Test Cases for ActionOutcome ---


def test_action_outcome_initialization_full():
    """
    Tests that ActionOutcome correctly initializes with all arguments provided.
    """
    # Arrange
    details_dict = {"reason": "test_reason", "value": 123}

    # Act
    outcome = ActionOutcome(
        success=True,
        message="Test successful",
        base_reward=10.5,
        details=details_dict,
    )

    # Assert
    assert outcome.success is True
    assert outcome.message == "Test successful"
    assert outcome.base_reward == 10.5
    assert outcome.details == details_dict
    # The 'reward' attribute should initially be the same as 'base_reward'
    assert outcome.reward == 10.5


def test_action_outcome_initialization_defaults():
    """
    Tests that ActionOutcome correctly initializes with default values when
    optional arguments are not provided.
    """
    # Act
    outcome = ActionOutcome(
        success=False,
        message="Test failed",
        base_reward=-1.0,
    )

    # Assert
    assert outcome.success is False
    assert outcome.message == "Test failed"
    assert outcome.base_reward == -1.0
    # The 'details' attribute should default to an empty dictionary
    assert outcome.details == {}
    assert outcome.reward == -1.0


def test_action_outcome_reward_is_mutable():
    """
    Tests that the 'reward' attribute can be modified after initialization,
    which is important for systems that apply subjective multipliers.
    """
    # Arrange
    outcome = ActionOutcome(
        success=True,
        message="Initial outcome",
        base_reward=5.0,
    )

    # Act
    outcome.reward = 15.0

    # Assert
    assert outcome.base_reward == 5.0
    assert outcome.reward == 15.0
