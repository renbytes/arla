# tests/agent-engine/systems/test_metrics_system.py

from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_engine.logging.exporter_interface import ExporterInterface
from agent_engine.logging.metrics_calculator_interface import (
    MetricsCalculatorInterface,
)
from agent_engine.systems.metrics_system import MetricsSystem


@pytest.fixture
def mock_simulation_state():
    """Fixture for a mocked SimulationState."""
    return MagicMock()


@pytest.fixture
def mock_calculators():
    """Fixture for a list of mocked MetricsCalculatorInterfaces."""
    calc1 = MagicMock(spec=MetricsCalculatorInterface)
    calc1.calculate_metrics.return_value = {"metric_a": 1.0, "metric_b": 2.0}

    calc2 = MagicMock(spec=MetricsCalculatorInterface)
    calc2.calculate_metrics.return_value = {"metric_c": 3.0}

    return [calc1, calc2]


@pytest.fixture
def mock_exporters():
    """Fixture for a list of mocked ExporterInterfaces."""
    return [AsyncMock(spec=ExporterInterface), AsyncMock(spec=ExporterInterface)]


@pytest.mark.asyncio
async def test_update_calculates_and_exports_metrics(mock_simulation_state, mock_calculators, mock_exporters):
    """
    Tests that the update method correctly calls all calculators, combines
    their metrics, and passes the result to all exporters.
    """
    # Arrange
    system = MetricsSystem(
        simulation_state=mock_simulation_state,
        config={},
        cognitive_scaffold=MagicMock(),
        calculators=mock_calculators,
        exporters=mock_exporters,
    )
    current_tick = 10

    # Act
    await system.update(current_tick)

    # Assert
    # Verify calculators were called
    for calc in mock_calculators:
        calc.calculate_metrics.assert_called_once_with(mock_simulation_state)

    # Verify exporters were called with the combined metrics
    expected_metrics = {"metric_a": 1.0, "metric_b": 2.0, "metric_c": 3.0}
    for exporter in mock_exporters:
        exporter.export_metrics.assert_awaited_once_with(current_tick, expected_metrics)


@pytest.mark.asyncio
async def test_update_handles_calculator_failure_gracefully(mock_simulation_state, mock_exporters):
    """
    Tests that the system continues to function and exports metrics from
    successful calculators even if one calculator fails.
    """
    # Arrange
    successful_calc = MagicMock(spec=MetricsCalculatorInterface)
    successful_calc.calculate_metrics.return_value = {"metric_a": 1.0}

    failing_calc = MagicMock(spec=MetricsCalculatorInterface)
    failing_calc.calculate_metrics.side_effect = Exception("Calculator failed")

    system = MetricsSystem(
        simulation_state=mock_simulation_state,
        config={},
        cognitive_scaffold=MagicMock(),
        calculators=[successful_calc, failing_calc],
        exporters=mock_exporters,
    )

    # Act
    await system.update(10)

    # Assert
    # The system should still export the metrics from the calculator that succeeded
    expected_metrics = {"metric_a": 1.0}
    for exporter in mock_exporters:
        exporter.export_metrics.assert_awaited_once_with(10, expected_metrics)


@pytest.mark.asyncio
async def test_update_handles_exporter_failure_gracefully(mock_simulation_state, mock_calculators):
    """
    Tests that a failure in one exporter does not prevent other exporters
    from receiving the metrics.
    """
    # Arrange
    successful_exporter = AsyncMock(spec=ExporterInterface)
    failing_exporter = AsyncMock(spec=ExporterInterface)
    failing_exporter.export_metrics.side_effect = Exception("Exporter failed")

    system = MetricsSystem(
        simulation_state=mock_simulation_state,
        config={},
        cognitive_scaffold=MagicMock(),
        calculators=mock_calculators,
        exporters=[successful_exporter, failing_exporter],
    )
    expected_metrics = {"metric_a": 1.0, "metric_b": 2.0, "metric_c": 3.0}

    # Act
    await system.update(10)

    # Assert
    # The successful exporter should have been called, despite the other one failing
    successful_exporter.export_metrics.assert_awaited_once_with(10, expected_metrics)
