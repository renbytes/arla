# tests/integration/run_bql_suite.py
import asyncio
import uuid
from typing import Dict, Any

# --- Imports for the BQL Assertion Framework ---
from agent_core.testing.bql import BQL
from agent_sim.infrastructure.database.async_database_manager import (
    AsyncDatabaseManager,
)
from agent_sim.infrastructure.database.bql_test_reporter import BQLTestReporter

# --- Imports for Sugarscape Dependencies (Mocked or real for integration) ---
from simulations.sugarscape_sim.run import (
    setup_and_run,
)  # The main simulation entry point
from simulations.sugarscape_sim.components import EnergyComponent, PositionComponent


def define_behavioral_assertions() -> BQL:
    """
    Defines the full test suite using BQL objects.
    """
    # 1. Map components required for assertions (These must be available in the DB logs)
    component_map = {
        "EnergyComponent": EnergyComponent,
        "PositionComponent": PositionComponent,
    }
    bql = BQL(component_map)

    print("\n[BQL Suite] Defining Behavioral Assertions...")

    # 2. ASSERT_ALWAYS: Physics/Integrity Check
    # Assertion 1: Energy must ALWAYS be non-negative. If this fails, the physics engine broke.
    bql.assert_always(bql.EnergyComponent.current_energy >= 0)

    # 3. ASSERT_EVENTUALLY: Survival/Goal Check
    # Assertion 2: Agents must EVENTUALLY reach a high energy level.
    bql.assert_eventually(bql.EnergyComponent.current_energy > 200.0)

    # 4. ASSERT_EVENTUALLY: Behavioral Check (Did any agent move?)
    # Assertion 3: At least one agent must EVENTUALLY be in the top right corner.
    bql.assert_eventually(bql.PositionComponent.x > 40, bql.PositionComponent.y > 40)

    return bql


async def main():
    """
    Main orchestration function: runs sim, then runs BQL report.
    """
    # --- Setup IDs and Config ---
    run_id = f"test_bql_{uuid.uuid4().hex[:8]}"
    task_id = "BQL_TEST_RUN"
    experiment_name = "BQL_Regression_Tests"

    # Use minimal config for fast testing
    config_overrides: Dict[str, Any] = {
        "simulation": {"steps": 50, "random_seed": 10},
        "agent": {"foundational": {"num_agents": 2}},
    }

    # 1. Define the assertions suite
    bql_suite = define_behavioral_assertions()

    # 2. Run the Simulation (Engine Phase)
    # Pass the assertions list directly to the SimulationManager constructor
    # (as modified in agent-engine/src/agent_engine/simulation/engine.py)
    print(f"\n[ORCHESTRATOR] Starting Simulation (Run: {run_id})...")

    try:
        # The core simulation execution
        await setup_and_run(
            run_id=run_id,
            task_id=task_id,
            experiment_name=experiment_name,
            config_overrides=config_overrides,
            behavioral_assertions=bql_suite.assertions,  # <--- THE KEY INJECTION
        )
        print(f"[ORCHESTRATOR] Simulation finished. Run ID: {run_id}")

    except Exception as e:
        print(f"[ORCHESTRATOR] FATAL ERROR during simulation run: {e}")
        return

    # 3. Execute Assertions and Report (Analysis Phase)
    db_manager = AsyncDatabaseManager()
    await db_manager.check_connection()

    reporter = BQLTestReporter(
        run_id=run_id, db_manager=db_manager, assertions=bql_suite.assertions
    )

    final_results = await reporter.execute_and_report()

    # Determine overall exit code for a CI/CD pipeline
    if all(final_results.values()):
        print("\n[CI/CD STATUS] All Behavioral Tests Passed.")
    else:
        print("\n[CI/CD STATUS] Some Behavioral Tests Failed.")


if __name__ == "__main__":
    asyncio.run(main())
