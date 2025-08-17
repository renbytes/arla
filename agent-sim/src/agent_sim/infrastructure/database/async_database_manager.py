# src/data/async_database_manager.py

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional, cast

from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .models import (
    AgentState,
    Event,
    Experiment,
    LearningCurve,
    Metric,
    ScaffoldInteraction,
    SimulationRun,
)


class AsyncDatabaseManager:
    """Asynchronous database manager using SQLAlchemy's async features."""

    def __init__(self) -> None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url or not database_url.startswith("postgresql+asyncpg"):
            raise ValueError("A valid DATABASE_URL for postgresql+asyncpg is required.")

        self.engine: AsyncEngine = create_async_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_recycle=1800,
            pool_timeout=5,
            connect_args={"timeout": 10},
        )
        self._session_factory = async_sessionmaker(bind=self.engine, expire_on_commit=False)
        self._is_operational = True

    async def check_connection(self, retries: int = 5, delay: float = 2.0) -> bool:
        """Actively checks the database connection with retries."""
        for attempt in range(retries):
            try:
                async with self.engine.connect() as conn:
                    await conn.run_sync(lambda c: c.execute(text("SELECT 1")))
                print("✅ Database connection successful.")
                self._is_operational = True
                return True
            except Exception as e:
                print(f"Attempt {attempt + 1}/{retries} failed to connect to DB: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay * (attempt + 1))
                else:
                    print("❌ Could not establish database connection after multiple retries.")
                    self._is_operational = False
                    return False
        return False

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Context manager for database sessions with proper error handling."""
        if not self._is_operational:
            yield  # type: ignore
            return

        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except (SQLAlchemyError, OSError) as e:
            print(f"DB Error: {e}. Tripping circuit breaker.")
            await session.rollback()
            self._is_operational = False
            raise
        finally:
            await session.close()

    async def update_simulation_run_status(
        self, run_id: uuid.UUID, status: str, error_message: Optional[str] = None
    ) -> None:
        """Updates the status and completion time of a simulation run."""
        if not self._is_operational:
            return
        try:
            async with self.get_session() as session:
                stmt = select(SimulationRun).where(SimulationRun.id == run_id)
                result = await session.execute(stmt)
                run = result.scalar_one_or_none()
                if run:
                    run.status = status
                    run.error_message = error_message
                    if status in ["completed", "failed"]:
                        run.completed_at = datetime.utcnow()
        except Exception as e:
            print(f"Failed to update simulation run status: {e}")
            self._is_operational = False

    async def log_event(
        self,
        simulation_id: uuid.UUID,
        tick: int,
        agent_id: str,
        action_type: str,
        success: bool,
        reward: float,
        message: str,
        details: Dict[str, Any],
    ) -> None:
        """Log an action event asynchronously."""
        if not self._is_operational:
            return
        try:
            async with self.get_session() as session:
                event = Event(
                    simulation_id=simulation_id,
                    tick=tick,
                    agent_id=agent_id,
                    action_type=action_type,
                    success=success,
                    reward=reward,
                    message=message,
                    details=details,
                )
                session.add(event)
        except Exception as e:
            print(f"Failed to log event: {e}")
            self._is_operational = False

    async def create_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        total_runs: int,
        simulation_package: str,
        mlflow_experiment_id: str,
    ) -> uuid.UUID:
        """Create a new experiment record asynchronously."""
        if not self._is_operational:
            # If DB is down, we can't create an experiment, so this must fail.
            raise ConnectionError("Database is not operational, cannot create experiment.")
        async with self.get_session() as session:
            experiment = Experiment(
                mlflow_experiment_id=mlflow_experiment_id,
                name=name,
                config=config,
                total_runs=total_runs,
                simulation_package=simulation_package,
                status="created",
            )
            session.add(experiment)
            await session.flush()
            await session.refresh(experiment)
            if experiment.id is None:
                raise RuntimeError("Failed to create experiment ID.")
            return cast(uuid.UUID, experiment.id)

    async def create_simulation_run(
        self,
        run_id: uuid.UUID,
        experiment_id: uuid.UUID,
        scenario_name: str,
        config: Dict[str, Any],
        task_id: str,
    ) -> None:
        """Create a new simulation run record asynchronously."""
        if not self._is_operational:
            return
        try:
            async with self.get_session() as session:
                run = SimulationRun(
                    id=run_id,
                    experiment_id=experiment_id,
                    scenario_name=scenario_name,
                    config=config,
                    random_seed=config.get("random_seed"),
                    task_id=task_id,
                    status="queued",
                )
                session.add(run)
        except Exception as e:
            print(f"Failed to create simulation run: {e}")
            self._is_operational = False

    async def log_agent_state(
        self,
        simulation_id: uuid.UUID,
        tick: int,
        agent_id: str,
        components_data: Dict[str, Any],
    ) -> None:
        """Log agent state for a specific tick asynchronously."""
        if not self._is_operational:
            return
        try:
            async with self.get_session() as session:
                state = AgentState(
                    simulation_id=simulation_id,
                    tick=tick,
                    agent_id=agent_id,
                    components_data=components_data,
                )
                session.add(state)
        except Exception as e:
            print(f"Failed to log agent state: {e}")
            self._is_operational = False

    async def log_metrics(self, simulation_id: uuid.UUID, tick: int, metrics_data: Dict[str, Any]) -> None:
        """Log aggregated metrics for a specific tick asynchronously."""
        if not self._is_operational:
            return
        try:
            async with self.get_session() as session:
                metric = Metric(simulation_id=simulation_id, tick=tick, **metrics_data)
                session.add(metric)
        except Exception as e:
            print(f"Failed to log metrics: {e}")
            self._is_operational = False

    async def log_scaffold_interaction(self, **kwargs: Any) -> None:
        """Logs a cognitive scaffold interaction to the database asynchronously."""
        if not self._is_operational:
            return
        try:
            async with self.get_session() as session:
                interaction = ScaffoldInteraction(**kwargs)
                session.add(interaction)
        except Exception as e:
            print(f"Failed to log scaffold interaction: {e}")
            self._is_operational = False

    async def log_learning_curve(self, simulation_id: uuid.UUID, tick: int, agent_id: str, q_loss: float) -> None:
        """Log learning curve data"""
        if not self._is_operational:
            return
        try:
            async with self.get_session() as session:
                learning_curve = LearningCurve(simulation_id=simulation_id, tick=tick, agent_id=agent_id, q_loss=q_loss)
                session.add(learning_curve)
        except Exception as e:
            print(f"Failed to log learning curve: {e}")
            self._is_operational = False
