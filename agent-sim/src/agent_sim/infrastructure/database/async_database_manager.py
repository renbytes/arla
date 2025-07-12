# src/data/async_database_manager.py

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from .models import (
    AgentState,
    Event,
    Experiment,
    LearningCurve,
    Metric,
    ScaffoldInteraction,
    SimulationRun,
)

DB_OPERATION_LOCK = asyncio.Lock()


class AsyncDatabaseManager:
    """Asynchronous database manager using SQLAlchemy's async features."""

    def __init__(self) -> None:
        self.engine: Optional[Any] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread_id: Optional[int] = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize the async engine with proper connection pooling"""
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL environment variable is not set")

        # Use NullPool to avoid connection reuse issues in multi-threaded environments
        self.engine = create_async_engine(
            db_url, poolclass=NullPool, echo=False, future=True
        )  # This prevents connection sharing issues
        self.session_factory = async_sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Context manager for database sessions with proper error handling"""
        if self.session_factory is None:
            raise RuntimeError("Database session factory not initialized.")
        async with DB_OPERATION_LOCK:  # Serialize all database operations
            async with self.session_factory() as session:
                try:
                    yield session
                    await session.commit()
                except Exception as e:
                    await session.rollback()
                    raise e

    async def update_simulation_run_status(
        self, run_id: uuid.UUID, status: str, error_message: Optional[str] = None
    ) -> None:
        """Updates the status and completion time of a simulation run."""
        async with self.get_session() as session:
            stmt = select(SimulationRun).where(SimulationRun.id == run_id)
            result = await session.execute(stmt)
            run = result.scalar_one_or_none()
            if run:
                run.status = status
                run.error_message = error_message
                if status in ["completed", "failed"]:
                    run.completed_at = datetime.utcnow()

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

    async def create_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        total_runs: int,
        simulation_package: str,
        mlflow_experiment_id: str,
    ) -> uuid.UUID:
        """Create a new experiment record asynchronously."""
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

    async def log_agent_state(
        self,
        simulation_id: uuid.UUID,
        tick: int,
        agent_id: str,
        components_data: Dict[str, Any],
    ) -> None:
        """Log agent state for a specific tick asynchronously."""
        async with self.get_session() as session:
            # Pass the generic dictionary to the AgentState constructor
            state = AgentState(
                simulation_id=simulation_id,
                tick=tick,
                agent_id=agent_id,
                components_data=components_data,
            )
            session.add(state)

    async def log_metrics(self, simulation_id: uuid.UUID, tick: int, metrics_data: Dict[str, Any]) -> None:
        """Log aggregated metrics for a specific tick asynchronously."""
        async with self.get_session() as session:
            metric = Metric(simulation_id=simulation_id, tick=tick, **metrics_data)
            session.add(metric)

    async def log_scaffold_interaction(self, **kwargs: Any) -> None:
        """Logs a cognitive scaffold interaction to the database asynchronously."""
        async with self.get_session() as session:
            interaction = ScaffoldInteraction(**kwargs)
            session.add(interaction)

    async def log_learning_curve(self, simulation_id: uuid.UUID, tick: int, agent_id: str, q_loss: float) -> None:
        """Log learning curve data"""
        async with self.get_session() as session:
            learning_curve = LearningCurve(simulation_id=simulation_id, tick=tick, agent_id=agent_id, q_loss=q_loss)
            session.add(learning_curve)

    def close(self) -> None:
        """This method is no longer needed in a fully async, session-per-operation model."""
        pass
