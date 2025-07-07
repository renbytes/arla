# src/data/models.py
"""
Defines the SQLAlchemy ORM models for the entire simulation database.
This includes tables for experiments, simulation runs, agent states, events,
and all LLM interactions via the Cognitive Scaffold.

The setup is asynchronous, designed to work with `asyncpg` for PostgreSQL
or `aiosqlite` for local testing.
"""

import os
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import DECIMAL, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid


# --- MODIFIED: Use a class-based Base for mypy compatibility ---
class Base(DeclarativeBase):
    pass


class Experiment(Base):
    """Represents a collection of simulation runs, defining a complete experiment."""

    __tablename__ = "experiments"

    # --- Use Mapped and mapped_column for all columns ---
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String, default="created")
    total_runs: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    simulation_runs: Mapped[List["SimulationRun"]] = relationship("SimulationRun", back_populates="experiment", cascade="all, delete-orphan")


class SimulationRun(Base):
    """Represents a single simulation run within a larger experiment."""

    __tablename__ = "simulation_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    experiment_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("experiments.id"), index=True)
    task_id: Mapped[Optional[str]] = mapped_column(String, index=True, nullable=True)
    scenario_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    random_seed: Mapped[Optional[int]] = mapped_column(Integer, index=True, nullable=True)
    config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String, default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="simulation_runs")


class AgentState(Base):
    """Logs a snapshot of an agent's state at a specific tick."""

    __tablename__ = "agent_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    simulation_id: Mapped[str] = mapped_column(String, ForeignKey("simulation_runs.id"), index=True)
    tick: Mapped[int] = mapped_column(Integer, index=True)
    agent_id: Mapped[str] = mapped_column(String, index=True)
    health: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    time_budget: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resources: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pos_x: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    pos_y: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    valence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    arousal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_goal: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    cognitive_dissonance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_tolerance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    identity_coherence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    identity_stability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    social_validation_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    identity_domains_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    value_multipliers: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    identity_embedding: Mapped[Optional[List[float]]] = mapped_column(JSONB, nullable=True)
    social_memory: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    __table_args__ = (UniqueConstraint("simulation_id", "tick", "agent_id", name="uq_agent_state_tick"),)


class Event(Base):
    """Represents a single, discrete action event that occurred during a simulation."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    simulation_id: Mapped[str] = mapped_column(String, ForeignKey("simulation_runs.id"), index=True)
    tick: Mapped[int] = mapped_column(Integer, index=True)
    agent_id: Mapped[str] = mapped_column(String, index=True)
    action_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    success: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)


class ScaffoldInteraction(Base):
    """Logs every prompt and completion pair sent to the Cognitive Scaffold."""

    __tablename__ = "scaffold_interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    simulation_id: Mapped[str] = mapped_column(String(255), ForeignKey("simulation_runs.id"), index=True)
    tick: Mapped[int] = mapped_column(Integer, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    agent_id: Mapped[str] = mapped_column(String(255), index=True)
    purpose: Mapped[str] = mapped_column(String(255), index=True)
    prompt: Mapped[str] = mapped_column(Text)
    llm_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 6), nullable=True)
    outcome_event_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("events.id"), nullable=True)
    outcome_reward: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 4), nullable=True)


class Metric(Base):
    """Stores aggregated metrics for an entire simulation at a specific tick."""

    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    simulation_id: Mapped[str] = mapped_column(String, ForeignKey("simulation_runs.id"), index=True)
    tick: Mapped[int] = mapped_column(Integer, index=True)
    active_agents: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    avg_reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_health: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_time_budget: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_resources: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    goal_distribution: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    avg_cognitive_dissonance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class LearningCurve(Base):
    """Stores data points for plotting agent learning curves, such as Q-loss."""

    __tablename__ = "learning_curves"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    simulation_id: Mapped[str] = mapped_column(String, ForeignKey("simulation_runs.id"), index=True)
    tick: Mapped[int] = mapped_column(Integer, index=True)
    agent_id: Mapped[str] = mapped_column(String, index=True)
    q_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


# --- Asynchronous Database Setup ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///data/agent_soul_sim.db")
async_engine = create_async_engine(DATABASE_URL)
async_session_maker = async_sessionmaker(bind=async_engine, expire_on_commit=False, class_=AsyncSession)


async def create_tables():
    """
    Creates all the tables defined in this file in the database asynchronously.
    This function is idempotent; it won't re-create existing tables.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
