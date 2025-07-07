#!/usr/bin/env python3
"""
create_agent_sim_structure.py

Run this script from the directory where you want the new `agent-sim/`
project folder to appear.

$ python create_agent_sim_structure.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path("agent-sim").resolve()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def write_file(path: Path, content: str | bytes = "") -> None:
    """Create parent dirs (if needed) and write text/bytes to the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    with path.open(mode, encoding=None if mode == "wb" else "utf-8") as f:
        f.write(content)


def py(content: str) -> str:
    """Strip common indentation for multi-line Python snippets."""
    return dedent(content).lstrip("\n")


# ---------------------------------------------------------------------------
# File map – {relative_path: content}
# ---------------------------------------------------------------------------

FILES: dict[str, str | bytes] = {
    # Workspace meta-files
    ".env": "OPENAI_API_KEY=\nDATABASE_URL=postgresql+asyncpg://user:pass@db/arla\n",
    ".gitignore": dedent(
        """
        # Byte-compiled / cache
        __pycache__/
        *.py[cod]
        # Virtualenv
        .venv/
        # IDE
        .idea/
        .vscode/
        # Jupyter
        *.ipynb_checkpoints/
        # Python tooling
        .pytest_cache/
        .mypy_cache/
        # Local env & secrets
        .env
        """
    ).strip()
    + "\n",
    "Dockerfile": dedent(
        """
        FROM python:3.11-slim

        WORKDIR /app
        COPY pyproject.toml .
        RUN pip install --upgrade pip && pip install -r <(python -c 'import tomllib,sys,json;print("\\n".join(tomllib.load(open("pyproject.toml","rb"))["project"]["dependencies"]))')

        COPY src ./src
        COPY notebooks ./notebooks
        ENV PYTHONPATH="/app/src:${PYTHONPATH}"
        CMD ["python", "-m", "src.main"]
        """
    ).lstrip(),
    "README.md": "# agent-sim\n\nA demo project that plugs a custom world into the ARLA cognitive engine.\n",
    # Poetry / PEP 621 metadata
    "pyproject.toml": dedent(
        """
        [project]
        name = "agent-sim"
        version = "0.1.0"
        description = "ARLA-powered simulation with a custom world layer."
        requires-python = ">=3.10"
        dependencies = [
            "arla-core>=0.1.0",
            "agent-engine @ git+https://github.com/your-org/agent-engine.git",
            "asyncpg",
            "sqlalchemy[asyncio]",
            "alembic",
            "celery[redis]",
        ]

        [tool.black]
        line-length = 88
        """
    ).lstrip(),
    # Empty notebook skeleton
    "notebooks/analysis.ipynb": json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, indent=2)
    + "\n",
    # --- source tree ---
    "src/__init__.py": "",
    "src/main.py": py(
        """
        \"\"\"Entry-point for running the simulation.

        In real usage you would parse CLI flags, load scenario YAML/JSON,
        instantiate the SimulationManager, inject providers, then call run().
        \"\"\"
        from simulations.soul_sim.world import run_world

        if __name__ == "__main__":
            run_world()
        """
    ),
    # Infrastructure → database
    "src/infrastructure/__init__.py": "",
    "src/infrastructure/database/__init__.py": "",
    "src/infrastructure/database/async_database_manager.py": py(
        """
        from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

        _engine: AsyncEngine | None = None


        def get_engine() -> AsyncEngine:
            global _engine
            if _engine is None:
                raise RuntimeError("Database engine not initialised. Call init_engine() first.")
            return _engine


        def init_engine(database_url: str, echo: bool = False) -> AsyncEngine:
            global _engine
            _engine = create_async_engine(database_url, echo=echo, future=True)
            return _engine
        """
    ),
    "src/infrastructure/database/models.py": py(
        """
        from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
        from sqlalchemy import Integer, String


        class Base(DeclarativeBase):
            pass


        class SimulationRun(Base):
            __tablename__ = "simulation_runs"

            id: Mapped[int] = mapped_column(Integer, primary_key=True)
            scenario_name: Mapped[str] = mapped_column(String(100))
            result_blob: Mapped[str] = mapped_column(String)
        """
    ),
    # Infrastructure → async task queue
    "src/infrastructure/tasks/__init__.py": "",
    "src/infrastructure/tasks/celery_app.py": py(
        """
        import os
        from celery import Celery

        broker_url = os.getenv("CELERY_BROKER", "redis://redis:6379/0")
        backend_url = os.getenv("CELERY_BACKEND", "redis://redis:6379/1")

        celery_app = Celery("agent_sim", broker=broker_url, backend=backend_url)

        celery_app.conf.update(
            task_serializer="json",
            accept_content=["json"],
            result_serializer="json",
            timezone="UTC",
            enable_utc=True,
        )
        """
    ),
    "src/infrastructure/tasks/simulation_tasks.py": py(
        """
        from celery_app import celery_app
        from simulations.soul_sim.world import run_world


        @celery_app.task(bind=True)
        def execute_simulation(self, scenario_path: str) -> str:
            \"\"\"Background task to run a simulation scenario.\"\"\"
            return run_world(scenario_path)
        """
    ),
    # Simulations package
    "src/simulations/__init__.py": "",
    "src/simulations/soul_sim/__init__.py": "",
    "src/simulations/soul_sim/components.py": "# World-specific ECS components go here.\n",
    "src/simulations/soul_sim/config.yaml": dedent(
        """
        simulation:
          steps: 100
          random_seed: 42
        learning:
          memory:
            reflection_interval: 50
        """
    ).lstrip(),
    "src/simulations/soul_sim/providers.py": py(
        """
        \"\"\"Concrete implementations of the provider interfaces needed by ARLA.\"\"\"
        from agent_core.environment.state_node_encoder_interface import StateNodeEncoderInterface
        from agent_core.environment.vitality_metrics_provider_interface import (
            VitalityMetricsProviderInterface,
        )
        from agent_core.environment.controllability_provider_interface import (
            ControllabilityProviderInterface,
        )


        class GridStateNodeEncoder(StateNodeEncoderInterface):
            def encode_state_for_causal_graph(self, *args, **kwargs):
                # Return a placeholder symbolic node
                return ("STATE", "placeholder")


        class GridVitalityProvider(VitalityMetricsProviderInterface):
            def get_normalized_vitality_metrics(self, *args, **kwargs):
                return {"health_norm": 0.5, "time_norm": 0.5, "resources_norm": 0.5}


        class GridControllabilityProvider(ControllabilityProviderInterface):
            def get_controllability_score(self, *args, **kwargs) -> float:
                return 0.5
        """
    ),
    "src/simulations/soul_sim/scenarios/default.json": json.dumps(
        {"name": "default", "description": "Starter scenario", "entities": []}, indent=2
    )
    + "\n",
    "src/simulations/soul_sim/systems.py": "# World-specific systems (Movement, Combat…) go here.\n",
    "src/simulations/soul_sim/world.py": py(
        """
        \"\"\"World bootstrap that wires providers into the agent-engine SimulationManager.\"\"\"
        from pathlib import Path
        from omegaconf import OmegaConf

        # Stubs – replace with real imports from agent_engine and your providers
        from agent_engine.simulation.engine import SimulationManager
        from simulations.soul_sim.providers import (
            GridControllabilityProvider,
            GridStateNodeEncoder,
            GridVitalityProvider,
        )


        def run_world(scenario_path: str | None = None) -> str:
            # 1. Load YAML/JSON config
            cfg_file = Path(__file__).with_name("config.yaml")
            config = OmegaConf.load(cfg_file)

            # 2. Create provider instances
            controllability = GridControllabilityProvider()
            vitality = GridVitalityProvider()
            state_node_encoder = GridStateNodeEncoder()

            # 3. Initialise SimulationManager (simplified)
            sim_mgr = SimulationManager(
                config=config,
                environment=None,  # Replace with your concrete EnvironmentInterface
                scenario_loader=lambda: None,
                action_generator=lambda *a, **kw: [],
                decision_selector=lambda *a, **kw: None,
            )

            # 4. Register cognitive systems with providers
            from agent_engine.systems.affect_system import AffectSystem

            sim_mgr.register_system(
                AffectSystem,
                vitality_metrics_provider=vitality,
                controllability_provider=controllability,
            )

            # Register other systems here…

            # 5. Run simulation (stub)
            sim_mgr.run()
            return "Simulation finished."
        """
    ),
}

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main() -> None:
    for rel_path, content in FILES.items():
        write_file(PROJECT_ROOT / rel_path, content)
    print(f"✔  Project skeleton created at: {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
