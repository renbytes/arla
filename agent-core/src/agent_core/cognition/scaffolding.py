# src/agent_core/cognition/scaffolding.py

import asyncio
from typing import Any, Coroutine

# We keep the query_llm import here, but the client itself will be initialized lazily.
from agent_core.cognition.ai_models.openai_client import query_llm


class CognitiveScaffold:
    """
    Explicit interface for all external LLM interactions.
    This class handles prompt construction, querying, and comprehensive logging.
    """

    def __init__(self, simulation_id: str, config: Any, db_logger: Any) -> None:
        """Initializes the scaffold.

        Args:
            simulation_id: The unique ID for the simulation run.
            config: A validated Pydantic configuration object.
            db_logger: A database logger instance.
        """
        self.db_logger = db_logger
        self.simulation_id = simulation_id
        self.config = config

    def query(self, agent_id: str, purpose: str, prompt: str, current_tick: int) -> str:
        """
        The single, unified method for all LLM calls.
        """
        # Pass the 'llm' sub-model directly to the query function.
        # This assumes the main config object has an `llm` attribute.
        response_text, tokens_used, cost = query_llm(prompt, llm_config=self.config.llm)

        # Uses the injected logger, which will be the real one during a simulation.
        log_coro = self.db_logger.log_scaffold_interaction(
            simulation_id=self.simulation_id,
            tick=current_tick,
            agent_id=agent_id,
            purpose=purpose,
            prompt=prompt,
            llm_response=response_text,
            tokens_used=tokens_used,
            cost_usd=cost,
        )

        # In a real run, this would be a real async runner.
        # For now, we adapt to call the method on the logger itself if it's the real one.
        if asyncio.iscoroutine(log_coro):
            asyncio.create_task(log_coro)

        return response_text


# The below is left for offline testing
class MockDbLogger:
    """A mock DB logger that provides awaitable no-op methods."""

    # This method is called from a synchronous context in CognitiveScaffold
    def log_scaffold_interaction(self, **kwargs: Any) -> Coroutine[Any, Any, None]:
        async def dummy_coro() -> None:
            pass

        return dummy_coro()

    # Add the other methods that LoggingSystem calls.
    # These need to be `async def` so they can be awaited.
    async def log_agent_state(self, **kwargs: Any) -> None:
        """Mocked placeholder for logging agent state."""
        pass

    async def log_event(self, **kwargs: Any) -> None:
        """Mocked placeholder for logging an event."""
        pass

    async def log_learning_curve(self, **kwargs: Any) -> None:
        """Mocked placeholder for logging learning curve data."""
        pass


class MockAsyncRunner:
    def run_async(self, coro: Coroutine[Any, Any, None]) -> None:
        # Instead of doing nothing, we close the coroutine.
        # This cleanly disposes of it and prevents the "never awaited"
        # RuntimeWarning during tests or runs using the mock.
        coro.close()


db_logger_instance = MockDbLogger()
async_runner_instance = MockAsyncRunner()
