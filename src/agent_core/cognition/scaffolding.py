# src/agent_core/cognition/scaffolding.py

from typing import Any, Dict

# NOTE: This will have a dependency on the `arla-engine` in the future,
# which will provide the DB logger and async runner. For now, we mock them.
class MockDbLogger:
    def log_scaffold_interaction(self, **kwargs):
        pass

class MockAsyncRunner:
    def run_async(self, coro):
        pass

db_logger_instance = MockDbLogger()
async_runner_instance = MockAsyncRunner()

# We keep the query_llm import here, but the client itself will be initialized lazily.
from src.cognition.ai_models.openai_client import query_llm


class CognitiveScaffold:
    """
    Explicit interface for all external LLM interactions.
    This class handles prompt construction, querying, and comprehensive logging.
    """

    def __init__(self, simulation_id: str, config: Dict[str, Any]):
        # In the final version, db_logger would be injected.
        self.db_logger = db_logger_instance
        self.simulation_id = simulation_id
        self.config = config

    def query(self, agent_id: str, purpose: str, prompt: str, current_tick: int) -> str:
        """
        The single, unified method for all LLM calls.
        """
        response_text, tokens_used, cost = query_llm(prompt, llm_config=self.config.get("llm", {}))

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
        async_runner_instance.run_async(log_coro)

        return response_text
