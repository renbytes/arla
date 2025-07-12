from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str
    completion_model: str
    temperature: float = Field(..., ge=0.0, le=1.0)
    max_tokens: int = Field(..., gt=0)
    reflection_prompt_prefix: str
    embedding_model: str
