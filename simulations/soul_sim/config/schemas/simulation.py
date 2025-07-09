from typing import Optional
from pydantic import BaseModel

class SimulationConfig(BaseModel):
    steps: int
    log_directory: str
    database_directory: str
    database_file: str
    enable_debug_logging: bool
    random_seed: Optional[int] = None
