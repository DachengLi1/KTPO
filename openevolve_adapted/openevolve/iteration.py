"""Result dataclass used by single_task_gym to report one evaluation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Result:
    """Resulting program and metrics from an iteration of OpenEvolve."""

    child_program: Optional[Any] = None
    parent: Optional[Any] = None
    child_metrics: Optional[Dict[str, float]] = None
    iteration_time: float = 0.0
    prompt: Optional[Any] = None
    llm_response: Optional[str] = None
    artifacts: Optional[Dict] = None
    runtime_environment_path: Optional[str] = None
    iteration: int = 0
