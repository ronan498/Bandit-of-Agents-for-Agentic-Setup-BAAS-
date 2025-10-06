from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class EvalResult:
    reward: float           # final scalar used by the bandit
    quality: float          # task quality in [0,1]
    metrics: Dict[str, Any] # extra metrics (e.g., rouge, length_penalty, cost, latency)

class TaskAdapter:
    """
    A thin interface between 'plans/arms' and 'task-specific evaluation'.
    """
    name: str = "base"

    def prepare(self, plan: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optionally modify/augment the plan using the task payload.
        Return what the arm's exec_fn should receive as 'plan' (or a wrapper).
        """
        return plan

    def evaluate(self, output: Dict[str, Any], payload: Dict[str, Any]) -> EvalResult:
        """
        Map an arm's raw output to (reward, quality, metrics).
        Must be implemented by concrete adapters.
        """
        raise NotImplementedError