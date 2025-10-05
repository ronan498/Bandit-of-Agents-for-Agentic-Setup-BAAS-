from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Optional

@dataclass
class ArmResult:
    success: bool
    reward: float
    logs: List[str]
    metrics: Dict[str, Any]

@dataclass
class ArmSpec:
    name: str
    description: str
    # Given a rich context dict, produce a concrete plan (steps/config)
    plan_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    # Execute the plan and return ArmResult
    exec_fn: Callable[[Dict[str, Any]], ArmResult]

_REGISTRY: List[ArmSpec] = []

def register_arm(arm: ArmSpec) -> None:
    # de-dup by name
    for a in _REGISTRY:
        if a.name == arm.name:
            return
    _REGISTRY.append(arm)

def all_arms() -> List[ArmSpec]:
    return list(_REGISTRY)

def get_arm_by_name(name: str) -> Optional[ArmSpec]:
    for a in _REGISTRY:
        if a.name == name:
            return a
    return None