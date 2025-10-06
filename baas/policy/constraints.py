from typing import List, Dict, Any

def allow_arm(name: str, constraints: Dict[str, Any]) -> bool:
    allow = constraints.get("allow_arms")
    deny = constraints.get("deny_arms")
    if allow and name not in allow:
        return False
    if deny and name in deny:
        return False
    return True

def violates_runtime_limits(metrics: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
    max_cost = constraints.get("max_cost_usd")
    max_lat  = constraints.get("max_latency_s")
    if max_cost is not None and metrics.get("usd_cost", 0.0) > max_cost:
        return True
    if max_lat  is not None and metrics.get("llm_latency_s", metrics.get("latency_s", 0.0)) > max_lat:
        return True
    return False