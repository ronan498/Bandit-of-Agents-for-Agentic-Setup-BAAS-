import math
import time
import importlib.util
from typing import Dict, Any, Tuple

def _has_module(name: str) -> int:
    return 1 if importlib.util.find_spec(name) is not None else 0

def env_checks(plan: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Fast, deterministic checks to ground reward.
    Returns (ok, metrics).
    """
    metrics = {}

    # Basic Python health check: small compute task
    t0 = time.perf_counter()
    s = sum(i*i for i in range(10_000))  # quick CPU test
    t1 = time.perf_counter()
    metrics["cpu_ms_small"] = (t1 - t0) * 1000.0
    metrics["compute_ok"] = 1 if s > 0 else 0

    # Optional ML stack presence signals (don’t install anything)
    metrics["has_torch"] = _has_module("torch")
    metrics["has_transformers"] = _has_module("transformers")

    # Tiny “agent step” smoke test: simulate a two-step plan succeeding if max_steps >= 3
    steps_budget = int(plan.get("max_steps", 3))
    metrics["steps_budget"] = steps_budget
    metrics["smoke_pass"] = 1 if steps_budget >= 3 else 0

    # Framework preference heuristic — purely for diversity of outcomes
    fw = plan.get("framework", "").lower()
    metrics["framework_langchain"] = 1 if "langchain" in fw else 0
    metrics["framework_autogen"] = 1 if "autogen" in fw else 0

    ok = bool(metrics["compute_ok"] and metrics["smoke_pass"])
    return ok, metrics

def composite_reward(metrics: Dict[str, Any]) -> float:
    """
    Map metrics to [0,1] reward. Keep it simple and deterministic:
      - base from compute_ok & smoke_pass
      - small bonus for frameworks we 'prefer' in this demo
      - gentle penalty if the tiny CPU test is slow
    """
    base = 0.0
    base += 0.5 * metrics.get("compute_ok", 0)
    base += 0.4 * metrics.get("smoke_pass", 0)

    # “Preference” bonus based on framework (purely demonstrative)
    if metrics.get("framework_langchain", 0):
        base += 0.05
    if metrics.get("framework_autogen", 0):
        base += 0.05

    # Soft penalty if CPU check is unusually slow (> 15 ms for this tiny loop)
    cpu_ms = float(metrics.get("cpu_ms_small", 0.0))
    if cpu_ms > 15.0:
        base -= min(0.1, (cpu_ms - 15.0) / 200.0)

    # Clamp to [0,1]
    return max(0.0, min(1.0, base))