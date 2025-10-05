from typing import Dict, Any
from baas.arms.registry import ArmSpec, ArmResult, register_arm
from baas.exec.validators import env_checks, composite_reward

def plan_basic_agentic(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple planner: choose a basic framework and minimal tools.
    Uses presence of npm (ctx['pkg_mgrs'][2]) as a weak proxy to pick LangChain.
    """
    plan = {
        "framework": "LangChain" if ctx["pkg_mgrs"][2] else "LLM-Agent-Base",
        "tools": ["python_repl", "search"],
        "max_steps": 3,  # will be tuned by inner_rl
    }
    return plan

def exec_basic_agentic(plan: Dict[str, Any]) -> ArmResult:
    ok, metrics = env_checks(plan)
    r = composite_reward(metrics)
    logs = [
        f"Setup {plan['framework']} agent with tools {plan['tools']}",
        f"Validation {'passed' if ok else 'failed'}",
        f"Metrics: {metrics}"
    ]
    return ArmResult(success=ok, reward=r, logs=logs, metrics=metrics)

def plan_reflective_agentic(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reflective planner: adds retry/reflect tools and slightly larger step budget.
    Uses presence of npm (ctx['pkg_mgrs'][2]) as a weak proxy to pick AutoGen.
    """
    plan = {
        "framework": "AutoGen" if ctx["pkg_mgrs"][2] else "CustomGraphAgent",
        "tools": ["python_repl", "search", "retry", "reflect"],
        "max_steps": 5,  # will be tuned by inner_rl
    }
    return plan

def exec_reflective_agentic(plan: Dict[str, Any]) -> ArmResult:
    ok, metrics = env_checks(plan)
    r = composite_reward(metrics)
    logs = [
        f"Setup {plan['framework']} reflective agent",
        f"Validation {'passed' if ok else 'failed'}",
        f"Metrics: {metrics}"
    ]
    return ArmResult(success=ok, reward=r, logs=logs, metrics=metrics)

# Register arms at import
register_arm(ArmSpec(
    name="basic_agentic",
    description="Simple planner-executor pattern with LangChain or base fallback.",
    plan_fn=plan_basic_agentic,
    exec_fn=exec_basic_agentic,
))
register_arm(ArmSpec(
    name="reflective_agentic",
    description="Adds reflection and retry to planner-executor pattern.",
    plan_fn=plan_reflective_agentic,
    exec_fn=exec_reflective_agentic,
))