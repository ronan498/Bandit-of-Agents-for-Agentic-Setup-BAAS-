from typing import Dict, Any
from baas.arms.registry import ArmSpec, ArmResult, register_arm
from baas.exec.validators import env_checks, composite_reward

def plan_graph_agentic(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic plan standing in for a LangGraph/graph-planner style pipeline:
    router -> tool -> reflect -> retry
    """
    # crude heuristic: if any ML stack present, assume we can tolerate more branching
    ml_present = any(ctx.get("ml_stack", []))
    max_steps = 5 if ml_present else 4
    return {
        "framework": "GraphPlanner",
        "tools": ["router", "python_repl", "reflect", "retry", "search"],
        "max_steps": max_steps,
    }

def exec_graph_agentic(plan: Dict[str, Any]) -> ArmResult:
    ok, metrics = env_checks(plan)
    # tiny bias: reward bonus if we used a router+reflect combo (already captured by smoke_pass)
    r = composite_reward(metrics)
    logs = [
        f"Graph pipeline with tools={plan['tools']} and max_steps={plan['max_steps']}",
        f"Validation {'passed' if ok else 'failed'}",
        f"Metrics: {metrics}",
    ]
    return ArmResult(success=ok, reward=r, logs=logs, metrics=metrics)

register_arm(ArmSpec(
    name="graph_agentic",
    description="Router -> tool -> reflect -> retry composition (graph-style planner).",
    plan_fn=plan_graph_agentic,
    exec_fn=exec_graph_agentic,
))