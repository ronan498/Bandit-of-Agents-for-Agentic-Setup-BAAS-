import os
import json
import time
from typing import Dict, Any

from baas.arms.registry import ArmSpec, ArmResult, register_arm
from baas.exec.validators import env_checks, composite_reward


# ---------------------------------------------------------------------
#  OpenAI Client Helper (graceful fallback)
# ---------------------------------------------------------------------
def _get_openai_client():
    """
    Return an OpenAI client if OPENAI_API_KEY is set, else None.
    Uses the modern 'openai>=1.x' SDK.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        return client
    except Exception:
        return None


def _summarise_context(ctx: Dict[str, Any]) -> str:
    """Compress context dict into JSON string."""
    return json.dumps(ctx, separators=(",", ":"))


# ---------------------------------------------------------------------
#  PLAN: Ask LLM for configuration (or fallback)
# ---------------------------------------------------------------------
def plan_llm_agentic(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask an LLM to propose a small plan JSON given the environment context.
    Falls back to a deterministic heuristic if no key or parsing fails.
    Expected JSON keys: framework:str, tools:list[str], max_steps:int
    """
    client = _get_openai_client()
    if client is None:
        # Fallback heuristic if no API key
        return {
            "framework": "LangGraph" if ctx["pkg_mgrs"][2] else "CustomGraphAgent",
            "tools": ["python_repl", "search", "router"],
            "max_steps": 4,
        }

    prompt = f"""
You are a systems agent that outputs ONLY a compact JSON object to set up an agentic AI pipeline.

Given this environment context (JSON): { _summarise_context(ctx) }

Return JSON with EXACT keys:
- "framework": string (e.g., "LangGraph", "AutoGen", "LiteAgent")
- "tools": array of strings (subset of ["python_repl","search","retry","reflect","router"])
- "max_steps": integer in [3,6]

No explanation. No markdown. Just the JSON object.
"""

    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise JSON-only planner."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        latency = time.time() - t0

        text = resp.choices[0].message.content.strip()
        plan = json.loads(text)

        # Sanity checks
        if not isinstance(plan, dict):
            raise ValueError("not a dict")
        if "framework" not in plan or "tools" not in plan or "max_steps" not in plan:
            raise ValueError("missing keys")
        if not isinstance(plan["tools"], list):
            raise ValueError("tools not list")

        plan["max_steps"] = int(plan["max_steps"])
        plan["max_steps"] = max(3, min(6, plan["max_steps"]))

        # Estimate tokens/cost (very rough)
        input_chars = len(prompt)
        output_chars = len(text)
        est_tokens = int((input_chars + output_chars) / 4)
        est_usd_cost = est_tokens * 0.000002  # mini-model ballpark

        # Attach metrics for later eval/logs
        plan["_llm_metrics"] = {
            "llm_ok": 1,
            "llm_latency_s": latency,
            "llm_out_len": len(text),
            "llm_text": text,
            "usd_cost": est_usd_cost,
        }

        return plan

    except Exception as e:
        # Robust fallback if LLM call or parsing fails
        return {
            "framework": "LiteAgent",
            "tools": ["python_repl", "search", "retry"],
            "max_steps": 4,
            "_llm_metrics": {
                "llm_ok": 0,
                "error": str(e),
                "llm_latency_s": 0.0,
                "llm_out_len": 0,
                "usd_cost": 0.0,
            },
        }


# ---------------------------------------------------------------------
#  EXEC: Run environment validation + reward aggregation
# ---------------------------------------------------------------------
def exec_llm_agentic(plan: Dict[str, Any]) -> ArmResult:
    """
    Execute a lightweight validation phase and compute reward based on environment readiness
    + any LLM metrics (latency, cost, etc.).
    """
    ok, metrics = env_checks(plan)
    llm_metrics = plan.get("_llm_metrics", {})

    # Merge LLM metrics into environment metrics
    metrics.update(llm_metrics)

    # Composite reward combines env checks + runtime feedback
    r = composite_reward(metrics)

    # Logs for explainability
    logs = [
        f"LLM-planned framework={plan.get('framework')}, tools={plan.get('tools')}, max_steps={plan.get('max_steps')}",
        f"Validation {'passed' if ok else 'failed'}",
        f"Metrics: {metrics}",
    ]

    # Put text into output dict (compatible with your _extract_text_from_result)
    text = metrics.get("llm_text", "")

    return ArmResult(
        success=ok,
        reward=r,
        logs=logs,
        metrics=metrics,   # includes "llm_text" (if LLM used) plus env checks
    )


# ---------------------------------------------------------------------
#  Register the arm
# ---------------------------------------------------------------------
register_arm(
    ArmSpec(
        name="llm_agentic",
        description="LLM proposes JSON plan dynamically; falls back safely if no key or failure.",
        plan_fn=plan_llm_agentic,
        exec_fn=exec_llm_agentic,
    )
)