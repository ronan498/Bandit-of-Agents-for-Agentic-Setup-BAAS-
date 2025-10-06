import os
import json
from typing import Dict, Any, Optional

from baas.arms.registry import ArmSpec, ArmResult, register_arm
from baas.exec.validators import env_checks, composite_reward

# --- Optional OpenAI client helper (graceful fallback) ---

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
    return json.dumps(ctx, separators=(",", ":"))

def plan_llm_agentic(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask an LLM to propose a small plan JSON given the environment context.
    Falls back to a deterministic heuristic if no key or parsing fails.
    Expected JSON keys: framework:str, tools:list[str], max_steps:int
    """
    client = _get_openai_client()
    if client is None:
        # Fallback heuristic
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
        # Prefer a small, cost-effective model; change if you like
        # If your account doesn't have this model, switch to any chat-completions-capable model you do have.
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise JSON-only planner."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        # The model should return raw JSON. Try to parse.
        plan = json.loads(text)
        # Minimal sanity checks
        if not isinstance(plan, dict):
            raise ValueError("not a dict")
        if "framework" not in plan or "tools" not in plan or "max_steps" not in plan:
            raise ValueError("missing keys")
        if not isinstance(plan["tools"], list):
            raise ValueError("tools not list")
        plan["max_steps"] = int(plan["max_steps"])
        # Clamp max_steps a bit
        plan["max_steps"] = max(3, min(6, plan["max_steps"]))
        return plan
    except Exception:
        # Robust fallback if LLM call or parsing fails
        return {
            "framework": "LiteAgent",
            "tools": ["python_repl", "search", "retry"],
            "max_steps": 4,
        }

def exec_llm_agentic(plan: Dict[str, Any]) -> ArmResult:
    ok, metrics = env_checks(plan)
    r = composite_reward(metrics)
    logs = [
        f"LLM-planned framework={plan.get('framework')}, tools={plan.get('tools')}, max_steps={plan.get('max_steps')}",
        f"Validation {'passed' if ok else 'failed'}",
        f"Metrics: {metrics}",
    ]
    return ArmResult(success=ok, reward=r, logs=logs, metrics=metrics)

# Register the arm at import
register_arm(ArmSpec(
    name="llm_agentic",
    description="Plan is proposed by an LLM (JSON-only), with safe fallback if no OPENAI_API_KEY.",
    plan_fn=plan_llm_agentic,
    exec_fn=exec_llm_agentic,
))