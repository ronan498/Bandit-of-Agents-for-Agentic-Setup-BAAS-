# experiments/run_task_demo.py
import os
import json
import time
import random
from collections import Counter
from typing import Dict, Any, List

import numpy as np

from baas.contexts import get_context, vectorize_context
from baas.bandit.linucb import LinUCB

# Arms (registration happens on import)
import baas.arms.patterns          # noqa: F401
import baas.arms.llm_planner       # noqa: F401
import baas.arms.graph_planner     # noqa: F401

from baas.arms.registry import all_arms
from baas.inner_rl.reinforce import KnobTuner, tune_and_execute

from baas.tasks.summarize import SummarizationAdapter
from baas.policy.constraints import allow_arm, violates_runtime_limits


# ----------------------- helpers -----------------------

LOG_PATH = "logs/baas.jsonl"


def _ensure_logdir():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def _load_samples() -> List[Dict[str, Any]]:
    """
    Load summarization samples from experiments/data/summarize.jsonl if present,
    else fall back to a small built-in set and repeat to ~50 examples for the demo.
    """
    path = "experiments/data/summarize.jsonl"
    if os.path.exists(path):
        with open(path, "r") as f:
            return [json.loads(l) for l in f if l.strip()]

    samples = [
        {
            "id": "ex1",
            "text": "BAAS selects among agent patterns to optimize quality, cost, and latency.",
            "reference": "BAAS chooses the best agent for a task balancing quality, cost, and speed.",
        },
        {
            "id": "ex2",
            "text": "Reflective agents can retry and self-evaluate; graphs orchestrate multi-step reasoning.",
            "reference": "Reflective agents retry and self-check; graph agents manage multi-step work.",
        },
        {
            "id": "ex3",
            "text": "Basic patterns are fast and cheap for straightforward prompts.",
            "reference": "Basic agents are quick and inexpensive for simple tasks.",
        },
    ]
    return (samples * 10)[:50]


def _extract_text_from_result(result) -> str:
    """
    Be liberal in what we accept:
    - result.text (if present)
    - result.output (str) or result.output["text"]
    - result.content / result.message (str or dict with 'text')
    - result.metrics['llm_text'] as a last resort
    """
    # direct
    txt = getattr(result, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    # common containers
    for attr in ("output", "content", "message"):
        val = getattr(result, attr, None)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, dict):
            t = val.get("text")
            if isinstance(t, str) and t.strip():
                return t

    # metrics fallback
    metrics = getattr(result, "metrics", {}) or {}
    mt = metrics.get("llm_text")
    if isinstance(mt, str) and mt.strip():
        return mt

    return ""


# ----------------------- main demo -----------------------

def run(T: int = 30, alpha: float = 1.0):
    """
    A task-focused demo:
    - Uses LinUCB over registered arms
    - Performs an inner-loop tune on a single knob (max_steps)
    - Evaluates outputs via SummarizationAdapter (quality/cost/latency)
    - Prints a clean, user-friendly trace and per-arm rollups
    """
    _ensure_logdir()

    # Context & arms
    ctx = get_context()
    x = np.array(vectorize_context(ctx), dtype=float)
    d = x.shape[0]

    arms = all_arms()
    if not arms:
        raise RuntimeError("No arms registered.")
    arm_index = {i: a for i, a in enumerate(arms)}

    # Bandit & tuners
    bandit = LinUCB(d=d, n_arms=len(arms), alpha=alpha)
    tuners = {a.name: KnobTuner(init_p=0.5, lr=0.15) for a in arms}

    # Task adapter & data
    adapter = SummarizationAdapter()
    samples = _load_samples()

    # Friendlier constraints (so small LLM calls can win if quality is better)
    constraints = {
        "max_latency_s": 2.0,
        "max_cost_usd": 0.01,
        "deny_arms": [],
        "w_quality": 1.0,
        "w_cost": 0.1,
        "w_latency": 0.1,
    }

    print(
        f"\nRunning BAAS Task Demo • task=summarize • context_dim={d} • "
        f"arms={', '.join([a.name for a in arms])}\n"
    )

    chosen: List[str] = []
    rewards: List[float] = []
    per_arm: Dict[str, Dict[str, float]] = {}
    t0 = time.time()

    for t in range(T):
        # Candidate selection (pre-filter by policy if needed)
        a_idx = bandit.select(x)
        arm = arm_index[a_idx]
        if not allow_arm(arm.name, constraints):
            # naive fallback: pick the next allowed arm
            for j in range(len(arms)):
                if allow_arm(arms[j].name, constraints):
                    a_idx, arm = j, arms[j]
                    break

        tuner = tuners[arm.name]

        # Build a plan for the chosen arm and prepare task payload
        plan = arm.plan_fn(ctx)
        sample = random.choice(samples)
        plan = adapter.prepare(plan, payload={"text": sample["text"], "constraints": constraints})

        # Inner-loop tune steps
        best_result, tuner = tune_and_execute(
            plan,
            arm.exec_fn,
            knob_key="max_steps",
            option0=3,
            option1=5,
            tuner=tuner,
            trials=2,
        )
        tuners[arm.name] = tuner

        # Extract text & evaluate task-specific utility
        pred_text = _extract_text_from_result(best_result)
        evalr = adapter.evaluate(
            {"text": pred_text, "metrics": best_result.metrics},
            {"text": plan["task_input"], "reference": sample.get("reference"), "constraints": constraints},
        )

        # Enforce runtime constraints post-hoc (auto-fallback idea stub)
        violated = violates_runtime_limits(evalr.metrics, constraints)

        # Bandit update with evaluated reward
        r = float(evalr.reward if not violated else 0.0)
        bandit.update(a_idx, x, r)

        rewards.append(r)
        chosen.append(arm.name)

        # Pretty line
        p = tuner.p
        steps = plan.get("max_steps", "?")
        q = float(evalr.quality)
        lat = float(evalr.metrics.get("latency_s", 0.0))
        cost = float(evalr.metrics.get("usd_cost", 0.0))
        viol = " ⚠" if violated else ""
        print(
            f"[{t:02d}] {arm.name:18s} r={r:0.2f} q={q:0.2f} $={cost:0.4f} "
            f"lat={lat:0.2f}s  steps→{steps}  p≈{p:0.2f}{viol}"
        )

        # Per-arm rollup stats
        s = per_arm.setdefault(arm.name, {"n": 0, "r": 0.0, "q": 0.0, "$": 0.0, "lat": 0.0})
        s["n"] += 1
        s["r"] += r
        s["q"] += q
        s["$"] += cost
        s["lat"] += lat

        # Log structured JSON for dashboards
        row = {
            "t": t,
            "arm": arm.name,
            "reward": r,
            "quality": q,
            "usd_cost": cost,
            "latency_s": lat,
            "tuned_p": p,
            "plan": plan,  # store the plan we executed
            "metrics": best_result.metrics,
            "explanation": (
                f"Chose {arm.name} given context; tuned steps to {steps}. "
                f"Quality={q:.2f}, Cost=${cost:.4f}, Latency={lat:.2f}s."
                + (" Violated constraints." if violated else "")
            ),
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")

    # Summary
    counts = Counter(chosen)
    print("\n──────────────── Summary ────────────────")
    for k, v in counts.items():
        print(f" • {k:18s} {v}")

    avg_all = float(np.mean(rewards)) if rewards else 0.0
    avg_last10 = float(np.mean(rewards[-10:])) if len(rewards) >= 10 else avg_all
    print(f"\nAvg reward (all):   {avg_all:.3f}")
    print(f"Avg reward (last10):{avg_last10:.3f}")

    # Simple recommendation heuristic
    rec = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "basic_agentic"
    why = {
        "basic_agentic": "fast, reliable for simple tasks",
        "reflective_agentic": "handles tricky prompts with retries",
        "llm_agentic": "balanced LLM output with low overhead",
        "graph_agentic": "best for multi-step reasoning",
    }.get(rec, "most effective recently")
    print(f"Recommended now:    {rec}")
    print(f"Why:                {why}")

    # Per-arm averages (easy to interpret)
    print("\nPer-arm averages (easier to read):")
    for name, s in per_arm.items():
        n = max(1, s["n"])
        print(
            f" - {name:18s}  reward={s['r']/n:0.3f}  quality={s['q']/n:0.3f}  "
            f"cost=${s['$']/n:0.4f}  latency={s['lat']/n:0.2f}s"
        )

    print(f"\nLogs:               {LOG_PATH}")
    print(f"Runtime:            {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run()