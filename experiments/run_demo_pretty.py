# experiments/run_demo_pretty.py
import time
from collections import Counter
import numpy as np

from baas.contexts import get_context, vectorize_context
from baas.bandit.linucb import LinUCB
import baas.arms.patterns  # registers arms
import baas.arms.llm_planner  # registers llm_agentic
import baas.arms.graph_planner  # registers graph_agentic
from baas.arms.registry import all_arms
from baas.inner_rl.reinforce import KnobTuner, tune_and_execute

def run(T=30, alpha=1.0):
    ctx = get_context()
    x = np.array(vectorize_context(ctx), dtype=float)
    arms = all_arms()
    bandit = LinUCB(d=x.shape[0], n_arms=len(arms), alpha=alpha)
    tuners = {a.name: KnobTuner(init_p=0.5, lr=0.15) for a in arms}

    history = []
    print(f"\nRunning BAAS demo • context_dim={len(x)} • arms={', '.join(a.name for a in arms)}\n")

    start = time.time()
    for t in range(T):
        a_idx = bandit.select(x)
        arm = arms[a_idx]
        tuner = tuners[arm.name]
        plan = arm.plan_fn(ctx)

        result, tuner = tune_and_execute(
            plan, arm.exec_fn,
            knob_key="max_steps", option0=3, option1=5,
            tuner=tuner, trials=2
        )
        tuners[arm.name] = tuner
        r = float(result.reward)
        bandit.update(a_idx, x, r)

        # live, compact line
        llm_note = ""
        if "llm_ok" in result.metrics:
            llm_note = f"  (LLM ok, {result.metrics.get('llm_latency_s', 0):.2f}s, out≈{result.metrics.get('llm_out_len', 0)})"

        print(f"[{t:02d}] {arm.name:<17} r={r:.2f}  steps→{'5' if tuner.p>0.5 else '3'}  p≈{tuner.p:>4.2f}{llm_note}")

        history.append({
            "t": t, "arm": arm.name, "reward": r, "p": tuner.p,
            "steps": 5 if tuner.p>0.5 else 3,
            **{f"m.{k}": v for k,v in result.metrics.items()}
        })

    dur = time.time() - start
    # summary
    counts = Counter(h["arm"] for h in history)
    avg = np.mean([h["reward"] for h in history])
    tail = np.mean([h["reward"] for h in history[-10:]]) if len(history)>=10 else avg
    best = counts.most_common(1)[0][0]

    print("\n──────────────── Summary ────────────────")
    print("Selections:")
    for k,v in counts.items():
        print(f" • {k:<17} {v:>3}")
    print(f"\nAvg reward (all):   {avg:.3f}")
    print(f"Avg reward (last10):{tail:.3f}")
    print(f"Recommended now:    {best}")
    # quick reason
    reasons = {
        "basic_agentic": "fast, reliable for simple tasks",
        "reflective_agentic": "adds self-check/retry for tougher queries",
        "llm_agentic": "uses the LLM directly; good for language-heavy tasks",
        "graph_agentic": "routes through a small tool graph; good when tasks vary"
    }
    print(f"Why:                {reasons.get(best,'learned highest reward in this context')}")
    print(f"\nRuntime:            {dur:.1f}s\n")

if __name__ == "__main__":
    run()