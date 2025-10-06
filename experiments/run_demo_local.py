import time
import numpy as np
from collections import Counter

from baas.contexts import get_context, vectorize_context
from baas.bandit.linucb import LinUCB
import baas.arms.patterns  # noqa: F401 (registers arms)
from baas.arms.registry import all_arms
from baas.inner_rl.reinforce import KnobTuner, tune_and_execute
import baas.arms.llm_planner  # noqa: F401
import baas.arms.graph_planner  # noqa: F401

# Optional: only import OpenAI client if/when we need it to avoid hard dependency at import time
def _try_llm_call():
    """
    Returns (ok: bool, latency_s: float, text_len: int, err: str|None)
    Makes a tiny OpenAI chat completion call to verify the LLM path is live.
    """
    try:
        from openai import OpenAI
        client = OpenAI()
        start = time.time()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "In one short sentence, define exploration in reinforcement learning."},
            ],
            max_tokens=30,
        )
        latency = time.time() - start
        text = resp.choices[0].message.content or ""
        return True, latency, len(text.strip()), None
    except Exception as e:
        return False, 0.0, 0, str(e)


def main(T: int = 50, alpha: float = 1.0):
    # 1) Build context vector from the *actual* environment
    ctx = get_context()
    x_vec = np.array(vectorize_context(ctx), dtype=float)
    d = x_vec.shape[0]

    # 2) Load registered arms
    arms = all_arms()
    n_arms = len(arms)
    if n_arms == 0:
        raise RuntimeError("No arms registered. Did you import baas.arms.patterns?")

    # 3) Bandit policy and inner-loop tuner per arm
    bandit = LinUCB(d=d, n_arms=n_arms, alpha=alpha)
    tuners = {a.name: KnobTuner(init_p=0.5, lr=0.15) for a in arms}

    rewards = []
    chosen = []
    print(f"[INFO] Context vector length: {d}, Arms: {[a.name for a in arms]}")

    # 4) Run a few rounds against your real context
    for t in range(T):
        a_idx = bandit.select(x_vec)
        arm = arms[a_idx]
        tuner = tuners[arm.name]

        # Plan, then inner-loop tune a knob (max_steps) before executing
        plan = arm.plan_fn(ctx)
        best_result, tuner = tune_and_execute(
            plan, arm.exec_fn,
            knob_key="max_steps",
            option0=3, option1=5,
            tuner=tuner, trials=2
        )
        tuners[arm.name] = tuner

        # Base reward from the arm's local validators
        r = float(best_result.reward)

        # Collect metrics and optionally augment with live LLM signal
        metrics = dict(best_result.metrics)

        # --- If this is the LLM arm, make a real OpenAI call and fold into reward ---
        if "llm" in arm.name:  # e.g., "llm_agentic"
            ok, latency, out_len, err = _try_llm_call()
            metrics.update({
                "llm_ok": int(ok),
                "llm_latency_s": round(latency, 3),
                "llm_out_len": out_len,
            })
            if not ok and err:
                print(f"[WARN] LLM call failed: {err}")

            # Blend: keep original signal but reward live LLM success.
            # Success => small boost; failure => slight penalty. Also nudge by latency (faster is better).
            if ok:
                # Map latency to a small bonus in [0, 0.05] (fast gets a bit more)
                lat_bonus = max(0.0, 0.05 - min(latency, 0.5) * 0.1)
                r = 0.6 * r + 0.4 * (0.9 + lat_bonus)
            else:
                r = 0.7 * r + 0.3 * 0.0  # failed LLM call: dampen reward slightly

        # Update the bandit with the (possibly) blended reward
        bandit.update(a_idx, x_vec, r)

        rewards.append(r)
        chosen.append(arm.name)

        print(
            f"[t={t:02d}] arm={arm.name:>20s} | reward={r:.2f} | "
            f"tuned_p≈{tuner.p:.2f} | plan={plan} | metrics={metrics}"
        )

    counts = Counter(chosen)
    print("\n=== Summary ===")
    print("Selection counts:", dict(counts))
    print(f"Avg reward: {np.mean(rewards):.3f}")
    print(f"Avg reward (last 10): {np.mean(rewards[-10:]):.3f}")


if __name__ == "__main__":
    main()