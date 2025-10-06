import numpy as np
from collections import Counter
from baas.contexts import get_context, vectorize_context
from baas.bandit.linucb import LinUCB
import baas.arms.patterns  # noqa: F401 (registers arms)
from baas.arms.registry import all_arms
from baas.inner_rl.reinforce import KnobTuner, tune_and_execute
import baas.arms.llm_planner  # noqa: F401
import baas.arms.graph_planner  # noqa: F401

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

        r = float(best_result.reward)
        bandit.update(a_idx, x_vec, r)

        rewards.append(r)
        chosen.append(arm.name)

        print(f"[t={t:02d}] arm={arm.name:>20s} | reward={r:.2f} | tuned_p≈{tuner.p:.2f} | plan={plan} | metrics={best_result.metrics}")

    counts = Counter(chosen)
    print("\n=== Summary ===")
    print("Selection counts:", dict(counts))
    print(f"Avg reward: {np.mean(rewards):.3f}")
    print(f"Avg reward (last 10): {np.mean(rewards[-10:]):.3f}")

if __name__ == "__main__":
    main()