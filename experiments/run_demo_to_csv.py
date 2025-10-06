import csv, os, time, importlib, platform, sys, shutil
from collections import Counter

# ---- Robust context probe + vector (works even if baas.contexts API changes) ----
def _probe_context_fallback():
    os_name = platform.system().lower()
    os_onehot = [1,0,0] if "linux" in os_name else [0,1,0] if "darwin" in os_name else [0,0,1]
    python_major, python_minor = sys.version_info.major, sys.version_info.minor
    gpu_nvidia = 1 if shutil.which("nvidia-smi") else 0
    cuda_toolkit = 1 if shutil.which("nvcc") else 0
    pkg_mgrs = [int(bool(shutil.which(x))) for x in ["pip","conda","poetry","mamba"]]
    cloud_envs = [int("CODESPACES" in os.environ), int("COLAB_GPU" in os.environ), int("KAGGLE_URL_BASE" in os.environ)]
    try:
        import torch  # noqa
        has_torch = 1
    except Exception:
        has_torch = 0
    try:
        import transformers  # noqa
        has_transformers = 1
    except Exception:
        has_transformers = 0
    ml_stack = [has_torch, has_transformers, 0, 0, 0]
    budget_flags = [0, 0, 0]
    return dict(
        os_onehot=os_onehot,
        python_major=python_major,
        python_minor=python_minor,
        gpu_nvidia=gpu_nvidia,
        cuda_toolkit=cuda_toolkit,
        pkg_mgrs=pkg_mgrs,
        cloud_envs=cloud_envs,
        ml_stack=ml_stack,
        budget_flags=budget_flags,
    )

# Try to use baas.contexts if available; otherwise fall back
probe_context = None
build_context_vector = None
try:
    ctxmod = importlib.import_module("baas.contexts")
    probe_context = getattr(ctxmod, "probe_context", None)
    build_context_vector = getattr(ctxmod, "build_context_vector", None)
except Exception:
    pass

if probe_context is None:
    probe_context = _probe_context_fallback

import numpy as np
def _build_vec_fallback(ctx):
    parts = []
    parts += ctx.get("os_onehot", [0,0,0])
    parts += [ctx.get("python_major", 0), ctx.get("python_minor", 0)]
    parts += [ctx.get("gpu_nvidia", 0), ctx.get("cuda_toolkit", 0)]
    parts += ctx.get("pkg_mgrs", [0,0,0,0])
    parts += ctx.get("cloud_envs", [0,0,0])
    parts += ctx.get("ml_stack", [0,0,0,0,0])
    parts += ctx.get("budget_flags", [0,0,0])
    return np.array(parts, dtype=float)

if build_context_vector is None:
    build_context_vector = _build_vec_fallback
# ---- end robust context block ----

from baas.bandit.linucb import LinUCB
from baas.arms.registry import all_arms
import baas.arms.patterns  # register defaults
import baas.arms.llm_planner  # register llm arm
import baas.arms.graph_planner  # register graph arm
# ---- Robust import of a tiny REINFORCE knob (fallback if package API differs) ----
try:
    from baas.inner_rl.reinforce import REINFORCEKnob  # type: ignore
except Exception:
    import math
    import numpy as np

    class REINFORCEKnob:
        """
        Minimal 2-choice policy-gradient knob with a moving baseline.
        choices: e.g., [3, 5]
        """
        def __init__(self, choices, lr=0.4, baseline_decay=0.9):
            assert len(choices) >= 2
            self.choices = list(choices)
            self.lr = lr
            self.baseline_decay = baseline_decay
            self.baseline = 0.0
            # single logit per choice
            self.logits = np.zeros(len(self.choices), dtype=float)
            self.last_idx = 0
            self.pi = 0.5  # report prob of the last sampled index 1 (for logging)

        def _probs(self):
            # numerically stable softmax
            m = np.max(self.logits)
            ex = np.exp(self.logits - m)
            p = ex / np.sum(ex)
            return p

        def sample(self):
            p = self._probs()
            self.last_idx = int(np.random.choice(len(self.choices), p=p))
            # convenience: if there are exactly 2 choices, expose prob of the "higher" one
            if len(self.choices) == 2:
                self.pi = float(p[1])
            return self.choices[self.last_idx]

        def update(self, reward):
            p = self._probs()
            adv = float(reward) - self.baseline
            grad = -p
            grad[self.last_idx] += 1.0
            self.logits += self.lr * adv * grad
            # update moving baseline
            self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * float(reward)
# ---- end fallback ----
import matplotlib.pyplot as plt

OUT_CSV = "experiments/demo_trace.csv"
OUT_PNG = "experiments/fig_selection.png"

def run(T=200):
    ctx = probe_context()
    x = build_context_vector(ctx)
    arms = all_arms()
    bandit = LinUCB(d=len(x), n_arms=len(arms), alpha=1.0)
    tuner = REINFORCEKnob(choices=[3, 5], lr=0.4, baseline_decay=0.9)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "arm", "reward", "tuned_p", "max_steps", "framework", "tools"])

        selections = []
        rewards = []

        for t in range(T):
            a = bandit.select(x)
            spec = arms[a]
            name = spec.name

            # plan + inner tuning
            plan = spec.plan_fn(ctx)
            max_steps = tuner.sample()
            plan["max_steps"] = max_steps

            # exec + reward
            result = spec.exec_fn(plan)
            r = float(result.reward)
            tuner.update(r)

            bandit.update(a, x, r)
            selections.append(name)
            rewards.append(r)

            w.writerow([t, name, f"{r:.3f}", f"{tuner.pi:.2f}", max_steps, plan.get("framework"), "|".join(plan.get("tools", []))])

    # plot selections
    counts_over_time = []
    c = Counter()
    for s in selections:
        c[s] += 1
        counts_over_time.append(dict(c))

    labels = [spec.name for spec in arms]
    xs = np.arange(1, len(selections) + 1)
    plt.figure()
    for label in labels:
        ys = [d.get(label, 0) for d in counts_over_time]
        plt.plot(xs, ys, label=label)
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Selections")
    plt.title("Arm Selections Over Time (Real-Context Demo)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    print(f"Saved CSV -> {OUT_CSV}")
    print(f"Saved plot -> {OUT_PNG}")

if __name__ == "__main__":
    run()