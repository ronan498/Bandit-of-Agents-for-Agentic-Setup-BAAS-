import numpy as np
import matplotlib.pyplot as plt

from baas.bandit.linucb import LinUCB
from baas.sim.user_models import SyntheticUser, oracle_reward

# -------- Non-contextual policies -------- #

class NonContextualEpsGreedy:
    """Epsilon-greedy that ignores context (treats rewards as arm-stationary)."""
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms, dtype=int)
        self.sums = np.zeros(n_arms, dtype=float)

    def select(self, _x):
        if np.random.rand() < self.epsilon or self.counts.sum() < self.n_arms:
            return np.random.randint(self.n_arms)
        means = np.divide(self.sums, np.maximum(self.counts, 1))
        return int(np.argmax(means))

    def update(self, arm: int, _x, r: float):
        self.counts[arm] += 1
        self.sums[arm] += r


class FixedBestAfterWarmup:
    """Round-robin warmup, then commit to the empirically best single arm (ignores context)."""
    def __init__(self, n_arms: int, warmup: int = 200):
        self.n_arms = n_arms
        self.warmup = warmup
        self.counts = np.zeros(n_arms, dtype=int)
        self.sums = np.zeros(n_arms, dtype=float)
        self.fixed_arm = None
        self.t = 0

    def select(self, _x):
        if self.fixed_arm is not None:
            return self.fixed_arm
        # round-robin during warmup
        if self.t < self.warmup:
            return self.t % self.n_arms
        # choose best and stick
        means = np.divide(self.sums, np.maximum(self.counts, 1))
        self.fixed_arm = int(np.argmax(means))
        return self.fixed_arm

    def update(self, arm: int, _x, r: float):
        self.t += 1
        if self.fixed_arm is None:
            self.counts[arm] += 1
            self.sums[arm] += r


# -------- Experiment harness -------- #

def run_once(T=3000, d=10, n_arms=6, alpha=1.0, seed=0):
    rng = np.random.default_rng(seed)
    user = SyntheticUser(d=d, n_arms=n_arms, seed=seed, noise_sd=0.05)

    # policies
    linucb = LinUCB(d=d, n_arms=n_arms, alpha=alpha)
    eps = NonContextualEpsGreedy(n_arms=n_arms, epsilon=0.1)
    fixed = FixedBestAfterWarmup(n_arms=n_arms, warmup=200)

    # trackers
    regs_lin, regs_eps, regs_fix = [], [], []

    for t in range(T):
        x = user.sample_context()

        # oracle for regret
        orw = oracle_reward(user, x)

        # LinUCB
        a_lin = linucb.select(x)
        r_lin = user.reward(a_lin, x)
        linucb.update(a_lin, x, r_lin)
        regs_lin.append(orw - r_lin)

        # epsilon greedy (non-contextual)
        a_eps = eps.select(x)
        r_eps = user.reward(a_eps, x)
        eps.update(a_eps, x, r_eps)
        regs_eps.append(orw - r_eps)

        # fixed after warmup
        a_fix = fixed.select(x)
        r_fix = user.reward(a_fix, x)
        fixed.update(a_fix, x, r_fix)
        regs_fix.append(orw - r_fix)

    return np.array(regs_lin), np.array(regs_eps), np.array(regs_fix)


def main():
    T, d, n_arms, alpha = 3000, 10, 6, 1.0
    seeds = [0, 1, 2, 3, 4]

    curves = {"LinUCB (contextual)": [], "ε-greedy (non-contextual)": [], "Fixed best after warmup": []}

    for s in seeds:
        rl, re, rf = run_once(T=T, d=d, n_arms=n_arms, alpha=alpha, seed=s)
        curves["LinUCB (contextual)"].append(np.cumsum(rl))
        curves["ε-greedy (non-contextual)"].append(np.cumsum(re))
        curves["Fixed best after warmup"].append(np.cumsum(rf))

    # mean across seeds
    xs = np.arange(1, T + 1)
    plt.figure()
    for label, arrs in curves.items():
        mean_curve = np.mean(np.stack(arrs, axis=0), axis=0)
        plt.plot(xs, mean_curve, label=label)
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Regret")
    plt.title("Contextual vs Non-contextual Policies (Synthetic Users)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/fig_regret.png", dpi=160)
    print("Saved plot to experiments/fig_regret.png")

    # print last-10% average regret for a quick table
    def tail_avg(curve): return float(np.mean(curve[int(0.9*T):]))
    for label, arrs in curves.items():
        mean_curve = np.mean(np.stack(arrs, axis=0), axis=0)
        per_step_regret = np.diff(np.concatenate([[0.0], mean_curve]))
        print(f"{label:28s}  tail avg per-step regret: {tail_avg(per_step_regret):.4f}")

if __name__ == "__main__":
    main()