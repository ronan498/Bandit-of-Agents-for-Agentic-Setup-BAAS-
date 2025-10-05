import numpy as np
from baas.bandit.linucb import LinUCB
from baas.sim.user_models import SyntheticUser, oracle_reward

def main(T: int = 5000, d: int = 10, n_arms: int = 6, alpha: float = 1.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    user = SyntheticUser(d=d, n_arms=n_arms, seed=seed, noise_sd=0.05)
    bandit = LinUCB(d=d, n_arms=n_arms, alpha=alpha)

    regrets = []
    rewards = []
    for t in range(T):
        x = user.sample_context()
        a = bandit.select(x)
        r = user.reward(a, x)
        bandit.update(a, x, r)

        # track metrics
        rewards.append(r)
        regrets.append(oracle_reward(user, x) - r)

    print(f"T={T}, d={d}, arms={n_arms}, alpha={alpha}")
    print(f"Avg reward (last 10%): {np.mean(rewards[int(0.9*T):]):.4f}")
    print(f"Avg regret  (last 10%): {np.mean(regrets[int(0.9*T):]):.4f}")

if __name__ == "__main__":
    main()