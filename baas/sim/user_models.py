import numpy as np

class SyntheticUser:
    """
    Toy simulator:
    - Context x ~ N(0, I_d)
    - Each arm a has hidden linear weights theta_a
    - Reward r = sigmoid(theta_a^T x) + noise, then clipped to [0,1]
    """
    def __init__(self, d: int, n_arms: int, seed: int = 0, noise_sd: float = 0.05):
        self.d = d
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)
        self.noise_sd = noise_sd
        # latent linear weights per arm
        self.theta = [self.rng.normal(0, 1, (d, 1)) for _ in range(n_arms)]

    def sample_context(self) -> np.ndarray:
        return self.rng.normal(0, 1, (self.d,))

    def _sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    def reward(self, arm: int, x: np.ndarray) -> float:
        x = x.reshape(-1, 1)
        mu = float(self.theta[arm].T @ x)
        base = self._sigmoid(mu)
        noise = self.rng.normal(0, self.noise_sd)
        r = np.clip(base + noise, 0.0, 1.0)
        return r

def oracle_reward(user: SyntheticUser, x: np.ndarray) -> float:
    """Best achievable expected reward for context x across all arms (with noise suppressed)."""
    x = x.reshape(-1, 1)
    vals = []
    for th in user.theta:
        mu = float(th.T @ x)
        vals.append(1.0 / (1.0 + np.exp(-mu)))
    return max(vals)