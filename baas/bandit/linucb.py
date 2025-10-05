import numpy as np

class LinUCB:
    """
    Simple contextual bandit using LinUCB algorithm.
    Each arm has its own linear model A, b.
    """
    def __init__(self, d: int, n_arms: int, alpha: float = 1.0):
        self.d = d
        self.n_arms = n_arms
        self.alpha = alpha
        self.A = [np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros((d, 1)) for _ in range(n_arms)]

    def select(self, x: np.ndarray) -> int:
        """Select arm using UCB criterion given context x."""
        x = x.reshape(-1, 1)
        scores = []
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mu = float(theta.T @ x)
            sigma = float(np.sqrt(x.T @ A_inv @ x))
            scores.append(mu + self.alpha * sigma)
        return int(np.argmax(scores))

    def update(self, arm: int, x: np.ndarray, reward: float):
        """Update linear model for chosen arm."""
        x = x.reshape(-1, 1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x