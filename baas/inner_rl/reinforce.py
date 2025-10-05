import math
import random
from typing import Dict, Any, Callable, Tuple

def bernoulli(p: float) -> int:
    return 1 if random.random() < p else 0

class KnobTuner:
    """
    Minimal REINFORCE-style tuner over a single discrete knob.
    - Maintains a probability p for choosing option1 vs option0.
    - After observing reward r in [0,1], updates p via policy gradient.
    This is intentionally tiny to keep the demo focused.
    """
    def __init__(self, init_p: float = 0.5, lr: float = 0.1, clip: float = 1e-3):
        self.p = max(min(init_p, 1.0 - clip), clip)
        self.lr = lr
        self.clip = clip

    def sample(self) -> Tuple[int, float]:
        """Return (action, logprob). action ∈ {0,1}."""
        a = bernoulli(self.p)
        # log π(a)
        logp = math.log(self.p if a == 1 else (1.0 - self.p))
        return a, logp

    def update(self, action: int, logp: float, reward: float, baseline: float = 0.0):
        """
        REINFORCE: ∇J ≈ (reward - baseline) * ∇logπ(a)
        For Bernoulli, ∂logπ/∂p = 1/p for a=1, and -1/(1-p) for a=0.
        """
        adv = reward - baseline
        if action == 1:
            grad = 1.0 / self.p
        else:
            grad = -1.0 / (1.0 - self.p)
        self.p += self.lr * adv * grad
        # clamp
        self.p = max(min(self.p, 1.0 - self.clip), self.clip)

def tune_and_execute(plan: Dict[str, Any],
                     exec_fn: Callable[[Dict[str, Any]], Any],
                     knob_key: str = "max_steps",
                     option0: int = 3,
                     option1: int = 5,
                     tuner: KnobTuner = None,
                     trials: int = 2):
    """
    Try a couple executions with different knob settings using a tiny policy.
    Return the best result and update the tuner.
    """
    if tuner is None:
        tuner = KnobTuner()

    best_result = None
    baseline = 0.0
    for _ in range(trials):
        a, logp = tuner.sample()
        plan[knob_key] = option1 if a == 1 else option0
        result = exec_fn(plan)
        r = float(result.reward)
        tuner.update(a, logp, r, baseline=baseline)
        baseline = 0.9 * baseline + 0.1 * r  # moving baseline
        if (best_result is None) or (r > float(best_result.reward)):
            best_result = result

    return best_result, tuner