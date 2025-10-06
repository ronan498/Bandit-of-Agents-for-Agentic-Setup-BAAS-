def utility(quality: float, cost: float, latency: float,
            wq: float = 1.0, wc: float = 1.0, wl: float = 0.2) -> float:
    return wq*quality - wc*cost - wl*latency