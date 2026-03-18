"""
training/normalizer.py  –  Futó reward normalizáló

Welford online algoritmus: numerikusan stabil, O(1) update.
v4: reward BB-ben mérve (payoffs[0] / bb) mielőtt ide kerül.
"""


class RunningMeanStd:
    """
    Online mean/variance tracker + normalizer.

    clip: szélső értékek levágása (default ±10σ)
    """

    def __init__(self, epsilon: float = 1e-8, clip: float = 10.0):
        self.mean    = 0.0
        self.M2      = 0.0
        self.count   = 0.0
        self.epsilon = epsilon
        self.clip    = clip

    def update(self, x: float):
        self.count += 1.0
        delta       = x - self.mean
        self.mean  += delta / self.count
        self.M2    += delta * (x - self.mean)

    @property
    def var(self) -> float:
        return self.M2 / (self.count - 1.0) if self.count > 1 else 1.0

    def normalize(self, x: float) -> float:
        std  = max(self.var, 0.0) ** 0.5
        norm = (x - self.mean) / (std + self.epsilon)
        return float(max(-self.clip, min(self.clip, norm)))

    def state_dict(self) -> dict:
        return {'mean': self.mean, 'M2': self.M2, 'count': self.count}

    def load_state_dict(self, d: dict):
        self.mean  = d.get('mean',  0.0)
        self.M2    = d.get('M2',    0.0)
        self.count = d.get('count', 0.0)
