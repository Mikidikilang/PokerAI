"""
training/opponent_pool.py  –  Self-play ellenfél pool

POOL_SIZE=15, SNAPSHOT_INTERVAL=1000 epizód.
50% self-play (current model) + 50% random pool tag.
Pool modellek CPU-n tárolódnak (VRAM megőrzés).
"""

import copy
import random
import collections

from core.model import AdvancedPokerAI


class OpponentPool:
    POOL_SIZE          = 15
    SNAPSHOT_INTERVAL  = 1_000

    def __init__(self, model_class, model_kwargs: dict):
        self.model_class  = model_class
        self.model_kwargs = model_kwargs
        self._pool: collections.deque = collections.deque(maxlen=self.POOL_SIZE)

    def snapshot(self, model: AdvancedPokerAI):
        """Aktuális modell deepcopy-ja CPU-ra → pool-ba."""
        clone = self.model_class(**self.model_kwargs)
        clone.load_state_dict(copy.deepcopy(
            {k: v.cpu() for k, v in model.state_dict().items()}
        ))
        clone.eval()
        self._pool.append(clone)

    def get_opponent(self, current_model: AdvancedPokerAI) -> AdvancedPokerAI:
        """50% current, 50% random pool tag."""
        if not self._pool or random.random() < 0.5:
            return current_model
        return random.choice(list(self._pool))

    def __len__(self):
        return len(self._pool)
