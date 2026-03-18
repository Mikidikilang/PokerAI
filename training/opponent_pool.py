"""
training/opponent_pool.py  –  Self-play ellenfél pool
"""
import copy, random, collections
from core.model import AdvancedPokerAI

class OpponentPool:
    POOL_SIZE=15; SNAPSHOT_INTERVAL=1_000
    def __init__(self, model_class, model_kwargs):
        self.model_class=model_class; self.model_kwargs=model_kwargs
        self._pool=collections.deque(maxlen=self.POOL_SIZE)
    def snapshot(self, model):
        clone=self.model_class(**self.model_kwargs)
        clone.load_state_dict(copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()}))
        clone.eval(); self._pool.append(clone)
    def get_opponent(self, current_model):
        if not self._pool or random.random()<0.5: return current_model
        return random.choice(list(self._pool))
    def __len__(self): return len(self._pool)
