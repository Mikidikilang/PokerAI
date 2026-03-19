"""
training/opponent_pool.py  –  Self-play ellenfél pool + szabály-alapú botok

v4 VÁLTOZÁS: Bot integráció
  Ha bot_ratio > 0, a get_opponent() néha szabály-alapú botokat ad vissza.
  Visszafelé kompatibilis: ha bot_ratio=0 (default), pontosan úgy működik mint előbb.

HASZNÁLAT:
  # Régi mód (változatlan):
  pool = OpponentPool(AdvancedPokerAI, model_kwargs, device)

  # Új mód (botokkal):
  pool = OpponentPool(AdvancedPokerAI, model_kwargs, device,
                       bot_ratio=0.2,
                       bot_types=['fish', 'nit', 'calling_station', 'lag'],
                       num_players=2, state_size=215)
  
  opp = pool.get_opponent(current_model)
  # 40% self-play, 40% snapshot, 20% random bot
"""

import copy, random, collections, logging
import torch
from core.model import AdvancedPokerAI

logger = logging.getLogger("PokerAI")


class OpponentPool:
    POOL_SIZE = 15
    SNAPSHOT_INTERVAL = 1_000

    def __init__(self, model_class, model_kwargs, device=None,
                 bot_ratio=0.0, bot_types=None,
                 num_players=None, state_size=None):
        """
        model_class:  AdvancedPokerAI osztály
        model_kwargs: dict – state_size, action_size, hidden_size
        device:       torch device
        bot_ratio:    float 0-1 – botok aránya a get_opponent()-ben
                      0.0 = nincs bot (régi viselkedés)
                      0.2 = 20% bot, 40% self-play, 40% snapshot
        bot_types:    list – ['fish', 'nit', 'calling_station', 'lag']
        num_players:  int – szükséges ha bot_ratio > 0
        state_size:   int – szükséges ha bot_ratio > 0
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self._pool = collections.deque(maxlen=self.POOL_SIZE)
        self._device = device or torch.device('cpu')

        # ── Bot integráció ────────────────────────────────────────────────
        self._bot_ratio = max(0.0, min(1.0, bot_ratio))
        self._bots = []

        if self._bot_ratio > 0 and bot_types:
            if num_players is None or state_size is None:
                raise ValueError(
                    "num_players és state_size szükséges ha bot_ratio > 0"
                )
            try:
                from .opponent_archetypes import create_bot
            except ImportError:
                from training.opponent_archetypes import create_bot

            action_size = model_kwargs.get('action_size', 7)
            for bt in bot_types:
                bot = create_bot(bt, num_players, state_size, action_size)
                self._bots.append(bot)
                logger.info(f"  Bot hozzáadva: {bot}")

            logger.info(
                f"OpponentPool: {len(self._bots)} bot, "
                f"ratio={self._bot_ratio:.0%} | "
                f"self-play={(1-self._bot_ratio)/2:.0%} | "
                f"snapshot={(1-self._bot_ratio)/2:.0%}"
            )

    def snapshot(self, model):
        """Modell snapshot mentése a pool-ba."""
        clone = self.model_class(**self.model_kwargs)
        # torch.compile() _orig_mod. prefix kezelése
        sd = {
            (k.replace('_orig_mod.', '', 1) if k.startswith('_orig_mod.') else k): v
            for k, v in model.state_dict().items()
        }
        clone.load_state_dict(copy.deepcopy(sd))
        clone.to(self._device)
        clone.eval()
        self._pool.append(clone)

    def get_opponent(self, current_model):
        """
        Ellenfél kiválasztása.

        Ha bot_ratio = 0 (default):
          50% current_model (self-play)
          50% random snapshot

        Ha bot_ratio > 0 (pl. 0.2):
          bot_ratio%  random bot         (20%)
          (1-bot_ratio)/2%  self-play    (40%)
          (1-bot_ratio)/2%  snapshot     (40%)
        """
        r = random.random()

        # Bot kiválasztás
        if self._bots and r < self._bot_ratio:
            return random.choice(self._bots)

        # Self-play vs snapshot
        # A maradék arányt 50/50 osztjuk
        remaining_start = self._bot_ratio
        midpoint = remaining_start + (1.0 - remaining_start) / 2.0

        if r < midpoint:
            # Self-play
            return current_model
        else:
            # Snapshot
            if self._pool:
                return random.choice(list(self._pool))
            return current_model

    # ── Diagnosztika ──────────────────────────────────────────────────────

    @property
    def bot_count(self):
        return len(self._bots)

    @property
    def bot_ratio(self):
        return self._bot_ratio

    @bot_ratio.setter
    def bot_ratio(self, value):
        """Bot arány dinamikus módosítása tréning közben."""
        self._bot_ratio = max(0.0, min(1.0, value))
        logger.info(f"OpponentPool bot_ratio → {self._bot_ratio:.0%}")

    def stats(self):
        return {
            'pool_size': len(self._pool),
            'max_pool': self.POOL_SIZE,
            'bot_count': len(self._bots),
            'bot_ratio': self._bot_ratio,
            'bot_types': [str(b) for b in self._bots],
        }

    def __len__(self):
        return len(self._pool)

    def __repr__(self):
        return (f"OpponentPool(snapshots={len(self._pool)}, "
                f"bots={len(self._bots)}, ratio={self._bot_ratio:.0%})")
