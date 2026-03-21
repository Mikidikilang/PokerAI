"""
training/opponent_pool.py  –  Self-play ellenfél pool + Phase 2 exploitative botok

TRÉNING FÁZISOK:
  Phase 1 (PHASE_SELF_PLAY):
    50% self-play + 50% snapshot
    Stabil konvergencia, Nash-irányú tanulás
    Beállítás: OpponentPool(..., phase=OpponentPool.PHASE_SELF_PLAY)

  Phase 2 (PHASE_EXPLOITATIVE):
    30% self-play + 30% snapshot + 40% botok (10% fish/nit/cs/lag egyenként)
    Exploitative play, HUD tanulás, sizing korreláció javítása
    Beállítás: OpponentPool(..., phase=OpponentPool.PHASE_EXPLOITATIVE)
    Ajánlott: 10-12M ep után, ha a modell erősen konvergált self-play-re

  A fázis futás közben is váltható: pool.set_phase(OpponentPool.PHASE_EXPLOITATIVE)

VISSZAFELÉ KOMPATIBILITÁS:
  A bot_ratio paraméter megmaradt Phase 1-ben (ha valaki nem akarja a phase rendszert).
  Ha phase=PHASE_SELF_PLAY és bot_ratio>0, az eredeti viselkedés érvényes.
  Ha phase=PHASE_EXPLOITATIVE, a bot_ratio-t a phase 2 logika felülírja.
"""

import copy, random, collections, logging
import torch
from core.model import AdvancedPokerAI

logger = logging.getLogger("PokerAI")


class OpponentPool:
    POOL_SIZE          = 15
    SNAPSHOT_INTERVAL  = 1_000

    # Tréning fázis konstansok
    PHASE_SELF_PLAY    = 1  # 50/50 self-play + snapshot (eredeti)
    PHASE_EXPLOITATIVE = 2  # 30/30/10/10/10/10 (Phase 2)

    # Phase 2 sampling célok (összesen 1.0)
    _P2_SELF_PLAY  = 0.30
    _P2_SNAPSHOT   = 0.30
    _P2_BOTS       = 0.40  # egyenletesen elosztva a botok közt

    def __init__(self, model_class, model_kwargs, device=None,
                 phase: int = PHASE_SELF_PLAY,
                 bot_ratio: float = 0.0,
                 bot_types: list = None,
                 bot_weights: list = None,
                 num_players: int = None,
                 state_size: int = None):
        """
        Args:
            model_class:  AdvancedPokerAI osztály
            model_kwargs: dict – state_size, action_size, hidden_size
            device:       torch device
            phase:        PHASE_SELF_PLAY (1) vagy PHASE_EXPLOITATIVE (2)
            bot_ratio:    Phase 1-ben: botok aránya [0, 1]. Phase 2-ben figyelmen kívül hagyva.
            bot_types:    ['fish', 'nit', 'calling_station', 'lag']
            num_players:  botok létrehozásához szükséges
            state_size:   botok létrehozásához szükséges
        """
        self.model_class  = model_class
        self.model_kwargs = model_kwargs
        self._pool        = collections.deque(maxlen=self.POOL_SIZE)
        self._device      = device or torch.device('cpu')
        self._phase       = phase
        self._bot_ratio   = max(0.0, min(1.0, bot_ratio))
        self._bots: list  = []

        # Botok inicializálása
        if bot_types and (num_players is not None) and (state_size is not None):
            try:
                from .opponent_archetypes import create_bot
            except ImportError:
                from training.opponent_archetypes import create_bot

            action_size = model_kwargs.get('action_size', 7)
            for bt in bot_types:
                try:
                    bot = create_bot(bt, num_players, state_size, action_size)
                    self._bots.append(bot)
                except ValueError as e:
                    logger.warning(f"Bot létrehozási hiba ({bt}): {e}")

        # A bot_weights alapértelmezése: egyenlő eloszlás
        if bot_weights is not None and len(bot_weights) == len(self._bots):
            total = sum(bot_weights)
            self._bot_weights = [w / total for w in bot_weights]
        else:
            n = max(len(self._bots), 1)
            self._bot_weights = [1.0 / n] * len(self._bots)

        if self._phase == self.PHASE_EXPLOITATIVE:
            logger.info(
                f"OpponentPool Phase 2 – EXPLOITATIVE | "
                f"{len(self._bots)} bot | "
                f"eloszlás: {self._P2_SELF_PLAY:.0%} self / "
                f"{self._P2_SNAPSHOT:.0%} snap / "
                f"{self._P2_BOTS:.0%} botok"
            )
            if self._bots:
                for j, b in enumerate(self._bots):
                    logger.info(f"  {b.__class__.__name__}: {self._bot_weights[j] * self._P2_BOTS:.0%}")
        elif self._bot_ratio > 0 and self._bots:
            logger.info(
                f"OpponentPool Phase 1 | bot_ratio={self._bot_ratio:.0%} | "
                f"{len(self._bots)} bot"
            )
        else:
            logger.info("OpponentPool Phase 1 – pure self-play")

    # ── Fázisváltás futás közben ──────────────────────────────────────────────

    def set_phase(self, phase: int):
        """
        Fázis váltása tréning közben.

        Példa:
            # 10M ep után kapcsolj át exploitative mode-ra
            if total_collected >= 10_000_000:
                pool.set_phase(OpponentPool.PHASE_EXPLOITATIVE)
        """
        if phase not in (self.PHASE_SELF_PLAY, self.PHASE_EXPLOITATIVE):
            raise ValueError(f"Érvénytelen fázis: {phase}")
        old = self._phase
        self._phase = phase
        logger.info(
            f"OpponentPool fázisváltás: {old} → {phase} "
            f"({'exploitative' if phase == self.PHASE_EXPLOITATIVE else 'self-play'})"
        )

    # ── Snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self, model):
        """Modell snapshot mentése a pool-ba (mindkét fázisban szükséges)."""
        clone = self.model_class(**self.model_kwargs)
        sd = {
            (k.replace('_orig_mod.', '', 1) if k.startswith('_orig_mod.') else k): v
            for k, v in model.state_dict().items()
        }
        clone.load_state_dict(copy.deepcopy(sd))
        clone.to(self._device)
        clone.eval()
        self._pool.append(clone)

    # ── Ellenfél kiválasztás ──────────────────────────────────────────────────

    def get_opponent(self, current_model):
        """
        Ellenfél kiválasztása az aktuális fázis alapján.

        Phase 1: bot_ratio-alapú (legacy)
          bot_ratio=0.0 → 50% self, 50% snapshot
          bot_ratio=0.2 → 20% random bot, 40% self, 40% snapshot

        Phase 2: explicit exploitative eloszlás
          30% self-play
          30% snapshot (ha van; különben self-play)
          40% botok (10% egyenként ha 4 bot van)
        """
        if self._phase == self.PHASE_EXPLOITATIVE and self._bots:
            return self._get_opponent_phase2(current_model)
        return self._get_opponent_phase1(current_model)

    def _get_opponent_phase2(self, current_model):
        """
        Phase 2 eloszlás: 30% self / 30% snapshot / 40% botok.

        A botok egyenlő eséllyel kerülnek kiválasztásra (10% mindegyik ha 4 van).
        Ha nincs elég snapshot, a snapshot-slot self-play-ra esik vissza.
        """
        r = random.random()

        if r < self._P2_SELF_PLAY:
            # 30%: self-play
            return current_model

        elif r < self._P2_SELF_PLAY + self._P2_SNAPSHOT:
            # 30%: historical snapshot
            if self._pool:
                return random.choice(list(self._pool))
            return current_model  # fallback ha még nincs snapshot

        else:
            # 40%: botok – súlyozott véletlenszerű kiválasztás
            # Súlyozott véletlenszerű bot kiválasztás
            rnd = random.random()
            cumulative = 0.0
            for bot_idx, weight in enumerate(self._bot_weights):
                cumulative += weight
                if rnd <= cumulative:
                    return self._bots[bot_idx]
            return self._bots[-1]  # fallback

    def _get_opponent_phase1(self, current_model):
        """
        Phase 1 eloszlás (eredeti logika megőrizve visszafelé kompatibilitáshoz).

        bot_ratio=0.0: 50% self, 50% snapshot
        bot_ratio=0.2: 20% random bot, 40% self, 40% snapshot
        """
        r = random.random()

        if self._bots and r < self._bot_ratio:
            return random.choice(self._bots)

        remaining_start = self._bot_ratio
        midpoint = remaining_start + (1.0 - remaining_start) / 2.0

        if r < midpoint:
            return current_model
        else:
            if self._pool:
                return random.choice(list(self._pool))
            return current_model

    # ── Diagnosztika ──────────────────────────────────────────────────────────

    @property
    def current_phase(self) -> int:
        return self._phase

    @property
    def phase_name(self) -> str:
        return "exploitative" if self._phase == self.PHASE_EXPLOITATIVE else "self-play"

    def stats(self) -> dict:
        n_bots = len(self._bots)
        if self._phase == self.PHASE_EXPLOITATIVE and n_bots > 0:
            dist = {
                'self_play': f"{self._P2_SELF_PLAY:.0%}",
                'snapshot':  f"{self._P2_SNAPSHOT:.0%}",
                **{
                    b.__class__.__name__: f"{self._bot_weights[j] * self._P2_BOTS:.0%}"
                    for j, b in enumerate(self._bots)
                },
            }
        else:
            half = (1 - self._bot_ratio) / 2
            dist = {
                'self_play': f"{half:.0%}",
                'snapshot':  f"{half:.0%}",
                'bots_total': f"{self._bot_ratio:.0%}",
            }
        return {
            'phase':       self._phase,
            'phase_name':  self.phase_name,
            'pool_size':   len(self._pool),
            'max_pool':    self.POOL_SIZE,
            'bot_count':   n_bots,
            'bot_types':   [b.__class__.__name__ for b in self._bots],
            'distribution': dist,
        }

    def __len__(self):
        return len(self._pool)

    def __repr__(self):
        return (
            f"OpponentPool(phase={self.phase_name}, "
            f"snapshots={len(self._pool)}, "
            f"bots={[b.__class__.__name__ for b in self._bots]})"
        )
