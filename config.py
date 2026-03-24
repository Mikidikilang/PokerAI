"""
config.py -- PokerAI v4 központi konfiguráció

[CONFIG-FULL v4.3] Minden tréning paraméter egy helyen, kód-módosítás nélkül
állítható.  A config.json felülírja a Python default értékeket – így más
modellt, más stílust, más reward shaping-et CSAK a JSON-ban kell beállítani.

Új mezők (v4.3):
  lr_scheduler              – LR scheduler típusa: 'cosine' | 'linear' | 'none'
  reset_optimizer_on_load   – True → checkpoint betöltésekor optimizer/scheduler
                              nulláról indul (degenerált modell újratanításához)
  allin_penalty_*           – All-in spam elleni közvetlen büntetés
  fold_bonus_*              – Gyenge lapokkal való fold jutalmazása
  stack_blindness_penalty_* – Rövid stacknél min-raise büntetése

Adatfolyam:
  config.json["config"]
      ↓  launcher.build_training_config()
  TrainingConfig
      ↓  runner.run_training_session()
  PPOTrainer / BatchedSyncCollector / OpponentPool
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """
    PPO tréning konfiguráció – minden paraméter módosítható a config.json-ban.
    """

    # ── Env & collection ──────────────────────────────────────────────────────

    num_envs: int = 512
    """Párhuzamos rlcard env-ek száma. GPU memória szerint skálázandó."""

    buffer_collect_size: int = 2048
    """Ennyi epizód gyűlik össze PPO update előtt."""

    max_steps_per_hand: int = 500
    """Ennyi lépés után az env deaktiválódik (végtelen kéz védelem)."""

    # ── Modell architektúra ───────────────────────────────────────────────────

    hidden_size: int = 512
    """Residual és fusion blokkok rejtett dimenziója."""

    gru_hidden: int | None = None
    """GRU encoder belső mérete. None → hidden_size // 4 (auto)."""

    # ── PPO hiperparaméterek ──────────────────────────────────────────────────

    learning_rate: float = 3e-4
    """Adam optimizer kezdeti LR."""

    clip_eps: float = 0.2
    """PPO clip epsilon."""

    ppo_epochs: int = 8
    """Minibatch epochs száma update-enként."""

    minibatch_size: int = 256
    """Minibatch méret a PPO update-ben."""

    value_coef: float = 0.5
    """Critic loss súlya."""

    entropy_coef: float = 0.01
    """Kezdeti entropy bónusz."""

    entropy_final: float = 0.001
    """Végső entropy bónusz."""

    entropy_decay: int = 30_000_000
    """Ennyi update-ciklus alatt csökken az entropy_coef entropy_final-ra.
    
    Degenerált modell javításához ajánlott: 900_000
    (a modell gyorsabban konvergál és abbahagyja az all-in spammingét)."""

    max_grad_norm: float = 0.5
    """Gradient clipping normája."""

    gamma: float = 0.99
    """Diszkont faktor."""

    gae_lambda: float = 0.95
    """GAE lambda."""

    # ── LR Scheduler ─────────────────────────────────────────────────────────

    lr_scheduler: str = "cosine"
    """LR scheduler típusa.
    
    Értékek:
      'cosine'  – CosineAnnealingLR (eredeti, oszcillál)
      'linear'  – LinearLR (egyirányú csökkentés, ajánlott degenerált modellnél)
      'none'    – nincs scheduler, LR konstans marad
    """

    lr_t_max: int = 500
    """CosineAnnealingLR vagy LinearLR periódus (update-ciklusban).
    
    Figyelem: ha a modell 17M epizódnál tart és újraindítod a tréninget,
    ezt állítsd a hátralévő update-ciklusok számára (pl. 9765)."""

    lr_eta_min_ratio: float = 0.05
    """Minimális LR = learning_rate × eta_min_ratio (cosine/linear esetén)."""

    # ── Checkpoint / optimizer reset ─────────────────────────────────────────

    reset_optimizer_on_load: bool = False
    """Ha True, checkpoint betöltésekor kihagyja az optimizer és scheduler
    állapotát – friss LR-rel folytatja a tréninget.
    
    Mikor használd:
      - Degenerált modell javításakor (az optimizer momentum rossz irányba mutat)
      - LR scheduler csere után (cosine → linear)
      - Ha a tréning loss nem javul hiába folytatod
    
    Biztonságos: a modell súlyok (state_dict) mindig betöltődnek."""

    # ── Reward shaping – meglévő ──────────────────────────────────────────────

    draw_fold_penalty: float = 0.08
    """BB-büntetés ha postflop fold erős equity-vel."""

    draw_equity_threshold: float = 0.44
    """E felett a postflop fold büntetett."""

    street_reward_scale: float = 0.05
    """Street-átmenet equity delta szorzója."""

    # ── Reward shaping – All-in spam büntetés ────────────────────────────────

    allin_penalty_enabled: bool = False
    """All-in büntetés bekapcsolása.
    
    Hatása: ha a modell all-in-nel megy alacsony equity-vel,
    azonnali büntetést kap – megtanulja, hogy a 72o all-in rossz döntés.
    Ajánlott: True degenerált modellnél."""

    allin_penalty_equity_threshold: float = 0.45
    """E ALATT az all-in action büntetett.
    ~50% = átlagos kéz; 0.45 alatt gyengébb a mediánnál."""

    allin_penalty_amount: float = 0.15
    """All-in büntetés nagysága BB-ben. Ajánlott: 0.10–0.20."""

    # ── Reward shaping – Fold bónusz ─────────────────────────────────────────

    fold_bonus_enabled: bool = False
    """Fold bónusz bekapcsolása gyenge lapokhoz.
    
    Hatása: jutalmat kap a modell ha gyenge lappal dob – közvetlenül
    gyógyítja a '72o all-in' hibát azzal hogy a fold-ot vonzóbbá teszi.
    Ajánlott: True degenerált modellnél."""

    fold_bonus_equity_threshold: float = 0.38
    """E ALATT az fold action jutalmazott (gyenge kéz, helyes döntés dob)."""

    fold_bonus_amount: float = 0.05
    """Fold bónusz nagysága BB-ben. Ajánlott: 0.03–0.08."""

    # ── Reward shaping – Stack-vakság büntetés ────────────────────────────────

    stack_blindness_penalty_enabled: bool = False
    """Stack-blindness büntetés bekapcsolása.
    
    Hatása: rövid stacknél (pl. ≤15BB) a min-raise helyett push/fold
    (all-in vagy fold) a GTO stratégia. Ha mégis min-raise-el, bünteti.
    Ajánlott: True ha a modell short-stack döntéseiben hibák vannak."""

    stack_blindness_bb_threshold: float = 15.0
    """E ALATT (BB-ben mérve) a nem-all-in raise büntetett."""

    stack_blindness_penalty_amount: float = 0.10
    """Stack-blindness büntetés nagysága BB-ben."""

    # ── Mérföldkő rendszer ────────────────────────────────────────────────────

    milestone_interval: int = 2_000_000
    """Ennyi epizódonként ment snapshot + futtat sanity tesztet."""

    milestone_dir_root: str = "ModellNaplo"
    """Mérföldkő mentések gyökérmappája."""

    milestone_hands: int = 2000
    """Sanity teszt kézszám mérföldkőnél."""

    # ── Opponent pool ─────────────────────────────────────────────────────────

    opponent_bot_types: list = field(default_factory=lambda: [
        'fish', 'nit', 'calling_station', 'lag'
    ])
    opponent_bot_weights: list = field(default_factory=lambda: [0.8, 1.5, 0.2, 1.5])

    # ── Equity estimator ─────────────────────────────────────────────────────

    equity_n_sim: int = 100
    """Monte Carlo szimulációk száma (max; adaptív early stopping aktív)."""

    equity_min_sims: int = 30
    """Minimum szimulációk az early stopping előtt."""

    equity_cache_size: int = 100_000
    """Equity cache mérete (LRU)."""

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Checkpoint-ba menthető dict."""
        return {
            k: (list(v) if isinstance(v, list) else v)
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Checkpoint-ból visszaállítás."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
