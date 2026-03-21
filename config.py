"""
config.py  --  PokerAI v4 központi konfiguráció

[ARCH FIX] Korábban a konstansok szét voltak szórva modulszintű változókként:
  - training/runner.py: NUM_ENVS, BUFFER_COLLECT_SIZE, HIDDEN_SIZE, LEARNING_RATE ...
  - training/trainer.py: CLIP_EPS, PPO_EPOCHS, MINIBATCH, LR_T_MAX ...
  - training/collector.py: _MAX_STEPS_PER_HAND, _STREET_REWARD_SCALE ...

Mostantól egyetlen TrainingConfig dataclass tartalmaz minden konfigurálható
paramétert. A runner.py ezt példányosítja és adja át.

Használat:
    from config import TrainingConfig
    cfg = TrainingConfig()            # default értékek
    cfg = TrainingConfig(num_envs=256, hidden_size=256)   # overrides

Előnyök:
    - Kísérletezhetőség: egy helyen változtatható minden paraméter
    - Type safety: mypy strict mode kompatibilis
    - Dokumentált: minden paraméternek van docstring-je
    - Checkpoint-ba menthető: cfg.__dict__ → JSON
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """
    PPO tréning konfiguráció.
    Minden paraméter módosítható példányosításkor.
    """

    # ── Env & collection ──────────────────────────────────────────────────────
    num_envs: int = 512
    """Párhuzamos rlcard env-ek száma. GPU memória szerint skálázandó."""

    buffer_collect_size: int = 2048
    """Ennyi epizód gyűlik össze PPO update előtt. Nagyobb → stabilabb gradiens."""

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
    """PPO clip epsilon – policy ratio clamp: [1-ε, 1+ε]."""

    ppo_epochs: int = 8
    """Minibatch epochs száma update-enként."""

    minibatch_size: int = 256
    """Minibatch méret a PPO update-ben."""

    value_coef: float = 0.5
    """Critic loss súlya az összesített loss-ban."""

    entropy_coef: float = 0.01
    """Kezdeti entropy bónusz – exploration ösztönzés."""

    entropy_final: float = 0.001
    """Végső entropy bónusz (tréning végén)."""

    entropy_decay: int = 30_000_000
    """Ennyi update-ciklus alatt csökken az entropy_coef entropy_final-ra."""

    max_grad_norm: float = 0.5
    """Gradient clipping normája."""

    gamma: float = 0.99
    """Diszkont faktor."""

    gae_lambda: float = 0.95
    """GAE lambda – trade-off bias/variance."""

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    lr_t_max: int = 500
    """CosineAnnealingLR periódus (update-ciklusban)."""

    lr_eta_min_ratio: float = 0.05
    """Minimális LR = learning_rate × eta_min_ratio."""

    # ── Reward shaping ────────────────────────────────────────────────────────
    draw_fold_penalty: float = 0.08
    """BB-büntetés ha postflop fold erős equity-vel (>= draw_equity_threshold)."""

    draw_equity_threshold: float = 0.44
    """~8-9 outos draw szintje. E felett a fold büntetett."""

    street_reward_scale: float = 0.05
    """[RF-11] Street-átmenet equity delta szorzója. Alacsony szándékosan."""

    # ── Mérföldkő rendszer ────────────────────────────────────────────────────
    milestone_interval: int = 2_000_000
    """Ennyi epizódonként ment snapshot + futtat sanity tesztet."""

    milestone_dir_root: str = "ModellNaplo"
    """Mérföldkő mentések gyökérmappája."""

    # ── Opponent pool ─────────────────────────────────────────────────────────
    opponent_bot_types: list = field(default_factory=lambda: [
        'fish', 'nit', 'calling_station', 'lag'
    ])
    opponent_bot_weights: list = field(default_factory=lambda: [0.8, 1.5, 0.2, 1.5])

    # ── Equity estimator ─────────────────────────────────────────────────────
    equity_n_sim: int = 200
    """Monte Carlo szimulációk száma (maximum; adaptív early stopping)."""

    equity_cache_size: int = 20_000
    """Equity cache mérete. Nagyobb → kevesebb újraszámítás."""

    def to_dict(self) -> dict:
        """Checkpoint-ba menthető dict."""
        return {
            k: (list(v) if isinstance(v, list) else v)
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrainingConfig:
        """Checkpoint-ból visszaállítás."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
