"""
training/trainer.py  --  PPO Trainer (v4.3 CONFIG-FULL)

[CONFIG-FULL v4.3]
  Minden hiperparaméter a TrainingConfig-ból jön – a kód nem tartalmaz
  hardkódolt konstansokat. Az __init__ cfg=None paramétert fogad; ha
  megadod (runner.py-ból mindig megadja), az összes érték onnan jön.

  Új funkciók:
    lr_scheduler: 'cosine' | 'linear' | 'none'
      - 'cosine'  : eredeti CosineAnnealingLR (oszcillál, T_max ciklusonként)
      - 'linear'  : LinearLR (egyirányú csökkentés, ajánlott degenerált modellnél)
      - 'none'    : konstans LR, nincs scheduler
    
    reset_optimizer_on_load: bool
      - True  → load_state_dict() figyelmen kívül hagyja az optimizer és
                scheduler állapotát – friss LR-rel folytatja
      - False → hagyományos viselkedés (optimizer state visszatölt)
"""
import collections
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .buffer import PPOBuffer

logger = logging.getLogger("PokerAI")


class PPOTrainer:
    # ── Osztályszintű default értékek (fallback ha cfg=None) ─────────────────
    # Ezeket SOHA ne módosítsd kézzel – a config.json / TrainingConfig
    # felülírja őket az __init__-ben.
    CLIP_EPS         = 0.2
    PPO_EPOCHS       = 8
    MINIBATCH        = 256
    VALUE_COEF       = 0.5
    ENTROPY_COEF     = 0.01
    ENTROPY_FINAL    = 0.001
    ENTROPY_DECAY    = 30_000_000
    MAX_GRAD_NORM    = 0.5
    GAMMA            = 0.99
    GAE_LAM          = 0.95
    LR_T_MAX         = 500
    LR_ETA_MIN_RATIO = 0.05
    LR_SCHEDULER     = "cosine"

    def __init__(self, model, lr: float = 3e-4, device=None, cfg=None):
        """
        Args:
            model:  AdvancedPokerAI példány
            lr:     Alap tanulási ráta (cfg.learning_rate felülírja)
            device: torch.device
            cfg:    TrainingConfig – ha megadva, az összes hiperparaméter
                    ebből jön; a class-szintű defaultokat felülírja.
        """
        self.model  = model
        self.device = device or torch.device('cpu')

        # ── Hiperparaméterek betöltése cfg-ből ────────────────────────────
        if cfg is not None:
            self.CLIP_EPS         = cfg.clip_eps
            self.PPO_EPOCHS       = cfg.ppo_epochs
            self.MINIBATCH        = cfg.minibatch_size
            self.VALUE_COEF       = cfg.value_coef
            self.ENTROPY_COEF     = cfg.entropy_coef
            self.ENTROPY_FINAL    = cfg.entropy_final
            self.ENTROPY_DECAY    = cfg.entropy_decay
            self.MAX_GRAD_NORM    = cfg.max_grad_norm
            self.GAMMA            = cfg.gamma
            self.GAE_LAM          = cfg.gae_lambda
            self.LR_T_MAX         = cfg.lr_t_max
            self.LR_ETA_MIN_RATIO = cfg.lr_eta_min_ratio
            lr_scheduler_type     = getattr(cfg, 'lr_scheduler', 'cosine')
            self._reset_opt       = getattr(cfg, 'reset_optimizer_on_load', False)
            lr = cfg.learning_rate
        else:
            lr_scheduler_type = self.LR_SCHEDULER
            self._reset_opt   = False

        self._lr       = lr
        self.use_amp   = (self.device.type == 'cuda')

        # ── Optimizer ────────────────────────────────────────────────────
        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

        # ── LR Scheduler ─────────────────────────────────────────────────
        self.scheduler = self._build_scheduler(lr_scheduler_type, lr)
        self._lr_scheduler_type = lr_scheduler_type

        logger.info(
            f"PPOTrainer init | LR={lr:.2e} | scheduler={lr_scheduler_type} "
            f"| entropy_decay={self.ENTROPY_DECAY:,} "
            f"| reset_opt_on_load={self._reset_opt}"
        )

        self.scaler         = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self._total_updates = 0

    def _build_scheduler(self, scheduler_type: str, lr: float):
        """
        LR scheduler példányosítása típus alapján.

        Args:
            scheduler_type: 'cosine' | 'linear' | 'none'
            lr:             Alap LR (eta_min számításhoz)

        Returns:
            torch LR scheduler, vagy None ha 'none'
        """
        eta_min = lr * self.LR_ETA_MIN_RATIO

        if scheduler_type == "linear":
            # LinearLR: start_factor=1.0 → end_factor=eta_min_ratio,
            # LR_T_MAX update alatt lineárisan csökken. Nem oszcillál.
            # Ajánlott degenerált modell javításakor.
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.LR_ETA_MIN_RATIO,
                total_iters=self.LR_T_MAX,
            )

        elif scheduler_type == "none":
            # Nincs scheduler – LR konstans marad
            return None

        else:  # "cosine" (default)
            # CosineAnnealingLR: T_max update-ciklus után visszaugrik max LR-re
            # Figyelem: ez a degeneráció egyik oka volt (T_max=500 → 1M ep-ként reset)
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.LR_T_MAX,
                eta_min=eta_min,
            )

    def _entropy_coef(self) -> float:
        """Lineárisan interpolált entropy koefficiense az aktuális update-hez."""
        progress = min(self._total_updates / max(self.ENTROPY_DECAY, 1), 1.0)
        return self.ENTROPY_COEF + (self.ENTROPY_FINAL - self.ENTROPY_COEF) * progress

    def update(self, buffer: PPOBuffer, last_value: float = 0.0) -> dict:
        """
        PPO update.

        Args:
            buffer:     PPOBuffer – a gyűjtött tapasztalatok
            last_value: V(s_{T+1}) bootstrap érték (runner.py adja át)
        """
        if len(buffer) < 4:
            return {}

        advantages, returns = buffer.compute_gae(
            self.GAMMA, self.GAE_LAM, last_value=last_value
        )
        dev           = self.device
        old_log_probs = torch.stack(buffer.log_probs).to(dev)
        actions_t     = torch.stack(buffer.actions).view(-1).to(dev)
        states_t      = torch.stack(buffer.states).to(dev)
        advantages    = advantages.to(dev)
        returns_t     = returns.to(dev)

        metrics = collections.defaultdict(list)

        for _ in range(self.PPO_EPOCHS):
            idx = torch.randperm(len(buffer))
            for start in range(0, len(buffer), self.MINIBATCH):
                mb = idx[start: start + self.MINIBATCH]
                if len(mb) < 2:
                    continue

                mb_states  = states_t[mb]
                mb_legal   = [buffer.legal_actions[i.item()] for i in mb]
                mb_actions = actions_t[mb]
                mb_old_lp  = old_log_probs[mb]
                mb_adv     = advantages[mb]
                mb_ret     = returns_t[mb]

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    new_lp, entropy, new_val = self.model.evaluate_actions(
                        mb_states, mb_legal, mb_actions
                    )
                    ratio  = torch.exp(new_lp - mb_old_lp)
                    surr1  = ratio * mb_adv
                    surr2  = torch.clamp(
                        ratio, 1.0 - self.CLIP_EPS, 1.0 + self.CLIP_EPS
                    ) * mb_adv
                    actor_loss  = -torch.min(surr1, surr2).mean()
                    critic_loss = F.huber_loss(new_val, mb_ret)
                    ent_loss    = -entropy.mean()
                    loss = (
                        actor_loss
                        + self.VALUE_COEF  * critic_loss
                        + self._entropy_coef() * ent_loss
                    )

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.MAX_GRAD_NORM
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self._total_updates += 1

                metrics['actor'].append(actor_loss.item())
                metrics['critic'].append(critic_loss.item())
                metrics['entropy'].append(-ent_loss.item())

        if self.scheduler is not None:
            self.scheduler.step()

        buffer.reset()
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def state_dict(self) -> dict:
        return {
            'optimizer':        self.optimizer.state_dict(),
            'scheduler':        self.scheduler.state_dict() if self.scheduler else None,
            'scaler':           self.scaler.state_dict(),
            'total_updates':    self._total_updates,
            'lr_scheduler_type': self._lr_scheduler_type,
        }

    def load_state_dict(self, d: dict) -> None:
        """
        Checkpoint visszaállítása.

        Ha reset_optimizer_on_load=True (config.json-ban állítható!),
        az optimizer és scheduler állapotát kihagyja – friss LR-rel indul.
        Hasznos degenerált modell javításakor.
        """
        if self._reset_opt:
            logger.info(
                "reset_optimizer_on_load=True → optimizer és scheduler "
                "állapot kihagyva, friss LR-rel folytat."
            )
            # Csak a total_updates-t töltjük be (entropy decay helyes marad)
            self._total_updates = d.get('total_updates', 0)
            return

        # ── Normál betöltés ────────────────────────────────────────────
        if 'optimizer' in d:
            try:
                self.optimizer.load_state_dict(d['optimizer'])
            except (KeyError, ValueError) as e:
                logger.warning(
                    f"Optimizer state nem töltve be (architektúra mismatch): {e}"
                )

        if 'scheduler' in d and d['scheduler'] is not None and self.scheduler is not None:
            # Ha az elmentett scheduler típus más, mint a jelenlegi → kihagyjuk
            saved_type   = d.get('lr_scheduler_type', 'cosine')
            current_type = self._lr_scheduler_type
            if saved_type != current_type:
                logger.warning(
                    f"Scheduler típus változott ({saved_type} → {current_type}), "
                    f"scheduler állapot kihagyva (helyes viselkedés)."
                )
            else:
                try:
                    self.scheduler.load_state_dict(d['scheduler'])
                except (KeyError, ValueError) as e:
                    logger.warning(
                        f"Scheduler state nem töltve be: {e}"
                    )

        if 'scaler' in d:
            self.scaler.load_state_dict(d['scaler'])

        self._total_updates = d.get('total_updates', 0)
