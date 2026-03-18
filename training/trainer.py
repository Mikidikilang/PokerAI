"""
training/trainer.py  –  PPO Trainer

Hiperparaméterek (NOTES_MASTER.txt §10):
  CLIP_EPS=0.2, PPO_EPOCHS=8, MINIBATCH=256
  VALUE_COEF=0.5, ENTROPY_COEF 0.02→0.002 (50M update)
  MAX_GRAD_NORM=0.5, GAMMA=0.99, GAE_LAM=0.95
  Adam(lr=3e-4, eps=1e-5), StepLR(1M, 0.7)
"""

import collections
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .buffer import PPOBuffer


class PPOTrainer:
    CLIP_EPS      = 0.2
    PPO_EPOCHS    = 8
    MINIBATCH     = 256
    VALUE_COEF    = 0.5
    ENTROPY_COEF  = 0.02
    ENTROPY_FINAL = 0.002
    ENTROPY_DECAY = 50_000_000
    MAX_GRAD_NORM = 0.5
    GAMMA         = 0.99
    GAE_LAM       = 0.95

    def __init__(self, model, lr: float = 3e-4,
                 device: torch.device = None):
        self.model   = model
        self.device  = device or torch.device('cpu')
        self.use_amp = (self.device.type == 'cuda')

        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1_000_000, gamma=0.7
        )
        self.scaler         = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self._total_updates = 0

    def _entropy_coef(self) -> float:
        progress = min(self._total_updates / self.ENTROPY_DECAY, 1.0)
        return (self.ENTROPY_COEF
                + (self.ENTROPY_FINAL - self.ENTROPY_COEF) * progress)

    def update(self, buffer: PPOBuffer) -> dict:
        if len(buffer) < 4:
            return {}

        advantages, returns = buffer.compute_gae(self.GAMMA, self.GAE_LAM)

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
                    ratio       = torch.exp(new_lp - mb_old_lp)
                    surr1       = ratio * mb_adv
                    surr2       = torch.clamp(
                        ratio, 1.0 - self.CLIP_EPS, 1.0 + self.CLIP_EPS
                    ) * mb_adv
                    actor_loss  = -torch.min(surr1, surr2).mean()
                    critic_loss = F.huber_loss(new_val, mb_ret)
                    ent_loss    = -entropy.mean()
                    loss        = (actor_loss
                                   + self.VALUE_COEF   * critic_loss
                                   + self._entropy_coef() * ent_loss)

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

        self.scheduler.step()
        buffer.reset()
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def state_dict(self) -> dict:
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler':    self.scaler.state_dict(),
            'total_updates': self._total_updates,
        }

    def load_state_dict(self, d: dict):
        if 'optimizer' in d:
            self.optimizer.load_state_dict(d['optimizer'])
        if 'scheduler' in d:
            self.scheduler.load_state_dict(d['scheduler'])
        if 'scaler' in d:
            self.scaler.load_state_dict(d['scaler'])
        self._total_updates = d.get('total_updates', 0)
