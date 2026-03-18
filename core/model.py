"""
core/model.py  –  AdvancedPokerAI (v4)

Architektúra változatlan v3.3 óta (feedforward, nincs LSTM/hidden state).
Részletes changelog: NOTES_MASTER.txt §1-2.

Hálózat:
  state → LayerNorm → Linear(state_size→hidden)
        → 4× ResidualBlock (Pre-LN, GELU, expansion=2, dropout=0.05)
        → 2× TemporalBlock (Pre-LN, GELU, dropout=0.05)
        → Actor head:  LN → Linear(h→h/2) → GELU → Linear(h/2→actions)
        → Critic head: LN → Linear(h→h/2) → GELU → Linear(h/2→1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Pre-Norm reziduális blokk GELU aktivációval."""

    def __init__(self, hidden_size: int, expansion: int = 2, dropout: float = 0.05):
        super().__init__()
        inner     = hidden_size * expansion
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1  = nn.Linear(hidden_size, inner)
        self.fc2  = nn.Linear(inner, hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.drop(self.fc2(x))
        return x + residual


class TemporalBlock(nn.Module):
    """
    Feedforward blokk az LSTM helyett (v3.3 óta).
    Pre-Norm: LayerNorm → Linear → GELU → Linear + skip.
    Nincs hidden state → nincs PPO ratio torzítás.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1  = nn.Linear(hidden_size, hidden_size)
        self.fc2  = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.drop(self.fc2(x))
        return x + residual


class AdvancedPokerAI(nn.Module):
    """
    PPO Actor-Critic hálózat pókerhez.

    Paraméterek:
        state_size  – bemeneti obs vektor mérete (automatikusan kalkulált)
        action_size – absztrakt akciók száma (7)
        hidden_size – rejtett réteg mérete (512)
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_size = action_size

        self.input_norm = nn.LayerNorm(state_size)
        self.input_proj = nn.Linear(state_size, hidden_size)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, expansion=2, dropout=0.05)
            for _ in range(4)
        ])

        self.temporal = nn.ModuleList([
            TemporalBlock(hidden_size, dropout=0.05)
            for _ in range(2)
        ])

        self.actor_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, action_size),
        )

        self.critic_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() >= 2 and 'weight' in name:
                nn.init.orthogonal_(param, gain=math.sqrt(2))
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.orthogonal_(self.actor_head[-1].weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def _encode(self, state: torch.Tensor) -> torch.Tensor:
        """Input → ResBlocks → TemporalBlocks. Teljesen feedforward."""
        device = next(self.parameters()).device
        x = state.to(device)
        x = self.input_norm(x)
        x = F.gelu(self.input_proj(x))
        for block in self.res_blocks:
            x = block(x)
        for block in self.temporal:
            x = block(x)
        return x

    def forward(self, state: torch.Tensor, legal_actions: list,
                hidden_state=None):
        """
        Paraméterek:
            state        – (B, state_size) tensor
            legal_actions– list[int], engedélyezett akció indexek
            hidden_state – figyelmen kívül hagyva (API kompatibilitás)

        Visszatér: (action_probs, value, None)
        """
        x = self._encode(state)

        logits = self.actor_head(x)
        mask   = torch.full_like(logits, -1e9)
        for a in legal_actions:
            if 0 <= a < self.action_size:
                mask[:, a] = 0.0

        action_probs = F.softmax(logits + mask, dim=-1)
        value        = self.critic_head(x)
        return action_probs, value, None

    def get_action(self, state: torch.Tensor, legal_actions: list,
                   hidden_state=None, deterministic: bool = False):
        """
        Visszatér: (action, log_prob, entropy, value, None)
        """
        action_probs, value, _ = self.forward(state, legal_actions)
        dist   = torch.distributions.Categorical(action_probs)
        action = action_probs.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, None

    def evaluate_actions(self,
                         states,
                         legal_actions_batch: list,
                         actions):
        """
        Batch kiértékelés PPO update-hez.
        JAVÍTÁS v3.4-hez képest: egyetlen _encode() call!

        Visszatér: (log_probs, entropies, values) – mind (N,) tensor
        """
        if isinstance(states, (list, tuple)):
            states_t = torch.stack(list(states))
        else:
            states_t = states

        if isinstance(actions, (list, tuple)):
            actions_t = torch.stack(list(actions)).view(-1)
        else:
            actions_t = actions.view(-1)

        N      = states_t.shape[0]
        device = next(self.parameters()).device
        states_t  = states_t.to(device)
        actions_t = actions_t.to(device)

        mask = torch.full((N, self.action_size), -1e9, device=device)
        for i, legal in enumerate(legal_actions_batch):
            for a in legal:
                if 0 <= a < self.action_size:
                    mask[i, a] = 0.0

        # EGYETLEN encode pass – v3.4 dupla forward bug javítva
        x = self._encode(states_t)

        logits    = self.actor_head(x)
        probs     = F.softmax(logits + mask, dim=-1)
        dist      = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions_t)
        entropies = dist.entropy()
        values    = self.critic_head(x).squeeze(-1)

        return log_probs, entropies, values
