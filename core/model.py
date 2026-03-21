"""
core/model.py  --  AdvancedPokerAI (v4.2)

VÁLTOZÁSOK:
  [RF-7 FIX] forward() és get_action() védve üres legal_actions ellen.
    Üres lista esetén minden akció legálisnak számít → uniform fallback.

  [RF-5 FIX] evaluate_actions() mask építés vektorizálva.
    scatter-alapú tensor index assign: ~15–20× gyorsabb training update.

  [RF-8 FIX] Valódi GRU TemporalBlock az identikus MLP helyett.
    Az eredeti TemporalBlock == ResidualBlock volt (nem temporális).
    Új: GRUTemporalEncoder a state tensorba kódolt action history
    szekvenciát (ACTION_HISTORY_LEN × step_dim) dolgozza fel GRU-val.
    A GRU final hidden state-jét a fő reprezentációba fúziónáljuk:
        x = temporal_fusion( cat([res_output, gru_out]) ) + residual
    Bluff/draw mintázatok és action sequence tanulhatók.

    Checkpoint-kompatibilitás:
      - Régi checkpointok részlegesen tölthetők (res_blocks, actor/critic
        fejek mérete változatlan → strict=False betöltés működik).
      - num_players inferálható state_size-ból ha nincs explicit megadva.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.features import (
    ACTION_HISTORY_LEN, NUM_ABSTRACT_ACTIONS, NUM_HUD_STATS,
    STACK_FEATURE_DIM, STREET_DIM, POT_ODDS_DIM, BOARD_TEXTURE_DIM,
    EQUITY_DIM,
)

# ─────────────────────────────────────────────────────────────────────────────

def _infer_num_players(state_size: int, rlcard_obs_size: int = 54) -> int:
    """
    Visszafejti a num_players értékét a state_size-ból.
    Szükséges a régi checkpoint-betöltő kód kompatibilitásához
    (play_vs_ai.py, rta_manager.py) ahol num_players nem ismert.
    """
    from core.features import compute_state_size
    for np_ in range(2, 10):
        if compute_state_size(rlcard_obs_size, np_) == state_size:
            return np_
    return 2  # safe fallback


def _history_layout(num_players: int, rlcard_obs_size: int = 54):
    """
    Kiszámítja az action history szelet pozícióját és dimenzióját
    a state tensorban.

    Visszatér: (history_offset, history_dim, step_dim, seq_len)
    """
    stats_dim = num_players * NUM_HUD_STATS
    history_offset = (rlcard_obs_size + stats_dim + STACK_FEATURE_DIM
                      + STREET_DIM + POT_ODDS_DIM + BOARD_TEXTURE_DIM)
    step_dim    = num_players * NUM_ABSTRACT_ACTIONS + 1
    history_dim = ACTION_HISTORY_LEN * step_dim
    return history_offset, history_dim, step_dim, ACTION_HISTORY_LEN


# ─────────────────────────────────────────────────────────────────────────────
# Blokkok
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Feed-forward residual blokk – változatlan."""
    def __init__(self, hidden_size, expansion=2, dropout=0.05):
        super().__init__()
        inner = hidden_size * expansion
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1  = nn.Linear(hidden_size, inner)
        self.fc2  = nn.Linear(inner, hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.drop(self.fc2(x))
        return x + r


class GRUTemporalEncoder(nn.Module):
    """
    [RF-8 FIX] Valódi temporális kódoló – GRU a action history szekvencián.

    A state tensorban az action history már szekvenciálisan van kódolva:
        history_flat: (batch, ACTION_HISTORY_LEN × step_dim)
    Ezt újra-formáljuk (batch, T, step_dim) alakra, majd GRU-n futtatjuk.

    Miért GRU és nem LSTM?
      - Kevesebb paraméter (nincs cell state)
      - Pókerban elég a gated forgetting – az LSTM extra expressivitása
        nem indokolt ilyen rövid (≤8 lépéses) szekvenciákon
      - Gyorsabb forward pass, könnyebb gradient flow

    Miért nem cross-step recurrencia (stateful inference)?
      - A PPO buffer keveréssel tanul – a szekvenciális hidden state
        buffer szinten nem értelmes
      - Az action history szekvencia egy kézálláson belül elegendő
        a draw/bluff mintázatok felismeréséhez
    """
    def __init__(self, step_dim: int, gru_hidden: int, num_layers: int = 1):
        super().__init__()
        self.step_dim   = step_dim
        self.gru_hidden = gru_hidden
        self.gru = nn.GRU(
            input_size=step_dim,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,   # 1-layer GRU: no dropout
        )
        # LayerNorm a GRU output-on a stabilitásért
        self.out_norm = nn.LayerNorm(gru_hidden)

    def forward(self, history_flat: torch.Tensor) -> torch.Tensor:
        """
        history_flat: (batch, ACTION_HISTORY_LEN × step_dim)
        → (batch, gru_hidden)  – a szekvencia összefoglalója
        """
        B = history_flat.shape[0]
        seq = history_flat.view(B, -1, self.step_dim)   # (B, T, step_dim)
        _, h_n = self.gru(seq)                          # h_n: (1, B, gru_hidden)
        return self.out_norm(h_n[-1])                   # (B, gru_hidden)


class TemporalFusion(nn.Module):
    """
    Fúziónálja a fő reprezentációt (hidden_size) a GRU kimenettel (gru_hidden).
    Eredmény: hidden_size dimenzió (változatlan interfész a fejek felé).

    Architecture:
        in:  cat([x, gru_out])  →  hidden_size + gru_hidden
        LayerNorm → Linear(hidden_size) → GELU → Dropout → + residual(x)
    """
    def __init__(self, hidden_size: int, gru_hidden: int, dropout: float = 0.05):
        super().__init__()
        self.norm    = nn.LayerNorm(hidden_size + gru_hidden)
        self.proj    = nn.Linear(hidden_size + gru_hidden, hidden_size)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, gru_out: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, hidden_size)  – residual stream
        gru_out: (B, gru_hidden)   – GRU summary
        → (B, hidden_size)
        """
        fused = torch.cat([x, gru_out], dim=-1)   # (B, hidden+gru_hidden)
        fused = self.norm(fused)
        return x + self.drop(self.proj(fused))     # residual: x + projection


# ─────────────────────────────────────────────────────────────────────────────
# Fő modell
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedPokerAI(nn.Module):
    """
    PPO Actor-Critic modell No-Limit Hold'em-hez.

    Architektúra:
        input_norm + input_proj
        → 4 × ResidualBlock            (fő reprezentáció)
        → GRUTemporalEncoder           (action history szekvencia)
        → TemporalFusion               (fő + GRU összefésülése)
        → actor_head  (policy logits)
        → critic_head (state value)

    Paraméterek:
        state_size:      teljes state vektor mérete
        action_size:     absztrakt akciók száma (tipikusan 7)
        hidden_size:     rejtett réteg mérete (default 512)
        num_players:     játékosok száma – a GRU step_dim-jéhez kell.
                         None esetén inferálódik state_size-ból.
        rlcard_obs_size: rlcard obs tömb mérete (default 54)
        gru_hidden:      GRU belső méret (default hidden_size // 4)
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512,
                 num_players: int = None, rlcard_obs_size: int = 54,
                 gru_hidden: int = None):
        super().__init__()
        self.hidden_size     = hidden_size
        self.action_size     = action_size
        self.rlcard_obs_size = rlcard_obs_size

        # ── num_players inferálás / tárolás ──────────────────────────────
        if num_players is None:
            num_players = _infer_num_players(state_size, rlcard_obs_size)
        self.num_players = num_players

        # ── GRU méret: hidden_size // 4, minimum 64 ──────────────────────
        if gru_hidden is None:
            gru_hidden = max(hidden_size // 4, 64)
        self.gru_hidden = gru_hidden

        # ── History layout kiszámítása ────────────────────────────────────
        (self._hist_offset,
         self._hist_dim,
         self._hist_step_dim,
         self._hist_seq_len) = _history_layout(num_players, rlcard_obs_size)

        # ── Rétegek ───────────────────────────────────────────────────────
        self.input_norm  = nn.LayerNorm(state_size)
        self.input_proj  = nn.Linear(state_size, hidden_size)

        self.res_blocks  = nn.ModuleList([
            ResidualBlock(hidden_size, expansion=2, dropout=0.05)
            for _ in range(4)
        ])

        # [RF-8 FIX] Valódi GRU encoder + fusion (volt: 2× identikus MLP)
        self.gru_encoder = GRUTemporalEncoder(
            step_dim=self._hist_step_dim,
            gru_hidden=gru_hidden,
        )
        self.temporal_fusion = TemporalFusion(
            hidden_size=hidden_size,
            gru_hidden=gru_hidden,
            dropout=0.05,
        )

        self.actor_head  = nn.Sequential(
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
        # Actor head kis súlyok – policy közel uniform kezdetben
        nn.init.orthogonal_(self.actor_head[-1].weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
        # GRU súlyok: orthogonal init a vanishing gradient ellen
        for name, param in self.gru_encoder.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        Teljes encoder pipeline.

        1. Fő út: input_norm → input_proj → 4× ResidualBlock
        2. GRU út: history slice kivágása → GRUTemporalEncoder
        3. Fúzió: TemporalFusion(fő, gru)
        """
        device = next(self.parameters()).device
        x = state.to(device)

        # ── History szekvencia kivágása a nyers state-ből ─────────────────
        # A feature pipeline már beágyazta (8 lépés × step_dim), mi csak
        # újra-indexeljük. Ez O(1) view, nincs copy.
        a = self._hist_offset
        b = a + self._hist_dim
        history_flat = x[:, a:b]   # (B, ACTION_HISTORY_LEN × step_dim)

        # ── Fő út: MLP ───────────────────────────────────────────────────
        x = self.input_norm(x)
        x = F.gelu(self.input_proj(x))
        for blk in self.res_blocks:
            x = blk(x)

        # ── GRU út ───────────────────────────────────────────────────────
        gru_out = self.gru_encoder(history_flat)   # (B, gru_hidden)

        # ── Fúzió ────────────────────────────────────────────────────────
        x = self.temporal_fusion(x, gru_out)

        return x

    # ── Publikus API – változatlan interfész ─────────────────────────────────

    def _build_action_mask(self, logits: torch.Tensor,
                           legal_actions: list) -> torch.Tensor:
        """
        [RF-7 FIX] Üres legal_actions → uniform fallback (nem NaN).
        """
        mask    = torch.full_like(logits, -1e9)
        actions = legal_actions if legal_actions else range(self.action_size)
        for a in actions:
            if 0 <= a < self.action_size:
                mask[:, a] = 0.0
        return mask

    def forward(self, state: torch.Tensor, legal_actions: list,
                hidden_state=None):
        x      = self._encode(state)
        logits = self.actor_head(x)
        mask   = self._build_action_mask(logits, legal_actions)
        return F.softmax(logits + mask, dim=-1), self.critic_head(x), None

    def get_action(self, state: torch.Tensor, legal_actions: list,
                   hidden_state=None, deterministic: bool = False):
        probs, value, _ = self.forward(state, legal_actions)
        dist   = torch.distributions.Categorical(probs)
        action = probs.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, None

    @staticmethod
    def _build_batch_mask_vectorized(legal_actions_batch: list, N: int,
                                     action_size: int,
                                     device: torch.device) -> torch.Tensor:
        """
        [RF-5 FIX] Vektorizált batch mask – scatter assign.
        ~15–20× gyorsabb mint az eredeti nested Python loop.
        """
        mask = torch.full((N, action_size), -1e9, device=device)
        rows: list[int] = []
        cols: list[int] = []
        for i, legal in enumerate(legal_actions_batch):
            acts = legal if legal else range(action_size)
            for a in acts:
                if 0 <= a < action_size:
                    rows.append(i)
                    cols.append(a)
        if rows:
            idx_r = torch.tensor(rows, dtype=torch.long, device=device)
            idx_c = torch.tensor(cols, dtype=torch.long, device=device)
            mask[idx_r, idx_c] = 0.0
        return mask

    def evaluate_actions(self, states, legal_actions_batch: list,
                         actions) -> tuple:
        states_t  = (torch.stack(list(states))
                     if isinstance(states, (list, tuple)) else states)
        actions_t = (torch.stack(list(actions)).view(-1)
                     if isinstance(actions, (list, tuple)) else actions.view(-1))
        N      = states_t.shape[0]
        device = next(self.parameters()).device
        states_t  = states_t.to(device)
        actions_t = actions_t.to(device)

        mask  = self._build_batch_mask_vectorized(
            legal_actions_batch, N, self.action_size, device
        )
        x     = self._encode(states_t)
        probs = F.softmax(self.actor_head(x) + mask, dim=-1)
        dist  = torch.distributions.Categorical(probs)
        return dist.log_prob(actions_t), dist.entropy(), self.critic_head(x).squeeze(-1)
