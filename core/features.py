"""
core/features.py  –  v4 Feature Engineering

Teljes state vektor összerakása. Minden feature itt él.

STATE VEKTOR STRUKTÚRA (NOTES_MASTER.txt §4):
  [1] rlcard base obs             (dinamikus méret, BB-normált)
  [2] OpponentHUD stats           (num_players × 7)
  [3] Stack & blind features      (8 dim)
  [4] Street context              (4 dim, one-hot)
  [5] Pot odds & bet context      (4 dim)
  [6] Board texture               (6 dim)
  [7] Action history              (ACTION_HISTORY_LEN × (num_p×7+1))
  [8] Position encoding           (2 × num_players)

compute_state_size(rlcard_obs_size, num_players) → int
build_state_tensor(...)  → torch.Tensor (1, state_size)
"""

import collections
import numpy as np
import torch
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Konstansok
# ─────────────────────────────────────────────────────────────────────────────

ACTION_HISTORY_LEN  = 8   # utolsó N akció tárolva
NUM_ABSTRACT_ACTIONS = 7  # PokerActionMapper.NUM_CUSTOM_ACTIONS

STREET_PREFLOP = 0
STREET_FLOP    = 1
STREET_TURN    = 2
STREET_RIVER   = 3

# Board texture dim
BOARD_TEXTURE_DIM = 6

# Stack feature dim
STACK_FEATURE_DIM = 8

# Pot odds feature dim
POT_ODDS_DIM = 4

# Street dim
STREET_DIM = 4


# ─────────────────────────────────────────────────────────────────────────────
# State size kalkulátor
# ─────────────────────────────────────────────────────────────────────────────

EQUITY_DIM = 1  # Monte Carlo equity scalar


def compute_state_size(rlcard_obs_size: int, num_players: int) -> int:
    """
    Teljes state vektor méretének kiszámítása.
    Mindig innen számolj – ne hardkódolj STATE_SIZE-t!
    """
    opponent_stats_size  = num_players * NUM_ABSTRACT_ACTIONS  # HUD: 7 stat/player
    action_history_size  = ACTION_HISTORY_LEN * (num_players * NUM_ABSTRACT_ACTIONS + 1)
    position_size        = 2 * num_players

    return (
        rlcard_obs_size
        + opponent_stats_size
        + STACK_FEATURE_DIM
        + STREET_DIM
        + POT_ODDS_DIM
        + BOARD_TEXTURE_DIM
        + action_history_size
        + position_size
        + EQUITY_DIM
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Stack & Blind features  (8 dim)
# ─────────────────────────────────────────────────────────────────────────────

def compute_stack_features(state: dict, num_players: int,
                           bb: float, sb: float,
                           initial_stack: float) -> np.ndarray:
    """
    Stack és blind kontextus features.

    Visszatér: np.ndarray (8,)
      [0] spr             – stack/pot normált (0-1, cap=20BB)
      [1] m_ratio         – stack/(BB+SB), normált (0-1, cap=50)
      [2] blind_ratio     – sb/bb (általában 0.5, de custom blindoknál eltér)
      [3] active_ratio    – aktív játékosok aránya
      [4] depth_deep      – stack > 60BB
      [5] depth_mid       – 20 < stack ≤ 60BB
      [6] depth_short     – 10 < stack ≤ 20BB
      [7] depth_push      – stack ≤ 10BB (push-fold zóna)
    """
    raw       = state.get('raw_obs', {})
    all_chips = raw.get('all_chips', [initial_stack] * num_players)
    my_chips  = raw.get('my_chips',  initial_stack)

    # Pot becslés: kezdő stack összeg - maradék chipek
    # Ha rlcard adja a pot-ot, használjuk azt
    pot_raw   = raw.get('pot', None)
    if pot_raw is not None and pot_raw > 0:
        pot_size = float(pot_raw)
    else:
        total_in = initial_stack * num_players
        pot_size = max(1.0, total_in - sum(all_chips))

    bb_safe = max(bb, 1e-6)

    pot_in_bb   = pot_size / bb_safe
    stack_in_bb = my_chips / bb_safe

    spr         = min(stack_in_bb / max(pot_in_bb, 1.0), 20.0) / 20.0
    m_ratio     = min(my_chips / max(bb + sb, 1e-6), 50.0) / 50.0
    blind_ratio = sb / bb_safe
    active_ratio = sum(1 for c in all_chips if c > 0) / max(num_players, 1)

    # Stack depth one-hot (4 dim)
    depth_deep  = float(stack_in_bb > 60)
    depth_mid   = float(20 < stack_in_bb <= 60)
    depth_short = float(10 < stack_in_bb <= 20)
    depth_push  = float(stack_in_bb <= 10)

    return np.array(
        [spr, m_ratio, blind_ratio, active_ratio,
         depth_deep, depth_mid, depth_short, depth_push],
        dtype=np.float32
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Street context  (4 dim, one-hot)
# ─────────────────────────────────────────────────────────────────────────────

def encode_street(street: int) -> np.ndarray:
    """
    Street one-hot encoding.
    street: 0=preflop, 1=flop, 2=turn, 3=river
    Visszatér: np.ndarray (4,)
    """
    vec = np.zeros(4, dtype=np.float32)
    if 0 <= street < 4:
        vec[street] = 1.0
    return vec


def detect_street(state: dict) -> int:
    """
    Street detektálás rlcard state-ből.
    A public_cards (board lapok) számából következtetünk.
    """
    raw    = state.get('raw_obs', {})
    board  = raw.get('public_cards', [])
    n      = len(board)
    if n == 0:
        return STREET_PREFLOP
    elif n == 3:
        return STREET_FLOP
    elif n == 4:
        return STREET_TURN
    else:  # n >= 5
        return STREET_RIVER


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pot odds & bet context  (4 dim)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pot_odds_features(state: dict, bb: float,
                               initial_stack: float,
                               num_players: int) -> np.ndarray:
    """
    Pot odds és bet kontextus.

    Visszatér: np.ndarray (4,)
      [0] pot_odds_norm      – call_amount / (pot + call_amount), 0=nincs bet
      [1] call_amount_bb     – call összeg BB-ben, normált (cap=50BB)
      [2] is_facing_bet      – 1.0 ha van bet/raise amit reagálni kell
      [3] facing_bet_pot_pct – bet mérete pot%-ban (0-3x pot, normált)
    """
    raw        = state.get('raw_obs', {})
    all_chips  = raw.get('all_chips', [initial_stack] * num_players)
    my_chips   = raw.get('my_chips',  initial_stack)

    pot_raw    = raw.get('pot', None)
    if pot_raw is not None and pot_raw > 0:
        pot_size = float(pot_raw)
    else:
        total_in = initial_stack * num_players
        pot_size = max(1.0, total_in - sum(all_chips))

    # Legkisebb stack az ellenfelek közül (közelítő call amount)
    # Az rlcard nem mindig adja a 'call_amount'-ot közvetlenül
    # raw_obs-ból próbáljuk kinyerni, fallback: 0
    call_amount = float(raw.get('call_amount', 0.0))
    if call_amount == 0.0:
        # Közelítés: max tétek különbsége
        stakes     = raw.get('stakes', {})
        current_bets = raw.get('all_chips', [])
        # Ha nem elérhető, 0 marad
    bb_safe = max(bb, 1e-6)

    is_facing  = float(call_amount > 0.01)
    pot_odds   = call_amount / max(pot_size + call_amount, 1e-6) if is_facing else 0.0
    call_bb    = min(call_amount / bb_safe, 50.0) / 50.0
    bet_pot_pct = min(call_amount / max(pot_size, 1e-6), 3.0) / 3.0 if is_facing else 0.0

    return np.array(
        [pot_odds, call_bb, is_facing, bet_pot_pct],
        dtype=np.float32
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Board texture  (6 dim)
# ─────────────────────────────────────────────────────────────────────────────

CARD_RANK_MAP = {r: i for i, r in enumerate('23456789TJQKA')}


def compute_board_texture(state: dict) -> np.ndarray:
    """
    Board textúra feature-ök.

    Visszatér: np.ndarray (6,)
      [0] is_paired          – van-e pár a boardon
      [1] is_monotone        – mind azonos szín
      [2] is_two_tone        – pontosan 2 szín
      [3] max_connectedness  – leghosszabb összefüggő sorozat / 5
      [4] high_card_norm     – legmagasabb lap rank / 14
      [5] num_cards_norm     – board lapok száma / 5
    """
    raw   = state.get('raw_obs', {})
    board = raw.get('public_cards', [])

    if not board:
        return np.zeros(6, dtype=np.float32)

    ranks = []
    suits = []
    for card in board:
        if len(card) >= 2:
            rank_char = card[0].upper()
            suit_char = card[1].lower()
            if rank_char in CARD_RANK_MAP:
                ranks.append(CARD_RANK_MAP[rank_char])
            suits.append(suit_char)

    if not ranks:
        return np.zeros(6, dtype=np.float32)

    # Páros board
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    is_paired = float(any(c >= 2 for c in rank_counts.values()))

    # Szín
    unique_suits = len(set(suits))
    is_monotone  = float(unique_suits == 1)
    is_two_tone  = float(unique_suits == 2)

    # Connectedness: leghosszabb gap-nélküli sorozat
    sorted_ranks = sorted(set(ranks))
    max_conn = 1
    curr_conn = 1
    for i in range(1, len(sorted_ranks)):
        if sorted_ranks[i] - sorted_ranks[i-1] == 1:
            curr_conn += 1
            max_conn = max(max_conn, curr_conn)
        else:
            curr_conn = 1
    max_conn_norm = max_conn / 5.0

    high_card_norm  = max(ranks) / 12.0  # Ace rank=12
    num_cards_norm  = len(board) / 5.0

    return np.array(
        [is_paired, is_monotone, is_two_tone,
         max_conn_norm, high_card_norm, num_cards_norm],
        dtype=np.float32
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Position encoding  (2 × num_players dim)
# ─────────────────────────────────────────────────────────────────────────────

def encode_position(button_pos: int, my_player_id: int,
                    num_players: int) -> np.ndarray:
    """
    Pozíció kódolás.

    Visszatér: np.ndarray (2 × num_players,)
      [0:n]  button_one_hot       – melyik pozícióban van a button
      [n:2n] relative_pos_one_hot – hányan vannak előttem a button után
    """
    button_vec = np.zeros(num_players, dtype=np.float32)
    if 0 <= button_pos < num_players:
        button_vec[button_pos] = 1.0

    relative_pos = (my_player_id - button_pos - 1) % num_players
    relative_vec = np.zeros(num_players, dtype=np.float32)
    if 0 <= relative_pos < num_players:
        relative_vec[relative_pos] = 1.0

    return np.concatenate([button_vec, relative_vec])


# ─────────────────────────────────────────────────────────────────────────────
# 6. Action History Encoder
# ─────────────────────────────────────────────────────────────────────────────

class ActionHistoryEncoder:
    """
    v4 újítás: bet_size_norm is tárolódik akciónként.

    History bejegyzés: (player_id, abstract_action, bet_size_norm)
      bet_size_norm = bet_amount / pot_size  (0 ha fold/call/check)

    One-hot + scalar per slot:
      dim_per_action = num_players × num_actions + 1  (one-hot + bet size)
    """

    def __init__(self, num_players: int,
                 num_actions: int = NUM_ABSTRACT_ACTIONS):
        self.num_players     = num_players
        self.num_actions     = num_actions
        self.dim_per_action  = num_players * num_actions + 1  # +1 bet size
        self.total_dim       = ACTION_HISTORY_LEN * self.dim_per_action

    def encode_single(self, player_id: int, action: int,
                      bet_size_norm: float = 0.0) -> np.ndarray:
        vec = np.zeros(self.dim_per_action, dtype=np.float32)
        if 0 <= player_id < self.num_players and 0 <= action < self.num_actions:
            idx = player_id * self.num_actions + action
            vec[idx] = 1.0
        vec[-1] = min(float(bet_size_norm), 5.0) / 5.0  # normált 0-1
        return vec

    def encode_history(self, history: collections.deque) -> np.ndarray:
        """
        deque[(player_id, action, bet_size_norm)] → flat array (total_dim,)
        Régebbi bejegyzések az elejére kerülnek (időrendi sorrend).
        Ha rövidebb a history → nullával tölt.
        """
        result = np.zeros(self.total_dim, dtype=np.float32)
        for i, entry in enumerate(history):
            if i >= ACTION_HISTORY_LEN:
                break
            if len(entry) == 3:
                player, action, bet_norm = entry
            else:
                player, action = entry[0], entry[1]
                bet_norm = 0.0
            offset = i * self.dim_per_action
            result[offset:offset + self.dim_per_action] = \
                self.encode_single(player, action, bet_norm)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. Fő state tensor builder
# ─────────────────────────────────────────────────────────────────────────────

def build_state_tensor(
    state: dict,
    tracker,                        # OpponentHUDTracker
    action_history: collections.deque,
    history_encoder: ActionHistoryEncoder,
    num_players: int,
    my_player_id: int,
    bb: float,
    sb: float,
    initial_stack: float,
    street: Optional[int] = None,   # ha None: auto-detect
    equity: float = 0.5,            # MC equity becslés (0.5 = semleges)
) -> torch.Tensor:
    """
    Teljes v4 state vektor összerakása.

    Visszatér: torch.Tensor (1, state_size)

    FIGYELEM: ez a függvény feltételezi hogy a tracker és history
    már frissítve van az AKTUÁLIS lépés előtt.
    """
    # ── 1. rlcard base obs ────────────────────────────────────────────────────
    obs_arr = np.array(state['obs'], dtype=np.float32)

    # ── 2. OpponentHUD stats ──────────────────────────────────────────────────
    stats_arr = np.array(tracker.get_stats_vector(), dtype=np.float32)

    # ── 3. Stack & blind features ─────────────────────────────────────────────
    stack_arr = compute_stack_features(
        state, num_players, bb, sb, initial_stack
    )

    # ── 4. Street context ─────────────────────────────────────────────────────
    if street is None:
        street = detect_street(state)
    street_arr = encode_street(street)

    # ── 5. Pot odds & bet context ─────────────────────────────────────────────
    pot_arr = compute_pot_odds_features(
        state, bb, initial_stack, num_players
    )

    # ── 6. Board texture ──────────────────────────────────────────────────────
    board_arr = compute_board_texture(state)

    # ── 7. Action history ─────────────────────────────────────────────────────
    history_arr = history_encoder.encode_history(action_history)

    # ── 8. Position encoding ──────────────────────────────────────────────────
    raw_obs    = state.get('raw_obs', {})
    button_pos = raw_obs.get('button', 0)
    pos_arr    = encode_position(button_pos, my_player_id, num_players)

    # ── 9. Hand equity (1 dim) ────────────────────────────────────────────────
    equity_arr = np.array([float(equity)], dtype=np.float32)

    # ── Concatenate ───────────────────────────────────────────────────────────
    full = np.concatenate([
        obs_arr,
        stats_arr,
        stack_arr,
        street_arr,
        pot_arr,
        board_arr,
        history_arr,
        pos_arr,
        equity_arr,
    ])

    return torch.FloatTensor(full).unsqueeze(0)
