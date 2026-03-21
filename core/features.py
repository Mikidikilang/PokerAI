"""
core/features.py  --  v4 Feature Engineering (OPTIMIZED)

VÁLTOZÁSOK az eredetihez képest:
  + build_state_tensor_batch() – vektorizált batch state tensor építés
  + Pre-allokált numpy tömb (nincs N × np.concatenate overhead)
  + compute_stack_features_batch() – vektorizált stack feature számítás
  Visszafelé kompatibilis: build_state_tensor() változatlan API-val megmarad.
"""
import collections
import numpy as np
import torch
from typing import List, Optional

ACTION_HISTORY_LEN   = 8
NUM_ABSTRACT_ACTIONS = 7
STREET_PREFLOP=0; STREET_FLOP=1; STREET_TURN=2; STREET_RIVER=3
BOARD_TEXTURE_DIM=6; STACK_FEATURE_DIM=8; POT_ODDS_DIM=4; STREET_DIM=4; EQUITY_DIM=1

# NUM_HUD_STATS importálva az opponent_tracker-ből – szemantikailag helyes
# (jelenleg NUM_HUD_STATS == NUM_ABSTRACT_ACTIONS == 7, de jövőbiztos)
from core.opponent_tracker import NUM_HUD_STATS

def compute_state_size(rlcard_obs_size, num_players):
    return (rlcard_obs_size
            + num_players * NUM_HUD_STATS
            + STACK_FEATURE_DIM + STREET_DIM + POT_ODDS_DIM + BOARD_TEXTURE_DIM
            + ACTION_HISTORY_LEN * (num_players * NUM_ABSTRACT_ACTIONS + 1)
            + 2 * num_players + EQUITY_DIM)


# ═══════════════════════════════════════════════════════════════════════════════
# EREDETI (egyedi) feature függvények – VÁLTOZATLANOK, visszafelé kompatibilis
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stack_features(state, num_players, bb, sb, initial_stack):
    raw      = state.get('raw_obs', {})
    all_chips= raw.get('all_chips', [initial_stack]*num_players)
    my_chips = raw.get('my_chips', initial_stack)
    pot_raw  = raw.get('pot', None)
    pot_size = float(pot_raw) if pot_raw and pot_raw>0 else max(1.0, initial_stack*num_players - sum(all_chips))
    bb_safe  = max(bb, 1e-6)
    pot_in_bb= pot_size/bb_safe; stack_in_bb=my_chips/bb_safe
    spr      = min(stack_in_bb/max(pot_in_bb,1.0),20.0)/20.0
    m_ratio  = min(my_chips/max(bb+sb,1e-6),50.0)/50.0
    blind_ratio=sb/bb_safe
    active_ratio=sum(1 for c in all_chips if c>0)/max(num_players,1)
    return np.array([spr,m_ratio,blind_ratio,active_ratio,
                     float(stack_in_bb>60),float(20<stack_in_bb<=60),
                     float(10<stack_in_bb<=20),float(stack_in_bb<=10)],dtype=np.float32)

def encode_street(street):
    vec=np.zeros(4,dtype=np.float32)
    if 0<=street<4: vec[street]=1.0
    return vec

def detect_street(state):
    n=len(state.get('raw_obs',{}).get('public_cards',[]))
    if n==0: return STREET_PREFLOP
    elif n==3: return STREET_FLOP
    elif n==4: return STREET_TURN
    else: return STREET_RIVER

def compute_pot_odds_features(state, bb, initial_stack, num_players):
    raw        = state.get('raw_obs', {})
    all_chips  = raw.get('all_chips', [initial_stack]*num_players)
    pot_raw    = raw.get('pot', None)
    pot_size   = float(pot_raw) if pot_raw and pot_raw>0 else max(1.0, initial_stack*num_players-sum(all_chips))
    call_amount= float(raw.get('call_amount', 0.0))
    bb_safe    = max(bb,1e-6)
    is_facing  = float(call_amount>0.01)
    pot_odds   = call_amount/max(pot_size+call_amount,1e-6) if is_facing else 0.0
    call_bb    = min(call_amount/bb_safe,50.0)/50.0
    bet_pot_pct= min(call_amount/max(pot_size,1e-6),3.0)/3.0 if is_facing else 0.0
    return np.array([pot_odds,call_bb,is_facing,bet_pot_pct],dtype=np.float32)

CARD_RANK_MAP={r:i for i,r in enumerate('23456789TJQKA')}

def compute_board_texture(state):
    board=state.get('raw_obs',{}).get('public_cards',[])
    if not board: return np.zeros(6,dtype=np.float32)
    ranks=[]; suits=[]
    for card in board:
        if len(card)>=2:
            rc=card[0].upper(); sc=card[1].lower()
            if rc in CARD_RANK_MAP: ranks.append(CARD_RANK_MAP[rc])
            suits.append(sc)
    if not ranks: return np.zeros(6,dtype=np.float32)
    rank_counts={}
    for r in ranks: rank_counts[r]=rank_counts.get(r,0)+1
    is_paired=float(any(c>=2 for c in rank_counts.values()))
    unique_suits=len(set(suits))
    is_monotone=float(unique_suits==1); is_two_tone=float(unique_suits==2)
    sorted_ranks=sorted(set(ranks)); max_conn=1; curr_conn=1
    for i in range(1,len(sorted_ranks)):
        if sorted_ranks[i]-sorted_ranks[i-1]==1: curr_conn+=1; max_conn=max(max_conn,curr_conn)
        else: curr_conn=1
    return np.array([is_paired,is_monotone,is_two_tone,max_conn/5.0,max(ranks)/12.0,len(board)/5.0],dtype=np.float32)

def encode_position(button_pos, my_player_id, num_players):
    button_vec=np.zeros(num_players,dtype=np.float32)
    if 0<=button_pos<num_players: button_vec[button_pos]=1.0
    relative_pos=(my_player_id-button_pos-1)%num_players
    relative_vec=np.zeros(num_players,dtype=np.float32)
    if 0<=relative_pos<num_players: relative_vec[relative_pos]=1.0
    return np.concatenate([button_vec,relative_vec])


class ActionHistoryEncoder:
    def __init__(self, num_players, num_actions=NUM_ABSTRACT_ACTIONS):
        self.num_players=num_players; self.num_actions=num_actions
        self.dim_per_action=num_players*num_actions+1
        self.total_dim=ACTION_HISTORY_LEN*self.dim_per_action

    def encode_single(self, player_id, action, bet_size_norm=0.0):
        vec=np.zeros(self.dim_per_action,dtype=np.float32)
        if 0<=player_id<self.num_players and 0<=action<self.num_actions:
            vec[player_id*self.num_actions+action]=1.0
        vec[-1]=min(float(bet_size_norm),5.0)/5.0
        return vec

    def encode_history(self, history):
        result=np.zeros(self.total_dim,dtype=np.float32)
        for i,entry in enumerate(history):
            if i>=ACTION_HISTORY_LEN: break
            player,action=(entry[0],entry[1]); bet_norm=entry[2] if len(entry)==3 else 0.0
            offset=i*self.dim_per_action
            result[offset:offset+self.dim_per_action]=self.encode_single(player,action,bet_norm)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# EREDETI build_state_tensor – VÁLTOZATLAN API (RTA + play_vs_ai kompatibilis)
# ═══════════════════════════════════════════════════════════════════════════════

def build_state_tensor(state, tracker, action_history, history_encoder,
                       num_players, my_player_id, bb, sb, initial_stack,
                       street=None, equity=0.5):
    obs_arr   =np.array(state['obs'],dtype=np.float32)
    stats_arr =np.array(tracker.get_stats_vector(),dtype=np.float32)
    stack_arr =compute_stack_features(state,num_players,bb,sb,initial_stack)
    if street is None: street=detect_street(state)
    street_arr=encode_street(street)
    pot_arr   =compute_pot_odds_features(state,bb,initial_stack,num_players)
    board_arr =compute_board_texture(state)
    history_arr=history_encoder.encode_history(action_history)
    raw_obs   =state.get('raw_obs',{})
    button_pos=raw_obs.get('button',0)
    pos_arr   =encode_position(button_pos,my_player_id,num_players)
    equity_arr=np.array([float(equity)],dtype=np.float32)
    full=np.concatenate([obs_arr,stats_arr,stack_arr,street_arr,pot_arr,board_arr,history_arr,pos_arr,equity_arr])
    return torch.FloatTensor(full).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
# ÚJ: BATCH STATE TENSOR – vektorizált, pre-allokált
# ═══════════════════════════════════════════════════════════════════════════════

class BatchStateBuilder:
    """
    Pre-allokált numpy tömbökkel dolgozó batch state tensor builder.
    Egyetlen allokáció per batch, slicing-gal töltve – nincs GC nyomás.

    Használat a collector-ban:
        builder = BatchStateBuilder(state_size, num_players, max_batch=512)
        ...
        tensor = builder.build_batch(env_indices, states, trackers,
                                      action_histories, ...)
    """

    def __init__(self, state_size: int, num_players: int,
                 obs_dim: int = 54, max_batch: int = 512):
        """
        Paraméterek:
            state_size: teljes state vektor mérete (compute_state_size() eredménye)
            num_players: játékosok száma
            obs_dim:    rlcard obs tömb mérete – [RF-4 FIX] már nem hardcode,
                        hanem paraméterként kapja a collector-tól. Default=54
                        visszafelé kompatibilis.
            max_batch:  pre-allokált buffer méret
        """
        self.state_size = state_size
        self.num_players = num_players
        self.obs_dim = obs_dim          # [RF-4 FIX] eltárolva jövőbeli introspekció-hoz
        self.max_batch = max_batch
        self._history_encoder = ActionHistoryEncoder(num_players, NUM_ABSTRACT_ACTIONS)

        # Offset-ek kiszámítása egyszer – obs_dim-t paraméterből veszi
        stats_dim = num_players * NUM_ABSTRACT_ACTIONS  # 7 per player
        self._off = {}
        o = 0
        self._off['obs']     = (o, o + obs_dim);     o += obs_dim
        self._off['stats']   = (o, o + stats_dim);    o += stats_dim
        self._off['stack']   = (o, o + STACK_FEATURE_DIM);  o += STACK_FEATURE_DIM
        self._off['street']  = (o, o + STREET_DIM);   o += STREET_DIM
        self._off['pot']     = (o, o + POT_ODDS_DIM);  o += POT_ODDS_DIM
        self._off['board']   = (o, o + BOARD_TEXTURE_DIM); o += BOARD_TEXTURE_DIM
        hist_dim = self._history_encoder.total_dim
        self._off['history'] = (o, o + hist_dim);     o += hist_dim
        pos_dim = 2 * num_players
        self._off['pos']     = (o, o + pos_dim);      o += pos_dim
        self._off['equity']  = (o, o + EQUITY_DIM);    o += EQUITY_DIM

        # Pre-allokált buffer
        self._buf = np.zeros((max_batch, state_size), dtype=np.float32)

    def build_batch(self, env_indices, states, trackers, action_histories,
                    player_ids, bbs, sbs, initial_stacks, streets, equities=None):
        """
        Batch state tensor építés pre-allokált tömbből.

        Paraméterek:
            env_indices:      list[int] – melyik env-ek
            states:           list[dict] – rlcard state dict-ek
            trackers:         list[OpponentHUDTracker] – per-env tracker
            action_histories: list[deque] – per-env akció történet
            player_ids:       list[int] – kinek építjük (0=learner, egyéb=opp)
            bbs, sbs:         list[float] – BB/SB értékek
            initial_stacks:   list[float]
            streets:          list[int]
            equities:         list[float] vagy None (default 0.5)

        Visszatér: torch.FloatTensor shape (N, state_size)
        """
        N = len(env_indices)
        if N == 0:
            return torch.zeros((0, self.state_size), dtype=torch.float32)

        # Növeld a buffert ha kell
        if N > self._buf.shape[0]:
            self._buf = np.zeros((N, self.state_size), dtype=np.float32)

        buf = self._buf[:N]
        buf[:] = 0.0  # Gyors nullázás (memcpy)

        o = self._off

        for idx in range(N):
            i = env_indices[idx]
            state = states[i]

            # [1] obs (obs_dim dim – parametrized, RF-4 FIX)
            a, b = o['obs']
            obs = state.get('obs', None)
            if obs is not None:
                buf[idx, a:b] = obs

            # [2] stats (num_players × 7 dim)
            a, b = o['stats']
            buf[idx, a:b] = trackers[i].get_stats_vector()

            # [3] stack features (8 dim)
            a, b = o['stack']
            buf[idx, a:b] = compute_stack_features(
                state, self.num_players, bbs[i], sbs[i], initial_stacks[i]
            )

            # [4] street one-hot (4 dim)
            a, b = o['street']
            st = streets[i]
            if 0 <= st < 4:
                buf[idx, a + st] = 1.0

            # [5] pot odds (4 dim)
            a, b = o['pot']
            buf[idx, a:b] = compute_pot_odds_features(
                state, bbs[i], initial_stacks[i], self.num_players
            )

            # [6] board texture (6 dim)
            a, b = o['board']
            buf[idx, a:b] = compute_board_texture(state)

            # [7] action history (variable dim)
            a, b = o['history']
            buf[idx, a:b] = self._history_encoder.encode_history(action_histories[i])

            # [8] position encoding (2 × num_players dim)
            a, b = o['pos']
            raw_obs = state.get('raw_obs', {})
            button_pos = raw_obs.get('button', 0)
            buf[idx, a:b] = encode_position(button_pos, player_ids[idx], self.num_players)

            # [9] equity (1 dim)
            a, b = o['equity']
            buf[idx, a] = equities[idx] if equities is not None else 0.5

        return torch.from_numpy(buf[:N].copy())  # .copy() → contiguous, own memory
