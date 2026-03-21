"""
inference/realtime_assistant.py  –  RealtimePokerAssistant
"""
import collections
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from core.model import AdvancedPokerAI
from core.action_mapper import PokerActionMapper
from core.features import (
    ActionHistoryEncoder,
    build_state_tensor,
    detect_street,
    ACTION_HISTORY_LEN,
)
from core.opponent_tracker import OpponentHUDTracker
from core.equity import HandEquityEstimator
from utils.checkpoint_utils import safe_load_checkpoint

logger = logging.getLogger("PokerAI")


class RealtimePokerAssistant:
    def __init__(self, model_path: str, num_players: int = 6,
                 device: str = 'cpu', equity_sims: int = 500):
        self.num_players   = num_players
        self.device        = torch.device(device)
        self.action_mapper = PokerActionMapper()
        self.equity_est    = HandEquityEstimator(n_sim=equity_sims)

        logger.info(f"Modell betöltése: {model_path}")
        ck = safe_load_checkpoint(model_path, map_location=self.device)

        if isinstance(ck, dict) and 'state_dict' in ck:
            state_size  = ck.get('state_size',  492)
            action_size = ck.get('action_size', 7)
            sd          = ck['state_dict']
        else:
            raise ValueError(f"Érvénytelen checkpoint formátum: {model_path}")

        self.model = AdvancedPokerAI(state_size=state_size, action_size=action_size).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

        self._tracker         = OpponentHUDTracker(num_players)
        self._history_encoder = ActionHistoryEncoder(num_players, PokerActionMapper.NUM_CUSTOM_ACTIONS)
        self._action_history  = collections.deque(maxlen=ACTION_HISTORY_LEN)

        self._bb            = 2.0
        self._sb            = 1.0
        self._my_stack      = 100.0
        self._all_stacks    = [100.0] * num_players
        self._my_player_id  = 0
        self._button_pos    = 0
        self._street        = 0
        self._hand_started  = False

        logger.info(f"RealtimePokerAssistant kész | {num_players}p | state_size={state_size}")

    def new_hand(self, my_stack, all_stacks, bb, sb, my_player_id, button_pos):
        self._bb           = float(bb)
        self._sb           = float(sb)
        self._my_stack     = float(my_stack)
        self._all_stacks   = [float(s) for s in all_stacks]
        self._my_player_id = int(my_player_id)
        self._button_pos   = int(button_pos)
        self._street       = 0
        self._action_history.clear()
        self._hand_started = True

    def new_street(self, street: int):
        self._street = int(street)

    def reset_session(self):
        self._tracker.reset()
        self._action_history.clear()
        self._hand_started = False

    def record_opponent_action(self, player_id, action, bet_amount=0.0, pot_size=1.0):
        bet_norm = min(bet_amount / max(pot_size, 1e-6), 5.0) / 5.0
        self._action_history.append((player_id, action, bet_norm))
        self._tracker.record_action(player_id, action, street=self._street)

    def record_my_action(self, action, bet_amount=0.0, pot_size=1.0):
        bet_norm = min(bet_amount / max(pot_size, 1e-6), 5.0) / 5.0
        self._action_history.append((self._my_player_id, action, bet_norm))

    def get_recommendation(self, obs_vector, legal_actions, hole_cards=None,
                            board_cards=None, current_pot=None, call_amount=0.0):
        board_cards = board_cards or []
        hole_cards  = hole_cards  or []

        state = {
            'obs': np.array(obs_vector, dtype=np.float32),
            'raw_obs': {
                'my_chips': self._my_stack, 'all_chips': self._all_stacks,
                'pot': current_pot or 0.0, 'public_cards': board_cards,
                'hand': [], 'button': self._button_pos, 'call_amount': call_amount,
            },
            'legal_actions': [],
        }

        equity = 0.5
        if hole_cards and len(hole_cards) == 2:
            try:
                equity = self.equity_est.equity(hole_cards, board_cards,
                                                 num_opponents=max(self.num_players-1, 1))
            except Exception:
                equity = 0.5

        state_t = build_state_tensor(
            state, self._tracker, self._action_history, self._history_encoder,
            self.num_players, my_player_id=self._my_player_id,
            bb=self._bb, sb=self._sb, initial_stack=self._my_stack,
            street=self._street, equity=equity,
        )

        with torch.no_grad():
            action_probs, value, _ = self.model.forward(state_t.to(self.device), legal_actions)

        probs_np    = action_probs.squeeze(0).cpu().numpy()
        best_action = int(np.argmax(probs_np))
        confidence  = float(probs_np[best_action])

        sorted_idx = np.argsort(probs_np)[::-1]
        top3 = [(self.action_mapper.action_name(int(i)), float(probs_np[i]))
                for i in sorted_idx[:3] if int(i) in legal_actions]

        pot  = current_pot or 1.0
        spr  = self._my_stack / max(pot, 1e-6)
        m_   = self._my_stack / max(self._bb + self._sb, 1e-6)
        street_name = ['Preflop','Flop','Turn','River'][min(self._street, 3)]

        return {
            'action': best_action,
            'action_name': self.action_mapper.action_name(best_action),
            'confidence': confidence, 'top3': top3, 'equity': equity,
            'spr': round(spr, 2), 'm_ratio': round(m_, 1),
            'street_name': street_name, 'value_est': float(value.squeeze().cpu()),
            'explanation': f"{street_name} | {self.action_mapper.action_name(best_action)} ({confidence*100:.0f}%)",
        }
