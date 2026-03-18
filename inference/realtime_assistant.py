"""
inference/realtime_assistant.py  –  RealtimePokerAssistant

A modell wrapper valós idejű póker asszisztens funkcióhoz.
Ez a modul tréningtől független – csak inferencia.

Használat:
    assistant = RealtimePokerAssistant('6max_ppo_v4.pth', num_players=6)

    # Kéz kezdete
    assistant.new_hand(
        my_stack=150.0, all_stacks=[150]*6,
        bb=2.0, sb=1.0,
        my_player_id=2, button_pos=1
    )

    # Döntés kérése
    result = assistant.get_recommendation(
        obs_vector=obs,           # rlcard-kompatibilis obs array
        legal_actions=[1,2,3,4,5,6],
        hole_cards=['As','Kh'],   # opcionális, equity számításhoz
        board_cards=[],
    )
    print(result['action_name'], result['confidence'])

    # Ellenfél lépés rögzítése
    assistant.record_opponent_action(player_id=3, action=4, bet_amount=10.0)
"""

import collections
import logging
import os
import sys

# Projekt gyökér a path-ban (standalone futtatáshoz)
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

logger = logging.getLogger("PokerAI")


class RealtimePokerAssistant:
    """
    Realtime póker asszisztens.

    A modell egy tréninges checkpointból tölt be, és valós obs
    vektorokra ad akcióajánlást + magyarázatot.
    """

    def __init__(self, model_path: str, num_players: int = 6,
                 device: str = 'cpu', equity_sims: int = 500):
        self.num_players   = num_players
        self.device        = torch.device(device)
        self.action_mapper = PokerActionMapper()
        self.equity_est    = HandEquityEstimator(n_sim=equity_sims)

        # ── Modell betöltés ───────────────────────────────────────────────────
        logger.info(f"Modell betöltése: {model_path}")
        ck = torch.load(model_path, map_location=self.device,
                        weights_only=False)

        if isinstance(ck, dict) and 'state_dict' in ck:
            state_size  = ck.get('state_size',  492)
            action_size = ck.get('action_size', 7)
            sd          = ck['state_dict']
        else:
            raise ValueError(
                f"Érvénytelen checkpoint formátum: {model_path}\n"
                "Elvárt: dict {'state_dict': ..., 'state_size': ...}"
            )

        self.model = AdvancedPokerAI(
            state_size  = state_size,
            action_size = action_size,
        ).to(self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

        # ── Per-game state ────────────────────────────────────────────────────
        self._tracker         = OpponentHUDTracker(num_players)
        self._history_encoder = ActionHistoryEncoder(
            num_players, PokerActionMapper.NUM_CUSTOM_ACTIONS
        )
        self._action_history  = collections.deque(maxlen=ACTION_HISTORY_LEN)

        # Game context
        self._bb            = 2.0
        self._sb            = 1.0
        self._my_stack      = 100.0
        self._all_stacks    = [100.0] * num_players
        self._my_player_id  = 0
        self._button_pos    = 0
        self._street        = 0
        self._hand_started  = False

        logger.info(
            f"RealtimePokerAssistant kész | "
            f"{num_players}p | state_size={state_size}"
        )

    # ── Game context API ──────────────────────────────────────────────────────

    def new_hand(self, my_stack: float, all_stacks: list,
                 bb: float, sb: float,
                 my_player_id: int, button_pos: int):
        """
        Új kéz kezdetekor hívandó.
        Tracker NEM resetelődik (megőrzi az ellenfél statokat)!
        """
        self._bb           = float(bb)
        self._sb           = float(sb)
        self._my_stack     = float(my_stack)
        self._all_stacks   = [float(s) for s in all_stacks]
        self._my_player_id = int(my_player_id)
        self._button_pos   = int(button_pos)
        self._street       = 0  # preflop
        self._action_history.clear()
        self._hand_started = True
        logger.debug(
            f"Új kéz | pos={my_player_id} | "
            f"stack={my_stack:.1f}BB | BB={bb} SB={sb}"
        )

    def new_street(self, street: int):
        """
        Street váltáskor hívandó (0=preflop,1=flop,2=turn,3=river).
        """
        self._street = int(street)

    def reset_session(self):
        """Teljes reset (új ülős – tracker is törlődik)."""
        self._tracker.reset()
        self._action_history.clear()
        self._hand_started = False
        logger.info("Session reset.")

    # ── Akció rögzítés ────────────────────────────────────────────────────────

    def record_opponent_action(self, player_id: int,
                                action: int,
                                bet_amount: float = 0.0,
                                pot_size: float = 1.0):
        """
        Ellenfél akciójának rögzítése.

        action: absztrakt akció (0-6), PokerActionMapper szerint.
        bet_amount: a bet összege (0 ha fold/call/check).
        pot_size: aktuális pot (bet_norm számításhoz).
        """
        bet_norm = min(bet_amount / max(pot_size, 1e-6), 5.0) / 5.0
        self._action_history.append((player_id, action, bet_norm))
        self._tracker.record_action(
            player_id, action, street=self._street
        )

    def record_my_action(self, action: int,
                          bet_amount: float = 0.0,
                          pot_size: float = 1.0):
        """Saját akció rögzítése (history-ba)."""
        bet_norm = min(bet_amount / max(pot_size, 1e-6), 5.0) / 5.0
        self._action_history.append(
            (self._my_player_id, action, bet_norm)
        )

    # ── Fő ajánlás API ────────────────────────────────────────────────────────

    def get_recommendation(self, obs_vector,
                            legal_actions: list,
                            hole_cards: list = None,
                            board_cards: list = None,
                            current_pot: float = None,
                            call_amount: float = 0.0) -> dict:
        """
        Akcióajánlás az aktuális játékállapothoz.

        Paraméterek:
            obs_vector    – rlcard-kompatibilis obs (numpy array vagy list)
            legal_actions – absztrakt akció indexek listája
            hole_cards    – ['As','Kh'] saját lapok (equity számításhoz)
            board_cards   – ['Td','7c','2s'] közösségi lapok
            current_pot   – aktuális pot méret (ha None: obs-ból becsli)
            call_amount   – call összege (pot odds számításhoz)

        Visszatér: dict
            'action'       – int (absztrakt akció 0-6)
            'action_name'  – str pl. "Raise 50%"
            'confidence'   – float (ajánlott akció valószínűsége)
            'top3'         – [(action_name, prob), ...] top 3 akció
            'equity'       – float (kéz equity, 0.5 ha nem elérhető)
            'spr'          – float (stack/pot ratio)
            'm_ratio'      – float (stack/blinds)
            'street_name'  – str ('preflop'/'flop'/'turn'/'river')
            'explanation'  – str (rövid szöveges kontextus)
        """
        board_cards = board_cards or []
        hole_cards  = hole_cards  or []

        # ── State dict összerakása ────────────────────────────────────────────
        state = self._build_obs_dict(
            obs_vector, board_cards, call_amount, current_pot
        )

        # ── Equity becslés ───────────────────────────────────────────────────
        equity = 0.5
        if hole_cards and len(hole_cards) == 2:
            try:
                equity = self.equity_est.equity(
                    hole_cards, board_cards,
                    num_opponents=max(self.num_players - 1, 1)
                )
            except Exception:
                equity = 0.5

        # ── State tensor ─────────────────────────────────────────────────────
        state_t = build_state_tensor(
            state,
            self._tracker,
            self._action_history,
            self._history_encoder,
            self.num_players,
            my_player_id  = self._my_player_id,
            bb            = self._bb,
            sb            = self._sb,
            initial_stack = self._my_stack,
            street        = self._street,
            equity        = equity,
        )

        # ── Model forward ────────────────────────────────────────────────────
        with torch.no_grad():
            action_probs, value, _ = self.model.forward(
                state_t.to(self.device), legal_actions
            )

        probs_np = action_probs.squeeze(0).cpu().numpy()

        # ── Ajánlott akció ───────────────────────────────────────────────────
        best_action = int(np.argmax(probs_np))
        confidence  = float(probs_np[best_action])

        # Top-3
        sorted_idx = np.argsort(probs_np)[::-1]
        top3 = [
            (self.action_mapper.action_name(int(idx)), float(probs_np[idx]))
            for idx in sorted_idx[:3]
            if int(idx) in legal_actions
        ]

        # ── Kontextuális értékek ─────────────────────────────────────────────
        bb_s = max(self._bb, 1e-6)
        raw  = state.get('raw_obs', {})
        pot  = current_pot or float(raw.get('pot', 1.0))
        spr  = self._my_stack / max(pot, 1e-6)
        m_   = self._my_stack / max(self._bb + self._sb, 1e-6)

        street_names = ['Preflop', 'Flop', 'Turn', 'River']
        street_name  = street_names[min(self._street, 3)]

        explanation = self._explain(
            best_action, equity, spr, m_,
            confidence, street_name, call_amount
        )

        return {
            'action':       best_action,
            'action_name':  self.action_mapper.action_name(best_action),
            'confidence':   confidence,
            'top3':         top3,
            'equity':       equity,
            'spr':          round(spr, 2),
            'm_ratio':      round(m_, 1),
            'street_name':  street_name,
            'value_est':    float(value.squeeze().cpu()),
            'explanation':  explanation,
        }

    # ── Segédmetódusok ────────────────────────────────────────────────────────

    def _build_obs_dict(self, obs_vector, board_cards: list,
                         call_amount: float,
                         current_pot) -> dict:
        """
        Összerakja az rlcard-kompatibilis state dict-et a bemenetekből.
        """
        obs_arr = np.array(obs_vector, dtype=np.float32)
        return {
            'obs': obs_arr,
            'raw_obs': {
                'my_chips':      self._my_stack,
                'all_chips':     self._all_stacks,
                'pot':           current_pot or 0.0,
                'public_cards':  board_cards,
                'hand':          [],   # hole cards nem kell a feature engineeringhez
                'button':        self._button_pos,
                'call_amount':   call_amount,
            },
            'legal_actions': [],  # legal_actions külön paraméterként jön
        }

    def _explain(self, action: int, equity: float, spr: float,
                  m_ratio: float, confidence: float,
                  street_name: str, call_amount: float) -> str:
        """Rövid emberi magyarázat az ajánláshoz."""
        action_str = self.action_mapper.action_name(action)
        lines = [f"{street_name} | {action_str} ({confidence*100:.0f}%)"]

        if equity > 0.5:
            lines.append(f"Equity: {equity*100:.0f}% (kedvező)")
        elif equity < 0.35:
            lines.append(f"Equity: {equity*100:.0f}% (gyenge kéz)")

        if spr < 3:
            lines.append(f"SPR={spr:.1f} → push-or-fold zóna")
        elif spr > 15:
            lines.append(f"SPR={spr:.1f} → deep stack, pozícionális játék")

        if m_ratio < 10:
            lines.append(f"M={m_ratio:.0f} → veszélyzóna, consider shoving")

        if action == 0:
            lines.append("Fold: pot odds nem éri meg")
        elif action == 1 and call_amount > 0:
            po = call_amount / max(call_amount + 1.0, 1.0)
            lines.append(f"Call: pot odds ~{po*100:.0f}%")

        return " | ".join(lines)
