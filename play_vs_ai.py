#!/usr/bin/env python3
"""
play_vs_ai.py  –  Játssz a kitrénelt Poker AI modelleid ellen!

Használat:
    python play_vs_ai.py [--port 8080]

Majd nyisd meg a böngészőben: http://localhost:8080

A szerver automatikusan megtalálja a .pth fájlokat a mappában,
és a böngészős felületen kiválaszthatod melyik modell ellen játszol.
"""

import http.server
import json
import os
import sys
import glob
import collections
import random
import argparse
import webbrowser
import traceback
import logging
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from core.model import AdvancedPokerAI
from core.action_mapper import PokerActionMapper
from core.features import (
    ActionHistoryEncoder, build_state_tensor, detect_street,
    compute_state_size, ACTION_HISTORY_LEN
)
from core.opponent_tracker import OpponentHUDTracker
from core.equity import HandEquityEstimator
from utils.checkpoint_utils import safe_load_checkpoint

import rlcard

# ─── Logging setup: konzol + logs/ mappa ──────────────────────────────────────
from datetime import datetime

def _setup_logging():
    """Konzol (INFO) + fájl (DEBUG) logolás a logs/ mappába."""
    log = logging.getLogger("PlayVsAI")
    log.handlers.clear()
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Konzol – INFO szinttől
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.setLevel(logging.INFO)
    log.addHandler(console)

    # Fájl – logs/ mappába, session-időbélyeggel
    project_root = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = os.path.join(logs_dir, f"play_vs_ai_{timestamp}.log")
    fh = logging.FileHandler(session_file, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)

    # Fő összesítő log (mindig felülírja – utolsó futás)
    main_file = os.path.join(logs_dir, "play_vs_ai_latest.log")
    mh = logging.FileHandler(main_file, encoding="utf-8", mode="w")
    mh.setFormatter(fmt)
    mh.setLevel(logging.DEBUG)
    log.addHandler(mh)

    log.info(f"Session log: {session_file}")
    log.info(f"Latest log:  {main_file}")
    return log

logger = _setup_logging()

# ─────────────────────────────────────────────────────────────────────────────
# Card format conversion
# ─────────────────────────────────────────────────────────────────────────────

SUIT_MAP_RLCARD = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c'}
RANK_MAP_DISPLAY = {
    'A': 'A', '2': '2', '3': '3', '4': '4', '5': '5',
    '6': '6', '7': '7', '8': '8', '9': '9', 'T': '10',
    'J': 'J', 'Q': 'Q', 'K': 'K'
}

def rlcard_to_display(card_str):
    """'SA' → {'rank':'A','suit':'s','display':'A♠'}"""
    if not card_str or len(card_str) < 2:
        return None
    suit_char = card_str[0].upper()
    rank_char = card_str[1].upper()
    suit = SUIT_MAP_RLCARD.get(suit_char, 's')
    rank = RANK_MAP_DISPLAY.get(rank_char, rank_char)
    symbols = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    return {'rank': rank, 'suit': suit, 'display': f"{rank}{symbols[suit]}", 'raw': card_str}

def cards_to_equity_format(rlcard_cards):
    """'SA' → 'As' (equity.py formátum)"""
    result = []
    for c in rlcard_cards:
        if len(c) >= 2:
            suit = c[0].upper()
            rank = c[1].upper()
            suit_lower = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c'}.get(suit, 's')
            result.append(f"{rank}{suit_lower}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model Scanner
# ─────────────────────────────────────────────────────────────────────────────

def scan_models(base_dir='.'):
    """Megkeresi az összes .pth fájlt és kiolvassa a metaadatokat."""
    patterns = [
        os.path.join(base_dir, '*.pth'),
        os.path.join(base_dir, '**', '*.pth'),
    ]
    files = set()
    for p in patterns:
        files.update(glob.glob(p, recursive=True))

    models = []
    for f in sorted(files):
        try:
            ck = safe_load_checkpoint(f, map_location='cpu')
            if isinstance(ck, dict) and 'state_dict' in ck:
                state_size = ck.get('state_size', '?')
                action_size = ck.get('action_size', 7)
                episodes = ck.get('episodes_trained', 0)
                algorithm = ck.get('algorithm', 'unknown')

                # Próbáljuk kitalálni a játékosszámot a state_size-ból
                guessed_players = None
                if isinstance(state_size, int):
                    for np_ in range(2, 10):
                        if compute_state_size(54, np_) == state_size:
                            guessed_players = np_
                            break

                models.append({
                    'path': os.path.relpath(f, base_dir),
                    'abs_path': os.path.abspath(f),
                    'episodes': episodes,
                    'state_size': state_size,
                    'action_size': action_size,
                    'algorithm': algorithm,
                    'guessed_players': guessed_players,
                    'filename': os.path.basename(f),
                })
                logger.info(f"  Modell: {os.path.basename(f)} | "
                            f"{episodes:,} ep | state={state_size} | "
                            f"~{guessed_players or '?'}p")
        except Exception as e:
            logger.debug(f"  Skip {f}: {e}")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Game Session
# ─────────────────────────────────────────────────────────────────────────────

class GameSession:
    """Egy teljes játék session kezelése."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.env = None
        self.num_players = 0
        self.human_seat = 0
        self.action_mapper = PokerActionMapper()
        self.equity_est = HandEquityEstimator(n_sim=200)
        self.tracker = None
        self.history_encoder = None
        self.action_history = collections.deque(maxlen=ACTION_HISTORY_LEN)
        self.street = 0
        self.stacks = []
        self.initial_stack = 100.0
        self.bb = 2.0
        self.sb = 1.0
        self.hand_num = 0
        self.human_cards_cache = []
        self.hand_log = []  # akciók logja az aktuális kézben
        self.total_results = {'player': 0, 'ai': 0, 'hands': 0}
        self._active = False

    def load_model(self, model_path):
        """Modell betöltése."""
        logger.info(f"Modell betöltése: {model_path}")
        ck = safe_load_checkpoint(model_path, map_location=self.device)
        if not (isinstance(ck, dict) and 'state_dict' in ck):
            raise ValueError(f"Érvénytelen checkpoint: {model_path}")

        state_size = ck.get('state_size', 492)
        action_size = ck.get('action_size', 7)
        self.model = AdvancedPokerAI(
            state_size=state_size, action_size=action_size
        ).to(self.device)
        self.model.load_state_dict(ck['state_dict'])
        self.model.eval()
        self._state_size = state_size
        logger.info(f"Modell kész: state_size={state_size}, device={self.device}")
        return state_size, action_size

    def start_session(self, model_path, num_players, human_seat, bb=2, sb=1, stack=100):
        """Új session indítása."""
        self.load_model(model_path)
        self.num_players = num_players
        self.human_seat = human_seat
        self.bb = float(bb)
        self.sb = float(sb)
        self.initial_stack = float(stack)
        self.stacks = [float(stack)] * num_players
        self.hand_num = 0
        self.total_results = {'player': 0, 'ai': 0, 'hands': 0}

        self.env = rlcard.make('no-limit-holdem', config={
            'game_num_players': num_players,
        })
        self.tracker = OpponentHUDTracker(num_players)
        self.history_encoder = ActionHistoryEncoder(
            num_players, PokerActionMapper.NUM_CUSTOM_ACTIONS
        )
        self._active = True
        logger.info(f"Session indítva: {num_players}p, human=seat {human_seat}")
        return True

    def new_hand(self):
        """Új leosztás."""
        if not self._active:
            return {'error': 'Session nincs aktív'}

        self.hand_num += 1
        self.action_history.clear()
        self.hand_log = []
        self.street = 0

        try:
            logger.info(f"[Hand #{self.hand_num}] env.reset()...")
            state, player_id = self.env.reset()
            logger.info(f"[Hand #{self.hand_num}] reset OK, current_player={player_id}, human={self.human_seat}")
        except Exception as e:
            logger.error(f"Env reset hiba: {e}\n{traceback.format_exc()}")
            return {'error': str(e)}

        self.current_state = state
        self.current_player = player_id

        # Cache human cards from game internals
        try:
            hand_cards = self.env.game.players[self.human_seat].hand
            self.human_cards_cache = [c.get_index() for c in hand_cards]
            logger.info(f"[Hand #{self.hand_num}] Human cards: {self.human_cards_cache}")
        except Exception as e:
            logger.warning(f"Human cards cache hiba: {e}")
            self.human_cards_cache = state.get('raw_obs', {}).get('hand', [])

        # Run AI actions until human's turn
        logger.info(f"[Hand #{self.hand_num}] Running AI until human...")
        ai_actions = self._run_ai_until_human()
        logger.info(f"[Hand #{self.hand_num}] AI done, {len(ai_actions)} actions, is_over={self.env.is_over()}")
        return self._build_response(ai_actions)

    def human_action(self, abstract_action):
        """Emberi játékos akciója."""
        if not self._active or self.env.is_over():
            return {'error': 'Nem a te köröd vagy vége a kéznek'}

        abstract_action = int(abstract_action)
        raw_legal = self.current_state.get('legal_actions', [1])
        env_action = self.action_mapper.get_env_action(abstract_action, raw_legal)
        action_name = self.action_mapper.action_name(abstract_action)

        # Log
        self.hand_log.append({
            'player': self.human_seat,
            'player_name': 'Te',
            'action': abstract_action,
            'action_name': action_name,
            'street': self.street,
        })

        # Record
        self.action_history.append((self.human_seat, abstract_action, 0.0))
        self.tracker.record_action(
            self.human_seat, abstract_action, street=self.street
        )

        # Step
        try:
            new_state, new_player = self.env.step(env_action)
            self.current_state = new_state
            self.current_player = new_player
            self.street = detect_street(new_state)
        except Exception as e:
            logger.error(f"Step hiba: {e}")
            return {'error': str(e)}

        # Run AI
        ai_actions = self._run_ai_until_human()
        return self._build_response(ai_actions)

    def _run_ai_until_human(self):
        """AI játékosok léptetése amíg ember nem jön vagy vége."""
        ai_actions = []
        safety = 0
        while not self.env.is_over() and self.current_player != self.human_seat:
            safety += 1
            if safety > 200:
                logger.error("Végtelen loop védelem – kilépés")
                break
            try:
                info = self._ai_act()
                ai_actions.append(info)
            except Exception as e:
                logger.error(f"AI lépés hiba: {e}\n{traceback.format_exc()}")
                break
        return ai_actions

    def _ai_act(self):
        """Egy AI játékos lépése."""
        state = self.current_state
        player_id = self.current_player
        raw_legal = state.get('legal_actions', [1])
        abs_legal = self.action_mapper.get_abstract_legal_actions(raw_legal)

        # Equity számítás az AI saját lapjaival
        equity = 0.5
        try:
            ai_hand = self.env.game.players[player_id].hand
            ai_cards_eq = cards_to_equity_format([c.get_index() for c in ai_hand])
            board_eq = cards_to_equity_format(
                [c.get_index() for c in self.env.game.public_cards]
            )
            if len(ai_cards_eq) == 2:
                equity = self.equity_est.equity(
                    ai_cards_eq, board_eq,
                    num_opponents=max(self.num_players - 1, 1)
                )
        except Exception:
            pass

        # State tensor
        state_t = build_state_tensor(
            state, self.tracker, self.action_history, self.history_encoder,
            self.num_players, my_player_id=player_id,
            bb=self.bb, sb=self.sb, initial_stack=self.initial_stack,
            street=self.street, equity=equity,
        )

        # Model forward
        with torch.no_grad():
            action, log_prob, entropy, value, _ = self.model.get_action(
                state_t.to(self.device), abs_legal, deterministic=False
            )

        abstract_action = int(action.item())
        env_action = self.action_mapper.get_env_action(abstract_action, raw_legal)
        action_name = self.action_mapper.action_name(abstract_action)

        # Record
        self.action_history.append((player_id, abstract_action, 0.0))
        self.tracker.record_action(player_id, abstract_action, street=self.street)

        self.hand_log.append({
            'player': player_id,
            'player_name': f'AI #{player_id}',
            'action': abstract_action,
            'action_name': action_name,
            'street': self.street,
        })

        # Step
        new_state, new_player = self.env.step(env_action)
        self.current_state = new_state
        self.current_player = new_player
        self.street = detect_street(new_state)

        return {
            'player': player_id,
            'action': abstract_action,
            'action_name': action_name,
        }

    def _build_response(self, ai_actions=None):
        """Teljes játékállapot összerakása a frontend számára."""
        is_over = self.env.is_over()

        # Játékos adatok a game internals-ból
        players = []
        try:
            for i, p in enumerate(self.env.game.players):
                cards = [c.get_index() for c in p.hand] if p.hand else []
                # rlcard PlayerStatus enum → string konverzió
                raw_status = getattr(p, 'status', None)
                if raw_status is not None:
                    status_str = str(raw_status).lower()
                    if 'fold' in status_str:
                        status = 'folded'
                    elif 'allin' in status_str or 'all_in' in status_str:
                        status = 'allin'
                    else:
                        status = 'alive'
                else:
                    status = 'alive'

                players.append({
                    'seat': i,
                    'chips_in_pot': float(getattr(p, 'in_chips', 0)),
                    'stack': float(getattr(p, 'remained_chips', 0)),
                    'status': status,
                    'is_human': i == self.human_seat,
                    'name': 'Te' if i == self.human_seat else f'AI #{i}',
                    'cards': [rlcard_to_display(c) for c in cards] if (
                        i == self.human_seat or is_over
                    ) else None,
                })
        except Exception as e:
            logger.error(f"Player data hiba: {e}\n{traceback.format_exc()}")
            players = [{'seat': i, 'name': f'P{i}', 'stack': 0, 'chips_in_pot': 0,
                         'status': 'alive', 'is_human': i == self.human_seat, 'cards': None}
                        for i in range(self.num_players)]

        # Board
        board_cards = []
        try:
            board_cards = [
                rlcard_to_display(c.get_index())
                for c in self.env.game.public_cards
            ]
        except:
            pass

        # Pot
        pot = sum(p['chips_in_pot'] for p in players)

        # Human cards (always show)
        human_cards = [rlcard_to_display(c) for c in self.human_cards_cache]

        # Legal actions
        abs_legal = []
        legal_action_names = {}
        if not is_over and self.current_player == self.human_seat:
            raw_legal = self.current_state.get('legal_actions', [])
            abs_legal = self.action_mapper.get_abstract_legal_actions(raw_legal)
            legal_action_names = {
                a: self.action_mapper.action_name(a) for a in abs_legal
            }

        # Payoffs
        payoffs = None
        if is_over:
            try:
                p = self.env.get_payoffs()
                if p is None or len(p) == 0:
                    logger.warning("get_payoffs() üres vagy None értéket adott vissza")
                elif len(p) <= self.human_seat:
                    logger.warning(
                        f"get_payoffs() csak {len(p)} értéket adott, "
                        f"de human_seat={self.human_seat}"
                    )
                else:
                    payoffs = {i: float(p[i]) for i in range(len(p))}
                    human_result = float(p[self.human_seat])
                    self.total_results['hands'] += 1
                    if human_result > 0:
                        self.total_results['player'] += 1
                    elif human_result < 0:
                        self.total_results['ai'] += 1

                    # Update stacks
                    for i in range(min(len(p), self.num_players)):
                        self.stacks[i] += float(p[i])

            except Exception as e:
                logger.error(f"Payoffs hiba: {e}")

        # Equity (human szempontjából)
        equity = None
        try:
            if len(self.human_cards_cache) == 2:
                h_eq = cards_to_equity_format(self.human_cards_cache)
                b_eq = cards_to_equity_format(
                    [c.get_index() for c in self.env.game.public_cards]
                )
                equity = self.equity_est.equity(
                    h_eq, b_eq, num_opponents=max(self.num_players - 1, 1)
                )
        except:
            pass

        call_amount = 0
        if not is_over:
            try:
                human_chips = players[self.human_seat]['chips_in_pot']
                max_chips = max(p['chips_in_pot'] for p in players)
                call_amount = max_chips - human_chips
            except:
                pass

        return {
            'players': players,
            'board': board_cards,
            'pot': pot,
            'human_cards': human_cards,
            'human_seat': self.human_seat,
            'current_player': self.current_player if not is_over else -1,
            'legal_actions': abs_legal,
            'legal_action_names': legal_action_names,
            'is_over': is_over,
            'payoffs': payoffs,
            'ai_actions': ai_actions or [],
            'street': self.street,
            'street_name': ['Preflop', 'Flop', 'Turn', 'River'][min(self.street, 3)],
            'hand_num': self.hand_num,
            'hand_log': self.hand_log,
            'call_amount': call_amount,
            'stacks': self.stacks[:],
            'total_results': self.total_results.copy(),
            'equity': equity,
            'num_players': self.num_players,
            'bb': self.bb,
        }


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Server
# ─────────────────────────────────────────────────────────────────────────────

session = GameSession()
available_models = []

class GameHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.debug(f"HTTP: {args[0] if args else ''}")

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404, f'File not found: {filepath}')

    def _read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        if length > 0:
            return json.loads(self.rfile.read(length))
        return {}

    def _validate_start_params(self, body: dict) -> list:
        """
        /api/start paraméterek validálása.
        Visszatér: hibalista (üres = minden OK).
        """
        errors = []
        # num_players
        try:
            num_players = int(body.get('num_players', 6))
        except (TypeError, ValueError):
            errors.append("num_players nem egész szám")
            return errors  # többi check ettől függ
        if not 2 <= num_players <= 9:
            errors.append(
                f"num_players={num_players} érvénytelen (2–9 között kell)"
            )

        # human_seat
        try:
            human_seat = int(body.get('human_seat', 0))
        except (TypeError, ValueError):
            errors.append("human_seat nem egész szám")
            human_seat = -1
        if human_seat < 0 or human_seat >= num_players:
            errors.append(
                f"human_seat={human_seat} érvénytelen "
                f"(0 és {num_players - 1} között kell)"
            )

        # bb
        try:
            bb = float(body.get('bb', 2))
        except (TypeError, ValueError):
            errors.append("bb nem szám")
            bb = 0.0
        if bb <= 0:
            errors.append(f"bb={bb} érvénytelen (pozitív kell)")

        # stack
        try:
            stack = float(body.get('stack', 100))
        except (TypeError, ValueError):
            errors.append("stack nem szám")
            stack = 0.0
        if stack <= 0:
            errors.append(f"stack={stack} érvénytelen (pozitív kell)")
        elif bb > 0 and stack < bb * 2:
            errors.append(
                f"stack={stack} túl kicsi (legalább 2×bb={bb * 2:.1f} kell)"
            )

        return errors

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/' or path == '/index.html':
            html_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'play_vs_ai.html'
            )
            self._send_html(html_path)

        elif path == '/api/models':
            self._send_json({'models': available_models})

        elif path == '/api/state':
            if not session._active:
                self._send_json({'error': 'No active session'}, 400)
            else:
                self._send_json(session._build_response())

        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path

        try:
            body = self._read_body()

            if path == '/api/start':
                logger.info(f"POST /api/start: {body}")

                # Input validáció
                validation_errors = self._validate_start_params(body)
                if validation_errors:
                    msg = "; ".join(validation_errors)
                    logger.warning(f"Érvénytelen /api/start paraméterek: {msg}")
                    self._send_json({'error': f'Érvénytelen paraméterek: {msg}'}, 400)
                    return

                model_path = body.get('model_path', '')
                num_players = int(body.get('num_players', 6))
                human_seat = int(body.get('human_seat', 0))
                bb = float(body.get('bb', 2))
                sb = float(body.get('sb', 1))
                stack = float(body.get('stack', 100))

                # Resolve path
                base = os.path.dirname(os.path.abspath(__file__))
                full_path = os.path.join(base, model_path)
                if not os.path.exists(full_path):
                    self._send_json({'error': f'Model not found: {model_path}'}, 404)
                    return

                session.start_session(full_path, num_players, human_seat, bb, sb, stack)
                self._send_json({'ok': True, 'num_players': num_players})

            elif path == '/api/new_hand':
                logger.info("POST /api/new_hand")
                result = session.new_hand()
                logger.info(f"new_hand result keys: {list(result.keys()) if isinstance(result, dict) else 'not dict'}")
                self._send_json(result)

            elif path == '/api/action':
                action = int(body.get('action', 1))
                logger.info(f"POST /api/action: {action}")
                result = session.human_action(action)
                logger.info(f"action result: is_over={result.get('is_over')}, current={result.get('current_player')}")
                self._send_json(result)

            elif path == '/api/rescan':
                global available_models
                base = os.path.dirname(os.path.abspath(__file__))
                available_models = scan_models(base)
                self._send_json({'models': available_models})

            else:
                self.send_error(404)

        except Exception as e:
            logger.error(f"API hiba: {e}\n{traceback.format_exc()}")
            self._send_json({'error': str(e)}, 500)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Play vs your trained Poker AI')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    global available_models
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("\n" + "=" * 60)
    print("  🃏  POKER AI v4  –  Play vs Your Model")
    print("=" * 60)
    print(f"\n  Mappát vizsgálom: {base_dir}")
    available_models = scan_models(base_dir)
    print(f"  Talált modellek: {len(available_models)}")

    if not available_models:
        print("\n  ⚠ Nem találtam .pth fájlt!")
        print("  Tedd a kitrénelt modelljeidet ebbe a mappába.\n")

    html_path = os.path.join(base_dir, 'play_vs_ai.html')
    if not os.path.exists(html_path):
        print(f"\n  ⚠ Hiányzik: play_vs_ai.html")
        print(f"  Tedd a HTML fájlt ide: {html_path}\n")
        return

    server = http.server.HTTPServer(('0.0.0.0', args.port), GameHandler)
    url = f'http://localhost:{args.port}'
    print(f"\n  ✓ Szerver fut: {url}")
    print(f"  Device: {session.device}")
    print(f"\n  Ctrl+C a leállításhoz\n")
    print("=" * 60 + "\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Szerver leállítva.")
        server.server_close()


if __name__ == '__main__':
    main()
