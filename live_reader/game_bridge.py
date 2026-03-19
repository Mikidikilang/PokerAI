"""
live_reader/game_bridge.py  –  Állapotgép: ScreenReader → RTAManager

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MŰKÖDÉS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Minden ~0.5 másodpercben:
  1. ScreenReader.read_frame() → ScreenState
  2. DeltaDetector: mi változott az előző frame óta?
  3. Állapotgép: melyik RTAManager API-t hívjuk?
  4. Callback: javaslat megjelenítése a UI-nak

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HASZNÁLAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  bridge = GameBridge(
      screen_reader = reader,
      rta_manager   = manager,
      my_seat       = 4,
      bb            = 0.04,
      sb            = 0.02,
  )

  bridge.on_recommendation = lambda rec: print(rec['action_name'])
  bridge.run(fps=2.0)
"""

import logging
import time
from typing import Callable, Optional

from .data_types import (
    ScreenState, PlayerRead, CardRead, PlayerAction,
    FrameDelta, GamePhase,
)

logger = logging.getLogger("PokerAI.GameBridge")


# ─────────────────────────────────────────────────────────────────────────────
# Akció konverter: OCR szöveg → absztrakt akció (0-6)
# ─────────────────────────────────────────────────────────────────────────────

def ocr_action_to_abstract(action_type: str, amount: float,
                            pot_size: float, bb: float = 0.04) -> int:
    """
    OCR akció → absztrakt akció (0-6) konverzió.

      0 = Fold
      1 = Call / Check
      2 = Raise min (25%)
      3 = Raise 33%
      4 = Raise 50%
      5 = Raise 75%
      6 = Raise pot / All-in
    """
    action = action_type.lower().strip()

    if "fold" in action or "muck" in action:
        return 0
    if "check" in action:
        return 1
    if "call" in action:
        return 1
    if "all" in action:
        return 6

    if "raise" in action or "bet" in action:
        if pot_size <= 0:
            return 4
        ratio = amount / max(pot_size, 0.01)
        if ratio < 0.30:
            return 2
        elif ratio < 0.40:
            return 3
        elif ratio < 0.60:
            return 4
        elif ratio < 0.85:
            return 5
        else:
            return 6

    # Stack csökkenés alapján
    if amount > 0:
        if pot_size > 0:
            ratio = amount / pot_size
            if ratio >= 0.85: return 6
            elif ratio >= 0.60: return 5
            elif ratio >= 0.40: return 4
            elif ratio >= 0.25: return 3
            else: return 2
        return 4

    return 1  # default check/call


# ─────────────────────────────────────────────────────────────────────────────
# GameBridge – fő állapotgép
# ─────────────────────────────────────────────────────────────────────────────

class GameBridge:
    """
    Összeköti a ScreenReader-t az RTAManager-rel.
    
    Az állapotgép figyeli a képernyőt és automatikusan hívja a
    megfelelő RTAManager API-kat:
      - manage_table_change() – ha az asztalösszetétel változik
      - new_hand()            – új leosztás észlelésekor
      - new_street()          – board változáskor
      - record_opponent_action() – ellenfél lépésekor
      - get_recommendation()  – mielőtt mi lépünk
      - record_my_action()    – miután mi léptünk
    """

    def __init__(self,
                 screen_reader,
                 rta_manager,
                 my_seat: int = 0,
                 bb: float = 0.04,
                 sb: float = 0.02):
        """
        screen_reader: ScreenReader instance
        rta_manager:   RTAManager instance
        my_seat:       saját szék index
        bb, sb:        blind méretek (BoaBet $0.02/$0.04 micro)
        """
        self._reader = screen_reader
        self._manager = rta_manager
        self._my_seat = my_seat
        self._bb = bb
        self._sb = sb

        # Állapotgép
        self._phase = GamePhase.WAITING_FOR_HAND
        self._prev_state: Optional[ScreenState] = None
        self._prev_prev_state: Optional[ScreenState] = None

        # Kéz kontextus
        self._current_board: list = []
        self._current_street: int = 0
        self._hand_start_stacks: dict = {}   # username → induló stack
        self._actions_this_street: list = []
        self._table_initialized: bool = False
        self._current_num_players: int = 0
        self._last_recommendation: Optional[dict] = None
        self._hand_count: int = 0

        # Debounce: ennyi stabil frame után fogadjuk el a változást
        self._stable_board_count: int = 0
        self._stable_board_cards: list = []
        self._BOARD_STABLE_FRAMES = 2

        # Callbacks (a UI vagy konzol hívja)
        self.on_recommendation: Optional[Callable] = None
        self.on_hand_start: Optional[Callable] = None
        self.on_hand_end: Optional[Callable] = None
        self.on_state_update: Optional[Callable] = None
        self.on_opponent_action: Optional[Callable] = None

        logger.info(
            f"GameBridge inicializálva | seat={my_seat} | "
            f"BB=${bb} SB=${sb}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Fő loop
    # ═══════════════════════════════════════════════════════════════════════

    def run(self, fps: float = 2.0):
        """
        Fő végtelen loop. ~2 FPS-sel olvassa a képernyőt.
        Ctrl+C-vel leállítható.
        """
        interval = 1.0 / fps
        logger.info(f"GameBridge loop indítása ({fps} FPS, {interval*1000:.0f}ms/frame)")

        try:
            while True:
                start = time.time()
                self._tick()
                elapsed = time.time() - start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("GameBridge leállítva (Ctrl+C)")

    def tick_once(self) -> Optional[ScreenState]:
        """Egyetlen frame feldolgozása (teszteléshez)."""
        return self._tick()

    # ═══════════════════════════════════════════════════════════════════════
    # Belső logika
    # ═══════════════════════════════════════════════════════════════════════

    def _tick(self) -> Optional[ScreenState]:
        """Egyetlen frame feldolgozási ciklus."""
        try:
            state = self._reader.read_frame()
        except Exception as e:
            logger.error(f"read_frame hiba: {e}", exc_info=True)
            return None

        if not state.is_valid:
            return None

        # Delta detektálás
        delta = self._detect_changes(state)

        # Állapotgép feldolgozás
        self._process_delta(delta, state)

        # State callback
        if self.on_state_update:
            self.on_state_update(state)

        # Eltárolás
        self._prev_prev_state = self._prev_state
        self._prev_state = state

        return state

    # ═══════════════════════════════════════════════════════════════════════
    # Delta detektálás
    # ═══════════════════════════════════════════════════════════════════════

    def _detect_changes(self, current: ScreenState) -> FrameDelta:
        """Két egymást követő frame összehasonlítása."""
        delta = FrameDelta()
        prev = self._prev_state

        if prev is None:
            # Első frame
            if current.num_board_cards == 0 and len(current.my_hole_cards) == 2:
                delta.new_hand_detected = True
            return delta

        # ── Új kéz detektálás ────────────────────────────────────────────
        delta.new_hand_detected = self._detect_new_hand(prev, current)

        # ── Street változás ──────────────────────────────────────────────
        if not delta.new_hand_detected:
            new_street = self._detect_street_change(prev, current)
            if new_street is not None:
                delta.new_street = new_street
                new_board = [
                    c for c in current.board_cards
                    if c not in prev.board_cards
                ]
                delta.new_board_cards = new_board

        # ── Játékos akciók ───────────────────────────────────────────────
        if not delta.new_hand_detected:
            delta.player_actions = self._detect_player_actions(prev, current)

        # ── Aktív játékos változás ───────────────────────────────────────
        if current.active_player_seat != prev.active_player_seat:
            delta.active_player_changed = True
            delta.new_active_seat = current.active_player_seat

        # ── Saját sor változás ───────────────────────────────────────────
        if (current.active_player_seat == self._my_seat and
                prev.active_player_seat != self._my_seat):
            delta.my_turn_started = True

        if (prev.active_player_seat == self._my_seat and
                current.active_player_seat != self._my_seat):
            delta.my_turn_ended = True

        # ── Pot változás ─────────────────────────────────────────────────
        if (current.pot_total is not None and prev.pot_total is not None and
                abs(current.pot_total - prev.pot_total) > 0.01):
            delta.pot_changed = True
            delta.new_pot = current.pot_total

        # ── Kéz vége ────────────────────────────────────────────────────
        if not delta.new_hand_detected:
            delta.hand_over = self._detect_hand_over(prev, current)

        return delta

    def _detect_new_hand(self, prev: ScreenState, current: ScreenState) -> bool:
        """
        Új kéz detektálás – több jelzés kombinációja.
        
        Megbízható jelzések:
        1. Saját hole cards megváltoztak (más lapokat kaptunk)
        2. Board üres lett (előbb volt kártya, most nincs)
        3. Pot drasztikusan csökkent (a kéz végi kiosztás után)
        """
        # 1. Saját lapok megváltoztak
        prev_cards = set(c.to_equity_format() for c in prev.my_hole_cards)
        curr_cards = set(c.to_equity_format() for c in current.my_hole_cards)
        if (len(curr_cards) == 2 and len(prev_cards) == 2 and
                curr_cards != prev_cards):
            logger.info("Új kéz detektálva: hole cards megváltoztak")
            return True

        # 2. Board eltűnt + saját lapok megjelentek
        if (prev.num_board_cards > 0 and current.num_board_cards == 0 and
                len(current.my_hole_cards) == 2):
            logger.info("Új kéz detektálva: board eltűnt + új hole cards")
            return True

        # 3. Pot drasztikusan csökkent (kiosztás után)
        if (prev.pot_total and current.pot_total and
                current.pot_total < prev.pot_total * 0.3 and
                len(current.my_hole_cards) == 2):
            logger.info("Új kéz detektálva: pot csökkent")
            return True

        # 4. Első kéz: ha még nem volt kéz és van hole card
        if (self._phase == GamePhase.WAITING_FOR_HAND and
                len(current.my_hole_cards) == 2 and
                current.num_board_cards == 0):
            logger.info("Első kéz detektálva")
            return True

        return False

    def _detect_street_change(self, prev: ScreenState,
                               current: ScreenState) -> Optional[int]:
        """
        Street változás detektálás debounce-szal.
        Board kártya szám: 0→3 (flop), 3→4 (turn), 4→5 (river)
        """
        prev_n = prev.num_board_cards
        curr_n = current.num_board_cards

        if curr_n <= prev_n:
            self._stable_board_count = 0
            return None

        # Debounce: várjuk meg amíg stabil
        curr_cards_str = [c.to_equity_format() for c in current.board_cards]
        if curr_cards_str == self._stable_board_cards:
            self._stable_board_count += 1
        else:
            self._stable_board_cards = curr_cards_str
            self._stable_board_count = 1

        if self._stable_board_count >= self._BOARD_STABLE_FRAMES:
            self._stable_board_count = 0
            if curr_n == 3:
                return 1  # flop
            elif curr_n == 4:
                return 2  # turn
            elif curr_n == 5:
                return 3  # river

        return None

    def _detect_player_actions(self, prev: ScreenState,
                                current: ScreenState) -> list:
        """
        Játékos akciók detektálása a delta-ból.
        
        Módszerek:
        1. Akció badge szöveg megjelenése ("Fold", "Call", "Raise", "All-In")
        2. Stack csökkenés + pot növekedés → bet/raise/call
        3. Kártyák eltűnése → fold
        """
        actions = []
        prev_players = {p.seat_index: p for p in prev.players}
        curr_players = {p.seat_index: p for p in current.players}

        for seat_idx in curr_players:
            if seat_idx == self._my_seat:
                continue  # saját akciót külön kezeljük

            curr_p = curr_players.get(seat_idx)
            prev_p = prev_players.get(seat_idx)
            if not curr_p or not prev_p:
                continue

            username = curr_p.username or prev_p.username or f"seat_{seat_idx}"

            # 1. Akció badge változás
            if (curr_p.last_action_text and
                    curr_p.last_action_text != prev_p.last_action_text):
                from .screen_reader import TextReader
                action_type, amount = TextReader.parse_action(
                    curr_p.last_action_text
                )
                if action_type:
                    pot = current.pot_total or prev.pot_total or 1.0
                    abstract = ocr_action_to_abstract(
                        action_type, amount, pot, self._bb
                    )
                    actions.append(PlayerAction(
                        seat=seat_idx,
                        username=username,
                        action_type=action_type,
                        amount=amount,
                        abstract_action=abstract,
                    ))
                    continue

            # 2. Stack csökkenés detektálás
            if (prev_p.stack is not None and curr_p.stack is not None and
                    prev_p.stack > curr_p.stack + 0.01):
                diff = prev_p.stack - curr_p.stack
                pot = current.pot_total or 1.0

                # Mi lehetett? Call vagy raise
                if diff <= self._bb * 1.5:
                    action_type = "call"
                    abstract = 1
                else:
                    action_type = "raise"
                    abstract = ocr_action_to_abstract(
                        "raise", diff, pot, self._bb
                    )

                actions.append(PlayerAction(
                    seat=seat_idx,
                    username=username,
                    action_type=action_type,
                    amount=diff,
                    abstract_action=abstract,
                ))
                continue

            # 3. Fold detektálás (kártyák eltűntek)
            if prev_p.is_active and not curr_p.is_active:
                actions.append(PlayerAction(
                    seat=seat_idx,
                    username=username,
                    action_type="fold",
                    amount=0.0,
                    abstract_action=0,
                ))

        return actions

    def _detect_hand_over(self, prev: ScreenState,
                          current: ScreenState) -> bool:
        """Kéz vége detektálás."""
        if self._phase == GamePhase.WAITING_FOR_HAND:
            return False

        # Mindenki foldolt (1 aktív maradt)
        if current.num_active_players <= 1:
            return True

        # Showdown: más játékos kártyái face-up + river
        other_visible = sum(
            1 for p in current.players
            if p.seat_index != self._my_seat and len(p.cards) >= 2
        )
        if other_visible >= 1 and current.street >= 3:
            return True

        return False

    # ═══════════════════════════════════════════════════════════════════════
    # Állapotgép feldolgozás
    # ═══════════════════════════════════════════════════════════════════════

    def _process_delta(self, delta: FrameDelta, state: ScreenState):
        """A delta alapján az állapotgép léptetése és RTAManager hívások."""

        # ── Kéz vége ────────────────────────────────────────────────────
        if delta.hand_over and self._phase != GamePhase.WAITING_FOR_HAND:
            self._handle_hand_over(state)

        # ── Új kéz ──────────────────────────────────────────────────────
        if delta.new_hand_detected:
            self._handle_new_hand(state)

        # ── Street váltás ────────────────────────────────────────────────
        if delta.new_street is not None and self._phase == GamePhase.HAND_ACTIVE:
            self._handle_street_change(delta.new_street, state)

        # ── Ellenfél akciók ──────────────────────────────────────────────
        if delta.player_actions and self._phase in (
                GamePhase.HAND_ACTIVE, GamePhase.MY_TURN):
            for action in delta.player_actions:
                self._handle_opponent_action(action, state)

        # ── Saját akció rögzítése (ha az előző frame-ben mi voltunk) ─────
        if delta.my_turn_ended and self._last_recommendation:
            self._handle_my_action_done(state)

        # ── Én következem ────────────────────────────────────────────────
        if delta.my_turn_started and self._phase == GamePhase.HAND_ACTIVE:
            self._handle_my_turn(state)

    # ═══════════════════════════════════════════════════════════════════════
    # Handler metódusok
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_new_hand(self, state: ScreenState):
        """Új kéz → RTAManager.new_hand()"""
        self._hand_count += 1
        self._phase = GamePhase.HAND_ACTIVE
        self._current_street = 0
        self._current_board = []
        self._actions_this_street = []
        self._last_recommendation = None
        self._stable_board_count = 0
        self._stable_board_cards = []

        # Asztal összetétel ellenőrzés
        active_players = [p for p in state.players if p.is_active]
        num_active = len(active_players)

        # Seat map
        seat_map = {}
        all_stacks = {}
        for p in active_players:
            username = p.username or f"seat_{p.seat_index}"
            seat_map[p.seat_index] = username
            all_stacks[username] = p.stack or 100.0

        # Induló stackek mentése
        self._hand_start_stacks = {
            (p.username or f"seat_{p.seat_index}"): p.stack
            for p in active_players if p.stack
        }

        # Ha az asztalméret változott → manage_table_change
        if num_active != self._current_num_players or not self._table_initialized:
            self._current_num_players = num_active
            dealer_seat = state.dealer_seat or 0

            try:
                self._manager.manage_table_change(
                    num_players=num_active,
                    seat_map=seat_map,
                    my_seat=self._my_seat,
                    button_seat=dealer_seat,
                )
                self._table_initialized = True
            except Exception as e:
                logger.error(f"manage_table_change hiba: {e}", exc_info=True)
                return

        # new_hand() hívás
        my_player = next(
            (p for p in state.players if p.seat_index == self._my_seat),
            None
        )
        my_stack = my_player.stack if my_player and my_player.stack else 100.0

        try:
            self._manager.new_hand(
                my_stack=my_stack,
                all_stacks=all_stacks,
                bb=self._bb,
                sb=self._sb,
            )
        except Exception as e:
            logger.error(f"new_hand hiba: {e}", exc_info=True)
            return

        hole_cards = [c.to_equity_format() for c in state.my_hole_cards]
        logger.info(
            f"[Hand #{self._hand_count}] Új kéz | {num_active}p | "
            f"stack=${my_stack:.2f} | cards={hole_cards}"
        )

        if self.on_hand_start:
            self.on_hand_start({
                'hand_num': self._hand_count,
                'num_players': num_active,
                'my_stack': my_stack,
                'hole_cards': hole_cards,
                'bb': self._bb,
                'sb': self._sb,
            })

    def _handle_street_change(self, street: int, state: ScreenState):
        """Street váltás → RTAManager.new_street()"""
        self._current_street = street
        self._actions_this_street = []
        self._current_board = [c.to_equity_format() for c in state.board_cards]

        try:
            self._manager.new_street(street)
        except Exception as e:
            logger.error(f"new_street hiba: {e}", exc_info=True)

        street_names = {1: 'Flop', 2: 'Turn', 3: 'River'}
        logger.info(
            f"  Street: {street_names.get(street, '?')} | "
            f"board={self._current_board}"
        )

    def _handle_opponent_action(self, action: PlayerAction,
                                 state: ScreenState):
        """Ellenfél akció → RTAManager.record_opponent_action()"""
        pot = state.pot_total or 1.0

        # Kontextus (egyszerűsített – pontosabb verzió később)
        context = {
            'facing_3bet': False,
            'is_cbet_opp': False,
            'facing_cbet': False,
        }

        try:
            self._manager.record_opponent_action(
                username=action.username,
                action=action.abstract_action,
                bet_amount=action.amount,
                pot_size=pot,
                context=context,
            )
        except Exception as e:
            logger.error(f"record_opponent_action hiba: {e}", exc_info=True)

        self._actions_this_street.append(action)

        logger.info(
            f"  Opp: {action.username} → {action.action_type}"
            f"{f' ${action.amount:.2f}' if action.amount > 0 else ''}"
        )

        if self.on_opponent_action:
            self.on_opponent_action({
                'seat': action.seat,
                'username': action.username,
                'action': action.action_type,
                'amount': action.amount,
            })

    def _handle_my_turn(self, state: ScreenState):
        """Én következem → RTAManager.get_recommendation()"""
        self._phase = GamePhase.MY_TURN

        hole_cards = [c.to_equity_format() for c in state.my_hole_cards]
        board_cards = self._current_board
        pot = state.pot_total or 1.0
        call_amount = self._estimate_call_amount(state)

        # Legal actions: alapértelmezetten minden engedélyezett
        # Az RTAManager maszkolja a modell kimenetét
        legal_actions = [0, 1, 2, 3, 4, 5, 6]

        try:
            recommendation = self._manager.get_recommendation(
                legal_actions=legal_actions,
                hole_cards=hole_cards,
                board_cards=board_cards,
                current_pot=pot,
                call_amount=call_amount,
            )
        except Exception as e:
            logger.error(f"get_recommendation hiba: {e}", exc_info=True)
            recommendation = {
                'action_name': 'ERROR', 'confidence': 0,
                'equity': 0, 'explanation': str(e),
            }

        self._last_recommendation = recommendation

        logger.info(
            f"  ★ JAVASLAT: {recommendation.get('action_name', '?')} "
            f"({recommendation.get('confidence', 0)*100:.0f}%) | "
            f"Equity: {recommendation.get('equity', 0)*100:.0f}% | "
            f"Pot: ${pot:.2f}"
        )

        if self.on_recommendation:
            self.on_recommendation(recommendation)

    def _handle_my_action_done(self, state: ScreenState):
        """A saját akciónk megtörtént → record_my_action()"""
        if not self._last_recommendation:
            return

        action = self._last_recommendation.get('action', 1)
        pot = state.pot_total or 1.0

        # Bet összeg becslése a stack változásból
        bet_amount = 0.0
        if self._prev_state:
            my_prev = next(
                (p for p in self._prev_state.players
                 if p.seat_index == self._my_seat), None
            )
            my_curr = next(
                (p for p in state.players
                 if p.seat_index == self._my_seat), None
            )
            if (my_prev and my_curr and
                    my_prev.stack is not None and my_curr.stack is not None):
                bet_amount = max(0.0, my_prev.stack - my_curr.stack)

        try:
            self._manager.record_my_action(
                action=action,
                bet_amount=bet_amount,
                pot_size=pot,
            )
        except Exception as e:
            logger.error(f"record_my_action hiba: {e}", exc_info=True)

        self._phase = GamePhase.HAND_ACTIVE
        self._last_recommendation = None

    def _handle_hand_over(self, state: ScreenState):
        """Kéz vége."""
        self._phase = GamePhase.WAITING_FOR_HAND
        self._last_recommendation = None

        logger.info(f"  Kéz vége (#{self._hand_count})")

        if self.on_hand_end:
            self.on_hand_end({
                'hand_num': self._hand_count,
                'final_pot': state.pot_total,
            })

    # ═══════════════════════════════════════════════════════════════════════
    # Segéd metódusok
    # ═══════════════════════════════════════════════════════════════════════

    def _estimate_call_amount(self, state: ScreenState) -> float:
        """
        Call összeg becslés.
        
        Módszer:
        1. Ha van akció badge "Call $X.XX" → X.XX
        2. Ha van bet_this_round a többi játékosnál → max bet - saját bet
        3. Fallback: 0.0
        """
        my_player = next(
            (p for p in state.players if p.seat_index == self._my_seat),
            None
        )

        # 1. Akció gombok szövegéből
        # TODO: ha a YOLO detektálja a "Call $X.XX" gombot

        # 2. Bet összegek alapján
        if my_player:
            max_bet = max(
                (p.bet_this_round or 0.0 for p in state.players
                 if p.is_active), default=0.0
            )
            my_bet = my_player.bet_this_round or 0.0
            if max_bet > my_bet:
                return max_bet - my_bet

        # 3. Stack-alapú becslés
        if self._prev_state and my_player:
            prev_stacks = {
                p.seat_index: p.stack for p in self._prev_state.players
                if p.stack is not None
            }
            current_stacks = {
                p.seat_index: p.stack for p in state.players
                if p.stack is not None
            }
            # Ha valaki stack-je csökkent az előző frame óta → van bet
            for seat_idx in current_stacks:
                if seat_idx == self._my_seat:
                    continue
                prev_s = prev_stacks.get(seat_idx)
                curr_s = current_stacks.get(seat_idx)
                if prev_s and curr_s and prev_s > curr_s + 0.01:
                    return prev_s - curr_s

        return 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # Diagnosztika
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def phase(self) -> GamePhase:
        return self._phase

    @property
    def hand_count(self) -> int:
        return self._hand_count

    def status(self) -> dict:
        """Aktuális állapot summary."""
        return {
            'phase': self._phase.value,
            'hand_count': self._hand_count,
            'street': self._current_street,
            'board': self._current_board,
            'num_players': self._current_num_players,
            'table_initialized': self._table_initialized,
            'has_recommendation': self._last_recommendation is not None,
        }
