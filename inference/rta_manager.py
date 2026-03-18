"""
inference/rta_manager.py  –  RTAManager (RTA v4)

Online póker multi-asztalos realtime asszisztens manager.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEGOLDOTT PROBLÉMÁK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PROB-1: Dinamikus modellváltás
    ModelPool dict[int, AdvancedPokerAI] – lazy loading per asztalmméret.
    Minden modell a saját state_size-ával kompatibilis.

  PROB-2: Ellenfél memória megőrzése modellváltásnál
    GlobalPlayerTracker username → PlayerStats, SQLite-ban tárolva.
    Modellváltás, asztalváltás, kiülés esetén NEM resetelődik.

  PROB-3: ActionHistoryEncoder dimenziók
    Asztalméretnél az encoder újraépül a helyes num_players-szel.
    Action history törlődik (leosztások között elfogadható).

  PROB-4: Hosszú távú perzisztencia
    SQLite: ~8 MB / 50k játékos, ~80 MB / 500k játékos.
    Rolling window csak memóriában él (max 9 obj egyszerre).
    Auto-flush 50 akciónként, context manager garantált mentéssel.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AJÁNLOTT HASZNÁLAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  with RTAManager(
      model_paths = {6: '6max_ppo_v4.pth', 2: '2max_ppo_v4.pth'},
      db_path     = 'players.db',   # SQLite ellenfél adatbázis
  ) as manager:

      manager.manage_table_change(
          num_players = 6,
          seat_map    = {0: 'hero', 1: 'fish99', 2: 'reg42',
                         3: 'nit77', 4: 'aggro22', 5: 'loose_bill'},
          my_seat     = 0,
          button_seat = 5,
      )

      manager.new_hand(my_stack=150.0, bb=2.0, sb=1.0)

      result = manager.get_recommendation(
          obs_vector    = obs,
          legal_actions = [0,1,2,3,4,5,6],
          hole_cards    = ['As', 'Kh'],
          board_cards   = [],
          call_amount   = 4.0,
      )

      manager.record_opponent_action('fish99', action=4, bet_amount=10.0)
      manager.record_my_action(action=result['action'])
      manager.new_street(1)

  # kilépéskor automatikusan flush-ol

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FÁJL ELHELYEZÉSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  poker_ai_v4/
  └── inference/
      ├── __init__.py          ← RTAManager export hozzáadva
      ├── rta_manager.py       ← ez a fájl
      └── realtime_assistant.py ← változatlan
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
    ACTION_HISTORY_LEN,
)
from core.opponent_tracker import GlobalPlayerTracker
from core.equity import HandEquityEstimator

logger = logging.getLogger("PokerAI")

_UNKNOWN_PREFIX = "unknown_seat_"


# ─────────────────────────────────────────────────────────────────────────────
# ModelPool – lazy loading, egy modell per asztalmméret
# ─────────────────────────────────────────────────────────────────────────────

class ModelPool:
    """
    Betöltött modellek tárhelye. Lazy: csak első kérésre tölt be.
    """

    def __init__(self, model_paths: dict, device: torch.device):
        self._paths:  dict[int, str]             = {}
        self._models: dict[int, AdvancedPokerAI] = {}
        self._meta:   dict[int, dict]            = {}
        self._device  = device
        for num_players, path in model_paths.items():
            self._paths[int(num_players)] = path

    def get(self, num_players: int) -> tuple:
        """Visszaad: (model, state_size, action_size). Lazy betöltés."""
        num_players = int(num_players)
        if num_players not in self._models:
            self._load(num_players)
        return (
            self._models[num_players],
            self._meta[num_players]['state_size'],
            self._meta[num_players]['action_size'],
        )

    def available_sizes(self) -> list:
        return sorted(self._paths.keys())

    def preload_all(self):
        for n in self._paths:
            if n not in self._models:
                self._load(n)

    def _load(self, num_players: int):
        if num_players not in self._paths:
            raise ValueError(
                f"Nincs modell {num_players} játékoshoz. "
                f"Elérhető: {self.available_sizes()}"
            )
        path = self._paths[num_players]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modell fájl nem található: {path}")

        logger.info(f"ModelPool: betöltés {num_players}p → {path}")
        ck = torch.load(path, map_location=self._device, weights_only=False)
        if not (isinstance(ck, dict) and 'state_dict' in ck):
            raise ValueError(f"Érvénytelen checkpoint formátum: {path}")

        state_size  = ck.get('state_size',  492)
        action_size = ck.get('action_size', 7)
        model = AdvancedPokerAI(state_size=state_size,
                                action_size=action_size).to(self._device)
        model.load_state_dict(ck['state_dict'])
        model.eval()

        self._models[num_players] = model
        self._meta[num_players]   = {'state_size': state_size,
                                      'action_size': action_size}
        logger.info(f"ModelPool: {num_players}p kész | "
                    f"state_size={state_size} action_size={action_size}")

    def __repr__(self):
        return (f"ModelPool(paths={self.available_sizes()}, "
                f"loaded={list(self._models.keys())})")


# ─────────────────────────────────────────────────────────────────────────────
# SeatMapper – username ↔ lokális szék index fordítás
# ─────────────────────────────────────────────────────────────────────────────

class SeatMapper:
    """
    Az asztal aktuális szék→username mappingja.
    A lokális indexek (0..N-1) a széksorrend szerinti pozíciók,
    ez pontosan az amit a build_state_tensor() és ActionHistoryEncoder vár.
    """

    def __init__(self):
        self._seat_map:          dict[int, str] = {}
        self._local_order:       list[int]      = []
        self._username_to_local: dict[str, int] = {}
        self._my_seat       = 0
        self._my_local_idx  = 0

    def update(self, seat_map: dict, my_seat: int):
        """
        seat_map: {szék_idx: username}
        my_seat:  saját szék indexe
        """
        filled = {
            seat: (uname if uname else f"{_UNKNOWN_PREFIX}{seat}")
            for seat, uname in seat_map.items()
        }
        self._seat_map    = filled
        self._local_order = sorted(filled.keys())
        self._username_to_local = {
            filled[seat]: idx
            for idx, seat in enumerate(self._local_order)
        }
        self._my_seat      = my_seat
        self._my_local_idx = (self._local_order.index(my_seat)
                               if my_seat in self._local_order else 0)

    def local_index(self, username: str) -> int:
        return self._username_to_local.get(username, 0)

    def username(self, seat: int) -> str:
        return self._seat_map.get(seat, f"{_UNKNOWN_PREFIX}{seat}")

    def username_by_local(self, local_idx: int) -> str:
        if 0 <= local_idx < len(self._local_order):
            return self._seat_map.get(self._local_order[local_idx],
                                       f"{_UNKNOWN_PREFIX}{local_idx}")
        return f"{_UNKNOWN_PREFIX}{local_idx}"

    @property
    def my_local_idx(self) -> int:
        return self._my_local_idx

    @property
    def num_players(self) -> int:
        return len(self._seat_map)

    @property
    def ordered_usernames(self) -> list:
        return [self._seat_map[s] for s in self._local_order]

    def as_seat_map(self) -> dict:
        """Szék → username dict (GlobalPlayerTracker.preload_table()-hoz)."""
        return dict(self._seat_map)

    def build_local_stacks(self, stack_by_username: dict,
                            default_stack: float = 100.0) -> list:
        return [
            float(stack_by_username.get(self._seat_map.get(s, ''), default_stack))
            for s in self._local_order
        ]

    def __repr__(self):
        return f"SeatMapper({self._seat_map})"


# ─────────────────────────────────────────────────────────────────────────────
# _GlobalTrackerAdapter – bridge a build_state_tensor() felé
# ─────────────────────────────────────────────────────────────────────────────

class _GlobalTrackerAdapter:
    """
    Adapter: GlobalPlayerTracker + SeatMapper → get_stats_vector() interface.
    A build_state_tensor() OpponentHUDTracker.get_stats_vector()-t vár.
    Ez az adapter a SQLite-alapú GlobalPlayerTracker-t burkolja.
    Belső használatú.
    """

    def __init__(self, tracker: GlobalPlayerTracker, seat_mapper: SeatMapper):
        self._tracker     = tracker
        self._seat_mapper = seat_mapper

    def get_stats_vector(self) -> list:
        seat_map = {
            seat: self._seat_mapper.username(seat)
            for seat in range(self._seat_mapper.num_players)
        }
        return self._tracker.get_local_stats_vector(seat_map)


# ─────────────────────────────────────────────────────────────────────────────
# RTAManager – fő osztály
# ─────────────────────────────────────────────────────────────────────────────

class RTAManager:
    """
    Multi-asztalos realtime póker asszisztens manager.

    Egy RTAManager példány a teljes session alatt él.
    Context manager-ként használva session végén automatikusan flush-ol.

    API sorrendje egy tipikus kézben:
        1. manage_table_change()    – session elején és asztalméretnél
        2. new_hand()               – minden leosztás elején
        3. new_street()             – flop/turn/river érkezésekor
        4. record_opponent_action() – ellenfél lép
        5. get_recommendation()     – mielőtt te lépsz
        6. record_my_action()       – miután te léptél
    """

    def __init__(self,
                 model_paths:     dict,
                 db_path:         str  = None,
                 device:          str  = 'cpu',
                 equity_sims:     int  = 500,
                 tracker_memory:  int  = 1000):
        """
        model_paths:    dict[int, str]  – {num_players: '.pth fájl'}
                        Példa: {2: '2max_ppo_v4.pth', 6: '6max_ppo_v4.pth'}

        db_path:        SQLite adatbázis az ellenfél statisztikákhoz.
                        Automatikusan létrejön ha nem létezik.
                        None → csak memóriában él (teszteléshez).

        device:         'cpu' vagy 'cuda'
        equity_sims:    Monte Carlo szimulációk száma equity becsléshez
        tracker_memory: Rolling window mérete eseményekben (default: 1000)
        """
        self._device        = torch.device(device)
        self._pool          = ModelPool(model_paths, self._device)
        self._db_path       = db_path
        self._equity_est    = HandEquityEstimator(n_sim=equity_sims)
        self._action_mapper = PokerActionMapper()

        # GlobalPlayerTracker: SQLite-alapú, hetekig megőrzi az ellenfél statokat
        self._global_tracker = GlobalPlayerTracker(
            db_path=db_path,
            memory=tracker_memory,
        )

        # Aktuális asztal állapot
        self._seat_mapper     = SeatMapper()
        self._action_history  = collections.deque(maxlen=ACTION_HISTORY_LEN)
        self._history_encoder: ActionHistoryEncoder = None
        self._active_model:    AdvancedPokerAI      = None
        self._active_state_size:  int = 0
        self._active_action_size: int = 7

        # Játék kontextus
        self._bb           = 2.0
        self._sb           = 1.0
        self._my_stack     = 100.0
        self._all_stacks   = []
        self._button_seat  = 0
        self._street       = 0
        self._num_players  = 0
        self._hand_started = False

        logger.info(
            f"RTAManager inicializálva | "
            f"modellek: {list(model_paths.keys())} | "
            f"device: {device} | "
            f"db: {db_path or '(memória csak)'}"
        )
        if db_path:
            stats = self._global_tracker.db_stats()
            logger.info(
                f"Ellenfél DB: {stats['total_players']} játékos | "
                f"{stats['db_size_mb']} MB"
            )

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Session végén flush-ol a DB-be. Kivételeket nem nyel el."""
        self.flush_tracker()
        return False

    def flush_tracker(self):
        """
        Dirty lifetime értékek kiírása SQLite-ba.
        Context manager automatikusan hívja kilépéskor.
        Manuálisan is hívható pl. minden 100. kéz után.
        """
        self._global_tracker.flush()
        if self._db_path:
            logger.debug("RTAManager: tracker flush kész")

    # ── Asztal kezelés ────────────────────────────────────────────────────────

    def manage_table_change(self,
                             num_players: int,
                             seat_map:    dict,
                             my_seat:     int,
                             button_seat: int = 0):
        """
        Asztalméret változás vagy új asztal kezelése.

        Hívandó:
          - Session elején (első asztalnál)
          - Ha az asztalon ülők száma megváltozik
          - Ha új asztalhoz csatlakozunk

        A GlobalPlayerTracker NEM resetelődik – ellenfél statok megmaradnak.

        Paraméterek:
            num_players: int   – aktív játékosok száma (2-9)
            seat_map:    dict  – {szék_idx: username}
                                  Hiányzó/None username → 'unknown_seat_N'
            my_seat:     int   – saját szék indexe
            button_seat: int   – button szék indexe
        """
        num_players = int(num_players)
        logger.info(
            f"manage_table_change: {num_players}p | "
            f"my_seat={my_seat} | seats={seat_map}"
        )

        # 1. Modell lazy betöltése
        model, state_size, action_size = self._pool.get(num_players)
        self._active_model       = model
        self._active_state_size  = state_size
        self._active_action_size = action_size

        # 2. SeatMapper frissítése
        self._seat_mapper.update(seat_map, my_seat)
        self._num_players = num_players
        self._button_seat = button_seat

        # 3. ActionHistoryEncoder újraalkotása a helyes num_players-szel
        self._history_encoder = ActionHistoryEncoder(num_players, action_size)

        # 4. Action history törlése (inkompatibilis lenne a régi mérettel)
        self._action_history.clear()

        # 5. SQLite batch preload az asztalon ülő játékosokhoz
        #    Egyetlen DB query az összes játékosra egyszerre
        self._global_tracker.preload_table(self._seat_mapper.as_seat_map())

        logger.info(
            f"Aktív modell: {num_players}p | state_size={state_size} | "
            f"DB: {self._global_tracker.db_stats().get('total_players','?')} játékos"
        )

    def reset_session(self):
        """
        Teljes memória reset. DB NEM törlődik.
        Normál asztalváltáshoz NEM szükséges – azt manage_table_change kezeli.
        """
        self._global_tracker.reset()
        self._action_history.clear()
        self._hand_started = False
        logger.info("RTAManager: session reset (memória törölve, DB érintetlen)")

    # ── Kéz kezelés ──────────────────────────────────────────────────────────

    def new_hand(self,
                  my_stack:   float,
                  all_stacks: dict  = None,
                  bb:         float = 2.0,
                  sb:         float = 1.0):
        """
        Új leosztás kezdetekor hívandó.

        my_stack:   saját stack
        all_stacks: dict[username, float] – összes játékos stackje
                    Ha None: egyenlő stackek feltételezve.
        bb, sb:     nagy és kis vak értéke
        """
        self._bb       = float(bb)
        self._sb       = float(sb)
        self._my_stack = float(my_stack)
        self._street   = 0

        if all_stacks is not None:
            self._all_stacks = self._seat_mapper.build_local_stacks(
                all_stacks, default_stack=my_stack
            )
        else:
            self._all_stacks = [float(my_stack)] * self._num_players

        self._action_history.clear()
        self._hand_started = True

        logger.debug(
            f"new_hand | seat={self._seat_mapper.my_local_idx} | "
            f"stack={my_stack:.1f} | BB={bb} SB={sb}"
        )

    def new_street(self, street: int):
        """
        Street váltáskor hívandó.
        street: 0=preflop, 1=flop, 2=turn, 3=river
        """
        self._street = int(street)
        logger.debug(
            f"new_street: {['preflop','flop','turn','river'][min(street,3)]}"
        )

    # ── Akció rögzítés ────────────────────────────────────────────────────────

    def record_opponent_action(self,
                                username:   str,
                                action:     int,
                                bet_amount: float = 0.0,
                                pot_size:   float = 1.0,
                                context:    dict  = None):
        """
        Ellenfél akciójának rögzítése.

        username:   játékos azonosítója ('fish99')
        action:     absztrakt akció 0-6
        bet_amount: bet/raise összege (0 ha fold/call/check)
        pot_size:   aktuális pot mérete (bet normáláshoz)
        context:    {'facing_3bet': bool, 'is_cbet_opp': bool, 'facing_cbet': bool}
        """
        local_idx = self._seat_mapper.local_index(username)
        bet_norm  = min(bet_amount / max(pot_size, 1e-6), 5.0) / 5.0

        self._action_history.append((local_idx, action, bet_norm))
        self._global_tracker.record_action(
            username, action,
            street  = self._street,
            context = context,
        )

    def record_my_action(self,
                          action:     int,
                          bet_amount: float = 0.0,
                          pot_size:   float = 1.0):
        """Saját akció rögzítése az action history-ba."""
        my_idx   = self._seat_mapper.my_local_idx
        bet_norm = min(bet_amount / max(pot_size, 1e-6), 5.0) / 5.0
        self._action_history.append((my_idx, action, bet_norm))

    # ── Fő ajánlás API ────────────────────────────────────────────────────────

    def get_recommendation(self,
                            obs_vector:    object,
                            legal_actions: list,
                            hole_cards:    list  = None,
                            board_cards:   list  = None,
                            current_pot:   float = None,
                            call_amount:   float = 0.0) -> dict:
        """
        Akcióajánlás az aktuális játékállapothoz.

        Paraméterek:
            obs_vector:    rlcard obs array (numpy array vagy list)
            legal_actions: engedélyezett absztrakt akciók listája [0-6]
            hole_cards:    ['As','Kh'] saját lapok (equity számításhoz)
            board_cards:   ['Td','7c','2s'] közösségi lapok
            current_pot:   aktuális pot mérete
            call_amount:   call összege (pot odds számításhoz)

        Visszatér: dict
            'action'        – int: ajánlott absztrakt akció
            'action_name'   – str: pl. 'Raise 50%'
            'confidence'    – float: ajánlott akció valószínűsége
            'top3'          – [(action_name, prob), ...]: top 3 akció
            'equity'        – float: becsült kéz equity
            'spr'           – float: stack/pot ratio
            'm_ratio'       – float: stack/blindok
            'street_name'   – str: 'Preflop'/'Flop'/'Turn'/'River'
            'value_est'     – float: critic értékbecslés
            'explanation'   – str: rövid szöveges magyarázat
            'db_players'    – int: ismert játékosok száma az adatbázisban
        """
        if self._active_model is None:
            raise RuntimeError(
                "Nincs aktív modell. Hívd meg a manage_table_change()-t előbb."
            )

        board_cards = board_cards or []
        hole_cards  = hole_cards  or []

        # ── State dict összerakása ────────────────────────────────────────────
        state = self._build_obs_dict(obs_vector, board_cards,
                                      call_amount, current_pot)

        # ── Equity becslés ───────────────────────────────────────────────────
        equity = 0.5
        if len(hole_cards) == 2:
            try:
                equity = self._equity_est.equity(
                    hole_cards, board_cards,
                    num_opponents=max(self._num_players - 1, 1)
                )
            except Exception:
                equity = 0.5

        # ── State tensor összerakása ──────────────────────────────────────────
        tracker_adapter = _GlobalTrackerAdapter(
            self._global_tracker, self._seat_mapper
        )
        state_t = build_state_tensor(
            state,
            tracker_adapter,
            self._action_history,
            self._history_encoder,
            self._num_players,
            my_player_id  = self._seat_mapper.my_local_idx,
            bb            = self._bb,
            sb            = self._sb,
            initial_stack = self._my_stack,
            street        = self._street,
            equity        = equity,
        )

        # ── Model forward ────────────────────────────────────────────────────
        with torch.no_grad():
            action_probs, value, _ = self._active_model.forward(
                state_t.to(self._device), legal_actions
            )

        probs_np    = action_probs.squeeze(0).cpu().numpy()
        best_action = int(np.argmax(probs_np))
        confidence  = float(probs_np[best_action])

        sorted_idx = np.argsort(probs_np)[::-1]
        top3 = [
            (self._action_mapper.action_name(int(i)), float(probs_np[i]))
            for i in sorted_idx[:3]
            if int(i) in legal_actions
        ]

        # ── Kontextuális értékek ─────────────────────────────────────────────
        pot         = current_pot or 1.0
        spr         = self._my_stack / max(pot, 1e-6)
        m_ratio     = self._my_stack / max(self._bb + self._sb, 1e-6)
        street_name = ['Preflop','Flop','Turn','River'][min(self._street, 3)]
        explanation = self._explain(best_action, equity, spr, m_ratio,
                                     confidence, street_name, call_amount)

        db_stats = self._global_tracker.db_stats()

        return {
            'action':      best_action,
            'action_name': self._action_mapper.action_name(best_action),
            'confidence':  confidence,
            'top3':        top3,
            'equity':      equity,
            'spr':         round(spr, 2),
            'm_ratio':     round(m_ratio, 1),
            'street_name': street_name,
            'value_est':   float(value.squeeze().cpu()),
            'explanation': explanation,
            'db_players':  db_stats.get('total_players', len(self._global_tracker)),
        }

    # ── Segédmetódusok ────────────────────────────────────────────────────────

    def _build_obs_dict(self, obs_vector, board_cards: list,
                         call_amount: float, current_pot) -> dict:
        obs_arr = np.array(obs_vector, dtype=np.float32)
        stacks  = self._all_stacks or [self._my_stack] * self._num_players
        button_local = (
            self._seat_mapper.local_index(
                self._seat_mapper.username(self._button_seat)
            )
            if self._button_seat in range(self._num_players) else 0
        )
        return {
            'obs': obs_arr,
            'raw_obs': {
                'my_chips':     self._my_stack,
                'all_chips':    stacks,
                'pot':          current_pot or 0.0,
                'public_cards': board_cards,
                'hand':         [],
                'button':       button_local,
                'call_amount':  call_amount,
            },
            'legal_actions': [],
        }

    def _explain(self, action: int, equity: float, spr: float,
                  m_ratio: float, confidence: float,
                  street_name: str, call_amount: float) -> str:
        lines = [f"{street_name} | "
                 f"{self._action_mapper.action_name(action)} "
                 f"({confidence*100:.0f}%)"]
        if equity > 0.5:
            lines.append(f"Equity: {equity*100:.0f}% (kedvező)")
        elif equity < 0.35:
            lines.append(f"Equity: {equity*100:.0f}% (gyenge kéz)")
        if spr < 3:
            lines.append(f"SPR={spr:.1f} → push-or-fold")
        elif spr > 15:
            lines.append(f"SPR={spr:.1f} → deep stack")
        if m_ratio < 10:
            lines.append(f"M={m_ratio:.0f} → veszélyzóna")
        if action == 0:
            lines.append("Fold: pot odds nem éri meg")
        elif action == 1 and call_amount > 0:
            po = call_amount / max(call_amount + 1.0, 1.0)
            lines.append(f"Call: pot odds ~{po*100:.0f}%")
        return " | ".join(lines)

    # ── Diagnosztika ──────────────────────────────────────────────────────────

    def player_stats(self, username: str) -> dict:
        """Egy ellenfél összesített stat summaryja."""
        return self._global_tracker.player_summary(username)

    def all_player_stats(self) -> list:
        """Session közben érintett összes játékos summaryja."""
        return self._global_tracker.all_summaries()

    def top_players(self, n: int = 20) -> list:
        """Top N játékos az adatbázisból lifetime kézszám szerint."""
        return self._global_tracker.top_players_by_hands(n)

    def db_info(self) -> dict:
        """Adatbázis állapot és méret."""
        return self._global_tracker.db_stats()

    def current_table_info(self) -> dict:
        """Aktuális asztal állapota."""
        return {
            'num_players':  self._num_players,
            'my_local_idx': self._seat_mapper.my_local_idx,
            'seat_order':   self._seat_mapper.ordered_usernames,
            'street':       ['preflop','flop','turn','river'][min(self._street,3)],
            'bb':           self._bb,
            'sb':           self._sb,
            'my_stack':     self._my_stack,
            'model_loaded': self._active_model is not None,
            'state_size':   self._active_state_size,
            'db':           self._global_tracker.db_stats(),
        }

    def preload_all_models(self):
        """Összes modell előzetes betöltése startup-ban."""
        self._pool.preload_all()

    def __repr__(self):
        return (f"RTAManager("
                f"models={self._pool.available_sizes()}, "
                f"table={self._num_players}p, "
                f"db={self._db_path or 'memory'})")
