"""
inference/rta_manager.py  –  RTAManager (RTA v2)

Online póker multi-asztalos realtime asszisztens manager.

Megoldott problémák (NOTES_MASTER.txt §14 és Gemini review alapján):

  PROB-1: Dinamikus modellváltás
    A ModelPool dict[int, AdvancedPokerAI] tárolja az összes betöltött
    modellt. manage_table_change() vált köztük az asztalméret alapján.
    Minden modell a saját state_size-ával kompatibilis ActionHistoryEncoder-t
    és FeatureBuilder-t kap.

  PROB-2: Ellenfél memória megőrzése
    GlobalPlayerTracker (core/opponent_tracker.py) username → PlayerStats
    dict-ben tárolja az adatokat. Modellváltáskor NEM resetelődik.
    Leosztás között szintén megmarad.

  PROB-3: ActionHistoryEncoder dimenziók
    Asztalméretnél az action history törlődik (leosztások közt elfogadható),
    az encoder újra létrejön a helyes num_players-szel.
    A SeatMapper gondoskodik a username → lokális index fordításról.

Használat:
    manager = RTAManager({
        2: '2max_ppo_v4.pth',
        6: '6max_ppo_v4.pth',
        9: '9max_ppo_v4.pth',
    })

    # Asztal kezdete / asztalméret változás
    manager.manage_table_change(
        num_players = 6,
        seat_map    = {0: 'hero', 1: 'fish99', 2: 'reg42',
                       3: 'nit77', 4: 'aggro22', 5: 'unknown_seat_5'},
        my_seat     = 0,
        button_seat = 5,
    )

    # Kéz kezdete
    manager.new_hand(
        my_stack   = 150.0,
        all_stacks = {'fish99': 200.0, 'reg42': 80.0, ...},
        bb=2.0, sb=1.0,
    )

    # Döntés kérése
    result = manager.get_recommendation(
        obs_vector    = obs,
        legal_actions = [0,1,2,3,4,5,6],
        hole_cards    = ['As','Kh'],
        board_cards   = [],
        call_amount   = 4.0,
    )

    # Ellenfél akció rögzítése
    manager.record_opponent_action(
        username      = 'fish99',
        action        = 4,
        bet_amount    = 10.0,
        pot_size      = 20.0,
    )
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
    compute_state_size,
)
from core.opponent_tracker import GlobalPlayerTracker
from core.equity import HandEquityEstimator

logger = logging.getLogger("PokerAI")

# Fallback username ismeretlen székekhez
_UNKNOWN_PREFIX = "unknown_seat_"


# ─────────────────────────────────────────────────────────────────────────────
# ModelPool – lazy loading, egy modell per asztalmméret
# ─────────────────────────────────────────────────────────────────────────────

class ModelPool:
    """
    Betöltött modellek tárhelye.
    Lazy: csak akkor tölt be egy modellt, ha először kérik.
    """

    def __init__(self, model_paths: dict, device: torch.device):
        """
        model_paths: dict[int, str]  – {num_players: '.pth fájl elérési útja'}
        """
        self._paths:  dict[int, str]             = {}
        self._models: dict[int, AdvancedPokerAI] = {}
        self._meta:   dict[int, dict]            = {}
        self._device  = device

        for num_players, path in model_paths.items():
            self._paths[int(num_players)] = path

    def get(self, num_players: int) -> tuple:
        """
        Visszaad: (model, state_size, action_size)
        Lazy betöltés: első híváskor tölti be a checkpoint-ot.
        """
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
            raise ValueError(
                f"Érvénytelen checkpoint formátum: {path}\n"
                "Elvárt: dict {'state_dict': ..., 'state_size': ...}"
            )

        state_size  = ck.get('state_size',  492)
        action_size = ck.get('action_size', 7)

        model = AdvancedPokerAI(
            state_size  = state_size,
            action_size = action_size,
        ).to(self._device)
        model.load_state_dict(ck['state_dict'])
        model.eval()

        self._models[num_players] = model
        self._meta[num_players]   = {
            'state_size':  state_size,
            'action_size': action_size,
        }
        logger.info(
            f"ModelPool: {num_players}p kész | "
            f"state_size={state_size} action_size={action_size}"
        )

    def preload_all(self):
        """Összes modell előzetes betöltése (opcionális, startup-ban hívható)."""
        for num_players in self._paths:
            if num_players not in self._models:
                self._load(num_players)

    def __repr__(self):
        loaded = list(self._models.keys())
        return (f"ModelPool("
                f"paths={self.available_sizes()}, "
                f"loaded={loaded})")


# ─────────────────────────────────────────────────────────────────────────────
# SeatMapper – username ↔ lokális szék index fordítás
# ─────────────────────────────────────────────────────────────────────────────

class SeatMapper:
    """
    Az asztal aktuális szék→username mappingja.

    seat_map: dict[int, str]  pl. {0:'hero', 1:'fish99', 2:'reg42', ...}

    A lokális indexek a széksorrend szerinti pozíciók (0..N-1),
    ahol N az asztal jelenlegi mérete. Ez pontosan az amit a
    build_state_tensor() és az ActionHistoryEncoder vár.
    """

    def __init__(self):
        self._seat_map:     dict[int, str] = {}   # szék → username
        self._local_order:  list[int]      = []   # rendezett székek
        self._username_to_local: dict[str, int] = {}

    def update(self, seat_map: dict, my_seat: int):
        """
        seat_map: {szék_idx: username}  – az aktuális asztal térképe
        my_seat:  saját szék indexe
        """
        # Hiányzó username-ek fallback-je
        filled = {}
        for seat, uname in seat_map.items():
            filled[seat] = uname if uname else f"{_UNKNOWN_PREFIX}{seat}"
        self._seat_map   = filled
        self._local_order = sorted(filled.keys())

        # username → lokális index (0-tól N-1-ig, szék sorrendben)
        self._username_to_local = {
            uname: idx
            for idx, seat in enumerate(self._local_order)
            for uname in [filled[seat]]
        }

        self._my_seat       = my_seat
        self._my_local_idx  = self._local_order.index(my_seat) \
                              if my_seat in self._local_order else 0

    def local_index(self, username: str) -> int:
        """Username → lokális index (0..N-1)."""
        return self._username_to_local.get(username, 0)

    def username(self, seat: int) -> str:
        """Szék → username."""
        return self._seat_map.get(seat, f"{_UNKNOWN_PREFIX}{seat}")

    def username_by_local(self, local_idx: int) -> str:
        """Lokális index → username."""
        if 0 <= local_idx < len(self._local_order):
            seat = self._local_order[local_idx]
            return self._seat_map.get(seat, f"{_UNKNOWN_PREFIX}{seat}")
        return f"{_UNKNOWN_PREFIX}{local_idx}"

    @property
    def my_local_idx(self) -> int:
        return self._my_local_idx

    @property
    def num_players(self) -> int:
        return len(self._seat_map)

    @property
    def ordered_usernames(self) -> list:
        """Szék sorrend szerinti username lista."""
        return [self._seat_map[s] for s in self._local_order]

    def build_local_stacks(self, stack_by_username: dict,
                            default_stack: float = 100.0) -> list:
        """
        Username→stack dict → lokális sorrend szerinti stack lista.
        build_state_tensor()-hoz szükséges formátum.
        """
        return [
            float(stack_by_username.get(self._seat_map.get(s, ''), default_stack))
            for s in self._local_order
        ]

    def __repr__(self):
        return f"SeatMapper({self._seat_map})"


# ─────────────────────────────────────────────────────────────────────────────
# RTAManager – fő osztály
# ─────────────────────────────────────────────────────────────────────────────

class RTAManager:
    """
    Multi-asztalos realtime póker asszisztens manager.

    Egy RTAManager példány a teljes session alatt él.
    Belül tartja a GlobalPlayerTracker-t (ellenfél memória),
    a ModelPool-t (num_players → modell), és az aktuális asztal
    kontextusát (SeatMapper, ActionHistoryEncoder, stb.).

    Gyors összefoglaló:
        1. manage_table_change()  – asztal méret / összetétel változáskor
        2. new_hand()             – minden leosztás elején
        3. new_street()           – utca váltáskor (flop/turn/river)
        4. get_recommendation()   – saját lépés előtt
        5. record_opponent_action() / record_my_action()  – lépések után
    """

    def __init__(self, model_paths: dict,
                 device: str = 'cpu',
                 equity_sims: int = 500,
                 tracker_memory: int = 1000):
        """
        model_paths: dict[int, str]  – {num_players: '.pth elérési út'}
          Példa: {2: '2max_ppo_v4.pth', 6: '6max_ppo_v4.pth'}

        device:       'cpu' vagy 'cuda'
        equity_sims:  MC szimulációk száma equity becsléshez
        tracker_memory: OpponentHUDTracker rolling window mérete
        """
        self._device       = torch.device(device)
        self._pool         = ModelPool(model_paths, self._device)
        self._global_tracker = GlobalPlayerTracker(memory=tracker_memory)
        self._equity_est   = HandEquityEstimator(n_sim=equity_sims)
        self._action_mapper = PokerActionMapper()

        # Aktuális asztal állapot
        self._seat_mapper   = SeatMapper()
        self._action_history: collections.deque = collections.deque(
            maxlen=ACTION_HISTORY_LEN
        )
        self._history_encoder: ActionHistoryEncoder = None  # manage_table_change-ben init
        self._active_model:    AdvancedPokerAI = None
        self._active_state_size: int = 0
        self._active_action_size: int = 7

        # Játék kontextus
        self._bb            = 2.0
        self._sb            = 1.0
        self._my_stack      = 100.0
        self._all_stacks    = []
        self._button_seat   = 0
        self._street        = 0
        self._num_players   = 0
        self._hand_started  = False

        logger.info(
            f"RTAManager inicializálva | "
            f"modellek: {model_paths} | device: {device}"
        )

    # ── Asztal és session kezelés ─────────────────────────────────────────────

    def manage_table_change(self, num_players: int,
                             seat_map: dict,
                             my_seat: int,
                             button_seat: int = 0):
        """
        Asztalméret változás vagy új asztal kezelése.

        Kötelezően hívandó:
          - Session elején
          - Ha az asztalon lévő játékosok száma megváltozik
          - Ha új asztalhoz csatlakozunk

        FONTOS: a GlobalPlayerTracker NEM resetelődik – az ellenfél
        statisztikák megmaradnak. Csak az action history törlődik.

        Paraméterek:
            num_players: int        – aktív játékosok száma (2-9)
            seat_map:    dict       – {szék_idx: username}
              Hiányzó vagy None username → "unknown_seat_N" fallback
            my_seat:     int        – saját szék indexe
            button_seat: int        – button pozíció (szék index)
        """
        num_players = int(num_players)
        logger.info(
            f"manage_table_change: {num_players}p | "
            f"my_seat={my_seat} | seats={seat_map}"
        )

        # 1. Modell betöltése (lazy)
        model, state_size, action_size = self._pool.get(num_players)
        self._active_model      = model
        self._active_state_size = state_size
        self._active_action_size = action_size

        # 2. SeatMapper frissítése
        self._seat_mapper.update(seat_map, my_seat)
        self._num_players = num_players
        self._button_seat = button_seat

        # 3. ActionHistoryEncoder újraalkotása a helyes num_players-szel
        self._history_encoder = ActionHistoryEncoder(
            num_players, action_size
        )

        # 4. Action history törlése (asztalméretnél inkompatibilis lenne)
        self._action_history.clear()

        # 5. GlobalPlayerTracker NEM resetelődik
        logger.info(
            f"Aktív modell: {num_players}p | state_size={state_size} | "
            f"ismert ellenfelek: {len(self._global_tracker)}"
        )

    def reset_session(self):
        """
        Teljes reset (új ülős – tracker is törlődik).
        Normál asztalváltáshoz NEM szükséges.
        """
        self._global_tracker.reset()
        self._action_history.clear()
        self._hand_started = False
        logger.info("RTAManager session reset (tracker törölve).")

    # ── Kéz kezelés ──────────────────────────────────────────────────────────

    def new_hand(self, my_stack: float,
                  all_stacks: dict = None,
                  bb: float = 2.0, sb: float = 1.0):
        """
        Új kéz kezdetekor hívandó.

        Paraméterek:
            my_stack:   saját stack
            all_stacks: dict[username, float] – összes játékos stackje
                        (ha None: egyenlő stackek feltételezve)
            bb, sb:     vak értékek
        """
        self._bb         = float(bb)
        self._sb         = float(sb)
        self._my_stack   = float(my_stack)
        self._street     = 0

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
        logger.debug(f"new_street: {['preflop','flop','turn','river'][min(street,3)]}")

    # ── Akció rögzítés ────────────────────────────────────────────────────────

    def record_opponent_action(self, username: str,
                                action: int,
                                bet_amount: float = 0.0,
                                pot_size: float = 1.0,
                                context: dict = None):
        """
        Ellenfél akciójának rögzítése.

        username:   az ellenfél azonosítója (pl. 'fish99')
        action:     absztrakt akció (0-6)
        bet_amount: bet összege (0 ha fold/call/check)
        pot_size:   aktuális pot (bet_norm számításhoz)
        context:    {'facing_3bet': bool, 'is_cbet_opp': bool, ...}
        """
        local_idx = self._seat_mapper.local_index(username)
        bet_norm  = min(bet_amount / max(pot_size, 1e-6), 5.0) / 5.0

        self._action_history.append((local_idx, action, bet_norm))
        self._global_tracker.record_action(
            username, action,
            street  = self._street,
            context = context,
        )

    def record_my_action(self, action: int,
                          bet_amount: float = 0.0,
                          pot_size: float = 1.0):
        """Saját akció rögzítése (history-ba)."""
        my_idx   = self._seat_mapper.my_local_idx
        bet_norm = min(bet_amount / max(pot_size, 1e-6), 5.0) / 5.0
        self._action_history.append((my_idx, action, bet_norm))

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
            obs_vector:    rlcard-kompatibilis obs (numpy array vagy list)
            legal_actions: absztrakt akció indexek listája
            hole_cards:    ['As','Kh'] saját lapok (equity számításhoz)
            board_cards:   ['Td','7c','2s'] közösségi lapok
            current_pot:   pot méret (ha None: obs-ból becsli)
            call_amount:   call összege (pot odds számításhoz)

        Visszatér: dict (RealtimePokerAssistant-kompatibilis formátum)
            'action'       – int
            'action_name'  – str
            'confidence'   – float
            'top3'         – [(action_name, prob), ...]
            'equity'       – float
            'spr'          – float
            'm_ratio'      – float
            'street_name'  – str
            'value_est'    – float
            'explanation'  – str
            'known_players'– int (GlobalPlayerTracker mérete)
        """
        if self._active_model is None:
            raise RuntimeError(
                "Nincs aktív modell. Hívd meg a manage_table_change()-t "
                "mielőtt get_recommendation()-t hívnál."
            )

        board_cards = board_cards or []
        hole_cards  = hole_cards  or []

        # ── State dict összerakása ────────────────────────────────────────────
        state = self._build_obs_dict(obs_vector, board_cards,
                                      call_amount, current_pot)

        # ── Equity becslés ───────────────────────────────────────────────────
        equity = 0.5
        if hole_cards and len(hole_cards) == 2:
            try:
                equity = self._equity_est.equity(
                    hole_cards, board_cards,
                    num_opponents=max(self._num_players - 1, 1)
                )
            except Exception:
                equity = 0.5

        # ── HUD stats (GlobalPlayerTracker → lokális sorrend) ────────────────
        # Felülírjuk a state obs-ának stats részét a globális trackerből
        # A build_state_tensor egy OpponentHUDTracker.get_stats_vector()-t
        # vár, ezért egy adapter objektumot adunk át.
        tracker_adapter = _GlobalTrackerAdapter(
            self._global_tracker,
            self._seat_mapper
        )

        # ── State tensor ─────────────────────────────────────────────────────
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

        probs_np = action_probs.squeeze(0).cpu().numpy()

        # ── Ajánlott akció ───────────────────────────────────────────────────
        best_action = int(np.argmax(probs_np))
        confidence  = float(probs_np[best_action])

        sorted_idx = np.argsort(probs_np)[::-1]
        top3 = [
            (self._action_mapper.action_name(int(idx)), float(probs_np[idx]))
            for idx in sorted_idx[:3]
            if int(idx) in legal_actions
        ]

        # ── Kontextuális értékek ─────────────────────────────────────────────
        pot    = current_pot or 1.0
        spr    = self._my_stack / max(pot, 1e-6)
        m_     = self._my_stack / max(self._bb + self._sb, 1e-6)

        street_names = ['Preflop', 'Flop', 'Turn', 'River']
        street_name  = street_names[min(self._street, 3)]

        explanation = self._explain(
            best_action, equity, spr, m_,
            confidence, street_name, call_amount
        )

        return {
            'action':        best_action,
            'action_name':   self._action_mapper.action_name(best_action),
            'confidence':    confidence,
            'top3':          top3,
            'equity':        equity,
            'spr':           round(spr, 2),
            'm_ratio':       round(m_, 1),
            'street_name':   street_name,
            'value_est':     float(value.squeeze().cpu()),
            'explanation':   explanation,
            'known_players': len(self._global_tracker),
        }

    # ── Segédmetódusok ────────────────────────────────────────────────────────

    def _build_obs_dict(self, obs_vector, board_cards: list,
                         call_amount: float,
                         current_pot) -> dict:
        obs_arr = np.array(obs_vector, dtype=np.float32)

        # Lokális sorrend szerinti stack lista
        stacks = self._all_stacks if self._all_stacks else \
                 [self._my_stack] * self._num_players

        return {
            'obs': obs_arr,
            'raw_obs': {
                'my_chips':     self._my_stack,
                'all_chips':    stacks,
                'pot':          current_pot or 0.0,
                'public_cards': board_cards,
                'hand':         [],
                'button':       self._seat_mapper.local_index(
                                    self._seat_mapper.username(self._button_seat)
                                ) if self._button_seat in range(self._num_players) else 0,
                'call_amount':  call_amount,
            },
            'legal_actions': [],
        }

    def _explain(self, action: int, equity: float, spr: float,
                  m_ratio: float, confidence: float,
                  street_name: str, call_amount: float) -> str:
        action_str = self._action_mapper.action_name(action)
        lines = [f"{street_name} | {action_str} ({confidence*100:.0f}%)"]

        if equity > 0.5:
            lines.append(f"Equity: {equity*100:.0f}% (kedvező)")
        elif equity < 0.35:
            lines.append(f"Equity: {equity*100:.0f}% (gyenge kéz)")

        if spr < 3:
            lines.append(f"SPR={spr:.1f} → push-or-fold zóna")
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
        """Egy ellenfél stat summaryja."""
        return self._global_tracker.player_summary(username)

    def all_player_stats(self) -> list:
        """Összes ismert játékos summaryja."""
        return self._global_tracker.all_summaries()

    def current_table_info(self) -> dict:
        """Aktuális asztal állapota diagnosztikához."""
        return {
            'num_players':   self._num_players,
            'my_local_idx':  self._seat_mapper.my_local_idx,
            'seat_order':    self._seat_mapper.ordered_usernames,
            'street':        ['preflop','flop','turn','river'][min(self._street, 3)],
            'bb':            self._bb,
            'sb':            self._sb,
            'my_stack':      self._my_stack,
            'known_players': len(self._global_tracker),
            'model_loaded':  self._active_model is not None,
            'state_size':    self._active_state_size,
        }

    def preload_all_models(self):
        """Összes modell előzetes betöltése."""
        self._pool.preload_all()

    def __repr__(self):
        return (f"RTAManager("
                f"models={self._pool.available_sizes()}, "
                f"table={self._num_players}p, "
                f"known_players={len(self._global_tracker)})")


# ─────────────────────────────────────────────────────────────────────────────
# _GlobalTrackerAdapter – bridge a build_state_tensor() felé
# ─────────────────────────────────────────────────────────────────────────────

class _GlobalTrackerAdapter:
    """
    Adapter: GlobalPlayerTracker + SeatMapper → OpponentHUDTracker interface.

    A build_state_tensor() egy .get_stats_vector() metódust vár,
    ami flat list[float] formátumban adja vissza a HUD statokat.
    Ez az adapter a GlobalPlayerTracker.get_local_stats_vector()-ját
    hívja a SeatMapper aktuális seat_map-jével.

    Belső használatú, külső kódnak nem kell látnia.
    """

    def __init__(self, global_tracker: GlobalPlayerTracker,
                 seat_mapper: SeatMapper):
        self._tracker     = global_tracker
        self._seat_mapper = seat_mapper

    def get_stats_vector(self) -> list:
        # seat_map: szék_idx → username, szék sorrend szerint
        seat_map = {
            seat: self._seat_mapper.username(seat)
            for seat in range(self._seat_mapper.num_players)
        }
        return self._tracker.get_local_stats_vector(seat_map)
