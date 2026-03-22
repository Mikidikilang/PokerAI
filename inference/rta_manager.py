"""
inference/rta_manager.py  –  RTAManager (RTA v4.2.2)

Online póker multi-asztalos realtime asszisztens manager.

Változások v4.2.1 → v4.2.2:
  [BUGFIX-KRITIKUS-3] ModelPool._load(): a lokális `num_players` paramétert
    felülírta a ck.get('num_players', None) hívás. Ha a checkpoint nem
    tartalmazta ezt a mezőt (v4.1 előtti checkpointok), a modell
    self._models[None] kulcs alá került, és minden get(6) hívás
    KeyError/ValueError-t dobott. Javítás: különálló változó az
    rlcard-kompatibilis modell konstruáláshoz, az eredeti paraméter
    marad a dict kulcs.

  [QUALITY] Teljes type hint lefedettség az összes publikus metóduson.
  [QUALITY] Robusztus hibakezelés a ModelPool._load()-ban: részletesebb
    hibaüzenetek, checkpoint mező validáció.
  [QUALITY] Javított docstringek (Args/Returns/Raises szekciók).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEGOLDOTT PROBLÉMÁK (eredeti v4.2.1 listából megőrizve)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PROB-1: Dinamikus modellváltás
  PROB-2: Ellenfél memória megőrzése modellváltásnál
  PROB-3: ActionHistoryEncoder dimenziók
  PROB-4: Hosszú távú perzisztencia

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AJÁNLOTT HASZNÁLAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  with RTAManager(
      model_paths = {6: '6max_ppo_v4.pth', 2: '2max_ppo_v4.pth'},
      db_path     = 'players.db',
  ) as manager:

      manager.manage_table_change(
          num_players = 6,
          seat_map    = {0: 'hero', 1: 'fish99', ...},
          my_seat     = 0,
          button_seat = 5,
      )
      manager.new_hand(my_stack=150.0, bb=2.0, sb=1.0)
      result = manager.get_recommendation(
          legal_actions = [0,1,2,3,4,5,6],
          hole_cards    = ['As', 'Kh'],
      )
"""

from __future__ import annotations

import collections
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

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
from inference.obs_builder import ObsBuilder
from utils.checkpoint_utils import safe_load_checkpoint

logger = logging.getLogger("PokerAI")

_UNKNOWN_PREFIX: str = "unknown_seat_"


# ─────────────────────────────────────────────────────────────────────────────
# ModelPool – lazy loading, egy modell per asztalmméret
# ─────────────────────────────────────────────────────────────────────────────

class ModelPool:
    """
    Betöltött modellek tárhelye.  Lazy: csak első kérésre tölt be.

    A pool {num_players: model} mappingot tart fenn.  Minden modell
    a saját state_size-ával kompatibilis.

    Changelog:
        v4.2.2 – BUGFIX-KRITIKUS-3: A _load() metódusban a lokális
                 ``num_players`` paramétert felülírta a checkpoint
                 értéke.  Ez None-t eredményezett régi ckpt-oknál,
                 és self._models[None] kulcsot hozott létre, ami
                 minden subsequent get() hívást eltörte.
    """

    def __init__(
        self,
        model_paths: Dict[int, str],
        device: torch.device,
    ) -> None:
        """
        Args:
            model_paths: ``{num_players: pth_fájl_elérési_út}`` mapping.
                         Példa: ``{2: '2max_ppo_v4.pth', 6: '6max_ppo_v4.pth'}``
            device:      A torch device, amelyre a modellek kerülnek.
        """
        self._paths:  Dict[int, str]             = {}
        self._models: Dict[int, AdvancedPokerAI] = {}
        self._meta:   Dict[int, Dict]            = {}
        self._device: torch.device               = device

        for num_players, path in model_paths.items():
            self._paths[int(num_players)] = path

    # ── Publikus API ─────────────────────────────────────────────────────────

    def get(self, num_players: int) -> Tuple[AdvancedPokerAI, int, int]:
        """
        Visszaad egy ``(model, state_size, action_size)`` tuple-t.
        Szükség esetén lazy betöltés.

        Args:
            num_players: Asztalmméret (2–9).

        Returns:
            ``(AdvancedPokerAI, state_size, action_size)`` tuple.

        Raises:
            ValueError:      Ha nincs regisztrált modell erre a mérete.
            FileNotFoundError: Ha a .pth fájl nem létezik.
            RuntimeError:    Ha a checkpoint formátuma érvénytelen.
        """
        num_players = int(num_players)
        if num_players not in self._models:
            self._load(num_players)
        return (
            self._models[num_players],
            self._meta[num_players]["state_size"],
            self._meta[num_players]["action_size"],
        )

    def available_sizes(self) -> List[int]:
        """Visszaadja a regisztrált asztalméretek listáját."""
        return sorted(self._paths.keys())

    def preload_all(self) -> None:
        """Előre betölti az összes regisztrált modellt."""
        for n in list(self._paths.keys()):
            if n not in self._models:
                self._load(n)

    # ── Belső betöltés ───────────────────────────────────────────────────────

    def _load(self, num_players: int) -> None:
        """
        Betölt egy modellt a megadott asztalmmérethez.

        BUGFIX-KRITIKUS-3:
            Az eredeti kódban a ``num_players`` lokális paramétert
            felülírta: ``num_players = ck.get('num_players', None)``.
            Ha a checkpoint nem tartalmazta ezt a mezőt (v4.1 előtti
            checkpointok), ``num_players`` None-ra állt, és
            ``self._models[None]`` kulcs keletkezett.  Minden utólagos
            ``get(6)`` hívás ezt nem találta, így KeyError/ValueError
            kivételt dobott.

            Javítás: a checkpoint értéke ``ck_num_players`` változóba
            kerül.  A dict kulcs MINDIG az eredeti ``num_players``
            paraméter marad.  A modell konstruktora a checkpoint
            értékét kapja, ha érvényes – különben az eredeti paramétert.

        Args:
            num_players: Asztalmméret – ez lesz a dict kulcs.

        Raises:
            ValueError:      Ha nincs regisztrált path erre a méretre.
            FileNotFoundError: Ha a .pth fájl nem létezik a lemezen.
            RuntimeError:    Ha a checkpoint formátuma érvénytelen
                             (hiányzó 'state_dict' kulcs).
        """
        if num_players not in self._paths:
            raise ValueError(
                f"Nincs modell regisztrálva {num_players} játékoshoz. "
                f"Elérhető: {self.available_sizes()}"
            )

        path = self._paths[num_players]

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Modell fájl nem található: {path!r}  "
                f"(num_players={num_players})"
            )

        logger.info(f"ModelPool: betöltés {num_players}p → {path!r}")

        ck = safe_load_checkpoint(path, map_location=self._device)

        if not isinstance(ck, dict) or "state_dict" not in ck:
            raise RuntimeError(
                f"Érvénytelen checkpoint formátum: {path!r}  "
                f"(hiányzó 'state_dict' kulcs).  "
                f"Típus: {type(ck).__name__}"
            )

        # ── Checkpoint mezők kiolvasása ───────────────────────────────────
        state_size:  int = ck.get("state_size",  492)
        action_size: int = ck.get("action_size", 7)

        # BUGFIX-KRITIKUS-3: KÜLÖNÁLLÓ változó a checkpoint értékéhez.
        # NE írjuk felül a ``num_players`` paramétert!
        # A checkpoint 'num_players' mezője a GRU step_dim számításához
        # kell.  Ha hiányzik (régi ckpt), az eredeti paramétert használjuk.
        ck_num_players: Optional[int] = ck.get("num_players", None)
        rlcard_obs_size: int          = ck.get("rlcard_obs_size", 54)

        # A modell konstruálásához: ck_num_players elsőbbsége, ha érvényes
        model_num_players: int = (
            int(ck_num_players)
            if ck_num_players is not None
            else num_players
        )

        if ck_num_players is None:
            logger.warning(
                f"ModelPool: a checkpoint ({path!r}) nem tartalmaz "
                f"'num_players' mezőt (valószínűleg v4.1 előtti formátum). "
                f"Fallback: num_players={num_players} a paraméterből."
            )

        if ck_num_players is not None and int(ck_num_players) != num_players:
            logger.warning(
                f"ModelPool: a checkpoint 'num_players' ({ck_num_players}) "
                f"eltér a kért értéktől ({num_players}). "
                f"A checkpoint értéke ({ck_num_players}) kerül a modell "
                f"konstruktorába, de a dict kulcs {num_players} marad."
            )

        # ── Modell létrehozása és state_dict betöltése ────────────────────
        model = AdvancedPokerAI(
            state_size=state_size,
            action_size=action_size,
            num_players=model_num_players,
            rlcard_obs_size=rlcard_obs_size,
        ).to(self._device)

        incompatible = model.load_state_dict(ck["state_dict"], strict=False)
        if incompatible.missing_keys:
            logger.info(
                f"ModelPool: {len(incompatible.missing_keys)} hiányzó kulcs "
                f"(nulláról indul, pl. GRU rétegek régi ckpt-ban): "
                f"{incompatible.missing_keys[:5]}"
                + (" ..." if len(incompatible.missing_keys) > 5 else "")
            )
        if incompatible.unexpected_keys:
            logger.warning(
                f"ModelPool: {len(incompatible.unexpected_keys)} váratlan "
                f"kulcs a checkpointban (figyelmen kívül hagyva): "
                f"{incompatible.unexpected_keys[:5]}"
            )

        model.eval()

        # ── KULCSFONTOSSÁGÚ: az eredeti ``num_players`` PARAMÉTER a kulcs ──
        # NE használjuk ck_num_players-t vagy model_num_players-t itt!
        self._models[num_players] = model
        self._meta[num_players] = {
            "state_size":       state_size,
            "action_size":      action_size,
            "model_num_players": model_num_players,
            "rlcard_obs_size":  rlcard_obs_size,
            "path":             path,
        }

        logger.info(
            f"ModelPool: {num_players}p kész | "
            f"state_size={state_size} | "
            f"action_size={action_size} | "
            f"model_num_players={model_num_players} | "
            f"rlcard_obs_size={rlcard_obs_size}"
        )

    def __repr__(self) -> str:
        return (
            f"ModelPool("
            f"registered={self.available_sizes()}, "
            f"loaded={sorted(self._models.keys())})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SeatMapper – username ↔ lokális szék index fordítás
# ─────────────────────────────────────────────────────────────────────────────

class SeatMapper:
    """
    Az asztal aktuális szék→username mappingját tárolja.

    A lokális indexek (0..N-1) a széksorrend szerinti pozíciók,
    pontosan az amit a ``build_state_tensor()`` és
    ``ActionHistoryEncoder`` vár.
    """

    def __init__(self) -> None:
        self._seat_map:          Dict[int, str] = {}
        self._local_order:       List[int]      = []
        self._username_to_local: Dict[str, int] = {}
        self._my_seat:           int            = 0
        self._my_local_idx:      int            = 0

    def update(self, seat_map: Dict[int, str], my_seat: int) -> None:
        """
        Frissíti a szék mappinget.

        Args:
            seat_map: ``{szék_idx: username}`` mapping.
                      None vagy üres username → ``'unknown_seat_{idx}'``
            my_seat:  Saját szék indexe.
        """
        filled: Dict[int, str] = {
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
        self._my_local_idx = (
            self._local_order.index(my_seat)
            if my_seat in self._local_order
            else 0
        )

    def local_index(self, username: str) -> int:
        """Visszaadja a username lokális indexét (0..N-1). Fallback: 0."""
        return self._username_to_local.get(username, 0)

    def username(self, seat: int) -> str:
        """Visszaadja a szék username-jét."""
        return self._seat_map.get(seat, f"{_UNKNOWN_PREFIX}{seat}")

    def username_by_local(self, local_idx: int) -> str:
        """Visszaadja a lokális index alapján a username-t."""
        if 0 <= local_idx < len(self._local_order):
            return self._seat_map.get(
                self._local_order[local_idx],
                f"{_UNKNOWN_PREFIX}{local_idx}",
            )
        return f"{_UNKNOWN_PREFIX}{local_idx}"

    @property
    def my_local_idx(self) -> int:
        """Saját lokális index."""
        return self._my_local_idx

    @property
    def num_players(self) -> int:
        """Aktív játékosok száma az asztalon."""
        return len(self._seat_map)

    @property
    def ordered_usernames(self) -> List[str]:
        """Játékosok listája szék-index szerint rendezve."""
        return [self._seat_map[s] for s in self._local_order]

    def as_seat_map(self) -> Dict[int, str]:
        """``{szék_idx: username}`` dict – ``preload_table()``-hoz."""
        return dict(self._seat_map)

    def build_local_stacks(
        self,
        stack_by_username: Dict[str, float],
        default_stack: float = 100.0,
    ) -> List[float]:
        """
        Összegyűjti a stackeket lokális szék-sorrendben.

        Args:
            stack_by_username: ``{username: stack}`` mapping.
            default_stack:     Fallback érték ismeretlen játékosnál.

        Returns:
            Stack lista lokális sorrendben.
        """
        return [
            float(
                stack_by_username.get(
                    self._seat_map.get(s, ""), default_stack
                )
            )
            for s in self._local_order
        ]

    def __repr__(self) -> str:
        return f"SeatMapper({self._seat_map})"


# ─────────────────────────────────────────────────────────────────────────────
# _GlobalTrackerAdapter – bridge a build_state_tensor() felé
# ─────────────────────────────────────────────────────────────────────────────

class _GlobalTrackerAdapter:
    """
    Adapter: ``GlobalPlayerTracker`` + ``SeatMapper`` → ``get_stats_vector()``
    interface.

    A ``build_state_tensor()`` az ``OpponentHUDTracker.get_stats_vector()``
    interfészt várja.  Ez az adapter a SQLite-alapú
    ``GlobalPlayerTracker``-t burkolja azzal az interfésszel.

    Belső használatú – külső kód ne példányosítsa közvetlenül.
    """

    def __init__(
        self,
        tracker: GlobalPlayerTracker,
        seat_mapper: SeatMapper,
    ) -> None:
        self._tracker     = tracker
        self._seat_mapper = seat_mapper

    def get_stats_vector(self) -> List[float]:
        """Flat HUD stat vektor a ``build_state_tensor()`` számára."""
        seat_map: Dict[int, str] = {
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

    Egy ``RTAManager`` példány a teljes session alatt él.
    Context manager-ként használva session végén automatikusan flush-ol.

    Alapvető workflow::

        with RTAManager(
            model_paths = {6: '6max_ppo_v4.pth', 2: '2max_ppo_v4.pth'},
            db_path     = 'players.db',
        ) as manager:
            # 1. Asztal setup (session elején, asztalméretnél)
            manager.manage_table_change(6, seat_map, my_seat=0)

            # 2. Minden kéz elején
            manager.new_hand(my_stack=200.0, bb=2.0, sb=1.0)

            # 3. Leosztás közben (sorrendben)
            manager.record_opponent_action('fish99', action=4, bet_amount=10.0)
            result = manager.get_recommendation(
                legal_actions=[0,1,2,3,4,5,6],
                hole_cards=['As','Kh'],
            )
            manager.record_my_action(result['action'])
            manager.new_street(1)  # flop
    """

    def __init__(
        self,
        model_paths:    Dict[int, str],
        db_path:        Optional[str]  = None,
        device:         str            = "cpu",
        equity_sims:    int            = 500,
        tracker_memory: int            = 1000,
    ) -> None:
        """
        Args:
            model_paths:    ``{num_players: pth_fájl}`` mapping.
                            Példa: ``{2: '2max.pth', 6: '6max.pth'}``
            db_path:        SQLite fájl az ellenfél statisztikákhoz.
                            ``None`` → csak memória (teszteléshez).
            device:         ``'cpu'`` vagy ``'cuda'``.
            equity_sims:    Monte Carlo szimulációk száma equity becsléshez.
            tracker_memory: Rolling window mérete eseményekben.
        """
        self._device:        torch.device    = torch.device(device)
        self._pool:          ModelPool       = ModelPool(model_paths, self._device)
        self._db_path:       Optional[str]   = db_path
        self._equity_est:    HandEquityEstimator = HandEquityEstimator(
            n_sim=equity_sims
        )
        self._action_mapper: PokerActionMapper  = PokerActionMapper()

        self._global_tracker = GlobalPlayerTracker(
            db_path=db_path,
            memory=tracker_memory,
        )
        self._obs_builder = ObsBuilder()

        # Aktuális asztal állapot
        self._seat_mapper:      SeatMapper               = SeatMapper()
        self._action_history:   collections.deque        = collections.deque(
            maxlen=ACTION_HISTORY_LEN
        )
        self._history_encoder:  Optional[ActionHistoryEncoder] = None
        self._active_model:     Optional[AdvancedPokerAI]      = None
        self._active_state_size:  int = 0
        self._active_action_size: int = 7

        # Játék kontextus
        self._bb:           float = 2.0
        self._sb:           float = 1.0
        self._my_stack:     float = 100.0
        self._all_stacks:   List[float] = []
        self._button_seat:  int   = 0
        self._street:       int   = 0
        self._num_players:  int   = 0
        self._hand_started: bool  = False

        logger.info(
            f"RTAManager inicializálva | "
            f"modellek: {list(model_paths.keys())} | "
            f"device: {device} | "
            f"db: {db_path or '(csak memória)'}"
        )
        if db_path:
            stats = self._global_tracker.db_stats()
            logger.info(
                f"Ellenfél DB: {stats['total_players']} játékos | "
                f"{stats['db_size_mb']} MB"
            )

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "RTAManager":
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val:  Optional[BaseException],
        exc_tb:   object,
    ) -> bool:
        """Session végén flush-ol a DB-be. Kivételeket nem nyel el."""
        self.flush_tracker()
        return False

    def flush_tracker(self) -> None:
        """
        Dirty lifetime értékek kiírása SQLite-ba.

        Context manager automatikusan hívja kilépéskor.
        Manuálisan is hívható pl. minden 100. kéz után.
        """
        self._global_tracker.flush()
        if self._db_path:
            logger.debug("RTAManager: tracker flush kész")

    # ── Asztal kezelés ────────────────────────────────────────────────────────

    def manage_table_change(
        self,
        num_players: int,
        seat_map:    Dict[int, str],
        my_seat:     int,
        button_seat: int = 0,
    ) -> None:
        """
        Asztalméret változás vagy új asztal kezelése.

        Hívandó:
          - Session elején (első asztalnál)
          - Ha az asztalon ülők száma megváltozik
          - Ha új asztalhoz csatlakozunk

        A ``GlobalPlayerTracker`` **NEM resetelődik** – ellenfél statok
        megmaradnak.

        Args:
            num_players: Aktív játékosok száma (2–9).
            seat_map:    ``{szék_idx: username}`` mapping.
                         ``None`` vagy üres username → ``'unknown_seat_N'``.
            my_seat:     Saját szék indexe.
            button_seat: Button szék indexe.

        Raises:
            ValueError:      Ha nincs modell regisztrálva erre a méretre.
            FileNotFoundError: Ha a modell fájl nem létezik.
            RuntimeError:    Ha a checkpoint érvénytelen.
        """
        num_players = int(num_players)
        logger.info(
            f"manage_table_change: {num_players}p | "
            f"my_seat={my_seat} | seats={seat_map}"
        )

        # 1. Modell lazy betöltése (hibák továbbengedve a hívóhoz)
        model, state_size, action_size = self._pool.get(num_players)
        self._active_model       = model
        self._active_state_size  = state_size
        self._active_action_size = action_size

        # 2. SeatMapper frissítése
        self._seat_mapper.update(seat_map, my_seat)
        self._num_players  = num_players
        self._button_seat  = button_seat

        # 3. ActionHistoryEncoder újraalkotása a helyes num_players-szel
        self._history_encoder = ActionHistoryEncoder(num_players, action_size)

        # 4. Action history törlése (régi méret inkompatibilis lenne)
        self._action_history.clear()

        # 5. SQLite batch preload az asztalon ülőknek
        self._global_tracker.preload_table(
            self._seat_mapper.as_seat_map()
        )

        logger.info(
            f"Aktív modell: {num_players}p | "
            f"state_size={state_size} | "
            f"DB: {self._global_tracker.db_stats().get('total_players','?')} "
            f"játékos"
        )

    def reset_session(self) -> None:
        """
        Teljes memória reset.  DB **NEM** törlődik.

        Normál asztalváltáshoz NEM szükséges – azt
        ``manage_table_change()`` kezeli.
        """
        self._global_tracker.reset()
        self._action_history.clear()
        self._hand_started = False
        logger.info("RTAManager: session reset (memória törölve, DB érintetlen)")

    # ── Kéz kezelés ──────────────────────────────────────────────────────────

    def new_hand(
        self,
        my_stack:   float,
        all_stacks: Optional[Dict[str, float]] = None,
        bb:         float = 2.0,
        sb:         float = 1.0,
    ) -> None:
        """
        Új leosztás kezdetekor hívandó.

        Args:
            my_stack:   Saját stack.
            all_stacks: ``{username: stack}`` mapping az összes játékoshoz.
                        ``None`` → egyenlő stackek feltételezve.
            bb:         Big blind értéke.
            sb:         Small blind értéke.
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

    def new_street(self, street: int) -> None:
        """
        Street váltáskor hívandó.

        Args:
            street: 0=preflop, 1=flop, 2=turn, 3=river.
        """
        self._street = int(street)
        logger.debug(
            f"new_street: "
            f"{['preflop','flop','turn','river'][min(street, 3)]}"
        )

    # ── Akció rögzítés ────────────────────────────────────────────────────────

    def record_opponent_action(
        self,
        username:   str,
        action:     int,
        bet_amount: float              = 0.0,
        pot_size:   float              = 1.0,
        context:    Optional[Dict]     = None,
    ) -> None:
        """
        Ellenfél akciójának rögzítése.

        Args:
            username:   Játékos azonosítója (pl. ``'fish99'``).
            action:     Absztrakt akció 0–6.
            bet_amount: Bet/raise összege (0 ha fold/call/check).
            pot_size:   Aktuális pot mérete (bet normáláshoz).
            context:    Opcionális kontextus dict, pl.
                        ``{'facing_3bet': bool, 'is_cbet_opp': bool,
                           'facing_cbet': bool}``.
        """
        local_idx = self._seat_mapper.local_index(username)
        bet_norm  = min(bet_amount / max(pot_size, 1e-6), 5.0) / 5.0

        self._action_history.append((local_idx, action, bet_norm))
        self._global_tracker.record_action(
            username, action,
            street  = self._street,
            context = context,
        )

    def record_my_action(
        self,
        action:     int,
        bet_amount: float = 0.0,
        pot_size:   float = 1.0,
    ) -> None:
        """
        Saját akció rögzítése az action history-ba.

        Args:
            action:     Absztrakt akció 0–6.
            bet_amount: Bet/raise összege.
            pot_size:   Aktuális pot mérete.
        """
        my_idx   = self._seat_mapper.my_local_idx
        bet_norm = min(bet_amount / max(pot_size, 1e-6), 5.0) / 5.0
        self._action_history.append((my_idx, action, bet_norm))

    # ── Fő ajánlás API ────────────────────────────────────────────────────────

    def get_recommendation(
        self,
        legal_actions: List[int],
        hole_cards:    Optional[List[str]] = None,
        board_cards:   Optional[List[str]] = None,
        current_pot:   Optional[float]     = None,
        call_amount:   float               = 0.0,
    ) -> Dict:
        """
        Akcióajánlás az aktuális játékállapothoz.

        Args:
            legal_actions: Engedélyezett absztrakt akciók (0–6).
            hole_cards:    Saját lapok (pl. ``['As','Kh']``).
            board_cards:   Közösségi lapok (pl. ``['Td','7c','2s']``).
            current_pot:   Aktuális pot mérete.
            call_amount:   Call összege (pot odds számításhoz).

        Returns:
            Dict a következő kulcsokkal:
                - ``action``       (int) ajánlott absztrakt akció
                - ``action_name``  (str) pl. ``'Raise 50%'``
                - ``confidence``   (float) ajánlott akció valószínűsége
                - ``top3``         list[(action_name, prob)]
                - ``equity``       (float) becsült kéz equity
                - ``spr``          (float) stack/pot ratio
                - ``m_ratio``      (float) stack/blindok
                - ``street_name``  (str) ``'Preflop'``/``'Flop'``/...
                - ``value_est``    (float) critic értékbecslés
                - ``explanation``  (str) rövid szöveges magyarázat
                - ``db_players``   (int) ismert játékosok száma a DB-ben

        Raises:
            RuntimeError: Ha ``manage_table_change()`` még nem volt hívva.
        """
        if self._active_model is None:
            raise RuntimeError(
                "Nincs aktív modell. "
                "Hívd meg a manage_table_change()-t előbb."
            )
        if self._history_encoder is None:
            raise RuntimeError(
                "History encoder nincs inicializálva. "
                "Hívd meg a manage_table_change()-t előbb."
            )

        board_cards = board_cards or []
        hole_cards  = hole_cards  or []

        # ── Obs vektor rekonstrukciója ────────────────────────────────────
        stacks  = self._all_stacks or [self._my_stack] * self._num_players
        obs_arr = self._obs_builder.build(
            hole_cards  = hole_cards,
            board_cards = board_cards,
            my_chips    = self._my_stack,
            all_chips   = stacks,
        )

        # ── State dict összerakása ────────────────────────────────────────
        button_local = self._seat_mapper.local_index(
            self._seat_mapper.username(self._button_seat)
        )
        state: Dict = {
            "obs": obs_arr,
            "raw_obs": {
                "my_chips":     self._my_stack,
                "all_chips":    stacks,
                "pot":          current_pot or 0.0,
                "public_cards": board_cards,
                "hand":         hole_cards,
                "button":       button_local,
                "call_amount":  call_amount,
            },
        }

        # ── Equity becslés ────────────────────────────────────────────────
        equity: float = 0.5
        if len(hole_cards) == 2:
            try:
                equity = self._equity_est.equity(
                    hole_cards, board_cards,
                    num_opponents=max(self._num_players - 1, 1),
                )
            except Exception as exc:
                logger.debug(f"Equity becslés hiba (fallback 0.5): {exc}")
                equity = 0.5

        # ── State tensor ─────────────────────────────────────────────────
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

        # ── Model forward ─────────────────────────────────────────────────
        with torch.no_grad():
            action_probs, value, _ = self._active_model.forward(
                state_t.to(self._device), legal_actions
            )

        probs_np    = action_probs.squeeze(0).cpu().numpy()
        best_action = int(np.argmax(probs_np))
        confidence  = float(probs_np[best_action])

        sorted_idx = np.argsort(probs_np)[::-1]
        top3: List[Tuple[str, float]] = [
            (self._action_mapper.action_name(int(i)), float(probs_np[i]))
            for i in sorted_idx[:3]
            if int(i) in legal_actions
        ]

        # ── Kontextuális értékek ──────────────────────────────────────────
        pot         = current_pot or 1.0
        spr         = self._my_stack / max(pot, 1e-6)
        m_ratio     = self._my_stack / max(self._bb + self._sb, 1e-6)
        street_name = ["Preflop", "Flop", "Turn", "River"][
            min(self._street, 3)
        ]
        explanation = self._explain(
            best_action, equity, spr, m_ratio,
            confidence, street_name, call_amount,
        )

        db_stats = self._global_tracker.db_stats()

        return {
            "action":      best_action,
            "action_name": self._action_mapper.action_name(best_action),
            "confidence":  confidence,
            "top3":        top3,
            "equity":      equity,
            "spr":         round(spr, 2),
            "m_ratio":     round(m_ratio, 1),
            "street_name": street_name,
            "value_est":   float(value.squeeze().cpu()),
            "explanation": explanation,
            "db_players":  db_stats.get("total_players", len(self._global_tracker)),
        }

    # ── Privát segédmetódusok ─────────────────────────────────────────────────

    def _explain(
        self,
        action:      int,
        equity:      float,
        spr:         float,
        m_ratio:     float,
        confidence:  float,
        street_name: str,
        call_amount: float,
    ) -> str:
        """
        Rövid szöveges magyarázat a javaslathoz.

        Args:
            action:      Ajánlott absztrakt akció.
            equity:      Becsült kéz equity.
            spr:         Stack/pot ratio.
            m_ratio:     Stack/blindok.
            confidence:  A javasolt akció valószínűsége.
            street_name: Az aktuális utca neve.
            call_amount: Call összege.

        Returns:
            Pipe-szeparált szöveges magyarázat string.
        """
        lines = [
            f"{street_name} | "
            f"{self._action_mapper.action_name(action)} "
            f"({confidence * 100:.0f}%)"
        ]
        if equity > 0.5:
            lines.append(f"Equity: {equity * 100:.0f}% (kedvező)")
        elif equity < 0.35:
            lines.append(f"Equity: {equity * 100:.0f}% (gyenge kéz)")
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
            lines.append(f"Call: pot odds ~{po * 100:.0f}%")
        return " | ".join(lines)

    # ── Diagnosztika ──────────────────────────────────────────────────────────

    def player_stats(self, username: str) -> Dict:
        """
        Egy ellenfél összesített stat summaryja.

        Args:
            username: Játékos azonosítója.

        Returns:
            Summary dict VPIP, PFR, AF stb. értékekkel.
        """
        return self._global_tracker.player_summary(username)

    def all_player_stats(self) -> List[Dict]:
        """Session közben érintett összes játékos summaryja."""
        return self._global_tracker.all_summaries()

    def top_players(self, n: int = 20) -> List[Dict]:
        """
        Top N játékos az adatbázisból lifetime kézszám szerint.

        Args:
            n: Visszaadandó játékosok száma.

        Returns:
            Lista dicts-el, amelyek tartalmazzák a játékos adatokat.
        """
        return self._global_tracker.top_players_by_hands(n)

    def db_info(self) -> Dict:
        """Adatbázis állapot és méret."""
        return self._global_tracker.db_stats()

    def current_table_info(self) -> Dict:
        """Aktuális asztal állapota diagnosztikai célokra."""
        return {
            "num_players":  self._num_players,
            "my_local_idx": self._seat_mapper.my_local_idx,
            "seat_order":   self._seat_mapper.ordered_usernames,
            "street":       ["preflop","flop","turn","river"][
                min(self._street, 3)
            ],
            "bb":           self._bb,
            "sb":           self._sb,
            "my_stack":     self._my_stack,
            "model_loaded": self._active_model is not None,
            "state_size":   self._active_state_size,
            "db":           self._global_tracker.db_stats(),
        }

    def preload_all_models(self) -> None:
        """Összes modell előzetes betöltése startup-ban."""
        self._pool.preload_all()

    def __repr__(self) -> str:
        return (
            f"RTAManager("
            f"models={self._pool.available_sizes()}, "
            f"table={self._num_players}p, "
            f"db={self._db_path or 'memory'})"
        )
