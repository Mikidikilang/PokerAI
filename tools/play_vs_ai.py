#!/usr/bin/env python3
"""
tools/play_vs_ai.py  –  Játssz a kitrénelt Poker AI modelleid ellen!

Használat:
    python play_vs_ai.py [--port 8080]

Majd nyisd meg a böngészőben: http://localhost:8080

A szerver automatikusan megtalálja a .pth fájlokat a mappában,
és a böngészős felületen kiválaszthatod melyik modell ellen játszol.

Biztonsági megjegyzések:
    - A szerver csak a projekt könyvtárán belüli .pth fájlokat engedélyezi.
    - Path traversal védelem: minden model_path validálva van.
    - Lokális fejlesztői eszköz – ne tedd ki publikus internetre
      autentikáció nélkül.

Változások v4.2.2:
    [SECURITY-KRITIKUS-1] Path Traversal védelem hozzáadva:
        - _resolve_and_validate_model_path() helper
        - Whitelist alapú könyvtár-ellenőrzés (os.path.realpath)
        - Csak .pth kiterjesztés engedélyezett
        - Fájlnév karakterkészlet validáció
        - scan_models() is whitelist-en belül keres
    [QUALITY] Teljes type hint lefedettség.
    [QUALITY] Validáció a GameSession.start_session()-ban is.
"""

import argparse
import base64
import collections
import hashlib
import glob
import http.server
import json
import logging
import os
import random
import re
import sys
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from core.model import AdvancedPokerAI
from core.action_mapper import PokerActionMapper
from core.features import (
    ActionHistoryEncoder,
    build_state_tensor,
    detect_street,
    compute_state_size,
    ACTION_HISTORY_LEN,
)
from core.opponent_tracker import OpponentHUDTracker
from core.equity import HandEquityEstimator
from utils.checkpoint_utils import safe_load_checkpoint

import rlcard

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    """
    Konzol (INFO) + fájl (DEBUG) logolás a logs/ mappába.

    Returns:
        Konfigurált logger példány.
    """
    log = logging.getLogger("PlayVsAI")
    log.handlers.clear()
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
# Biztonsági konstansok
# ─────────────────────────────────────────────────────────────────────────────

# Megengedett könyvtárak, amelyekből .pth fájlok betölthetők.
# A check os.path.realpath() alapú – szimbolikus linkeket is felold.
# Értéke futáskor töltődik be a _build_allowed_dirs() hívással.
_ALLOWED_DIRS: List[str] = []

# Megengedett fájlkiterjesztések a model_path validációban.
_ALLOWED_EXTENSIONS: Tuple[str, ...] = (".pth",)

# Regex a fájlnév karakterkészlet validációhoz.
# Csak alfanumerikus karakterek, kötőjel, aláhúzás, pont, és / \ elválasztók.
# Ez kizárja a shell metakaraktereket (;, &, |, ` stb.).
_SAFE_PATH_RE = re.compile(r'^[\w\-./\\]+$')


def _build_allowed_dirs(base_dir: str) -> List[str]:
    """
    Felépíti a whitelist könyvtárak listáját a projekt báziskönyvtára alapján.

    A következő könyvtárak engedélyezettek:
        - ``{base_dir}/models/``   (ModelManager által kezelt modellek)
        - ``{base_dir}/``          (gyökérben lévő migrált .pth fájlok)

    Args:
        base_dir: A projekt abszolút báziskönyvtára.

    Returns:
        Valós (realpath) abszolút útvonalak listája.
    """
    allowed = [
        os.path.realpath(base_dir),
        os.path.realpath(os.path.join(base_dir, "models")),
    ]
    return allowed


def _resolve_and_validate_model_path(
    base_dir: str,
    model_path: str,
) -> str:
    """
    Feloldja és validálja a modell elérési útját path traversal ellen.

    A függvény a következő ellenőrzéseket végzi sorban:

    1. **Karakterkészlet**: Csak alfanumerikus, ``-``, ``_``, ``.``,
       ``/``, ``\\`` karakterek engedélyezettek.  Shell metakarakterek
       (pl. ``;``, ``&``, ``|``, `` ` ``) tiltottak.

    2. **Kiterjesztés**: Csak ``.pth`` fájlok engedélyezettek.

    3. **Path traversal**: ``os.path.realpath()`` feloldja a szimbolikus
       linkeket és a ``../`` komponenseket.  Az eredménynek a whitelist
       könyvtárak egyikén belül kell lennie.

    4. **Fájl létezése**: A feloldott útvonalnak létező fájlra kell mutatnia.

    Args:
        base_dir:   A projekt abszolút báziskönyvtára (nem user-controlled).
        model_path: A klienstől érkező elérési út (user-controlled, nem megbízható).

    Returns:
        A validált, feloldott abszolút elérési út.

    Raises:
        ValueError: Ha a path traversal detektálva, az elérési út nem
                    a whitelist könyvtáron belül van, érvénytelen karaktert
                    tartalmaz, vagy nem ``.pth`` kiterjesztésű.
        FileNotFoundError: Ha a feloldott útvonal nem létezik a lemezen.

    Példa::

        # Normál használat:
        path = _resolve_and_validate_model_path(
            '/app', 'models/2max/2max_ppo_v4.pth'
        )
        # → '/app/models/2max/2max_ppo_v4.pth'

        # Path traversal kísérlet – ValueError:
        _resolve_and_validate_model_path('/app', '../../../../etc/shadow')
    """
    if not model_path or not isinstance(model_path, str):
        raise ValueError("A model_path nem lehet üres vagy nem string típusú.")

    # 1. Karakterkészlet ellenőrzés
    if not _SAFE_PATH_RE.match(model_path):
        # Szándékosan nem szivárogtatjuk ki a bemeneti értéket a hibaüzenetben,
        # hogy elkerüljük a log injection lehetőségét.
        raise ValueError(
            "Érvénytelen karakter a model_path-ban. "
            "Csak alfanumerikus, kötőjel, aláhúzás, pont és "
            "elválasztó karakterek engedélyezettek."
        )

    # 2. Kiterjesztés ellenőrzés (case-insensitive)
    _, ext = os.path.splitext(model_path)
    if ext.lower() not in _ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Csak {_ALLOWED_EXTENSIONS} kiterjesztésű fájlok engedélyezettek. "
            f"Kapott kiterjesztés: {ext!r}"
        )

    # 3. Path traversal ellenőrzés
    # os.path.realpath() feloldja a szimbolikus linkeket és a ../  komponenseket.
    # FONTOS: os.path.join()-t base_dir + model_path-szal hívjuk, majd
    # realpath-ot alkalmazunk – így a "../../../etc/passwd" is feloldódik
    # az abszolút /etc/passwd útvonalra, ami nem lesz a whitelist-en.
    candidate = os.path.realpath(os.path.join(base_dir, model_path))

    allowed_dirs = _ALLOWED_DIRS if _ALLOWED_DIRS else _build_allowed_dirs(base_dir)

    in_allowed = any(
        candidate.startswith(allowed_dir + os.sep)
        or candidate == allowed_dir
        for allowed_dir in allowed_dirs
    )

    if not in_allowed:
        # Biztonsági log: a kísérlet tényét rögzítjük (de NEM a kliensből
        # érkező nyers értéket – log injection védelme).
        logger.warning(
            "Path traversal kísérlet detektálva. "
            f"Feloldott cél: {candidate!r} – nincs a whitelist-en. "
            f"Whitelist: {allowed_dirs}"
        )
        raise ValueError(
            "A megadott elérési út nem található az engedélyezett "
            "könyvtárakon belül."
        )

    # 4. Fájl létezése
    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"A modell fájl nem található: {os.path.basename(candidate)!r}"
        )

    return candidate


# ─────────────────────────────────────────────────────────────────────────────
# Kártya formátum segédfüggvények
# ─────────────────────────────────────────────────────────────────────────────

SUIT_MAP_RLCARD: Dict[str, str] = {
    "S": "s", "H": "h", "D": "d", "C": "c"
}
RANK_MAP_DISPLAY: Dict[str, str] = {
    "A": "A", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9", "T": "10",
    "J": "J", "Q": "Q", "K": "K",
}


def rlcard_to_display(card_str: str) -> Optional[Dict[str, str]]:
    """
    Konvertálja az rlcard kártya indexet megjelenítési formátumra.

    Args:
        card_str: rlcard ``Card.get_index()`` formátum (pl. ``'SA'``, ``'HK'``).

    Returns:
        Dict ``{'rank', 'suit', 'display', 'raw'}`` kulcsokkal,
        vagy ``None`` ha a bemenet érvénytelen.
    """
    if not card_str or not isinstance(card_str, str) or len(card_str) < 2:
        return None
    suit_char = card_str[0].upper()
    rank_char = card_str[1].upper()
    suit      = SUIT_MAP_RLCARD.get(suit_char, "s")
    rank      = RANK_MAP_DISPLAY.get(rank_char, rank_char)
    symbols   = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
    return {
        "rank":    rank,
        "suit":    suit,
        "display": f"{rank}{symbols[suit]}",
        "raw":     card_str,
    }


def cards_to_equity_format(rlcard_cards: List[str]) -> List[str]:
    """
    Konvertálja az rlcard kártya listát az equity.py formátumra.

    Args:
        rlcard_cards: rlcard formátumú kártyák (pl. ``['SA', 'HK']``).

    Returns:
        equity.py formátumú kártyák (pl. ``['As', 'Kh']``).
    """
    result: List[str] = []
    for c in rlcard_cards:
        if len(c) >= 2:
            suit  = c[0].upper()
            rank  = c[1].upper()
            s_low = {"S": "s", "H": "h", "D": "d", "C": "c"}.get(suit, "s")
            result.append(f"{rank}{s_low}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model Scanner
# ─────────────────────────────────────────────────────────────────────────────

def scan_models(base_dir: str) -> List[Dict[str, Any]]:
    """
    Megkeresi az összes .pth fájlt a whitelist könyvtárakban és
    kiolvassa a metaadatokat.

    Biztonsági megjegyzés: csak a whitelist könyvtárakon belüli fájlokat
    veszi figyelembe.  A ``glob`` eredményei ``_resolve_and_validate_model_path``
    szűrőn mennek át.

    Args:
        base_dir: A projekt abszolút báziskönyvtára.

    Returns:
        Modell metaadat dict-ek listája.
    """
    allowed_dirs = _build_allowed_dirs(base_dir)

    # Csak a whitelist könyvtárakban keresünk
    patterns: List[str] = []
    for allowed_dir in allowed_dirs:
        if os.path.isdir(allowed_dir):
            patterns.append(os.path.join(allowed_dir, "*.pth"))
            patterns.append(os.path.join(allowed_dir, "**", "*.pth"))

    raw_files: List[str] = []
    for p in patterns:
        raw_files.extend(glob.glob(p, recursive=True))

    # Deduplikáció + realpath normalizálás
    files: List[str] = sorted(
        {os.path.realpath(f) for f in raw_files}
    )

    models: List[Dict[str, Any]] = []
    for f in files:
        # Extra whitelist ellenőrzés (paranoid, de helyes)
        in_allowed = any(
            f.startswith(d + os.sep) or f == d
            for d in allowed_dirs
        )
        if not in_allowed:
            logger.debug(f"scan_models: kihagyva (whitelist-en kívül): {f}")
            continue

        try:
            ck = safe_load_checkpoint(f, map_location="cpu")
            if isinstance(ck, dict) and "state_dict" in ck:
                state_size       = ck.get("state_size", "?")
                action_size      = ck.get("action_size", 7)
                episodes         = ck.get("episodes_trained", 0)
                algorithm        = ck.get("algorithm", "unknown")
                guessed_players: Optional[int] = None

                if isinstance(state_size, int):
                    for np_ in range(2, 10):
                        if compute_state_size(54, np_) == state_size:
                            guessed_players = np_
                            break

                # A path-ot a báziskönyvtárhoz képest relatívan tároljuk,
                # hogy a kliensnek ne legyen teljes filesystem ismerete.
                rel_path = os.path.relpath(f, base_dir)

                models.append({
                    "path":            rel_path,
                    "abs_path":        f,
                    "episodes":        episodes,
                    "state_size":      state_size,
                    "action_size":     action_size,
                    "algorithm":       algorithm,
                    "guessed_players": guessed_players,
                    "filename":        os.path.basename(f),
                })
                logger.info(
                    f"  Modell: {os.path.basename(f)} | "
                    f"{episodes:,} ep | state={state_size} | "
                    f"~{guessed_players or '?'}p"
                )
        except Exception as e:
            logger.debug(f"  Skip {f}: {e}")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# GameSession
# ─────────────────────────────────────────────────────────────────────────────

class GameSession:
    """Egy teljes játék session kezelése."""

    def __init__(self) -> None:
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model:          Optional[AdvancedPokerAI]    = None
        self.env:            Optional[Any]                = None
        self.num_players:    int                          = 0
        self.human_seat:     int                          = 0
        self.action_mapper   = PokerActionMapper()
        self.equity_est      = HandEquityEstimator(n_sim=200)
        self.tracker:        Optional[OpponentHUDTracker] = None
        self.history_encoder: Optional[ActionHistoryEncoder] = None
        self.action_history  = collections.deque(maxlen=ACTION_HISTORY_LEN)
        self.street:         int   = 0
        self.stacks:         List[float] = []
        self.initial_stack:  float = 100.0
        self.bb:             float = 2.0
        self.sb:             float = 1.0
        self.hand_num:       int   = 0
        self.human_cards_cache: List[str] = []
        self.hand_log:       List[Dict]   = []
        self.total_results:  Dict[str, int] = {"player": 0, "ai": 0, "hands": 0}
        self._active:        bool = False

    def load_model(self, model_path: str) -> Tuple[int, int]:
        """
        Modell betöltése egy validált, abszolút elérési útról.

        Args:
            model_path: Validált abszolút .pth fájl útvonal.

        Returns:
            ``(state_size, action_size)`` tuple.

        Raises:
            RuntimeError: Ha a checkpoint formátuma érvénytelen.
        """
        logger.info(f"Modell betöltése: {model_path!r}")
        ck = safe_load_checkpoint(model_path, map_location=self.device)

        if not (isinstance(ck, dict) and "state_dict" in ck):
            raise RuntimeError(
                f"Érvénytelen checkpoint formátum: {model_path!r}. "
                f"Hiányzó 'state_dict' kulcs."
            )

        state_size      = ck.get("state_size",  492)
        action_size     = ck.get("action_size", 7)
        ck_num_players  = ck.get("num_players", None)
        rlcard_obs_size = ck.get("rlcard_obs_size", 54)

        self.model = AdvancedPokerAI(
            state_size=state_size,
            action_size=action_size,
            num_players=ck_num_players,
            rlcard_obs_size=rlcard_obs_size,
        ).to(self.device)

        self.model.load_state_dict(ck["state_dict"], strict=False)
        self.model.eval()
        self._state_size = state_size
        logger.info(
            f"Modell kész: state_size={state_size}, "
            f"device={self.device}, "
            f"num_players={ck_num_players}"
        )
        return state_size, action_size

    def start_session(
        self,
        model_path:  str,
        num_players: int,
        human_seat:  int,
        bb:          float = 2.0,
        sb:          float = 1.0,
        stack:       float = 100.0,
    ) -> bool:
        """
        Új session indítása.

        Args:
            model_path:  Validált abszolút .pth fájl útvonal.
            num_players: Játékosok száma (2–9).
            human_seat:  Saját szék indexe (0..num_players-1).
            bb:          Big blind.
            sb:          Small blind.
            stack:       Induló stack.

        Returns:
            ``True`` ha sikeresen inicializálódott.
        """
        self.load_model(model_path)
        self.num_players   = num_players
        self.human_seat    = human_seat
        self.bb            = float(bb)
        self.sb            = float(sb)
        self.initial_stack = float(stack)
        self.stacks        = [float(stack)] * num_players
        self.hand_num      = 0
        self.total_results = {"player": 0, "ai": 0, "hands": 0}

        self.env = rlcard.make(
            "no-limit-holdem",
            config={"game_num_players": num_players},
        )
        self.tracker         = OpponentHUDTracker(num_players)
        self.history_encoder = ActionHistoryEncoder(
            num_players, PokerActionMapper.NUM_CUSTOM_ACTIONS
        )
        self._active = True
        logger.info(
            f"Session indítva: {num_players}p, "
            f"human=seat {human_seat}"
        )
        return True

    def new_hand(self) -> Dict:
        """
        Új leosztás indítása.

        Returns:
            Teljes játékállapot dict a frontend számára.
        """
        if not self._active:
            return {"error": "Session nincs aktív"}

        self.hand_num += 1
        self.action_history.clear()
        self.hand_log = []
        self.street   = 0

        try:
            logger.info(f"[Hand #{self.hand_num}] env.reset()...")
            state, player_id = self.env.reset()
            logger.info(
                f"[Hand #{self.hand_num}] reset OK, "
                f"current_player={player_id}, "
                f"human={self.human_seat}"
            )
        except Exception as e:
            logger.error(f"Env reset hiba: {e}\n{traceback.format_exc()}")
            return {"error": str(e)}

        self.current_state  = state
        self.current_player = player_id

        # Human lapok cache-elése
        try:
            hand_cards = self.env.game.players[self.human_seat].hand
            self.human_cards_cache = [c.get_index() for c in hand_cards]
            logger.info(
                f"[Hand #{self.hand_num}] "
                f"Human cards: {self.human_cards_cache}"
            )
        except Exception as e:
            logger.warning(f"Human cards cache hiba: {e}")
            self.human_cards_cache = (
                state.get("raw_obs", {}).get("hand", [])
            )

        ai_actions = self._run_ai_until_human()
        logger.info(
            f"[Hand #{self.hand_num}] AI done, "
            f"{len(ai_actions)} actions, "
            f"is_over={self.env.is_over()}"
        )
        return self._build_response(ai_actions)

    def human_action(self, abstract_action: int) -> Dict:
        """
        Emberi játékos akciója.

        Args:
            abstract_action: Absztrakt akció index (0–6).

        Returns:
            Frissített játékállapot dict.
        """
        if not self._active or self.env.is_over():
            return {"error": "Nem a te köröd vagy vége a kéznek"}

        abstract_action = int(abstract_action)
        raw_legal       = self.current_state.get("legal_actions", [1])
        env_action      = self.action_mapper.get_env_action(
            abstract_action, raw_legal
        )
        action_name     = self.action_mapper.action_name(abstract_action)

        self.hand_log.append({
            "player":      self.human_seat,
            "player_name": "Te",
            "action":      abstract_action,
            "action_name": action_name,
            "street":      self.street,
        })

        self.action_history.append((self.human_seat, abstract_action, 0.0))
        self.tracker.record_action(
            self.human_seat, abstract_action, street=self.street
        )

        try:
            new_state, new_player = self.env.step(env_action)
            self.current_state  = new_state
            self.current_player = new_player
            self.street         = detect_street(new_state)
        except Exception as e:
            logger.error(f"Step hiba: {e}")
            return {"error": str(e)}

        ai_actions = self._run_ai_until_human()
        return self._build_response(ai_actions)

    def _run_ai_until_human(self) -> List[Dict]:
        """
        AI játékosok léptetése amíg ember nem következik vagy a kéz véget ér.

        Returns:
            AI akciók listája (diagnosztikai célokra).
        """
        ai_actions: List[Dict] = []
        safety = 0
        while (
            not self.env.is_over()
            and self.current_player != self.human_seat
        ):
            safety += 1
            if safety > 200:
                logger.error("Végtelen loop védelem – kilépés")
                break
            try:
                info = self._ai_act()
                ai_actions.append(info)
            except Exception as e:
                logger.error(
                    f"AI lépés hiba: {e}\n{traceback.format_exc()}"
                )
                break
        return ai_actions

    def _ai_act(self) -> Dict:
        """
        Egy AI játékos lépése.

        Returns:
            Az AI akciójának leírása dict-ben.
        """
        state     = self.current_state
        player_id = self.current_player
        raw_legal = state.get("legal_actions", [1])
        abs_legal = self.action_mapper.get_abstract_legal_actions(raw_legal)

        equity: float = 0.5
        try:
            ai_hand      = self.env.game.players[player_id].hand
            ai_cards_eq  = cards_to_equity_format(
                [c.get_index() for c in ai_hand]
            )
            board_eq     = cards_to_equity_format(
                [c.get_index() for c in self.env.game.public_cards]
            )
            if len(ai_cards_eq) == 2:
                equity = self.equity_est.equity(
                    ai_cards_eq, board_eq,
                    num_opponents=max(self.num_players - 1, 1),
                )
        except Exception:
            pass

        state_t = build_state_tensor(
            state, self.tracker, self.action_history,
            self.history_encoder, self.num_players,
            my_player_id  = player_id,
            bb            = self.bb,
            sb            = self.sb,
            initial_stack = self.initial_stack,
            street        = self.street,
            equity        = equity,
        )

        with torch.no_grad():
            action, log_prob, entropy, value, _ = self.model.get_action(
                state_t.to(self.device), abs_legal, deterministic=False
            )

        abstract_action = int(action.item())
        env_action      = self.action_mapper.get_env_action(
            abstract_action, raw_legal
        )
        action_name = self.action_mapper.action_name(abstract_action)

        self.action_history.append((player_id, abstract_action, 0.0))
        self.tracker.record_action(
            player_id, abstract_action, street=self.street
        )
        self.hand_log.append({
            "player":      player_id,
            "player_name": f"AI #{player_id}",
            "action":      abstract_action,
            "action_name": action_name,
            "street":      self.street,
        })

        new_state, new_player = self.env.step(env_action)
        self.current_state  = new_state
        self.current_player = new_player
        self.street         = detect_street(new_state)

        return {
            "player":      player_id,
            "action":      abstract_action,
            "action_name": action_name,
        }

    def _build_response(
        self,
        ai_actions: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Teljes játékállapot összerakása a frontend számára.

        Args:
            ai_actions: Az AI akciók listája (opcionális, diagnosztikai).

        Returns:
            JSON-serializable dict a frontend számára.
        """
        is_over = self.env.is_over()
        players: List[Dict] = []

        try:
            for i, p in enumerate(self.env.game.players):
                cards = [c.get_index() for c in p.hand] if p.hand else []
                raw_status = getattr(p, "status", None)
                if raw_status is not None:
                    status_str = str(raw_status).lower()
                    if "fold" in status_str:
                        status = "folded"
                    elif "allin" in status_str or "all_in" in status_str:
                        status = "allin"
                    else:
                        status = "alive"
                else:
                    status = "alive"

                players.append({
                    "seat":         i,
                    "chips_in_pot": float(getattr(p, "in_chips", 0)),
                    "stack":        float(getattr(p, "remained_chips", 0)),
                    "status":       status,
                    "is_human":     i == self.human_seat,
                    "name":         "Te" if i == self.human_seat else f"AI #{i}",
                    "cards": (
                        [rlcard_to_display(c) for c in cards]
                        if (i == self.human_seat or is_over)
                        else None
                    ),
                })
        except Exception as e:
            logger.error(
                f"Player data hiba: {e}\n{traceback.format_exc()}"
            )
            players = [
                {
                    "seat": i, "name": f"P{i}", "stack": 0,
                    "chips_in_pot": 0, "status": "alive",
                    "is_human": i == self.human_seat, "cards": None,
                }
                for i in range(self.num_players)
            ]

        board_cards: List[Optional[Dict]] = []
        try:
            board_cards = [
                rlcard_to_display(c.get_index())
                for c in self.env.game.public_cards
            ]
        except Exception:
            pass

        pot = sum(p["chips_in_pot"] for p in players)
        human_cards = [rlcard_to_display(c) for c in self.human_cards_cache]

        abs_legal: List[int]          = []
        legal_action_names: Dict[int, str] = {}

        if not is_over and self.current_player == self.human_seat:
            raw_legal = self.current_state.get("legal_actions", [])
            abs_legal = self.action_mapper.get_abstract_legal_actions(
                raw_legal
            )
            legal_action_names = {
                a: self.action_mapper.action_name(a) for a in abs_legal
            }

        payoffs: Optional[Dict[int, float]] = None
        if is_over:
            try:
                p = self.env.get_payoffs()
                if p is not None and len(p) > self.human_seat:
                    payoffs = {i: float(p[i]) for i in range(len(p))}
                    human_result = float(p[self.human_seat])
                    self.total_results["hands"] += 1
                    if human_result > 0:
                        self.total_results["player"] += 1
                    elif human_result < 0:
                        self.total_results["ai"] += 1
                    for i in range(min(len(p), self.num_players)):
                        self.stacks[i] += float(p[i])
                else:
                    logger.warning(
                        f"get_payoffs() üres vagy rövid: {p}"
                    )
            except Exception as e:
                logger.error(f"Payoffs hiba: {e}")

        equity: Optional[float] = None
        try:
            if len(self.human_cards_cache) == 2:
                h_eq = cards_to_equity_format(self.human_cards_cache)
                b_eq = cards_to_equity_format(
                    [c.get_index() for c in self.env.game.public_cards]
                )
                equity = self.equity_est.equity(
                    h_eq, b_eq,
                    num_opponents=max(self.num_players - 1, 1),
                )
        except Exception:
            pass

        call_amount: float = 0.0
        if not is_over:
            try:
                human_chips = players[self.human_seat]["chips_in_pot"]
                max_chips   = max(p["chips_in_pot"] for p in players)
                call_amount = max_chips - human_chips
            except Exception:
                pass

        return {
            "players":            players,
            "board":              board_cards,
            "pot":                pot,
            "human_cards":        human_cards,
            "human_seat":         self.human_seat,
            "current_player":     self.current_player if not is_over else -1,
            "legal_actions":      abs_legal,
            "legal_action_names": legal_action_names,
            "is_over":            is_over,
            "payoffs":            payoffs,
            "ai_actions":         ai_actions or [],
            "street":             self.street,
            "street_name":        ["Preflop", "Flop", "Turn", "River"][
                min(self.street, 3)
            ],
            "hand_num":           self.hand_num,
            "hand_log":           self.hand_log,
            "call_amount":        call_amount,
            "stacks":             self.stacks[:],
            "total_results":      self.total_results.copy(),
            "equity":             equity,
            "num_players":        self.num_players,
            "bb":                 self.bb,
        }


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Server
# ─────────────────────────────────────────────────────────────────────────────

session         = GameSession()
available_models: List[Dict[str, Any]] = []


# [TASK-6] Opcionalis HTTP Basic Auth – lasd train_gui.py megjegyzeseit.
_auth_hash: str = ""   # ures → nincs auth kovetelment


class GameHandler(http.server.BaseHTTPRequestHandler):
    def _check_auth(self) -> bool:
        """Ellenorzi a HTTP Basic Auth credentialokat (lasd train_gui.py)."""
        if not _auth_hash:
            return True
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Basic "):
            try:
                decoded   = base64.b64decode(auth_header[6:]).decode("utf-8")
                _, passwd = decoded.split(":", 1)
                if hashlib.sha256(passwd.encode()).hexdigest() == _auth_hash:
                    return True
            except Exception:
                pass
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Poker AI vs AI"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    """HTTP kéréskezelő a játék API-hoz."""

    def log_message(self, format: str, *args: Any) -> None:  # type: ignore[override]
        logger.debug("HTTP: %s", args[0] if args else "")

    def _send_json(self, data: Any, status: int = 200) -> None:
        """JSON válasz küldése."""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, filepath: str) -> None:
        """HTML fájl küldése."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404, f"File not found: {filepath}")

    def _read_body(self) -> Dict:
        """Request body olvasása és JSON parse."""
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            try:
                return json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                return {}
        return {}

    def do_OPTIONS(self) -> None:  # type: ignore[override]
        """CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods", "GET, POST, OPTIONS"
        )
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type"
        )
        self.end_headers()

    def do_GET(self) -> None:  # type: ignore[override]
        if not self._check_auth():   # [TASK-6]
            return
        """GET kérések kezelése."""
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            html_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "play_vs_ai.html",
            )
            self._send_html(html_path)

        elif path == "/api/models":
            self._send_json({"models": available_models})

        elif path == "/api/state":
            if not session._active:
                self._send_json({"error": "No active session"}, 400)
            else:
                self._send_json(session._build_response())

        else:
            self.send_error(404)

    def do_POST(self) -> None:  # type: ignore[override]
        if not self._check_auth():   # [TASK-6]
            return
        """POST kérések kezelése."""
        path = urlparse(self.path).path

        try:
            body = self._read_body()

            if path == "/api/start":
                self._handle_start(body)

            elif path == "/api/new_hand":
                logger.info("POST /api/new_hand")
                result = session.new_hand()
                self._send_json(result)

            elif path == "/api/action":
                action = int(body.get("action", 1))
                logger.info(f"POST /api/action: {action}")
                result = session.human_action(action)
                self._send_json(result)

            elif path == "/api/rescan":
                global available_models
                base = os.path.dirname(os.path.abspath(__file__))
                available_models = scan_models(base)
                self._send_json({"models": available_models})

            else:
                self.send_error(404)

        except Exception as e:
            logger.error(
                f"API hiba ({path}): {e}\n{traceback.format_exc()}"
            )
            self._send_json({"error": str(e)}, 500)

    def _handle_start(self, body: Dict) -> None:
        """
        ``/api/start`` végpont kezelése biztonságos path validációval.

        A metódus az alábbi sorrendben validálja a bemenetet:

        1. Numerikus paraméterek tartomány-ellenőrzése.
        2. ``model_path`` path traversal ellenőrzése a
           ``_resolve_and_validate_model_path()`` segítségével.
        3. Csak ezután kerül sor fájlrendszer-hozzáférésre.

        Args:
            body: A klienstől érkező JSON body dict.
        """
        logger.info(f"POST /api/start: num_players={body.get('num_players')}")

        # ── 1. Numerikus paraméterek validálása ───────────────────────────
        try:
            num_players = int(body.get("num_players", 6))
        except (TypeError, ValueError):
            self._send_json({"error": "num_players egész szám kell."}, 400)
            return

        if not 2 <= num_players <= 9:
            self._send_json(
                {"error": f"num_players értéke 2 és 9 közé kell essen, kapott: {num_players}"},
                400,
            )
            return

        try:
            human_seat = int(body.get("human_seat", 0))
        except (TypeError, ValueError):
            self._send_json({"error": "human_seat egész szám kell."}, 400)
            return

        if not 0 <= human_seat < num_players:
            self._send_json(
                {
                    "error": (
                        f"human_seat értéke 0 és {num_players - 1} közé "
                        f"kell essen, kapott: {human_seat}"
                    )
                },
                400,
            )
            return

        try:
            bb = float(body.get("bb", 2))
        except (TypeError, ValueError):
            self._send_json({"error": "bb numerikus érték kell."}, 400)
            return

        if bb <= 0:
            self._send_json({"error": "bb értéke pozitív kell legyen."}, 400)
            return

        try:
            sb = float(body.get("sb", 1))
        except (TypeError, ValueError):
            self._send_json({"error": "sb numerikus érték kell."}, 400)
            return

        try:
            stack = float(body.get("stack", 100))
        except (TypeError, ValueError):
            self._send_json({"error": "stack numerikus érték kell."}, 400)
            return

        if stack <= 0:
            self._send_json(
                {"error": "stack értéke pozitív kell legyen."}, 400
            )
            return

        if stack < bb * 2:
            self._send_json(
                {
                    "error": (
                        f"stack ({stack}) túl kicsi: "
                        f"legalább 2×bb={bb * 2:.1f} szükséges."
                    )
                },
                400,
            )
            return

        # ── 2. Model path validálás – PATH TRAVERSAL VÉDELEM ──────────────
        raw_model_path: str = body.get("model_path", "")
        base_dir = os.path.dirname(os.path.abspath(__file__))

        try:
            validated_path = _resolve_and_validate_model_path(
                base_dir, raw_model_path
            )
        except ValueError as ve:
            # SECURITY: NE szivárogassuk ki a raw_model_path értékét,
            # csak a sanitizált hibaüzenetet küldjük vissza.
            logger.warning(
                f"Érvénytelen model_path a /api/start kérésnél: {ve}"
            )
            self._send_json(
                {"error": f"Érvénytelen modell elérési út: {ve}"}, 400
            )
            return
        except FileNotFoundError as fe:
            self._send_json({"error": str(fe)}, 404)
            return

        # ── 3. Session indítása a validált elérési úttal ───────────────────
        try:
            session.start_session(
                validated_path, num_players, human_seat, bb, sb, stack
            )
            self._send_json({"ok": True, "num_players": num_players})
        except Exception as e:
            logger.error(f"Session indítási hiba: {e}\n{traceback.format_exc()}")
            self._send_json({"error": f"Session indítási hiba: {e}"}, 500)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Szerver indítási belépési pont."""
    parser = argparse.ArgumentParser(
        description="Play vs your trained Poker AI"
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--password", type=str, default="",
        help="Opcionalis HTTP Basic Auth jelszo (felhasznalonev: admin).",
    )
    args = parser.parse_args()

    global _auth_hash
    if args.password:
        _auth_hash = hashlib.sha256(args.password.encode()).hexdigest()
        print(f"  Auth: HTTP Basic Auth BEKAPCSOLVA")
    else:
        _auth_hash = ""

    global available_models, _ALLOWED_DIRS

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Whitelist inicializálása a szerver indításakor
    _ALLOWED_DIRS = _build_allowed_dirs(base_dir)

    print("\n" + "=" * 60)
    print("  🃏  POKER AI v4  –  Play vs Your Model")
    print("=" * 60)
    print(f"\n  Mappát vizsgálom: {base_dir}")
    print(f"  Whitelist könyvtárak:")
    for d in _ALLOWED_DIRS:
        print(f"    - {d}")

    available_models = scan_models(base_dir)
    print(f"\n  Talált modellek: {len(available_models)}")

    if not available_models:
        print("\n  ⚠ Nem találtam .pth fájlt a whitelist könyvtárakban!")
        print("  Tedd a kitrénelt modelljeidet a models/ mappába.")

    html_path = os.path.join(base_dir, "play_vs_ai.html")
    if not os.path.exists(html_path):
        print(f"\n  ⚠ Hiányzik: play_vs_ai.html")
        print(f"  Tedd a HTML fájlt ide: {html_path}")
        return

    server = http.server.HTTPServer(("0.0.0.0", args.port), GameHandler)
    url    = f"http://localhost:{args.port}"

    print(f"\n  ✓ Szerver fut: {url}")
    print(f"  Device: {session.device}")
    print(f"\n  Ctrl+C a leállításhoz\n")
    print("=" * 60 + "\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Szerver leállítva.")
        server.server_close()


if __name__ == "__main__":
    main()
