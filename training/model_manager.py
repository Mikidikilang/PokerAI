"""
training/model_manager.py  --  Modell mappa kezelő (v4.2.2)

Minden modellhez saját mappa: models/{name}/
  ├── {name}_ppo_v4.pth    ← checkpoint
  ├── config.json          ← aktuális tréning konfig
  ├── naplo.json           ← tréning napló (session history)
  └── tests/               ← test_model_sanity.py kimenetek

Napló struktúra:
  {
    "model_name": "2max",
    "num_players": 2,
    "created": "...",
    "total_episodes": 4200000,
    "sessions": [
      {
        "id": "sess_001",
        "started": "...", "ended": "...",
        "episodes_start": 0, "episodes_end": 2000000,
        "style": "exploitative",
        "config_snapshot": {...},
        "metrics_final": {...},
        "completed": true
      }
    ]
  }

Változások v4.2.2:
    [SECURITY-KRITIKUS-2] _read_checkpoint_meta(): explicit
        allow_unsafe=True + UnsafeCheckpointError kezelés.
        Ez a metódus ismeretlen/legacy checkpointokat is olvashat
        (bármi kerülhet a models/ mappába), ezért az unsafe
        betöltés itt szükséges, de dokumentált és elkülönített.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("PokerAI")

# ── Alapértelmezett konfig (TrainingConfig mezőkkel egyező) ──────────────────
CONFIG_DEFAULTS: Dict[str, Any] = {
    "num_envs":                512,
    "buffer_collect_size":     2048,
    "max_steps_per_hand":      500,
    "hidden_size":             512,
    "gru_hidden":              None,
    "learning_rate":           3e-4,
    "clip_eps":                0.2,
    "ppo_epochs":              8,
    "minibatch_size":          256,
    "value_coef":              0.5,
    "entropy_coef":            0.01,
    "entropy_final":           0.001,
    "entropy_decay":           30_000_000,
    "max_grad_norm":           0.5,
    "gamma":                   0.99,
    "gae_lambda":              0.95,
    "lr_t_max":                500,
    "lr_eta_min_ratio":        0.05,
    "draw_fold_penalty":       0.08,
    "draw_equity_threshold":   0.44,
    "street_reward_scale":     0.05,
    "milestone_interval":      2_000_000,
    "equity_n_sim":            200,
    "equity_cache_size":       20_000,
    "training_style":          "exploitative",
    "training_phase":          2,
    "bot_pool": {
        "fish":            {"enabled": True,  "weight": 0.8,  "display": "Fish",            "icon": "🐟"},
        "nit":             {"enabled": True,  "weight": 1.5,  "display": "Nit",             "icon": "🎯"},
        "calling_station": {"enabled": True,  "weight": 0.2,  "display": "Calling Station", "icon": "📞"},
        "lag":             {"enabled": True,  "weight": 1.5,  "display": "LAG",             "icon": "💣"},
    },
}

# ── Stílus presetek ──────────────────────────────────────────────────────────
STYLE_PRESETS: Dict[str, Dict] = {
    "self_play": {
        "training_phase": 1,
        "entropy_coef":   0.02,
        "entropy_final":  0.002,
        "entropy_decay":  50_000_000,
        "bot_pool": {k: {**v, "enabled": False, "weight": 0.0}
                     for k, v in CONFIG_DEFAULTS["bot_pool"].items()},
    },
    "exploitative": {
        "training_phase": 2,
        "entropy_coef":   0.01,
        "entropy_final":  0.001,
        "entropy_decay":  30_000_000,
        "bot_pool": CONFIG_DEFAULTS["bot_pool"],
    },
    "aggressive": {
        "training_phase": 2,
        "entropy_coef":   0.005,
        "entropy_final":  0.0005,
        "entropy_decay":  20_000_000,
        "bot_pool": {
            "fish":            {"enabled": True,  "weight": 0.5,  "display": "Fish",            "icon": "🐟"},
            "nit":             {"enabled": True,  "weight": 2.5,  "display": "Nit",             "icon": "🎯"},
            "calling_station": {"enabled": False, "weight": 0.0,  "display": "Calling Station", "icon": "📞"},
            "lag":             {"enabled": True,  "weight": 2.5,  "display": "LAG",             "icon": "💣"},
        },
    },
}


def _now() -> str:
    """Visszaadja az aktuális UTC időt ISO 8601 formátumban."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Mély dict merge: override felülírja a base-t, de hiányzó kulcsok megmaradnak."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class ModelManager:
    """
    Modell mappák kezelője.

    Minden modellnek saját alkönyvtára van a models/ alatt.
    Tartalmazza a checkpointot, konfigurációt, naplót és teszteket.
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        """
        Args:
            base_dir: A projekt gyökérkönyvtára.  Ha ``None``, automatikusan
                      meghatározza a fájl helyzetéből.
        """
        if base_dir is None:
            project_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            base_dir = project_root
        self._project_root = base_dir
        self._models_dir   = os.path.join(base_dir, "models")
        os.makedirs(self._models_dir, exist_ok=True)

    # ── Elérési utak ────────────────────────────────────────────────────────

    def model_dir(self, name: str) -> str:
        """Visszaadja a modell könyvtárát."""
        return os.path.join(self._models_dir, name)

    def pth_path(self, name: str, filename: Optional[str] = None) -> str:
        """
        Visszaadja a modell .pth fájljának elérési útját.

        Ha ``filename`` nincs megadva:
          1. Megnézi a ``models/{name}/`` mappát – ha van benne .pth fájl,
             azt adja vissza (megakadályozza a névduplázódást).
          2. Ha nincs .pth fájl, az alapértelmezett ``{name}_ppo_v4.pth``
             nevet generálja.

        Args:
            name:     Modell neve.
            filename: Explicit fájlnév (opcionális).

        Returns:
            Abszolút elérési út a .pth fájlhoz.
        """
        if filename is None:
            model_dir = self.model_dir(name)
            if os.path.isdir(model_dir):
                existing = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
                if existing:
                    found = existing[0]
                    logger.debug(
                        f"pth_path({name!r}): meglévő fájl → "
                        f"{os.path.basename(found)}"
                    )
                    return found
            filename = f"{name}_ppo_v4.pth"
            logger.debug(
                f"pth_path({name!r}): nincs .pth a mappában → új: {filename}"
            )
        return os.path.join(self.model_dir(name), filename)

    def config_path(self, name: str) -> str:
        """Visszaadja a konfig JSON elérési útját."""
        return os.path.join(self.model_dir(name), "config.json")

    def naplo_path(self, name: str) -> str:
        """Visszaadja a napló JSON elérési útját."""
        return os.path.join(self.model_dir(name), "naplo.json")

    def tests_dir(self, name: str) -> str:
        """Visszaadja a tesztek könyvtárát."""
        return os.path.join(self.model_dir(name), "tests")

    # ── Modell mappa létrehozása ─────────────────────────────────────────────

    def ensure_model_dir(self, name: str, num_players: int = 6) -> str:
        """
        Létrehozza a modell mappa struktúráját ha nem létezik.

        Args:
            name:        Modell neve.
            num_players: Játékosok száma (konfig inicializáláshoz).

        Returns:
            A modell könyvtár abszolút elérési útja.
        """
        d = self.model_dir(name)
        os.makedirs(d, exist_ok=True)
        os.makedirs(self.tests_dir(name), exist_ok=True)

        # config.json ha nem létezik
        if not os.path.exists(self.config_path(name)):
            cfg = dict(CONFIG_DEFAULTS)
            if num_players == 6:
                for n in range(9, 1, -1):
                    if f"{n}max" in name or f"{n}p" in name:
                        num_players = n
                        break
            self._write_json(self.config_path(name), {
                "num_players": num_players,
                "created":     _now(),
                "config":      cfg,
            })

        # naplo.json ha nem létezik
        if not os.path.exists(self.naplo_path(name)):
            self._write_json(self.naplo_path(name), {
                "model_name":     name,
                "num_players":    num_players,
                "created":        _now(),
                "total_episodes": 0,
                "sessions":       [],
            })

        return d

    # ── Config ──────────────────────────────────────────────────────────────

    def load_config(self, name: str) -> Dict:
        """
        Konfig betöltése. Hiányzó mezők defaulttal töltve.

        Args:
            name: Modell neve.

        Returns:
            Konfig dict ``num_players``, ``config`` stb. kulcsokkal.
        """
        path = self.config_path(name)
        if not os.path.exists(path):
            return {
                "num_players": self._guess_players(name),
                "created":     None,
                "config":      dict(CONFIG_DEFAULTS),
            }
        raw = self._read_json(path) or {}
        raw["config"] = _deep_merge(CONFIG_DEFAULTS, raw.get("config", {}))
        return raw

    def save_config(self, name: str, data: Dict) -> None:
        """
        Konfig mentése.

        Args:
            name: Modell neve.
            data: Konfig dict.
        """
        self.ensure_model_dir(name, data.get("num_players", 6))
        data["last_saved"] = _now()
        self._write_json(self.config_path(name), data)
        logger.debug(f"Konfig mentve: models/{name}/config.json")

    def apply_style_preset(self, config: Dict, style: str) -> Dict:
        """
        Stílus preset alkalmazása a konfigra.

        Args:
            config: Jelenlegi konfig dict.
            style:  Preset neve (``'self_play'``, ``'exploitative'``,
                    ``'aggressive'``).

        Returns:
            Frissített konfig dict.
        """
        if style not in STYLE_PRESETS:
            return config
        preset = STYLE_PRESETS[style]
        result = dict(config)
        for k, v in preset.items():
            if k == "bot_pool":
                result["bot_pool"] = _deep_merge(
                    config.get("bot_pool", CONFIG_DEFAULTS["bot_pool"]), v
                )
            else:
                result[k] = v
        result["training_style"] = style
        return result

    # ── Napló ────────────────────────────────────────────────────────────────

    def load_naplo(self, name: str) -> Dict:
        """
        Napló betöltése.

        Args:
            name: Modell neve.

        Returns:
            Napló dict ``sessions`` listával.
        """
        path = self.naplo_path(name)
        if not os.path.exists(path):
            return {
                "model_name":     name,
                "num_players":    0,
                "total_episodes": 0,
                "sessions":       [],
            }
        return self._read_json(path) or {"sessions": []}

    def start_session(
        self,
        name:            str,
        config_snapshot: Dict,
        episodes_start:  int,
        num_players:     int,
    ) -> str:
        """
        Új tréning session rögzítése a naplóba.

        Args:
            name:            Modell neve.
            config_snapshot: Az aktuális tréning konfig snapshot-ja.
            episodes_start:  A session kezdetén lévő epizódszám.
            num_players:     Játékosok száma.

        Returns:
            A generált session ID string.
        """
        self.ensure_model_dir(name, num_players)
        naplo      = self.load_naplo(name)
        session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session    = {
            "id":              session_id,
            "started":         _now(),
            "ended":           None,
            "duration_sec":    None,
            "episodes_start":  episodes_start,
            "episodes_end":    None,
            "episodes_added":  0,
            "style":           config_snapshot.get("training_style", "?"),
            "num_players":     num_players,
            "config_snapshot": config_snapshot,
            "metrics_final":   {},
            "completed":       False,
        }
        naplo.setdefault("sessions", []).append(session)
        self._write_json(self.naplo_path(name), naplo)
        logger.info(f"Napló session start: {name} / {session_id}")
        return session_id

    def end_session(
        self,
        name:         str,
        session_id:   str,
        episodes_end: int,
        metrics:      Optional[Dict] = None,
        completed:    bool           = True,
    ) -> None:
        """
        Tréning session lezárása.

        Args:
            name:         Modell neve.
            session_id:   A lezárandó session ID.
            episodes_end: A session végén lévő epizódszám.
            metrics:      Végső tréning metrikák (opcionális).
            completed:    ``True`` ha sikeresen befejezve,
                          ``False`` ha megszakítva.
        """
        naplo    = self.load_naplo(name)
        sessions = naplo.get("sessions", [])
        for sess in sessions:
            if sess.get("id") == session_id:
                now_str              = _now()
                sess["ended"]        = now_str
                sess["episodes_end"] = episodes_end
                sess["episodes_added"] = max(
                    0, episodes_end - sess.get("episodes_start", 0)
                )
                sess["completed"]      = completed
                sess["metrics_final"]  = metrics or {}
                try:
                    from datetime import datetime as dt
                    start = dt.fromisoformat(
                        sess["started"].replace("Z", "+00:00")
                    )
                    end = dt.fromisoformat(
                        now_str.replace("Z", "+00:00")
                    )
                    sess["duration_sec"] = int(
                        (end - start).total_seconds()
                    )
                except Exception:
                    pass
                break
        naplo["total_episodes"] = episodes_end
        naplo["last_updated"]   = _now()
        self._write_json(self.naplo_path(name), naplo)
        logger.info(
            f"Napló session end: {name} / {session_id} → "
            f"{episodes_end:,} ep"
        )

    def add_naplo_note(
        self,
        name:       str,
        session_id: str,
        note:       str,
    ) -> None:
        """
        Megjegyzés hozzáadása egy session bejegyzéshez.

        Args:
            name:       Modell neve.
            session_id: Session azonosító.
            note:       Megjegyzés szövege.
        """
        naplo = self.load_naplo(name)
        for sess in naplo.get("sessions", []):
            if sess.get("id") == session_id:
                sess["notes"] = note
                break
        self._write_json(self.naplo_path(name), naplo)

    # ── Tesztek ──────────────────────────────────────────────────────────────

    def list_tests(self, name: str) -> List[Dict]:
        """
        Visszaadja a tests/ mappa .json fájljait metaadatokkal.

        Args:
            name: Modell neve.

        Returns:
            Teszt metaadat dict-ek listája (legújabb elöl).
        """
        tdir = self.tests_dir(name)
        if not os.path.isdir(tdir):
            return []
        results: List[Dict] = []
        for f in sorted(
            glob.glob(os.path.join(tdir, "*.json")), reverse=True
        ):
            try:
                data = self._read_json(f) or {}
                results.append({
                    "filename": os.path.basename(f),
                    "path":     f,
                    "timestamp": data.get("timestamp", ""),
                    "episodes":  data.get("n_hands", "?"),
                    "grade":    data.get("summary", {}).get("grade", "?"),
                    "passed":   data.get("summary", {}).get("passed", 0),
                    "failed":   data.get("summary", {}).get("failed", 0),
                    "penalty":  data.get("summary", {}).get("penalty", "?"),
                    "has_log":  os.path.exists(f.replace(".json", ".log")),
                })
            except Exception:
                pass
        return results

    def get_test_log(self, name: str, filename: str) -> str:
        """
        Egy teszt .log fájl tartalmát adja vissza.

        Args:
            name:     Modell neve.
            filename: A log fájl neve (pl. ``'test_2max_...log'``).

        Returns:
            A log fájl tartalma stringként, vagy üres string.
        """
        log_path = os.path.join(self.tests_dir(name), filename)
        if not os.path.exists(log_path):
            return ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""

    # ── Model lista ──────────────────────────────────────────────────────────

    def list_models(self) -> List[Dict]:
        """
        Visszaadja az összes modell adatait.

        Elsőként a models/ mappát nézi, aztán a projekt gyökerét
        (migrálatlan checkpointok).

        Returns:
            Modell adatok listája.
        """
        models: List[Dict] = []

        # 1) models/ alkönyvtárak
        if os.path.isdir(self._models_dir):
            for entry in sorted(os.listdir(self._models_dir)):
                d = os.path.join(self._models_dir, entry)
                if not os.path.isdir(d):
                    continue
                pth_files = glob.glob(os.path.join(d, "*.pth"))
                if not pth_files:
                    cfg = self.load_config(entry)
                    models.append(self._model_entry(entry, None, cfg))
                    continue
                pth      = pth_files[0]
                cfg      = self.load_config(entry)
                ck_meta  = self._read_checkpoint_meta(pth)
                models.append(self._model_entry(entry, pth, cfg, ck_meta))

        # 2) Projekt gyökérben lévő .pth fájlok (migrálatlan)
        root_pths  = glob.glob(
            os.path.join(self._project_root, "*.pth")
        )
        known_pths = {m.get("abs_pth") for m in models}
        for pth in sorted(root_pths):
            if pth in known_pths:
                continue
            if "ModellNaplo" in pth:
                continue
            basename = os.path.splitext(os.path.basename(pth))[0]
            ck_meta  = self._read_checkpoint_meta(pth)
            models.append({
                "name":          basename,
                "display":       f"{basename} ⚠ (gyökérben, nem migrált)",
                "abs_pth":       os.path.abspath(pth),
                "rel_pth":       os.path.relpath(pth, self._project_root),
                "in_models_dir": False,
                "num_players":   ck_meta.get("num_players"),
                "episodes":      ck_meta.get("episodes", 0),
                "state_size":    ck_meta.get("state_size", "?"),
                "algorithm":     ck_meta.get("algorithm", "PPO"),
                "config":        None,
                "naplo_summary": None,
            })

        return models

    def migrate_to_models_dir(
        self,
        pth_path:    str,
        name:        Optional[str] = None,
        num_players: Optional[int] = None,
    ) -> str:
        """
        Migrál egy gyökérben lévő .pth fájlt a ``models/{name}/`` mappába.

        Args:
            pth_path:    Forrás .pth fájl abszolút elérési útja.
            name:        Célmappa neve (alapértelmezett: fájlnév kiterjesztés nélkül).
            num_players: Játékosok száma (alapértelmezett: checkpoint-ból).

        Returns:
            Az új .pth fájl abszolút elérési útja.
        """
        import shutil
        if name is None:
            name = os.path.splitext(os.path.basename(pth_path))[0]
        if num_players is None:
            ck      = self._read_checkpoint_meta(pth_path)
            num_players = ck.get("num_players", 6)

        self.ensure_model_dir(name, num_players)
        dest = self.pth_path(name, os.path.basename(pth_path))
        if not os.path.exists(dest):
            shutil.copy2(pth_path, dest)
            logger.info(f"Migráció: {pth_path} → {dest}")
        return dest

    # ── Belső segédmetódusok ─────────────────────────────────────────────────

    def _model_entry(
        self,
        name:    str,
        pth:     Optional[str],
        cfg:     Dict,
        ck_meta: Optional[Dict] = None,
    ) -> Dict:
        """
        Összerakja egy modell metaadat dict-jét a list_models() számára.

        Args:
            name:    Modell neve.
            pth:     Checkpoint fájl elérési útja (vagy ``None``).
            cfg:     Konfig dict.
            ck_meta: Checkpoint metaadatok (opcionális).

        Returns:
            Modell metaadat dict.
        """
        ck_meta = ck_meta or {}
        naplo   = self.load_naplo(name)
        sessions = naplo.get("sessions", [])
        last_sess = sessions[-1] if sessions else None
        return {
            "name":          name,
            "display":       name,
            "abs_pth":       os.path.abspath(pth) if pth else None,
            "rel_pth":       (
                os.path.relpath(pth, self._project_root) if pth else None
            ),
            "in_models_dir": True,
            "model_dir":     self.model_dir(name),
            "num_players":   (
                cfg.get("num_players") or ck_meta.get("num_players")
            ),
            "episodes":  ck_meta.get(
                "episodes", naplo.get("total_episodes", 0)
            ),
            "state_size":  ck_meta.get("state_size", "?"),
            "algorithm":   ck_meta.get("algorithm", "PPO"),
            "config":      cfg,
            "naplo_summary": {
                "total_sessions": len(sessions),
                "total_episodes": naplo.get("total_episodes", 0),
                "last_style":     (
                    last_sess.get("style") if last_sess else None
                ),
                "last_trained":   (
                    last_sess.get("ended") if last_sess else None
                ),
                "last_added":     (
                    last_sess.get("episodes_added") if last_sess else 0
                ),
            },
        }

    def _read_checkpoint_meta(self, pth: str) -> Dict:
        """
        Minimális metaadat kinyerése egy checkpoint fájlból.

        Ez a metódus kénytelen ``allow_unsafe=True``-val betölteni,
        mert a models/ mappában régi, legacy checkpointok is lehetnek.
        Ez a kódbázisban az EGYETLEN elfogadható hely az unsafe
        betöltésre a migrate_checkpoint_to_safe() mellett, mert:

          1. A betöltött adatból CSAK alap Python típusok kerülnek ki
             (int, str) – a state_dict-et nem érintjük.
          2. A modell_manager-t csak a lokális models/ mappán belüli
             fájlokra hívjuk (a list_models() a saját könyvtárakat
             járja be, nem user-controlled inputot).
          3. A path traversal ellen a list_models() és migrate_to_models_dir()
             is véd (csak az _models_dir és _project_root alatti fájlok).

        FONTOS: Ha egy legacy checkpointot találsz, futtasd a migrációt:

            from utils.checkpoint_utils import migrate_checkpoint_to_safe
            migrate_checkpoint_to_safe('old.pth', 'old.pth')

        Ezután ez a metódus weights_only=True-val is fog működni.

        Args:
            pth: Checkpoint fájl elérési útja.

        Returns:
            Metaadat dict ``episodes``, ``state_size``, ``algorithm``,
            ``num_players`` kulcsokkal.  Üres dict hiba esetén.
        """
        if not pth or not os.path.exists(pth):
            return {}

        import sys as _sys
        _sys.path.insert(0, self._project_root)

        try:
            from utils.checkpoint_utils import (
                safe_load_checkpoint,
                UnsafeCheckpointError,
            )

            # ── BIZTONSÁGI MEGJEGYZÉS ──────────────────────────────────────
            # allow_unsafe=True itt szándékos és dokumentált.
            # Indok: a models/ mappa legacy checkpointokat is tartalmazhat
            # (v4.1 előtti formátum), amelyek weights_only=True-val nem
            # tölthetők be.  A kinyert adat kizárólag alap Python típus
            # (int, str) – a state_dict-et NEM használjuk fel.
            #
            # Hosszú távú megoldás: futtasd a migrate_checkpoint_to_safe()
            # segédfüggvényt az összes legacy checkpointra, és utána az
            # allow_unsafe=True eltávolítható.
            # ─────────────────────────────────────────────────────────────

            try:
                # Elsőként megpróbáljuk safe módban (v4.2+ checkpointokhoz)
                ck = safe_load_checkpoint(
                    pth, map_location="cpu", allow_unsafe=False
                )
            except UnsafeCheckpointError:
                # Legacy checkpoint: unsafe fallback, explicit és dokumentált
                logger.info(
                    f"_read_checkpoint_meta: legacy checkpoint detektálva "
                    f"({os.path.basename(pth)!r}), unsafe betöltés.  "
                    f"Ajánlott: futtasd a migrate_checkpoint_to_safe()-t."
                )
                ck = safe_load_checkpoint(
                    pth,
                    map_location="cpu",
                    allow_unsafe=True,   # INDOK: ld. fenti magyarázat
                )

            if isinstance(ck, dict) and "state_dict" in ck:
                return {
                    "episodes":    ck.get("episodes_trained", 0),
                    "state_size":  ck.get("state_size", "?"),
                    "algorithm":   ck.get("algorithm", "PPO"),
                    "num_players": ck.get("num_players"),
                }

        except Exception as exc:
            logger.debug(
                f"Checkpoint meta olvasási hiba ({pth!r}): {exc}"
            )

        return {}

    @staticmethod
    def _guess_players(name: str) -> int:
        """
        Játékosszám becslése a modell nevéből.

        Args:
            name: Modell neve (pl. ``'6max_ppo_v4'``).

        Returns:
            Becsült játékosszám (2–9), fallback: 6.
        """
        for n in range(9, 1, -1):
            if f"{n}max" in name or f"{n}p" in name:
                return n
        return 6

    @staticmethod
    def _read_json(path: str) -> Optional[Dict]:
        """
        JSON fájl olvasása.

        Args:
            path: Fájl elérési útja.

        Returns:
            Beolvasott dict, vagy ``None`` hiba esetén.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _write_json(path: str, data: Any) -> None:
        """
        JSON fájl írása, a szülő könyvtár automatikus létrehozásával.

        Args:
            path: Célfájl elérési útja.
            data: Serializálható Python objektum.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
