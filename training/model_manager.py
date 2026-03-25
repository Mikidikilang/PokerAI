"""
training/model_manager.py  --  Modell mappa kezelő (v4.3 CONFIG-FULL)

[CONFIG-FULL v4.3] CONFIG_DEFAULTS bővítve az összes új TrainingConfig mezővel:
  - lr_scheduler, reset_optimizer_on_load
  - allin_penalty_*, fold_bonus_*, stack_blindness_penalty_*

Minden új modell config.json-ja automatikusan tartalmazza ezeket a default értékekkel.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("PokerAI")

# ── Alapértelmezett konfig – MINDIG szinkronban kell legyen TrainingConfig-gal ──
CONFIG_DEFAULTS: Dict[str, Any] = {
    # Env & collection
    "num_envs":                512,
    "buffer_collect_size":     2048,
    "max_steps_per_hand":      500,

    # Modell architektúra
    "hidden_size":             512,
    "gru_hidden":              None,

    # PPO hiperparaméterek
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

    # LR Scheduler
    "lr_scheduler":            "cosine",
    "lr_t_max":                500,
    "lr_eta_min_ratio":        0.05,

    # Checkpoint / optimizer reset
    "reset_optimizer_on_load": False,

    # Reward shaping – meglévő
    "draw_fold_penalty":       0.08,
    "draw_equity_threshold":   0.44,
    "street_reward_scale":     0.05,

    # Reward shaping – All-in spam büntetés (ÚJ)
    "allin_penalty_enabled":              False,
    "allin_penalty_equity_threshold":     0.45,
    "allin_penalty_amount":               0.15,

    # Reward shaping – Fold bónusz (ÚJ)
    "fold_bonus_enabled":                 False,
    "fold_bonus_equity_threshold":        0.38,
    "fold_bonus_amount":                  0.05,

    # Reward shaping – Stack-blindness büntetés (ÚJ)
    "stack_blindness_penalty_enabled":    False,
    "stack_blindness_bb_threshold":       15.0,
    "stack_blindness_penalty_amount":     0.10,

    # Mérföldkő
    "milestone_interval":      2_000_000,
    "equity_n_sim":            100,
    "equity_cache_size":       100_000,
    "training_style":          "exploitative",
    "training_phase":          2,

    # Bot pool
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
        "lr_scheduler":   "cosine",
        "reset_optimizer_on_load": False,
        # Büntetések kikapcsolt self-play-ben
        "allin_penalty_enabled": False,
        "fold_bonus_enabled": False,
        "stack_blindness_penalty_enabled": False,
        "bot_pool": {k: {**v, "enabled": False, "weight": 0.0}
                     for k, v in CONFIG_DEFAULTS["bot_pool"].items()},
    },
    "exploitative": {
        "training_phase": 2,
        "entropy_coef":   0.01,
        "entropy_final":  0.001,
        "entropy_decay":  30_000_000,
        "lr_scheduler":   "cosine",
        "reset_optimizer_on_load": False,
        "allin_penalty_enabled": False,
        "fold_bonus_enabled": False,
        "stack_blindness_penalty_enabled": False,
        "bot_pool": CONFIG_DEFAULTS["bot_pool"],
    },
    "aggressive": {
        "training_phase": 2,
        "entropy_coef":   0.005,
        "entropy_final":  0.0005,
        "entropy_decay":  20_000_000,
        "lr_scheduler":   "cosine",
        "reset_optimizer_on_load": False,
        "allin_penalty_enabled": False,
        "fold_bonus_enabled": False,
        "stack_blindness_penalty_enabled": False,
        "bot_pool": {
            "fish":            {"enabled": True,  "weight": 0.5,  "display": "Fish",            "icon": "🐟"},
            "nit":             {"enabled": True,  "weight": 2.5,  "display": "Nit",             "icon": "🎯"},
            "calling_station": {"enabled": False, "weight": 0.0,  "display": "Calling Station", "icon": "📞"},
            "lag":             {"enabled": True,  "weight": 2.5,  "display": "LAG",             "icon": "💣"},
        },
    },
    # [ÚJ] Degenerált modell javítása – minden büntetés bekapcsolt
    "degeneration_fix": {
        "training_phase":  2,
        "entropy_coef":    0.01,
        "entropy_final":   0.001,
        "entropy_decay":   900_000,          # gyors konvergencia
        "lr_scheduler":    "linear",          # egyirányú csökkentés
        "reset_optimizer_on_load": True,      # friss optimizer
        "allin_penalty_enabled":           True,
        "allin_penalty_equity_threshold":  0.45,
        "allin_penalty_amount":            0.15,
        "fold_bonus_enabled":              True,
        "fold_bonus_equity_threshold":     0.38,
        "fold_bonus_amount":               0.05,
        "stack_blindness_penalty_enabled": True,
        "stack_blindness_bb_threshold":    15.0,
        "stack_blindness_penalty_amount":  0.10,
        "bot_pool": {
            "fish":            {"enabled": True,  "weight": 0.8,  "display": "Fish",            "icon": "🐟"},
            "nit":             {"enabled": True,  "weight": 0.6,  "display": "Nit",             "icon": "🎯"},
            "calling_station": {"enabled": True,  "weight": 2.0,  "display": "Calling Station", "icon": "📞"},
            "lag":             {"enabled": True,  "weight": 0.6,  "display": "LAG",             "icon": "💣"},
        },
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# _now_iso alias (session_config.py kompatibilitáshoz)
_now_iso = _now


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class ModelManager:
    """Modell mappák kezelője. models/{name}/ struktúra."""

    def __init__(self, base_dir: Optional[str] = None) -> None:
        if base_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_dir = project_root
        self._project_root = base_dir
        self._models_dir   = os.path.join(base_dir, "models")
        os.makedirs(self._models_dir, exist_ok=True)

    def model_dir(self, name: str) -> str:
        return os.path.join(self._models_dir, name)

    def pth_path(self, name: str, filename: Optional[str] = None) -> str:
        if filename is None:
            model_dir = self.model_dir(name)
            if os.path.isdir(model_dir):
                existing = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
                if existing:
                    return existing[0]
            filename = f"{name}_ppo_v4.pth"
        return os.path.join(self.model_dir(name), filename)

    def config_path(self, name: str) -> str:
        return os.path.join(self.model_dir(name), "config.json")

    def naplo_path(self, name: str) -> str:
        return os.path.join(self.model_dir(name), "naplo.json")

    def tests_dir(self, name: str) -> str:
        return os.path.join(self.model_dir(name), "tests")

    def lifecycle_path(self, name: str) -> str:
        """Az életút-napló (LifecycleLogger) JSON fájljának útvonala."""
        return os.path.join(self.model_dir(name), "lifecycle.json")

    def ensure_model_dir(self, name: str, num_players: int = 6) -> str:
        d = self.model_dir(name)
        os.makedirs(d, exist_ok=True)
        os.makedirs(self.tests_dir(name), exist_ok=True)

        if not os.path.exists(self.config_path(name)):
            cfg = dict(CONFIG_DEFAULTS)
            self._write_json(self.config_path(name), {
                "num_players": num_players,
                "created":     _now(),
                "config":      cfg,
            })

        if not os.path.exists(self.naplo_path(name)):
            self._write_json(self.naplo_path(name), {
                "model_name":     name,
                "num_players":    num_players,
                "created":        _now(),
                "total_episodes": 0,
                "sessions":       [],
            })
        return d

    def load_config(self, name: str) -> Dict:
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
        self.ensure_model_dir(name, data.get("num_players", 6))
        data["last_saved"] = _now()
        self._write_json(self.config_path(name), data)
        logger.debug(f"Konfig mentve: models/{name}/config.json")

    def apply_style_preset(self, config: Dict, style: str) -> Dict:
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

    def load_naplo(self, name: str) -> Dict:
        path = self.naplo_path(name)
        if not os.path.exists(path):
            return {"model_name": name, "num_players": 0, "total_episodes": 0, "sessions": []}
        return self._read_json(path) or {"sessions": []}

    def start_session(self, name: str, config_snapshot: Dict, episodes_start: int, num_players: int) -> str:
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
        return session_id

    def end_session(self, name: str, session_id: str, episodes_end: int,
                    metrics: Optional[Dict] = None, completed: bool = True) -> None:
        naplo = self.load_naplo(name)
        for sess in naplo.get("sessions", []):
            if sess.get("id") == session_id:
                now_str              = _now()
                sess["ended"]        = now_str
                sess["episodes_end"] = episodes_end
                sess["episodes_added"] = max(0, episodes_end - sess.get("episodes_start", 0))
                sess["completed"]      = completed
                sess["metrics_final"]  = metrics or {}
                try:
                    from datetime import datetime as dt
                    start = dt.fromisoformat(sess["started"].replace("Z", "+00:00"))
                    end   = dt.fromisoformat(now_str.replace("Z", "+00:00"))
                    sess["duration_sec"] = int((end - start).total_seconds())
                except Exception:
                    pass
                break
        naplo["total_episodes"] = episodes_end
        naplo["last_updated"]   = _now()
        self._write_json(self.naplo_path(name), naplo)

    def add_naplo_note(self, name: str, session_id: str, note: str) -> None:
        naplo = self.load_naplo(name)
        for sess in naplo.get("sessions", []):
            if sess.get("id") == session_id:
                sess["notes"] = note
                break
        self._write_json(self.naplo_path(name), naplo)

    def list_tests(self, name: str) -> List[Dict]:
        tdir = self.tests_dir(name)
        if not os.path.isdir(tdir):
            return []
        results = []
        for f in sorted(glob.glob(os.path.join(tdir, "*.json")), reverse=True):
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
        log_path = os.path.join(self.tests_dir(name), filename)
        if not os.path.exists(log_path):
            return ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""

    def list_models(self) -> List[Dict]:
        models = []
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
                pth     = pth_files[0]
                cfg     = self.load_config(entry)
                ck_meta = self._read_checkpoint_meta(pth)
                models.append(self._model_entry(entry, pth, cfg, ck_meta))

        root_pths  = glob.glob(os.path.join(self._project_root, "*.pth"))
        known_pths = {m.get("abs_pth") for m in models}
        for pth in sorted(root_pths):
            if pth in known_pths or "ModellNaplo" in pth:
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

    def migrate_to_models_dir(self, pth_path: str, name: Optional[str] = None,
                               num_players: Optional[int] = None) -> str:
        import shutil
        if name is None:
            name = os.path.splitext(os.path.basename(pth_path))[0]
        if num_players is None:
            ck = self._read_checkpoint_meta(pth_path)
            num_players = ck.get("num_players", 6)
        self.ensure_model_dir(name, num_players)
        dest = self.pth_path(name, os.path.basename(pth_path))
        if not os.path.exists(dest):
            shutil.copy2(pth_path, dest)
            logger.info(f"Migráció: {pth_path} → {dest}")
        return dest

    def _model_entry(self, name, pth, cfg, ck_meta=None):
        ck_meta = ck_meta or {}
        naplo   = self.load_naplo(name)
        sessions = naplo.get("sessions", [])
        last_sess = sessions[-1] if sessions else None
        return {
            "name":          name,
            "display":       name,
            "abs_pth":       os.path.abspath(pth) if pth else None,
            "rel_pth":       os.path.relpath(pth, self._project_root) if pth else None,
            "in_models_dir": True,
            "model_dir":     self.model_dir(name),
            "num_players":   cfg.get("num_players") or ck_meta.get("num_players"),
            "episodes":      ck_meta.get("episodes", naplo.get("total_episodes", 0)),
            "state_size":    ck_meta.get("state_size", "?"),
            "algorithm":     ck_meta.get("algorithm", "PPO"),
            "config":        cfg,
            "naplo_summary": {
                "total_sessions": len(sessions),
                "total_episodes": naplo.get("total_episodes", 0),
                "last_style":     last_sess.get("style") if last_sess else None,
                "last_trained":   last_sess.get("ended") if last_sess else None,
                "last_added":     last_sess.get("episodes_added") if last_sess else 0,
            },
        }

    def _read_checkpoint_meta(self, pth: str) -> Dict:
        if not pth or not os.path.exists(pth):
            return {}
        import sys as _sys
        _sys.path.insert(0, self._project_root)
        try:
            from utils.checkpoint_utils import safe_load_checkpoint, UnsafeCheckpointError
            try:
                ck = safe_load_checkpoint(pth, map_location="cpu", allow_unsafe=False)
            except UnsafeCheckpointError:
                ck = safe_load_checkpoint(pth, map_location="cpu", allow_unsafe=True)
            if isinstance(ck, dict) and "state_dict" in ck:
                return {
                    "episodes":    ck.get("episodes_trained", 0),
                    "state_size":  ck.get("state_size", "?"),
                    "algorithm":   ck.get("algorithm", "PPO"),
                    "num_players": ck.get("num_players"),
                }
        except Exception as exc:
            logger.debug(f"Checkpoint meta olvasási hiba ({pth!r}): {exc}")
        return {}

    @staticmethod
    def _guess_players(name: str) -> int:
        for n in range(9, 1, -1):
            if f"{n}max" in name or f"{n}p" in name:
                return n
        return 6

    @staticmethod
    def _read_json(path: str) -> Optional[Dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _write_json(path: str, data: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# LifecycleLogger  --  Életút-naplózó (JSON Lifecycle Log)
# ─────────────────────────────────────────────────────────────────────────────

import shutil
import time
import uuid


class LifecycleLogger:
    """
    A modell teljes életútját rögzítő naplózó.

    Minden tréning session-höz elmenti a teljes TrainingConfig snapshotot,
    a PPO metrikákat, és a mérföldkő teszteredményeket egyetlen koherens
    naplo.json-ban – a ModelManager naplo.json-jától elkülönülve.

    Atomikus írás: naplo.json.tmp → os.replace() (crash-biztonságos).

    Tipikus használat:
        lifecycle = LifecycleLogger(manager.lifecycle_path(name))
        lifecycle.start_session(cfg.to_dict(), start_episode)
        lifecycle.log_milestone(1_000_000, "sanity_check", results_dict)
        lifecycle.close_session(end_episode, {"mean_policy_loss": -0.012, ...})
    """

    def __init__(self, log_path: str, model_id: str = "PokerAI") -> None:
        self.log_path = log_path
        self.model_id = model_id
        self._active_idx: Optional[int] = None

        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        self._data: Dict[str, Any] = self._load_or_init()

    # ── Belső segédek ────────────────────────────────────────────────────────

    def _load_or_init(self) -> Dict[str, Any]:
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(
                    f"[Lifecycle] Napló betöltve: {self.log_path} "
                    f"({len(data.get('sessions', []))} session, "
                    f"{data.get('total_episodes_trained', 0):,} ep)"
                )
                return data
            except json.JSONDecodeError:
                bak = f"{self.log_path}.bak"
                logger.warning(f"[Lifecycle] Sérült JSON – biztonsági mentés: {bak}")
                shutil.copy(self.log_path, bak)
        logger.info(f"[Lifecycle] Új lifecycle napló: {self.log_path}")
        return {
            "model_id": self.model_id,
            "created_at": self._ts(),
            "total_episodes_trained": 0,
            "sessions": [],
        }

    def _save(self) -> None:
        tmp = f"{self.log_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, self.log_path)

    @staticmethod
    def _ts() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _active(self) -> Dict[str, Any]:
        if self._active_idx is None:
            raise RuntimeError("[Lifecycle] Nincs aktív session – hívj start_session()-t.")
        return self._data["sessions"][self._active_idx]

    # ── Publikus API ─────────────────────────────────────────────────────────

    def start_session(self, config_dict: Dict[str, Any], start_episode: int) -> str:
        """Új session nyitása a config teljes snapshotjával."""
        if self._active_idx is not None:
            logger.warning("[Lifecycle] Árva nyitott session lezárása.")
            self.close_session(start_episode, {"system_note": "Automatikus lezárás (újraindítás)"})

        sid = f"sess_{uuid.uuid4().hex[:8]}"
        self._data["sessions"].append({
            "session_id":    sid,
            "start_time":    self._ts(),
            "end_time":      None,
            "episode_range": {"start": start_episode, "end": None},
            "config":        config_dict,
            "metrics":       {},
            "milestones":    [],
        })
        self._active_idx = len(self._data["sessions"]) - 1
        self._save()
        logger.info(f"[Lifecycle] Session nyitva: {sid} (ep {start_episode:,})")
        return sid

    def log_milestone(self, episode: int, test_name: str, results: Dict[str, Any]) -> None:
        """Mérföldkő teszteredmény beillesztése az aktív session alá."""
        sess = self._active()
        if episode > self._data.get("total_episodes_trained", 0):
            self._data["total_episodes_trained"] = episode
        sess["milestones"].append({
            "episode":   episode,
            "timestamp": self._ts(),
            "test_name": test_name,
            "results":   results,
        })
        self._save()
        logger.info(f"[Lifecycle] Mérföldkő: {test_name} @ ep {episode:,}")

    def close_session(self, end_episode: int, metrics: Dict[str, Any]) -> None:
        """Session lezárása, metrikák véglegesítése."""
        if self._active_idx is None:
            return
        sess = self._active()
        sess["end_time"] = self._ts()
        sess["episode_range"]["end"] = end_episode
        sess["metrics"] = metrics
        if end_episode > self._data.get("total_episodes_trained", 0):
            self._data["total_episodes_trained"] = end_episode
        sid = sess["session_id"]
        self._active_idx = None
        self._save()
        logger.info(f"[Lifecycle] Session lezárva: {sid} (ep {end_episode:,})")

    def get_total_episodes(self) -> int:
        return self._data.get("total_episodes_trained", 0)
