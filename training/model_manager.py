"""
training/model_manager.py  --  Modell mappa kezelő (v4.2.2)

Minden modellhez saját mappa: models/{name}/
  ├── {name}_ppo_v4.pth    ← checkpoint
  ├── config.json          ← aktuális trening konfig
  ├── naplo.json           ← trening napló (session history)
  └── tests/               ← test_model_sanity.py kimenetek

Változások v4.2.2-BUGFIX:
    [BUG-2] _read_checkpoint_meta(): robusztusabb checkpoint felismerés.
        Korábban csak akkor adott vissza episodes értéket, ha a checkpoint-ban
        volt "state_dict" kulcs. Ha a fájl kulso forrásból érkezett (más
        projekt, régi formátum, vagy más kulcsnévvel mentve), a metódus
        ures dict-et adott vissza -> a GUI 0 ep-t mutatott.

        Javítás: ha nincs "state_dict", a metódus megpróbálja más
        lehetséges kulcsokból kiolvasni az epizódszámot, és a modell
        súlyaiból megbecsülni, hogy a checkpoint tele van-e.
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
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class ModelManager:

    def __init__(self, base_dir: Optional[str] = None) -> None:
        if base_dir is None:
            project_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            base_dir = project_root
        self._project_root = base_dir
        self._models_dir   = os.path.join(base_dir, "models")
        os.makedirs(self._models_dir, exist_ok=True)

    # ── Eleresi utak ────────────────────────────────────────────────────────

    def model_dir(self, name: str) -> str:
        return os.path.join(self._models_dir, name)

    def pth_path(self, name: str, filename: Optional[str] = None) -> str:
        if filename is None:
            model_dir = self.model_dir(name)
            if os.path.isdir(model_dir):
                existing = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
                if existing:
                    found = existing[0]
                    logger.debug(
                        f"pth_path({name!r}): meglévő fájl -> "
                        f"{os.path.basename(found)}"
                    )
                    return found
            filename = f"{name}_ppo_v4.pth"
            logger.debug(
                f"pth_path({name!r}): nincs .pth a mappában -> új: {filename}"
            )
        return os.path.join(self.model_dir(name), filename)

    def config_path(self, name: str) -> str:
        return os.path.join(self.model_dir(name), "config.json")

    def naplo_path(self, name: str) -> str:
        return os.path.join(self.model_dir(name), "naplo.json")

    def tests_dir(self, name: str) -> str:
        return os.path.join(self.model_dir(name), "tests")

    # ── Modell mappa letrehozása ─────────────────────────────────────────────

    def ensure_model_dir(self, name: str, num_players: int = 6) -> str:
        d = self.model_dir(name)
        os.makedirs(d, exist_ok=True)
        os.makedirs(self.tests_dir(name), exist_ok=True)

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

    # ── Napló ────────────────────────────────────────────────────────────────

    def load_naplo(self, name: str) -> Dict:
        path = self.naplo_path(name)
        if not os.path.exists(path):
            return {
                "model_name":     name,
                "num_players":    0,
                "total_episodes": 0,
                "sessions":       [],
            }
        return self._read_json(path) or {"sessions": []}

    def start_session(self, name, config_snapshot, episodes_start, num_players):
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
        logger.info(f"Naplo session start: {name} / {session_id}")
        return session_id

    def end_session(self, name, session_id, episodes_end, metrics=None, completed=True):
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
            f"Naplo session end: {name} / {session_id} -> {episodes_end:,} ep"
        )

    def add_naplo_note(self, name, session_id, note):
        naplo = self.load_naplo(name)
        for sess in naplo.get("sessions", []):
            if sess.get("id") == session_id:
                sess["notes"] = note
                break
        self._write_json(self.naplo_path(name), naplo)

    # ── Tesztek ──────────────────────────────────────────────────────────────

    def list_tests(self, name: str) -> List[Dict]:
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
        models: List[Dict] = []

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

        root_pths  = glob.glob(os.path.join(self._project_root, "*.pth"))
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
                "display":       f"{basename} [WARN] (gyokerben, nem migralt)",
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

    def migrate_to_models_dir(self, pth_path, name=None, num_players=None):
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
            logger.info(f"Migracio: {pth_path} -> {dest}")
        return dest

    # ── Belső segédmetódusok ─────────────────────────────────────────────────

    def _model_entry(self, name, pth, cfg, ck_meta=None):
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

        [BUG-2 FIX] Robusztusabb checkpoint felismerés:
            Korábban a metódus csak akkor adott vissza episodes értéket,
            ha a betöltött dict-ben volt "state_dict" kulcs. Ez kizárta:
              - Kulso forrásból érkező checkpointokat (más kulcsnév)
              - Régi formátumú fájlokat
              - Olyan checkpointokat ahol a state_dict közvetlenül
                a root dict-ben van (nem state_dict kulcs alatt)

            Javítás: fallback lánc az epizódszám kiolvasásához:
              1. Elsődleges: "episodes_trained" kulcs (v4.2 formátum)
              2. Másodlagos: "episode", "episodes", "total_episodes" kulcsok
              3. Ha semmit sem talál, de a dict-ben vannak tensor értékek
                 (=modell súlyok), akkor a modell fel van töltve, csak az
                 epizódszám hiányzik -> None-t adunk vissza (GUI kezeli)

            A GUI-ban a None értéket "? ep" felirattal kezeljük, ami
            jelzi hogy a checkpoint létezik, de az epizódszám ismeretlen.
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

            try:
                ck = safe_load_checkpoint(pth, map_location="cpu", allow_unsafe=False)
            except UnsafeCheckpointError:
                logger.info(
                    f"_read_checkpoint_meta: legacy checkpoint detektálva "
                    f"({os.path.basename(pth)!r}), unsafe betöltés."
                )
                ck = safe_load_checkpoint(pth, map_location="cpu", allow_unsafe=True)

            if not isinstance(ck, dict):
                return {}

            # ── [BUG-2 FIX] Robusztus episodes kiolvasás ─────────────────
            # Az eredeti kód csak "state_dict" kulcs esetén futott le.
            # Most fallback lánccal próbálkozunk.

            # 1. Próbálkozás: v4.2 standard formátum (state_dict + meta)
            if "state_dict" in ck:
                return {
                    "episodes":    ck.get("episodes_trained", 0),
                    "state_size":  ck.get("state_size", "?"),
                    "algorithm":   ck.get("algorithm", "PPO"),
                    "num_players": ck.get("num_players"),
                }

            # 2. Próbálkozás: meta kulcsok közvetlenül a root dict-ben
            # (pl. ha valaki torch.save({"episodes_trained": N, ...}) hívott
            # state_dict wrapper nélkül)
            _EP_KEYS = ("episodes_trained", "episode", "episodes", "total_episodes")
            for ep_key in _EP_KEYS:
                if ep_key in ck:
                    logger.debug(
                        f"_read_checkpoint_meta: nem-standard formátum, "
                        f"episodes kulcs: {ep_key!r} ({os.path.basename(pth)})"
                    )
                    return {
                        "episodes":    ck.get(ep_key, 0),
                        "state_size":  ck.get("state_size", "?"),
                        "algorithm":   ck.get("algorithm", ck.get("algo", "PPO")),
                        "num_players": ck.get("num_players", ck.get("n_players")),
                    }

            # 3. Próbálkozás: a dict tensor értékeket tartalmaz
            # (= közvetlenül state_dict lett mentve, nem wrapper)
            # Pl.: torch.save(model.state_dict(), path)
            import torch
            has_tensors = any(
                isinstance(v, torch.Tensor) for v in ck.values()
            )
            if has_tensors:
                logger.debug(
                    f"_read_checkpoint_meta: raw state_dict formátum "
                    f"({os.path.basename(pth)}) – epizódszám ismeretlen."
                )
                # Epizódszám ismeretlen, de a checkpoint létezik és tele van.
                # None jelzi, hogy a fájl érvényes, csak a meta hiányzik.
                return {
                    "episodes":    None,
                    "state_size":  "?",
                    "algorithm":   "PPO",
                    "num_players": None,
                }

        except Exception as exc:
            logger.debug(
                f"Checkpoint meta olvasási hiba ({pth!r}): {exc}"
            )

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
