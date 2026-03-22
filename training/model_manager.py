"""
training/model_manager.py  --  Modell mappa kezelő

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
"""
import glob
import json
import logging
import os
import uuid
from datetime import datetime, timezone

logger = logging.getLogger("PokerAI")

# ── Alapértelmezett konfig (TrainingConfig mezőkkel egyező) ──────────────────
CONFIG_DEFAULTS = {
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
STYLE_PRESETS = {
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


def _deep_merge(base: dict, override: dict) -> dict:
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

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            # Projekt gyökér: ez a fájl training/ alatt van, tehát egy szinttel feljebb
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_dir = project_root
        self._project_root = base_dir
        self._models_dir   = os.path.join(base_dir, "models")
        os.makedirs(self._models_dir, exist_ok=True)

    # ── Elérési utak ────────────────────────────────────────────────────────

    def model_dir(self, name: str) -> str:
        return os.path.join(self._models_dir, name)

    def pth_path(self, name: str, filename: str = None) -> str:
        """
        Visszaadja a modell .pth fájljának elérési útját.

        Ha filename nincs megadva:
          1. Megnézi a models/{name}/ mappát – ha van benne .pth fájl,
             azt adja vissza (migrált vagy régi névkonvenciójú modelleknél
             ez megakadályozza a névduplázódást, pl. name_ppo_v4_ppo_v4.pth).
          2. Ha nincs .pth fájl, az alapértelmezett {name}_ppo_v4.pth nevet
             generálja (új modell esetén).

        BUGFIX: korábban mindig {name}_ppo_v4.pth-t adott vissza, ami
        migrált modelleknél eltért a tényleges fájlnévtől → az
        os.path.exists() False lett → episodes_trained=0 → 0-ról indult.
        """
        if filename is None:
            model_dir = self.model_dir(name)
            if os.path.isdir(model_dir):
                existing = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
                if existing:
                    found = existing[0]
                    logger.debug(
                        f"pth_path({name!r}): meglévő fájl → {os.path.basename(found)}"
                    )
                    return found
            filename = f"{name}_ppo_v4.pth"
            logger.debug(
                f"pth_path({name!r}): nincs .pth a mappában → új: {filename}"
            )
        return os.path.join(self.model_dir(name), filename)

    def config_path(self, name: str) -> str:
        return os.path.join(self.model_dir(name), "config.json")

    def naplo_path(self, name: str) -> str:
        return os.path.join(self.model_dir(name), "naplo.json")

    def tests_dir(self, name: str) -> str:
        return os.path.join(self.model_dir(name), "tests")

    # ── Modell mappa létrehozása ─────────────────────────────────────────────

    def ensure_model_dir(self, name: str, num_players: int = 6):
        """Létrehozza a modell mappa struktúráját ha nem létezik."""
        d = self.model_dir(name)
        os.makedirs(d, exist_ok=True)
        os.makedirs(self.tests_dir(name), exist_ok=True)

        # config.json ha nem létezik
        if not os.path.exists(self.config_path(name)):
            cfg = dict(CONFIG_DEFAULTS)
            # Játékosszámot a névből próbálja kitalálni ha nincs megadva
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

    def load_config(self, name: str) -> dict:
        """Konfig betöltése. Hiányzó mezők defaulttal töltve."""
        path = self.config_path(name)
        if not os.path.exists(path):
            return {
                "num_players": self._guess_players(name),
                "created": None,
                "config": dict(CONFIG_DEFAULTS),
            }
        raw = self._read_json(path) or {}
        raw["config"] = _deep_merge(CONFIG_DEFAULTS, raw.get("config", {}))
        return raw

    def save_config(self, name: str, data: dict):
        """Konfig mentése."""
        self.ensure_model_dir(name, data.get("num_players", 6))
        data["last_saved"] = _now()
        self._write_json(self.config_path(name), data)
        logger.debug(f"Konfig mentve: models/{name}/config.json")

    def apply_style_preset(self, config: dict, style: str) -> dict:
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

    def load_naplo(self, name: str) -> dict:
        path = self.naplo_path(name)
        if not os.path.exists(path):
            return {"model_name": name, "num_players": 0, "total_episodes": 0, "sessions": []}
        return self._read_json(path) or {"sessions": []}

    def start_session(self, name: str, config_snapshot: dict,
                      episodes_start: int, num_players: int) -> str:
        """
        Új tréning session rögzítése a naplóba.
        Visszatér: session_id (str)
        """
        self.ensure_model_dir(name, num_players)
        naplo = self.load_naplo(name)
        session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = {
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

    def end_session(self, name: str, session_id: str,
                    episodes_end: int, metrics: dict = None, completed: bool = True):
        """Tréning session lezárása."""
        naplo = self.load_naplo(name)
        sessions = naplo.get("sessions", [])
        for sess in sessions:
            if sess.get("id") == session_id:
                now_str = _now()
                sess["ended"] = now_str
                sess["episodes_end"] = episodes_end
                sess["episodes_added"] = max(0, episodes_end - sess.get("episodes_start", 0))
                sess["completed"] = completed
                sess["metrics_final"] = metrics or {}
                # Időtartam kiszámítása
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
        logger.info(f"Napló session end: {name} / {session_id} → {episodes_end:,} ep")

    def add_naplo_note(self, name: str, session_id: str, note: str):
        naplo = self.load_naplo(name)
        for sess in naplo.get("sessions", []):
            if sess.get("id") == session_id:
                sess["notes"] = note
                break
        self._write_json(self.naplo_path(name), naplo)

    # ── Tesztek ──────────────────────────────────────────────────────────────

    def list_tests(self, name: str) -> list:
        """Visszaadja a tests/ mappa .json fájljait metaadatokkal."""
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
        """Egy teszt .log fájl tartalmát adja vissza."""
        log_path = os.path.join(self.tests_dir(name), filename)
        if not os.path.exists(log_path):
            return ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""

    # ── Model lista ──────────────────────────────────────────────────────────

    def list_models(self) -> list:
        """
        Visszaadja az összes modell adatait.
        Elsőként a models/ mappát nézi, aztán a projekt gyökerét (migráció).
        """
        models = []

        # 1) models/ alkönyvtárak
        if os.path.isdir(self._models_dir):
            for entry in sorted(os.listdir(self._models_dir)):
                d = os.path.join(self._models_dir, entry)
                if not os.path.isdir(d):
                    continue
                pth_files = glob.glob(os.path.join(d, "*.pth"))
                if not pth_files:
                    # Még nincs .pth, de létező modellmappa (pl. most hozták létre)
                    cfg = self.load_config(entry)
                    models.append(self._model_entry(entry, None, cfg))
                    continue
                pth = pth_files[0]  # az első .pth fájlt vesszük
                cfg = self.load_config(entry)
                ck_meta = self._read_checkpoint_meta(pth)
                models.append(self._model_entry(entry, pth, cfg, ck_meta))

        # 2) Projekt gyökérben lévő .pth fájlok (migrálatlan)
        root_pths = glob.glob(os.path.join(self._project_root, "*.pth"))
        known_pths = {m.get("abs_pth") for m in models}
        for pth in sorted(root_pths):
            if pth in known_pths:
                continue
            if "ModellNaplo" in pth:
                continue
            basename = os.path.splitext(os.path.basename(pth))[0]
            ck_meta = self._read_checkpoint_meta(pth)
            models.append({
                "name":        basename,
                "display":     f"{basename} ⚠ (gyökérben, nem migrált)",
                "abs_pth":     os.path.abspath(pth),
                "rel_pth":     os.path.relpath(pth, self._project_root),
                "in_models_dir": False,
                "num_players": ck_meta.get("num_players"),
                "episodes":    ck_meta.get("episodes", 0),
                "state_size":  ck_meta.get("state_size", "?"),
                "algorithm":   ck_meta.get("algorithm", "PPO"),
                "config":      None,
                "naplo_summary": None,
            })

        return models

    def migrate_to_models_dir(self, pth_path: str, name: str = None,
                               num_players: int = None) -> str:
        """
        Migrál egy gyökérben lévő .pth fájlt a models/{name}/ mappába.
        Visszatér: új pth elérési út.
        """
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

    # ── Segéd ────────────────────────────────────────────────────────────────

    def _model_entry(self, name: str, pth: str | None,
                     cfg: dict, ck_meta: dict = None) -> dict:
        ck_meta = ck_meta or {}
        naplo = self.load_naplo(name)
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
                "total_sessions":  len(sessions),
                "total_episodes":  naplo.get("total_episodes", 0),
                "last_style":      last_sess.get("style") if last_sess else None,
                "last_trained":    last_sess.get("ended") if last_sess else None,
                "last_added":      last_sess.get("episodes_added") if last_sess else 0,
            },
        }

    def _read_checkpoint_meta(self, pth: str) -> dict:
        if not pth or not os.path.exists(pth):
            return {}
        try:
            import sys
            sys.path.insert(0, self._project_root)
            from utils.checkpoint_utils import safe_load_checkpoint
            ck = safe_load_checkpoint(pth, map_location="cpu")
            if isinstance(ck, dict) and "state_dict" in ck:
                return {
                    "episodes":    ck.get("episodes_trained", 0),
                    "state_size":  ck.get("state_size", "?"),
                    "algorithm":   ck.get("algorithm", "PPO"),
                    "num_players": ck.get("num_players"),
                }
        except Exception as e:
            logger.debug(f"Checkpoint meta olvasási hiba ({pth}): {e}")
        return {}

    @staticmethod
    def _guess_players(name: str) -> int:
        for n in range(9, 1, -1):
            if f"{n}max" in name or f"{n}p" in name:
                return n
        return 6

    @staticmethod
    def _read_json(path: str) -> dict | None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _write_json(path: str, data):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
