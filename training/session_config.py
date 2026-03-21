"""
training/session_config.py  --  Tréning session konfiguráció perzisztencia

Minden modell (.pth) saját konfigurációt kap, amit JSON-ban tárol.
Betöltéskor a hiányzó mezők a TrainingConfig defaultjaival töltődnek.

Használat:
    from training.session_config import SessionConfigManager
    mgr = SessionConfigManager()
    cfg = mgr.load("2max_ppo_v4.pth")   # → dict
    mgr.save("2max_ppo_v4.pth", cfg)
    mgr.update_meta("2max_ppo_v4.pth", episodes=4_200_000)
"""
import json
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger("PokerAI")

CONFIG_FILE = "training_configs.json"

# ── Default értékek (TrainingConfig alapján, de dict formában) ───────────────
DEFAULTS = {
    # Env
    "num_envs":               512,
    "buffer_collect_size":    2048,
    "max_steps_per_hand":     500,
    # Modell
    "hidden_size":            512,
    "gru_hidden":             None,
    # PPO
    "learning_rate":          3e-4,
    "clip_eps":               0.2,
    "ppo_epochs":             8,
    "minibatch_size":         256,
    "value_coef":             0.5,
    "entropy_coef":           0.01,
    "entropy_final":          0.001,
    "entropy_decay":          30_000_000,
    "max_grad_norm":          0.5,
    "gamma":                  0.99,
    "gae_lambda":             0.95,
    # LR Scheduler
    "lr_t_max":               500,
    "lr_eta_min_ratio":       0.05,
    # Reward shaping
    "draw_fold_penalty":      0.08,
    "draw_equity_threshold":  0.44,
    "street_reward_scale":    0.05,
    # Milestone
    "milestone_interval":     2_000_000,
    # Opponent pool
    "training_phase":         2,        # 1=self-play, 2=exploitative
    "training_style":         "exploitative",  # "self_play" | "exploitative" | "custom"
    "bot_pool": {
        "fish":            {"enabled": True,  "weight": 0.8,  "display": "Fish 🐟"},
        "nit":             {"enabled": True,  "weight": 1.5,  "display": "Nit 🎯"},
        "calling_station": {"enabled": True,  "weight": 0.2,  "display": "Calling Station 📞"},
        "lag":             {"enabled": True,  "weight": 1.5,  "display": "LAG 💣"},
    },
    # Equity
    "equity_n_sim":           200,
    "equity_cache_size":      20_000,
}

# ── Stílus presetek ──────────────────────────────────────────────────────────
STYLE_PRESETS = {
    "self_play": {
        "training_phase":   1,
        "entropy_coef":     0.02,
        "entropy_final":    0.002,
        "entropy_decay":    50_000_000,
        "bot_pool": {
            "fish":            {"enabled": False, "weight": 0.0},
            "nit":             {"enabled": False, "weight": 0.0},
            "calling_station": {"enabled": False, "weight": 0.0},
            "lag":             {"enabled": False, "weight": 0.0},
        },
    },
    "exploitative": {
        "training_phase":   2,
        "entropy_coef":     0.01,
        "entropy_final":    0.001,
        "entropy_decay":    30_000_000,
        "bot_pool": {
            "fish":            {"enabled": True, "weight": 0.8},
            "nit":             {"enabled": True, "weight": 1.5},
            "calling_station": {"enabled": True, "weight": 0.2},
            "lag":             {"enabled": True, "weight": 1.5},
        },
    },
    "aggressive": {
        "training_phase":   2,
        "entropy_coef":     0.005,
        "entropy_final":    0.0005,
        "entropy_decay":    20_000_000,
        "bot_pool": {
            "fish":            {"enabled": True, "weight": 0.5},
            "nit":             {"enabled": True, "weight": 2.5},
            "calling_station": {"enabled": False, "weight": 0.0},
            "lag":             {"enabled": True, "weight": 2.5},
        },
    },
}


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _deep_merge(base: dict, override: dict) -> dict:
    """Mély dict merge: override felülírja a base-t, de hiányzó kulcsok megmaradnak."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class SessionConfigManager:
    """
    Minden modell fájlhoz tárolt tréning konfiguráció kezelője.

    Adatfájl: training_configs.json (projekt gyökér, gitignored ha hozzáadod)

    Struktúra:
        {
          "model_configs": {
            "2max_ppo_v4.pth": {
              "num_players": 2,
              "last_used": "2026-03-21T...",
              "last_episodes": 4200000,
              "config": { ...tréning paraméterek... }
            }
          }
        }
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(project_root, CONFIG_FILE)
        self._path = config_path
        self._data = self._load_file()

    # ── Fájl I/O ────────────────────────────────────────────────────────────

    def _load_file(self) -> dict:
        if not os.path.exists(self._path):
            return {"model_configs": {}}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"session_config betöltési hiba ({self._path}): {e}")
            return {"model_configs": {}}

    def _save_file(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.error(f"session_config mentési hiba: {e}")

    # ── Publikus API ────────────────────────────────────────────────────────

    def load(self, model_filename: str) -> dict:
        """
        Adott modell konfigurációja. Hiányzó mezők default értékkel töltve.
        Visszatér: flat config dict (num_players, config mezők, meta).
        """
        basename = os.path.basename(model_filename)
        stored = self._data.get("model_configs", {}).get(basename, {})
        stored_config = stored.get("config", {})

        merged_config = _deep_merge(DEFAULTS, stored_config)

        return {
            "num_players": stored.get("num_players", self._guess_players(basename)),
            "last_used":   stored.get("last_used", None),
            "last_episodes": stored.get("last_episodes", 0),
            "config":      merged_config,
        }

    def save(self, model_filename: str, full_config: dict):
        """
        Konfiguráció mentése. full_config = load() által visszaadott dict.
        """
        basename = os.path.basename(model_filename)
        if "model_configs" not in self._data:
            self._data["model_configs"] = {}
        existing = self._data["model_configs"].get(basename, {})
        existing["num_players"]    = full_config.get("num_players", 2)
        existing["last_used"]      = _now_iso()
        existing["last_episodes"]  = full_config.get("last_episodes", 0)
        existing["config"]         = full_config.get("config", {})
        self._data["model_configs"][basename] = existing
        self._save_file()
        logger.debug(f"Session config mentve: {basename}")

    def update_meta(self, model_filename: str, episodes: int = None,
                    num_players: int = None):
        """Csak metaadatok frissítése (tréning közben hívható)."""
        basename = os.path.basename(model_filename)
        if "model_configs" not in self._data:
            self._data["model_configs"] = {}
        entry = self._data["model_configs"].get(basename, {})
        if episodes is not None:
            entry["last_episodes"] = episodes
        if num_players is not None:
            entry["num_players"] = num_players
        entry["last_used"] = _now_iso()
        self._data["model_configs"][basename] = entry
        self._save_file()

    def apply_style_preset(self, config: dict, style: str) -> dict:
        """
        Stílus preset alkalmazása a config-ra.
        Megtartja a user által módosított egyéb értékeket.
        """
        if style not in STYLE_PRESETS:
            logger.warning(f"Ismeretlen stílus preset: {style}")
            return config
        preset = STYLE_PRESETS[style]
        result = dict(config)
        result.update({k: v for k, v in preset.items() if k != "bot_pool"})
        if "bot_pool" in preset:
            merged_pool = _deep_merge(
                config.get("bot_pool", DEFAULTS["bot_pool"]),
                preset["bot_pool"]
            )
            result["bot_pool"] = merged_pool
        result["training_style"] = style
        return result

    def all_model_names(self) -> list:
        """Összes ismert modell neve."""
        return list(self._data.get("model_configs", {}).keys())

    def get_defaults(self) -> dict:
        """Alap default konfiguráció visszaadása."""
        return {"num_players": 6, "last_used": None, "last_episodes": 0, "config": dict(DEFAULTS)}

    # ── Segéd ───────────────────────────────────────────────────────────────

    @staticmethod
    def _guess_players(filename: str) -> int:
        """Fájlnévből próbálja kitalálni a játékosszámot."""
        for n in range(9, 1, -1):
            if f"{n}max" in filename or f"{n}p" in filename:
                return n
        return 6
