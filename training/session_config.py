"""
training/session_config.py  --  DEPRECATED

[TASK-7 CLEANUP] Ez a modul redundáns volt a training/model_manager.py-val.
A ModelManager tartalmazza az összes szükséges funkciót:
  - load_config()         ← SessionConfigManager.load()
  - save_config()         ← SessionConfigManager.save()
  - apply_style_preset()  ← SessionConfigManager.apply_style_preset()
  - _guess_players()      ← SessionConfigManager._guess_players()
  - CONFIG_DEFAULTS       ← DEFAULTS
  - STYLE_PRESETS         ← STYLE_PRESETS

Visszafelé kompatibilis stub: a SessionConfigManager importok nem törnek,
de a konstruktor figyelmeztetést logol hogy térj át ModelManager-re.

Migrációs útmutató:
    # Régi:
    from training.session_config import SessionConfigManager
    mgr = SessionConfigManager()
    cfg = mgr.load("model.pth")
    mgr.save("model.pth", cfg)

    # Új:
    from training.model_manager import ModelManager
    mgr = ModelManager()
    cfg = mgr.load_config("model")   # kiterjesztés nélkül
    mgr.save_config("model", cfg)
"""

import logging
import warnings

logger = logging.getLogger("PokerAI")

# Re-export a model_manager-ből, hogy a régi importok ne törjenek
from training.model_manager import (
    ModelManager,
    CONFIG_DEFAULTS as DEFAULTS,
    STYLE_PRESETS,
)

# _deep_merge és _now_iso stub – régi közvetlen importokhoz
from training.model_manager import _deep_merge  # type: ignore[attr-defined]

try:
    from training.model_manager import _now_iso  # type: ignore[attr-defined]
except ImportError:
    from datetime import datetime, timezone
    def _now_iso():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SessionConfigManager:
    """
    DEPRECATED – Használd helyette a ModelManager-t.

    Ez a stub wrapper biztosítja a visszafelé kompatibilitást,
    de minden hívás deprecation warningot log-ol.
    """

    def __init__(self, config_path: str = None):
        warnings.warn(
            "SessionConfigManager deprecated – használd a ModelManager-t "
            "(training/model_manager.py). Lásd session_config.py migrációs útmutatót.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "SessionConfigManager: DEPRECATED. Térj át ModelManager-re. "
            "Részletek: training/session_config.py"
        )
        self._mgr = ModelManager()

    def load(self, model_filename: str) -> dict:
        import os
        name = os.path.splitext(os.path.basename(model_filename))[0]
        return self._mgr.load_config(name)

    def save(self, model_filename: str, full_config: dict):
        import os
        name = os.path.splitext(os.path.basename(model_filename))[0]
        self._mgr.save_config(name, full_config)

    def update_meta(self, model_filename: str, episodes: int = None,
                    num_players: int = None):
        import os
        name = os.path.splitext(os.path.basename(model_filename))[0]
        self._mgr.update_session_meta(name, episodes=episodes,
                                       num_players=num_players)

    def apply_style_preset(self, config: dict, style: str) -> dict:
        return self._mgr.apply_style_preset(config, style)

    def all_model_names(self) -> list:
        return self._mgr.list_models()

    def get_defaults(self) -> dict:
        return {
            "num_players": 6,
            "last_used": None,
            "last_episodes": 0,
            "config": dict(DEFAULTS),
        }

    @staticmethod
    def _guess_players(filename: str) -> int:
        return ModelManager._guess_players(filename)
