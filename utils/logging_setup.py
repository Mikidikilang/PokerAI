"""
utils/logging_setup.py  –  Logger konfiguráció (v4)

Minden session külön log fájlt kap a logs/ mappában:
  logs/
    session_2p_20260318_140532.log
    session_6p_20260318_141200.log
    training.log  <- mindig az utolsó session (régi viselkedés is megmarad)
"""

import sys
import os
import logging
from datetime import datetime


def setup_logging(log_file: str = "training.log",
                  num_players: int = None,
                  level: int = logging.DEBUG):
    """
    Logger beállítása két helyre:
      1. stdout (konzol, csak INFO+)
      2. logs/session_Xp_YYYYMMDD_HHMMSS.log  (egyedi session, DEBUG+)
      3. logs/training.log  (mindig az utolsó session, felülírva)

    Visszatér: (logger, session_log_path)
    """
    logger = logging.getLogger("PokerAI")
    logger.handlers.clear()
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Konzol
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    # logs/ mappa
    logs_dir = _get_logs_dir()

    # 2. Session-specifikus log
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = (f"session_{num_players}p_{timestamp}.log"
                    if num_players is not None
                    else f"session_{timestamp}.log")
    session_path = os.path.join(logs_dir, session_name)
    sfh = logging.FileHandler(session_path, encoding="utf-8")
    sfh.setFormatter(fmt)
    sfh.setLevel(logging.DEBUG)
    logger.addHandler(sfh)

    # 3. Főlog (felülírja az előző session-t)
    main_path = os.path.join(logs_dir, log_file)
    mfh = logging.FileHandler(main_path, encoding="utf-8", mode='w')
    mfh.setFormatter(fmt)
    mfh.setLevel(logging.DEBUG)
    logger.addHandler(mfh)

    logger.info(f"Session log: {session_path}")
    logger.info(f"Folog:       {main_path}")

    return logger, session_path


def _get_logs_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir     = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def list_sessions() -> list:
    """Összes session log fájl időrendi sorrendben."""
    logs_dir = _get_logs_dir()
    files = [f for f in os.listdir(logs_dir)
             if f.startswith("session_") and f.endswith(".log")]
    return sorted(files)
