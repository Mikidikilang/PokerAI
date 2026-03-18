"""
utils/logging_setup.py  –  Logger konfiguráció (v4)
"""
import sys, os, logging
from datetime import datetime

def setup_logging(log_file="training.log", num_players=None, level=logging.DEBUG):
    logger = logging.getLogger("PokerAI")
    logger.handlers.clear()
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    logs_dir = _get_logs_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = (f"session_{num_players}p_{timestamp}.log" if num_players else f"session_{timestamp}.log")
    session_path = os.path.join(logs_dir, session_name)
    sfh = logging.FileHandler(session_path, encoding="utf-8")
    sfh.setFormatter(fmt); sfh.setLevel(logging.DEBUG)
    logger.addHandler(sfh)
    main_path = os.path.join(logs_dir, log_file)
    mfh = logging.FileHandler(main_path, encoding="utf-8", mode="w")
    mfh.setFormatter(fmt); mfh.setLevel(logging.DEBUG)
    logger.addHandler(mfh)
    logger.info(f"Session log: {session_path}")
    logger.info(f"Folog:       {main_path}")
    return logger, session_path

def _get_logs_dir():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir

def list_sessions():
    logs_dir = _get_logs_dir()
    files = [f for f in os.listdir(logs_dir) if f.startswith("session_") and f.endswith(".log")]
    return sorted(files)
