"""
training/launcher.py  --  Közös tréning indító logika (GUI + CLI)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ezt a modult hívja meg MINDKÉT belépési pont:

  train.py               ← közvetlen CLI / RunPod / headless indítás
  _train_session_cli.py  ← train_gui.py által spawnolt subprocess

Így a GUI-ból és a CLI-ből indított tréning 100%-ban kompatibilis:
  • azonos ModelManager útvonalak  (models/{name}/...)
  • azonos konfig-összerakás       (build_training_config)
  • azonos checkpoint formátum     (runner._save_checkpoint változatlan)
  • azonos naplo.json kezelés      (mgr.start_session / end_session)

GUI → leállítás → RunPod CLI folytatás: hibátlanul működik (és fordítva).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import atexit
import glob as _g
import logging
import os

from config import TrainingConfig
from training.model_manager import ModelManager, CONFIG_DEFAULTS
from training.runner import run_training_session
from utils.checkpoint_utils import safe_load_checkpoint

logger = logging.getLogger("PokerAI")

# Projekt gyökér (ez a fájl training/ alatt van, tehát egy szinttel feljebb)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Publikus API ─────────────────────────────────────────────────────────────

def build_training_config(
    config_dict: dict,
    model_name: str,
    mgr: ModelManager,
    milestone_interval: int | None = None,
    milestone_hands: int | None = None,
) -> TrainingConfig:
    """
    GUI-kompatibilis config_dict-ből épít TrainingConfig-ot.

    A logika azonos azzal, amit korábban a _train_session_cli.py és a
    train_gui.py /api/start végpontja egyaránt elvégzett (duplikáltan).
    Most ez az egyetlen forrás igazság.

    Args:
        config_dict:        A modell config.json „config" szekciója,
                            pl. ModelManager.load_config(name)["config"].
        model_name:         Modell neve (mgr.tests_dir() híváshoz).
        mgr:                Inicializált ModelManager példány.
        milestone_interval: Ha nem None, felülírja a config értékét.
                            Hasznos RunPod-on (pl. 500_000).
        milestone_hands:    Ha nem None, felülírja a config értékét.
    """
    raw = dict(config_dict)
    bot_pool = raw.pop("bot_pool", CONFIG_DEFAULTS["bot_pool"])
    raw.pop("training_style", None)
    raw.pop("training_phase", None)

    valid = set(TrainingConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in raw.items() if k in valid}

    # Bot pool összerakás – pontosan úgy, ahogy a GUI csinálja
    filtered["opponent_bot_types"] = [
        k for k, v in bot_pool.items() if v.get("enabled", True)
    ]
    filtered["opponent_bot_weights"] = [
        v.get("weight", 1.0) for k, v in bot_pool.items() if v.get("enabled", True)
    ]

    # Mérföldkő snapshot-ok ide kerülnek (Drive-on vagy lokálisan is működik)
    filtered["milestone_dir_root"] = mgr.tests_dir(model_name)

    if milestone_interval is not None:
        filtered["milestone_interval"] = milestone_interval
        logger.info(f"milestone_interval felülírva: {milestone_interval:,} ep")

    if milestone_hands is not None:
        filtered["milestone_hands"] = milestone_hands
        logger.info(f"milestone_hands felülírva: {milestone_hands}")

    try:
        return TrainingConfig(**filtered)
    except Exception as exc:
        logger.warning(f"TrainingConfig hiba ({exc}), default konfig használata.")
        return TrainingConfig()


def launch(
    model_name: str,
    pth_path: str,
    num_players: int,
    episodes: int,
    cfg: TrainingConfig,
    mgr: ModelManager,
    session_id: str | None = None,
    output_base: str | None = None,
) -> None:
    """
    Elindít egy tréning sessiont és beregisztrálja az atexit cleanup-ot.

    Pontosan ezt csinálta korábban a _train_session_cli.py main()-je,
    de kód-duplikáció nélkül. A train.py és _train_session_cli.py
    egyaránt ezt hívja.

    Args:
        model_name:  Modell azonosítója (napló célokra).
        pth_path:    A .pth checkpoint abszolút elérési útja.
        num_players: Játékosok száma.
        episodes:    Futtatandó epizódok száma (a checkpoint tetején).
        cfg:         Összerakott TrainingConfig (build_training_config-ból).
        mgr:         Inicializált ModelManager.
        session_id:  Ha a train_gui.py már nyitott egy session bejegyzést,
                     ide adjuk; különben automatikusan nyit egy újat.
        output_base: Külső mentési gyökér (RunPod workspace, Google Drive).
                     None → projekt gyökér (_BASE_DIR).
    """
    _output_base = output_base or _BASE_DIR

    # ── Kiindulási epizódszám a checkpointból ────────────────────────────────
    episodes_start = 0
    if os.path.exists(pth_path):
        try:
            ck = safe_load_checkpoint(pth_path, map_location="cpu")
            if isinstance(ck, dict):
                episodes_start = ck.get("episodes_trained", 0)
        except Exception:
            pass

    # ── Modell mappa + naplo.json biztosítása ────────────────────────────────
    mgr.ensure_model_dir(model_name, num_players)

    # Ha a GUI már nyitott egy session bejegyzést (session_id átadva),
    # azt folytatjuk – nem nyitunk duplikált bejegyzést.
    _session_id = session_id or mgr.start_session(
        model_name, {}, episodes_start, num_players
    )

    # ── atexit: naplo.json lezárása tréning végén / megszakításkor ───────────
    def _cleanup():
        try:
            ep_end = episodes_start + episodes
            if os.path.exists(pth_path):
                ck = safe_load_checkpoint(pth_path, map_location="cpu")
                if isinstance(ck, dict):
                    ep_end = ck.get("episodes_trained", ep_end)

            # Utolsó metrikák kinyerése a legfrissebb session logból
            metrics: dict = {}
            log_dirs = [
                os.path.join(_BASE_DIR, "logs"),
                os.path.join(_output_base, "logs"),
            ]
            all_logs: list[str] = []
            for ld in log_dirs:
                all_logs.extend(_g.glob(os.path.join(ld, "session_*.log")))
            all_logs = sorted(set(all_logs))
            if all_logs:
                with open(all_logs[-1], "r", errors="replace") as f:
                    lines = f.readlines()
                for line in reversed(lines[-50:]):
                    if "Actor" in line and "Ep" in line:
                        for part in line.split("|"):
                            p = part.strip()
                            if p.startswith("Actor "):
                                try: metrics["actor_loss"] = float(p.split()[-1])
                                except: pass
                            elif p.startswith("Critic "):
                                try: metrics["critic_loss"] = float(p.split()[-1])
                                except: pass
                            elif p.startswith("Ent "):
                                try: metrics["entropy"] = float(p.split()[-1])
                                except: pass
                        break

            mgr.end_session(model_name, _session_id, ep_end, metrics, completed=True)
        except Exception as exc:
            logger.warning(f"Launcher cleanup hiba: {exc}")

    atexit.register(_cleanup)

    # ── Indítási összefoglaló (konzolra és logba) ────────────────────────────
    _banner = [
        f"{'='*62}",
        f"  \U0001f0cf  POKER AI v4 – Tréning indul",
        f"{'='*62}",
        f"  Modell      : {model_name}",
        f"  Checkpoint  : {pth_path}",
        f"  Kezdő ep.   : {episodes_start:,}",
        f"  Futtatandó  : {episodes:,}",
        f"  Cél ep.     : {episodes_start + episodes:,}",
        f"  Mérföldkő   : minden {cfg.milestone_interval:,} ep",
        f"  Tests mappa : {cfg.milestone_dir_root}",
        f"{'='*62}",
    ]
    for line in _banner:
        print(line)
        logger.info(line)

    # ── Tréning indítása ─────────────────────────────────────────────────────
    run_training_session(
        num_players=num_players,
        filename=pth_path,
        episodes_to_run=episodes,
        cfg=cfg,
    )
