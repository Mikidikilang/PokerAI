"""
training/launcher.py  --  Közös tréning indító logika (GUI + CLI)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ezt a modult hívja meg MINDKÉT belépési pont:

  train.py               ← közvetlen CLI / RunPod / headless indítás
  _train_session_cli.py  ← train_gui.py által spawnolt subprocess

Változások v4.2.2-BUGFIX:
    [BUG-3] launch() + _cleanup(): unsafe checkpoint fallback.
        safe_load_checkpoint() allow_unsafe=False-szal UnsafeCheckpointError-t
        dobott legacy checkpointoknál → episodes_start=0 csendben.
        Következmény: a GUI mindig 0-ról indult, naplo.json sosem frissült.
        Javítás: _load_ck() helper safe→unsafe fallback lánccal.

    [BUG-1] _cleanup(): log minta javítva session_*.log → train_ui_*.log.
        A _open_session_log() mindig train_ui_{name}_{ts}.log névvel ment,
        ezért a session_*.log minta sosem talált semmit → ures metrics
        kerültek az end_session hívásba.
"""
from __future__ import annotations

import atexit
import glob as _g
import logging
import os

from config import TrainingConfig
from training.model_manager import ModelManager, CONFIG_DEFAULTS
from training.runner import run_training_session
from utils.checkpoint_utils import safe_load_checkpoint, UnsafeCheckpointError

logger = logging.getLogger("PokerAI")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_ck(path: str) -> dict | None:
    """
    Checkpoint betoltese safe, unsafe fallback-kel.

    [BUG-3 FIX] A legacy checkpointok (RunPod-on vagy más eszközön mentett
    v4.1 előtti fájlok) weights_only=False-t igényelnek. Saját, helyi
    fájloknál ez elfogadható.

    Returns:
        Betöltött dict, vagy None ha a fájl nem létezik / nem olvasható.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        return safe_load_checkpoint(path, map_location="cpu", allow_unsafe=False)
    except UnsafeCheckpointError:
        logger.info(
            f"Legacy checkpoint ({os.path.basename(path)!r}), "
            f"unsafe betöltés. Ajánlott: futtasd a migrate_checkpoint_to_safe()-t."
        )
        try:
            return safe_load_checkpoint(path, map_location="cpu", allow_unsafe=True)
        except Exception as exc:
            logger.warning(f"Checkpoint betoltesi hiba ({path!r}): {exc}")
            return None
    except Exception as exc:
        logger.warning(f"Checkpoint betoltesi hiba ({path!r}): {exc}")
        return None


def build_training_config(
    config_dict: dict,
    model_name: str,
    mgr: ModelManager,
    milestone_interval: int | None = None,
    milestone_hands: int | None = None,
) -> TrainingConfig:
    """
    GUI-kompatibilis config_dict-ből épít TrainingConfig-ot.
    """
    raw = dict(config_dict)
    bot_pool = raw.pop("bot_pool", CONFIG_DEFAULTS["bot_pool"])
    raw.pop("training_style", None)
    raw.pop("training_phase", None)

    valid    = set(TrainingConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in raw.items() if k in valid}

    filtered["opponent_bot_types"] = [
        k for k, v in bot_pool.items() if v.get("enabled", True)
    ]
    filtered["opponent_bot_weights"] = [
        v.get("weight", 1.0) for k, v in bot_pool.items() if v.get("enabled", True)
    ]

    filtered["milestone_dir_root"] = mgr.tests_dir(model_name)

    if milestone_interval is not None:
        filtered["milestone_interval"] = milestone_interval
        logger.info(f"milestone_interval feluliriva: {milestone_interval:,} ep")

    if milestone_hands is not None:
        filtered["milestone_hands"] = milestone_hands
        logger.info(f"milestone_hands feluliriva: {milestone_hands}")

    try:
        return TrainingConfig(**filtered)
    except Exception as exc:
        logger.warning(f"TrainingConfig hiba ({exc}), default konfig hasznalata.")
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
    """
    _output_base = output_base or _BASE_DIR

    # [BUG-3 FIX] _load_ck() safe+unsafe fallback-kel olvassa a checkpointot
    episodes_start = 0
    ck = _load_ck(pth_path)
    if ck is not None and isinstance(ck, dict):
        episodes_start = ck.get("episodes_trained", 0)

    mgr.ensure_model_dir(model_name, num_players)

    _session_id = session_id or mgr.start_session(
        model_name, {}, episodes_start, num_players
    )

    # atexit: naplo.json lezárása tréning végén / megszakításkor
    def _cleanup():
        try:
            ep_end = episodes_start + episodes
            # [BUG-3 FIX] unsafe fallback a cleanup-ban is
            ck_end = _load_ck(pth_path)
            if ck_end is not None and isinstance(ck_end, dict):
                ep_end = ck_end.get("episodes_trained", ep_end)

            # [BUG-1 FIX] Helyes log minta: train_ui_*.log (nem session_*.log)
            metrics: dict = {}
            log_dirs = [
                os.path.join(_BASE_DIR, "logs"),
                os.path.join(_output_base, "logs"),
            ]
            all_logs: list[str] = []
            for ld in log_dirs:
                # train_ui_*.log – a _open_session_log() által generált név
                all_logs.extend(_g.glob(os.path.join(ld, "train_ui_*.log")))
            all_logs = sorted(set(all_logs))
            if all_logs:
                with open(all_logs[-1], "r", errors="replace") as f:
                    lines = f.readlines()
                for line in reversed(lines[-50:]):
                    if "Actor" in line and "Ep" in line:
                        for part in line.split("|"):
                            p = part.strip()
                            if p.startswith("Actor "):
                                try:
                                    metrics["actor_loss"] = float(p.split()[-1])
                                except Exception:
                                    pass
                            elif p.startswith("Critic "):
                                try:
                                    metrics["critic_loss"] = float(p.split()[-1])
                                except Exception:
                                    pass
                            elif p.startswith("Ent "):
                                try:
                                    metrics["entropy"] = float(p.split()[-1])
                                except Exception:
                                    pass
                        break

            mgr.end_session(model_name, _session_id, ep_end, metrics, completed=True)
        except Exception as exc:
            logger.warning(f"Launcher cleanup hiba: {exc}")

    atexit.register(_cleanup)

    # Indítási összefoglaló
    _banner = [
        f"{'='*62}",
        f"  POKER AI v4 -- Trening indul",
        f"{'='*62}",
        f"  Modell      : {model_name}",
        f"  Checkpoint  : {pth_path}",
        f"  Kezdo ep.   : {episodes_start:,}",
        f"  Futtatando  : {episodes:,}",
        f"  Cel ep.     : {episodes_start + episodes:,}",
        f"  Merfoldko   : minden {cfg.milestone_interval:,} ep",
        f"  Tests mappa : {cfg.milestone_dir_root}",
        f"{'='*62}",
    ]
    for line in _banner:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", errors="replace").decode("ascii"))
        logger.info(line)

    run_training_session(
        num_players=num_players,
        filename=pth_path,
        episodes_to_run=episodes,
        cfg=cfg,
    )
