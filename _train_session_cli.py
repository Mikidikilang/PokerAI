#!/usr/bin/env python3
"""_train_session_cli.py -- CLI wrapper. Ne futtasd közvetlenül."""
import sys, os, json, argparse, atexit, glob as _g
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.runner import run_training_session
from training.model_manager import ModelManager, CONFIG_DEFAULTS
from config import TrainingConfig
from utils.checkpoint_utils import safe_load_checkpoint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mgr = ModelManager(BASE_DIR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",  required=True)
    parser.add_argument("--pth-path",    required=True)
    parser.add_argument("--players",     type=int, default=6)
    parser.add_argument("--episodes",    type=int, default=100_000)
    parser.add_argument("--config-json", default="{}")
    parser.add_argument("--session-id",  default=None)
    args = parser.parse_args()

    raw      = json.loads(args.config_json)
    bot_pool = raw.pop("bot_pool", CONFIG_DEFAULTS["bot_pool"])
    raw.pop("training_style", None); raw.pop("training_phase", None)

    valid = set(TrainingConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in raw.items() if k in valid}
    filtered["opponent_bot_types"]   = [k for k,v in bot_pool.items() if v.get("enabled",True)]
    filtered["opponent_bot_weights"] = [v.get("weight",1.0) for k,v in bot_pool.items() if v.get("enabled",True)]
    filtered["milestone_dir_root"]   = mgr.tests_dir(args.model_name)

    try:    cfg = TrainingConfig(**filtered)
    except: cfg = TrainingConfig()

    episodes_start = 0
    if os.path.exists(args.pth_path):
        try:
            ck = safe_load_checkpoint(args.pth_path, map_location="cpu")
            if isinstance(ck, dict):
                episodes_start = ck.get("episodes_trained", 0)
        except: pass

    session_id = args.session_id or mgr.start_session(
        args.model_name, json.loads(args.config_json), episodes_start, args.players)

    def _cleanup():
        try:
            ep_end = episodes_start + args.episodes
            if os.path.exists(args.pth_path):
                ck = safe_load_checkpoint(args.pth_path, map_location="cpu")
                if isinstance(ck, dict): ep_end = ck.get("episodes_trained", ep_end)
            metrics = {}
            logs = sorted(_g.glob(os.path.join(BASE_DIR, "logs", "session_*.log")))
            if logs:
                with open(logs[-1], "r", errors="replace") as f:
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
            mgr.end_session(args.model_name, session_id, ep_end, metrics, completed=True)
        except Exception as ex:
            print(f"Cleanup hiba: {ex}")
    atexit.register(_cleanup)

    run_training_session(
        num_players=args.players,
        filename=args.pth_path,
        episodes_to_run=args.episodes,
        cfg=cfg,
    )

if __name__ == "__main__":
    main()
