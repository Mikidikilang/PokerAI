#!/usr/bin/env python3
"""
_train_session_cli.py -- CLI wrapper a train_gui.py számára (és Colab-hoz).

[COLAB MOD v1] Új argumentumok:
  --milestone-interval INT
        Felülírja a config-ban lévő milestone_interval értékét.
        Lokális GUI futtatásnál: nincs megadva → marad a config értéke (2_000_000).
        Colab futtatásnál: tipikusan 500_000.

  --drive-output-dir PATH
        Ha meg van adva, a ModelManager (és így a config.json, naplo.json,
        a .pth checkpoint és a tests/ mappa) ebbe a külső könyvtárba kerül.
        Lokális futtatásnál: nincs megadva → projekt gyökér alatt models/.
        Colab futtatásnál: /content/drive/MyDrive/PokerAI_Models

VISSZAFELÉ KOMPATIBILITÁS:
  Mindkét új argumentum opcionális és default értéke None / nem alkalmazott.
  A lokális GUI (train_gui.py) nem adja meg őket → semmi sem változik.
"""
import sys, os, json, argparse, atexit, glob as _g
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.runner import run_training_session
from training.model_manager import ModelManager, CONFIG_DEFAULTS
from config import TrainingConfig
from utils.checkpoint_utils import safe_load_checkpoint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="Poker AI v4 – CLI tréning indító (GUI és Colab kompatibilis)"
    )
    parser.add_argument("--model-name",  required=True,
                        help="Modell neve (pl. '2max')")
    parser.add_argument("--pth-path",    required=True,
                        help="Checkpoint .pth elérési útja")
    parser.add_argument("--players",     type=int, default=6,
                        help="Játékosok száma (2-9)")
    parser.add_argument("--episodes",    type=int, default=100_000,
                        help="Futtatandó epizódok száma")
    parser.add_argument("--config-json", default="{}",
                        help="TrainingConfig override JSON string")
    parser.add_argument("--session-id",  default=None,
                        help="Folytatott session ID (opcionális)")

    # ── [COLAB MOD v1] Új argumentumok ──────────────────────────────────
    parser.add_argument(
        "--milestone-interval", type=int, default=None,
        help=(
            "Mérföldkő mentési intervallum (epizód). "
            "Felülírja a config-ban lévő értéket. "
            "Pl. 500000 Colab-on, elhagyható lokálisan."
        )
    )
    parser.add_argument(
        "--drive-output-dir", default=None,
        help=(
            "Külső mentési könyvtár (pl. Google Drive csatolási pont). "
            "Ha megadva, a ModelManager ide menti a modell mappát "
            "(config.json, naplo.json, .pth, tests/). "
            "Lokálisan hagyható üresen."
        )
    )
    # ────────────────────────────────────────────────────────────────────
    args = parser.parse_args()

    # ── ModelManager inicializálása ──────────────────────────────────────
    # Ha --drive-output-dir meg van adva: Drive-ra ment.
    # Ha nincs: a projekt gyökér alatt lévő models/ mappába (eredeti viselkedés).
    if args.drive_output_dir:
        output_base = args.drive_output_dir
        os.makedirs(output_base, exist_ok=True)
    else:
        output_base = BASE_DIR

    mgr = ModelManager(output_base)

    # ── Config összerakás ────────────────────────────────────────────────
    raw      = json.loads(args.config_json)
    bot_pool = raw.pop("bot_pool", CONFIG_DEFAULTS["bot_pool"])
    raw.pop("training_style", None)
    raw.pop("training_phase", None)

    valid = set(TrainingConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in raw.items() if k in valid}
    filtered["opponent_bot_types"]   = [k for k, v in bot_pool.items() if v.get("enabled", True)]
    filtered["opponent_bot_weights"] = [v.get("weight", 1.0) for k, v in bot_pool.items() if v.get("enabled", True)]

    # Milestone mentési gyökér = modell tests/ mappája (Drive-on vagy lokálisan)
    filtered["milestone_dir_root"] = mgr.tests_dir(args.model_name)

    # [COLAB MOD v1] milestone_interval felülírása ha CLI argumentumból jön
    if args.milestone_interval is not None:
        filtered["milestone_interval"] = args.milestone_interval
        print(
            f"[COLAB] milestone_interval felülírva: "
            f"{args.milestone_interval:,} epizód"
        )

    try:
        cfg = TrainingConfig(**filtered)
    except Exception as e:
        print(f"TrainingConfig hiba ({e}), default konfig használata.")
        cfg = TrainingConfig()

    # ── Checkpoint ellenőrzés (folytatott tréning) ───────────────────────
    episodes_start = 0
    if os.path.exists(args.pth_path):
        try:
            ck = safe_load_checkpoint(args.pth_path, map_location="cpu")
            if isinstance(ck, dict):
                episodes_start = ck.get("episodes_trained", 0)
        except Exception:
            pass

    # ── Modell mappa biztosítása (Drive-on is) ───────────────────────────
    mgr.ensure_model_dir(args.model_name, args.players)

    # ── Session napló indítása ────────────────────────────────────────────
    session_id = args.session_id or mgr.start_session(
        args.model_name, json.loads(args.config_json), episodes_start, args.players
    )

    # ── Cleanup regisztráció (tréning végi napló lezárás) ────────────────
    def _cleanup():
        try:
            ep_end = episodes_start + args.episodes
            if os.path.exists(args.pth_path):
                ck = safe_load_checkpoint(args.pth_path, map_location="cpu")
                if isinstance(ck, dict):
                    ep_end = ck.get("episodes_trained", ep_end)
            metrics = {}
            # Log keresése – lokális és Drive esetén is működik
            log_dirs = [
                os.path.join(BASE_DIR, "logs"),
                os.path.join(output_base, "logs"),
            ]
            logs = []
            for ld in log_dirs:
                logs.extend(_g.glob(os.path.join(ld, "session_*.log")))
            logs = sorted(set(logs))
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

    # ── Tréning indítása ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Modell:    {args.model_name}")
    print(f"  Checkpoint:{args.pth_path}")
    print(f"  Epizódok:  {args.episodes:,}")
    print(f"  Mérföldkő: minden {cfg.milestone_interval:,} ep")
    print(f"  Tests mappa: {cfg.milestone_dir_root}")
    print(f"{'='*60}\n")

    run_training_session(
        num_players=args.players,
        filename=args.pth_path,
        episodes_to_run=args.episodes,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
