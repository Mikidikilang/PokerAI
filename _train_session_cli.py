#!/usr/bin/env python3
"""
_train_session_cli.py  --  GUI subprocess wrapper (ne futtasd közvetlenül!)

Ezt a fájlt a train_gui.py indítja subprocessként a /api/start végponton.
Közvetlen CLI / RunPod indításhoz használd a train.py-t!

A tényleges logika a training/launcher.py-ban van, amit ez a script
és a train.py is meghív – így a GUI és a CLI 100%-ban kompatibilis:
azonos mentési rendszer, azonos konfig, azonos checkpoint formátum.

Argumentumok (belső, GUI által generált):
  --model-name        Modell neve
  --pth-path          Checkpoint .pth fájl elérési útja
  --players           Játékosok száma (2-9)
  --episodes          Futtatandó epizódok száma
  --config-json       TrainingConfig override JSON string (GUI-ból)
  --session-id        Napló session ID (GUI által előre megnyitott)
  --milestone-interval Mérföldkő intervallum felülírása (opcionális)
  --drive-output-dir  Külső mentési könyvtár (Google Drive / RunPod)
"""
import sys
import os
import json
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logging_setup import setup_logging
from training.model_manager import ModelManager, CONFIG_DEFAULTS
from training.launcher import build_training_config, launch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Poker AI v4 – GUI subprocess wrapper "
            "(közvetlen futtatáshoz használd a train.py-t)"
        )
    )
    parser.add_argument("--model-name",         required=True)
    parser.add_argument("--pth-path",           required=True)
    parser.add_argument("--players",            type=int,  default=6)
    parser.add_argument("--episodes",           type=int,  default=100_000)
    parser.add_argument("--config-json",        default="{}")
    parser.add_argument("--session-id",         default=None)
    parser.add_argument("--milestone-interval", type=int,  default=None)
    parser.add_argument("--drive-output-dir",   default=None)
    args = parser.parse_args()

    setup_logging("training.log", num_players=args.players)

    output_base = args.drive_output_dir or BASE_DIR
    if args.drive_output_dir:
        os.makedirs(args.drive_output_dir, exist_ok=True)

    mgr = ModelManager(output_base)

    config_dict = json.loads(args.config_json)

    cfg = build_training_config(
        config_dict=config_dict,
        model_name=args.model_name,
        mgr=mgr,
        milestone_interval=args.milestone_interval,
    )

    launch(
        model_name=args.model_name,
        pth_path=args.pth_path,
        num_players=args.players,
        episodes=args.episodes,
        cfg=cfg,
        mgr=mgr,
        session_id=args.session_id,
        output_base=output_base,
    )


if __name__ == "__main__":
    main()
