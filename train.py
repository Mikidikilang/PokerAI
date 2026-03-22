#!/usr/bin/env python3
"""
train.py  --  Poker AI v4 parancssoros tréning indító (RunPod / headless)

100%-ban kompatibilis a train_gui.py-al:
  • Ugyanazt a ModelManager-alapú mentési rendszert használja
  • A checkpoint formátum változatlan
  • GUI-ból leállított tréning CLI-ből folytatható (és fordítva)
  • A naplo.json session bejegyzések mindkét esetben azonosak

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PÉLDÁK:

  # Új 2-max modell, exploitative mód, 1 millió iteráció:
  python train.py --model_name 2max --mode exploitative --iterations 1000000

  # Folytatás GUI-ból leállított 6-max modelltől (--players nem kell,
  # betölti a config.json-ból):
  python train.py --model_name 6max --iterations 500000

  # RunPod, egyedi save_freq és test_hands:
  python train.py --model_name 2max --mode exploitative \\
                  --iterations 2000000 --save_freq 500000 --test_hands 500

  # Külső mentési könyvtár (pl. /workspace persistent storage):
  python train.py --model_name 2max --iterations 1000000 \\
                  --output_dir /workspace/poker_models
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging

from utils.logging_setup import setup_logging
from training.model_manager import ModelManager, CONFIG_DEFAULTS, STYLE_PRESETS
from training.launcher import build_training_config, launch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mód neve a CLI-n → STYLE_PRESETS kulcs a model_manager-ben
_MODE_TO_PRESET = {
    "selfplay":    "self_play",
    "exploitative": "exploitative",
    "aggressive":  "aggressive",
    "custom":      None,   # None = ne alkalmazz presetet, maradjon a meglévő config
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description=(
            "Poker AI v4 – Parancssoros tréning indító\n"
            "Kompatibilis a train_gui.py-al (azonos ModelManager + checkpoint)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MÓDOK (--mode):
  selfplay      Csak önmaga ellen játszik (magas entropy, exploration fókusz)
  exploitative  Különböző botok ellen: fish 🐟, nit 🎯, LAG 💣, calling station 📞
  aggressive    Erős LAG + nit mix (alacsony entropy, tight-aggressive)
  custom        A modell meglévő konfigurációját használja változatlanul

MÉRFÖLDKŐ (--save_freq):
  Minden N. epizódnál snapshot + automatikus sanity teszt fut.
  Lokálisan ajánlott: 2_000_000 (default)
  RunPod-on ajánlott: 500_000

PÉLDÁK:
  python train.py --model_name 2max --mode exploitative --iterations 1000000
  python train.py --model_name 6max --iterations 500000 --save_freq 200000
  python train.py --model_name 2max --mode selfplay --iterations 2000000 \\
                  --save_freq 500000 --test_hands 500 --output_dir /workspace/models
""",
    )

    # ── Kötelező ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model_name",
        required=True,
        metavar="NÉV",
        help=(
            "Modell azonosítója (pl. '2max', '6max', 'my_experiment'). "
            "Ha a models/ mappában már létezik, betölti onnan a konfigurációt "
            "és a checkpointot → folytatja az eddigi tréninget."
        ),
    )

    # ── Tréning paraméterek ───────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        choices=list(_MODE_TO_PRESET.keys()),
        default=None,
        metavar="MÓD",
        help=(
            "Tanítási mód / stílus preset: selfplay | exploitative | aggressive | custom. "
            "Ha nincs megadva: a modell meglévő konfigurációját használja "
            "(új modellnél: 'exploitative')."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Futtatandó iterációk (epizódok) száma. "
            "Ha nincs megadva, interaktívan kéri be (headless futtatásnál mindig add meg!)."
        ),
    )
    parser.add_argument(
        "--players",
        type=int,
        default=None,
        choices=range(2, 10),
        metavar="2-9",
        help=(
            "Játékosok száma (2-9). "
            "Ha a modell már létezik, a config.json-ból tölti be – "
            "ekkor nem szükséges megadni. Új modellnél default: 6."
        ),
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Mérföldkő mentési intervallum epizódban. "
            "Felülírja a konfigurációban tárolt értéket. "
            "Lokálisan: ~2_000_000 | RunPod-on: ~500_000."
        ),
    )
    parser.add_argument(
        "--test_hands",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Mérföldkő sanity teszthez használt kézszám. "
            "Felülírja a konfigurációban tárolt értéket. "
            "Default: 2000 (lokálisan), csökkenthető pl. 500-ra gyors tesztnél."
        ),
    )

    # ── Infrastruktúra ────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir",
        default=None,
        metavar="PATH",
        help=(
            "Külső mentési gyökér (pl. /workspace/models RunPod-on vagy "
            "Google Drive csatolási pont). "
            "Ha nincs megadva: projekt models/ mappájába ment."
        ),
    )
    parser.add_argument(
        "--session_id",
        default=None,
        help=(
            "Folytatott napló session ID (opcionális). "
            "Normál esetben automatikusan generálódik; "
            "csak speciális esetben add meg kézzel."
        ),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── Logging ──────────────────────────────────────────────────────────────
    setup_logging("training.log")
    logger = logging.getLogger("PokerAI")

    # ── ModelManager inicializálása ───────────────────────────────────────────
    output_base = args.output_dir or BASE_DIR
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Külső mentési könyvtár: {args.output_dir}")

    mgr = ModelManager(output_base)

    # ── Meglévő konfiguráció betöltése ────────────────────────────────────────
    existing_cfg_data = mgr.load_config(args.model_name)
    stored_config     = existing_cfg_data.get("config", dict(CONFIG_DEFAULTS))
    stored_players    = existing_cfg_data.get("num_players")

    # Játékosszám: CLI > tárolt config > default (6)
    num_players = args.players or stored_players or 6
    logger.info(
        f"Játékosszám: {num_players} "
        f"({'CLI' if args.players else 'config.json' if stored_players else 'default'})"
    )

    # ── Iterációk száma (interaktív fallback headless-ellenes) ───────────────
    if args.iterations is None:
        try:
            ep_str = input(
                f"\nHány iterációt futtassunk? "
                f"[{args.model_name}, {num_players}p] "
                f"(pl. 500000): "
            ).strip()
            iterations = int(ep_str)
        except (ValueError, EOFError, KeyboardInterrupt):
            parser.error(
                "--iterations nincs megadva és nem-interaktív módban fut. "
                "Add meg a --iterations N argumentumot."
            )
    else:
        iterations = args.iterations

    # ── Mód / stílus preset alkalmazása ──────────────────────────────────────
    # Ha --mode nincs megadva, a meglévő training_style-t veszi alapul.
    # Új modellnél (nincs training_style a config-ban): exploitative default.
    effective_mode = args.mode or stored_config.get("training_style", "exploitative")
    preset_key = _MODE_TO_PRESET.get(effective_mode)

    if preset_key is not None and preset_key in STYLE_PRESETS:
        # Preset alkalmazása (pontosan úgy, ahogy a GUI /api/apply_preset végpontja csinálja)
        config_to_use = mgr.apply_style_preset(stored_config, preset_key)
        logger.info(f"Stílus preset alkalmazva: '{effective_mode}' (kulcs: '{preset_key}')")
    else:
        # 'custom' mód vagy ismeretlen preset → meglévő konfig változatlan
        config_to_use = stored_config
        logger.info(f"Mód: 'custom' – meglévő konfig változatlan.")

    # ── TrainingConfig összerakása ────────────────────────────────────────────
    cfg = build_training_config(
        config_dict=config_to_use,
        model_name=args.model_name,
        mgr=mgr,
        milestone_interval=args.save_freq,
        milestone_hands=args.test_hands,
    )

    # ── Modell mappa + pth elérési út ────────────────────────────────────────
    mgr.ensure_model_dir(args.model_name, num_players)
    pth_path = mgr.pth_path(args.model_name)

    # ── Konfiguráció mentése – visszaolvasható a GUI-ból is ──────────────────
    updated_cfg_data = dict(existing_cfg_data)
    updated_cfg_data["num_players"] = num_players
    updated_cfg_data["config"]      = config_to_use
    mgr.save_config(args.model_name, updated_cfg_data)
    logger.info(f"Konfig mentve: models/{args.model_name}/config.json")

    # ── Tréning indítása (azonos launcher mint a GUI subprocess-énél) ─────────
    launch(
        model_name=args.model_name,
        pth_path=pth_path,
        num_players=num_players,
        episodes=iterations,
        cfg=cfg,
        mgr=mgr,
        session_id=args.session_id,
        output_base=output_base,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.getLogger("PokerAI").info("Tréning megszakítva (Ctrl+C).")
    except SystemExit:
        raise
    except Exception:
        logging.getLogger("PokerAI").critical(
            "Nem kezelt kivétel:", exc_info=True
        )
        sys.exit(1)
