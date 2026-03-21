#!/usr/bin/env python3
"""
run_live.py  –  Élő BoaBet póker olvasó + RTA javaslatok

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HASZNÁLAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python run_live.py --yolo models/poker_yolo_best.pt
                     --model 9max_ppo_v4.pth
                     --my-seat 7
                     --bb 0.04 --sb 0.02

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ELŐFELTÉTELEK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Betanított YOLO modell (poker_yolo_best.pt)
  2. Betanított PPO modell (9max_ppo_v4.pth vagy 6max stb.)
  3. BoaBet nyitva a böngészőben

  pip install mss opencv-python ultralytics pytesseract numpy
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging():
    """Konzol + fájl logolás."""
    logger = logging.getLogger("PokerAI")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Konzol – INFO
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # Fájl – DEBUG
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(f"logs/live_{ts}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    return logger


def print_recommendation(rec: dict):
    """Javaslat szép kiírása a konzolra."""
    action = rec.get('action_name', '?')
    conf = rec.get('confidence', 0) * 100
    equity = rec.get('equity', 0) * 100
    spr = rec.get('spr', 0)
    explanation = rec.get('explanation', '')

    print()
    print("  ╔═══════════════════════════════════════════════╗")
    print(f"  ║  ★ {action:^20s}  ({conf:.0f}%)         ║")
    print(f"  ║  Equity: {equity:.0f}%  |  SPR: {spr:.1f}                 ║")
    print(f"  ╚═══════════════════════════════════════════════╝")

    if rec.get('top3'):
        top3_str = " | ".join(
            f"{name} {p*100:.0f}%" for name, p in rec['top3']
        )
        print(f"  Top 3: {top3_str}")

    if explanation:
        print(f"  {explanation}")
    print()


def print_hand_start(info: dict):
    """Új kéz kiírása."""
    cards = info.get('hole_cards', [])
    cards_str = ' '.join(cards) if cards else '??'
    print(f"\n  ═══ Hand #{info.get('hand_num', 0)} ═══  "
          f"Lapok: {cards_str}  |  "
          f"Stack: ${info.get('my_stack', 0):.2f}  |  "
          f"{info.get('num_players', 0)}p")


def print_opponent_action(info: dict):
    """Ellenfél akció kiírása."""
    amount_str = f" ${info['amount']:.2f}" if info.get('amount', 0) > 0 else ""
    print(f"    {info.get('username', '?')}: {info.get('action', '?')}{amount_str}")


def main():
    parser = argparse.ArgumentParser(
        description='Élő BoaBet póker olvasó + AI javaslatok'
    )
    parser.add_argument(
        '--yolo', required=True,
        help='YOLO modell elérési útja (poker_yolo_best.pt)'
    )
    parser.add_argument(
        '--model', default=None,
        help='PPO modell .pth fájl (None = nincs RTA, csak olvasás)'
    )
    parser.add_argument(
        '--db', default='players.db',
        help='Ellenfél adatbázis (default: players.db)'
    )
    parser.add_argument(
        '--my-seat', type=int, default=7,
        help='Saját szék index (default: 7, alsó közép a BoaBet-en)'
    )
    parser.add_argument(
        '--bb', type=float, default=0.04,
        help='Big blind (default: $0.04)'
    )
    parser.add_argument(
        '--sb', type=float, default=0.02,
        help='Small blind (default: $0.02)'
    )
    parser.add_argument(
        '--fps', type=float, default=2.0,
        help='Képernyőolvasás FPS (default: 2.0)'
    )
    parser.add_argument(
        '--seat-config', default=None,
        help='Szék pozíciók JSON (default: beépített BoaBet 9-max)'
    )
    parser.add_argument(
        '--monitor', type=int, default=1,
        help='Monitor index (default: 1)'
    )
    parser.add_argument(
        '--region', default=None,
        help='Fix screenshot régió: "x,y,w,h"'
    )
    parser.add_argument(
        '--confidence', type=float, default=0.5,
        help='YOLO minimum confidence (default: 0.5)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Debug mód: menti a detekciós képeket'
    )
    parser.add_argument(
        '--read-only', action='store_true',
        help='Csak olvasás, nincs RTAManager (teszteléshez)'
    )
    args = parser.parse_args()

    logger = setup_logging()

    # ── Capture region ────────────────────────────────────────────────────
    capture_region = None
    if args.region:
        parts = [int(x) for x in args.region.split(',')]
        capture_region = {
            'left': parts[0], 'top': parts[1],
            'width': parts[2], 'height': parts[3],
        }

    # ── ScreenReader ──────────────────────────────────────────────────────
    from live_reader.screen_reader import ScreenReader

    reader = ScreenReader(
        yolo_model_path=args.yolo,
        seat_config_path=args.seat_config,
        my_seat=args.my_seat,
        window_title="BoaBet",
        yolo_confidence=args.confidence,
        monitor_index=args.monitor,
        capture_region=capture_region,
    )

    print("\n" + "=" * 55)
    print("  🃏  BoaBet Live Reader + AI Assistant")
    print("=" * 55)
    print(f"  YOLO modell:   {args.yolo}")
    print(f"  Saját szék:    {args.my_seat}")
    print(f"  Blindok:       ${args.sb}/${args.bb}")
    print(f"  FPS:           {args.fps}")

    # ── Read-only mód (RTAManager nélkül) ─────────────────────────────────
    if args.read_only or args.model is None:
        print(f"\n  ⚡ READ-ONLY mód (nincs AI javaslat)")
        print(f"  Ctrl+C a leállításhoz")
        print("=" * 55)

        try:
            frame_count = 0
            while True:
                state = reader.read_frame()
                frame_count += 1
                if state.is_valid:
                    board_str = ' '.join(str(c) for c in state.board_cards) or '-'
                    hole_str = ' '.join(str(c) for c in state.my_hole_cards) or '-'
                    pot_str = f"${state.pot_total:.2f}" if state.pot_total else "-"
                    active = state.active_player_seat

                    print(
                        f"  [{frame_count:4d}] "
                        f"{state.frame_ms:5.0f}ms | "
                        f"Board: {board_str:20s} | "
                        f"Hand: {hole_str:8s} | "
                        f"Pot: {pot_str:8s} | "
                        f"Active: seat {active} | "
                        f"Errors: {len(state.ocr_errors)}"
                    )

                    if args.debug and state.raw_detections:
                        os.makedirs("debug", exist_ok=True)
                        # TODO: save_debug_frame
                else:
                    print(f"  [{frame_count:4d}] INVALID FRAME")

                time.sleep(1.0 / args.fps)

        except KeyboardInterrupt:
            print("\n  Leállítva.")
        return

    # ── RTAManager ────────────────────────────────────────────────────────
    from inference.rta_manager import RTAManager

    # Próbáljuk kitalálni a játékosszámot a modell state_size-ából
    import torch
    from core.features import compute_state_size
    from utils.checkpoint_utils import safe_load_checkpoint

    ck = safe_load_checkpoint(args.model, map_location='cpu')
    state_size = ck.get('state_size', 475)
    guessed_players = 9  # default
    for np_ in range(2, 10):
        if compute_state_size(54, np_) == state_size:
            guessed_players = np_
            break

    print(f"  PPO modell:    {args.model} ({guessed_players}p)")
    print(f"  Ellenfél DB:   {args.db}")

    manager = RTAManager(
        model_paths={guessed_players: args.model},
        db_path=args.db,
        device='cpu',
        equity_sims=300,
    )

    # ── GameBridge ────────────────────────────────────────────────────────
    from live_reader.game_bridge import GameBridge

    bridge = GameBridge(
        screen_reader=reader,
        rta_manager=manager,
        my_seat=args.my_seat,
        bb=args.bb,
        sb=args.sb,
    )

    # Callbacks
    bridge.on_recommendation = print_recommendation
    bridge.on_hand_start = print_hand_start
    bridge.on_opponent_action = print_opponent_action

    print(f"\n  ✅ Minden kész! Játssz a BoaBet-en és figyeld a javaslatokat.")
    print(f"  Ctrl+C a leállításhoz")
    print("=" * 55 + "\n")

    # ── Futtatás ──────────────────────────────────────────────────────────
    bridge.run(fps=args.fps)


if __name__ == '__main__':
    main()
