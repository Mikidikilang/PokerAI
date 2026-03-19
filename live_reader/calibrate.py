#!/usr/bin/env python3
"""
live_reader/calibrate.py  –  Interaktív szék pozíció kalibráló

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HASZNÁLAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Nyisd meg a BoaBet-et egy teli asztalon
  2. Futtasd:
  
     python -m live_reader.calibrate --num-seats 9 --output configs/boabet_9max.json

  3. Megjelenik a screenshot – kattints a kért elemekre:
     - Minden szék közepére (játékos név/stack közepe)
     - A board közepére
     - A pot szövegre
  
  4. A program elmenti a configs/ mappába

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description='BoaBet szék pozíció kalibráló'
    )
    parser.add_argument(
        '--num-seats', type=int, default=9,
        help='Székek száma (default: 9)'
    )
    parser.add_argument(
        '--output', '-o', default='configs/boabet_9max.json',
        help='Kimeneti JSON fájl'
    )
    parser.add_argument(
        '--monitor', type=int, default=1,
        help='Monitor index (default: 1)'
    )
    parser.add_argument(
        '--screenshot', default=None,
        help='Kész screenshot fájl (ahelyett hogy élőben csinálna)'
    )
    args = parser.parse_args()

    try:
        import mss
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"\nHiányzó csomag: {e}")
        print("pip install mss opencv-python numpy")
        sys.exit(1)

    # Screenshot készítés vagy betöltés
    if args.screenshot and os.path.exists(args.screenshot):
        print(f"Screenshot betöltése: {args.screenshot}")
        image = cv2.imread(args.screenshot)
    else:
        print("Screenshot készítése...")
        with mss.mss() as sct:
            monitor = sct.monitors[args.monitor]
            screenshot = sct.grab(monitor)
            image = np.array(screenshot)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        print(f"  Méret: {image.shape[1]}x{image.shape[0]}")

    img_h, img_w = image.shape[:2]
    clicks = []
    current_step = [0]

    steps = []
    for i in range(args.num_seats):
        steps.append(f"Szék {i} közepe (játékos név/stack)")
    steps.append("Board közép (közösségi lapok területe)")
    steps.append("Pot szöveg közepe")

    print(f"\n{'='*55}")
    print(f"  Kattints a kért elemekre a képen!")
    print(f"  Összesen {len(steps)} kattintás szükséges.")
    print(f"  ESC = kilépés")
    print(f"{'='*55}\n")

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            step = current_step[0]
            if step >= len(steps):
                return
            
            x_rel = x / img_w
            y_rel = y / img_h
            clicks.append((x, y, x_rel, y_rel))
            
            # Rajzolj jelölőt
            cv2.circle(image, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(image, f"{step}", (x + 12, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            print(f"  [{step}] {steps[step]}: ({x_rel:.3f}, {y_rel:.3f})")
            
            current_step[0] += 1
            
            if current_step[0] < len(steps):
                title = f"KALIBRACIO - {steps[current_step[0]]} - Kattints!"
                cv2.setWindowTitle("calibration", title)
            else:
                cv2.setWindowTitle("calibration", "KESZ! Nyomj ESC-et a menteshez")
                print("\n  ✅ Minden pozíció rögzítve! Nyomj ESC-et.")

    cv2.namedWindow("calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("calibration", min(img_w, 1400), min(img_h, 800))
    cv2.setWindowTitle("calibration", f"KALIBRACIO - {steps[0]} - Kattints!")
    cv2.setMouseCallback("calibration", on_click)

    while True:
        cv2.imshow("calibration", image)
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

    if current_step[0] < args.num_seats:
        print("\n  ⚠ Nem minden szék lett kalibrálva!")
        print("  Futtasd újra és kattints minden elemre.")
        return

    # Config összerakása
    config = {
        'num_seats': args.num_seats,
        'image_size': [img_w, img_h],
        'seats': {},
        'board_region': {},
        'pot_region': {},
    }

    for i in range(args.num_seats):
        if i < len(clicks):
            _, _, x_rel, y_rel = clicks[i]
            config['seats'][str(i)] = {
                'x': round(x_rel, 4),
                'y': round(y_rel, 4),
                'r': 0.08,  # sugár
            }

    # Board
    board_idx = args.num_seats
    if board_idx < len(clicks):
        _, _, bx, by = clicks[board_idx]
        config['board_region'] = {
            'x': round(bx - 0.15, 4),
            'y': round(by - 0.05, 4),
            'w': 0.30,
            'h': 0.12,
        }

    # Pot
    pot_idx = args.num_seats + 1
    if pot_idx < len(clicks):
        _, _, px, py = clicks[pot_idx]
        config['pot_region'] = {
            'x': round(px - 0.10, 4),
            'y': round(py - 0.02, 4),
            'w': 0.20,
            'h': 0.05,
        }

    # Mentés
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Konfiguráció mentve: {args.output}")
    print(f"  Székek: {args.num_seats}")
    print(f"  Használat: python run_live.py --seat-config {args.output}")


if __name__ == '__main__':
    main()
