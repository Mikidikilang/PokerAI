#!/usr/bin/env python3
"""
live_reader/collect_screenshots.py  –  Screenshot gyűjtő a YOLO tréninghez

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HASZNÁLAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Nyisd meg a BoaBet-et a böngészőben
  2. Ülj le egy asztalhoz (play money jó teszteléshez)
  3. Futtasd:
  
     python -m live_reader.collect_screenshots --output screenshots/ --interval 3

  4. Játssz néhány kézet (vagy csak nézd a játékot)
     A script 3 másodpercenként ment egy screenshotot
  
  5. Ctrl+C leállítás
  6. Az eredmény: screenshots/ mappában PNG fájlok

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CÉL: 500-1000 screenshot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  A változatosság a kulcs! Gyűjts screenshotokat:
  - Különböző játékfázisokból (preflop, flop, turn, river)
  - Különböző számú játékossal (2-9 fő)
  - Fold-ok közben (kevesebb játékos, "Fold" badge-ek)
  - All-in szituációkból
  - Showdown-ból (látható kártyák)
  - Üres asztalról is (kontroll)

  ~3 másodperces intervallumal ~20 perc játék alatt 400 kép gyűlik.
  30-40 perc játékkal 600-800 kép lesz – ez bőven elég.
"""

import argparse
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser(
        description='BoaBet screenshot gyűjtő YOLO tréninghez'
    )
    parser.add_argument(
        '--output', '-o', default='screenshots',
        help='Kimeneti mappa (default: screenshots/)'
    )
    parser.add_argument(
        '--interval', '-i', type=float, default=3.0,
        help='Screenshotok közti szünet másodpercben (default: 3)'
    )
    parser.add_argument(
        '--monitor', '-m', type=int, default=1,
        help='Monitor index (1 = elsődleges, default: 1)'
    )
    parser.add_argument(
        '--region', '-r', type=str, default=None,
        help='Fix régió: "x,y,width,height" (pl. "0,100,1440,780")'
    )
    parser.add_argument(
        '--max', type=int, default=1000,
        help='Maximum screenshotok száma (default: 1000)'
    )
    args = parser.parse_args()

    # Importok
    try:
        import mss
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"\nHiányzó csomag: {e}")
        print("Telepítsd: pip install mss opencv-python numpy")
        sys.exit(1)

    # Kimeneti mappa
    os.makedirs(args.output, exist_ok=True)

    # Régió parse
    capture_region = None
    if args.region:
        try:
            parts = [int(x) for x in args.region.split(',')]
            capture_region = {
                'left': parts[0], 'top': parts[1],
                'width': parts[2], 'height': parts[3],
            }
            print(f"Fix régió: {capture_region}")
        except (ValueError, IndexError):
            print(f"Hibás régió formátum: {args.region}")
            print("Helyes formátum: x,y,width,height (pl. 0,100,1440,780)")
            sys.exit(1)

    print("=" * 60)
    print("  📸  BoaBet Screenshot Gyűjtő")
    print("=" * 60)
    print(f"\n  Kimeneti mappa:  {os.path.abspath(args.output)}")
    print(f"  Intervallum:     {args.interval} mp")
    print(f"  Monitor:         {args.monitor}")
    print(f"  Maximum:         {args.max} kép")
    print(f"\n  Ctrl+C a leállításhoz")
    print("=" * 60)

    count = 0
    start_time = time.time()

    try:
        with mss.mss() as sct:
            monitors = sct.monitors
            print(f"\n  Elérhető monitorok: {len(monitors) - 1}")
            for i, m in enumerate(monitors[1:], 1):
                print(f"    Monitor {i}: {m['width']}x{m['height']} "
                      f"at ({m['left']}, {m['top']})")

            if capture_region:
                monitor = capture_region
            else:
                monitor = monitors[args.monitor]

            print(f"\n  Használt terület: {monitor}")
            print(f"\n  Gyűjtés indítása...\n")

            while count < args.max:
                # Screenshot
                screenshot = sct.grab(monitor)
                image = np.array(screenshot)
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

                # Mentés
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"boabet_{timestamp}_{count:04d}.png"
                filepath = os.path.join(args.output, filename)
                cv2.imwrite(filepath, image)

                count += 1
                elapsed = time.time() - start_time
                fps = count / max(elapsed, 1)

                print(f"  [{count:4d}] {filename}  "
                      f"({image.shape[1]}x{image.shape[0]})  "
                      f"({fps:.1f} kép/perc)")

                # Várás
                time.sleep(args.interval)

    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Kész! {count} screenshot mentve → {os.path.abspath(args.output)}")
    print(f"  Idő: {elapsed:.0f} mp ({elapsed/60:.1f} perc)")
    print(f"\n  Következő lépés:")
    print(f"    1. Töltsd fel a képeket Roboflow-ra (https://roboflow.com)")
    print(f"    2. Címkézd be a YOLO_TRAINING_GUIDE.md szerint")
    print(f"    3. Exportáld YOLOv8 formátumban")
    print(f"    4. Tanítsd be: python -m live_reader.train_yolo")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
