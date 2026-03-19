#!/usr/bin/env python3
"""
live_reader/train_yolo.py  –  YOLO modell tréning póker asztal detektáláshoz

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ELŐFELTÉTELEK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Screenshotok gyűjtése: python -m live_reader.collect_screenshots
  2. Címkézés Roboflow-ban (lásd YOLO_TRAINING_GUIDE.md)
  3. Export: YOLOv8 formátum → dataset/ mappa

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HASZNÁLAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Alap tréning
  python -m live_reader.train_yolo --data dataset/data.yaml --epochs 100

  # Folytatás
  python -m live_reader.train_yolo --data dataset/data.yaml --resume models/last.pt

  # Kisebb modell (gyorsabb)
  python -m live_reader.train_yolo --data dataset/data.yaml --model yolov8n.pt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRÉNING IDŐK (közelítő)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  GPU              | 500 kép / 100 epoch | 1000 kép / 150 epoch
  -----------------+--------------------+---------------------
  RTX 3060         | ~30 perc            | ~60 perc
  RTX 3070/3080    | ~20 perc            | ~40 perc
  RTX 4070/4080    | ~15 perc            | ~25 perc
  Google Colab T4  | ~45 perc            | ~90 perc
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description='YOLO modell tréning póker asztal detektáláshoz'
    )
    parser.add_argument(
        '--data', required=True,
        help='data.yaml elérési útja (Roboflow export-ból)'
    )
    parser.add_argument(
        '--model', default='yolov8s.pt',
        help='Alap modell (yolov8n.pt=nano/gyors, yolov8s.pt=small/ajánlott, '
             'yolov8m.pt=medium/pontosabb). Default: yolov8s.pt'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Tréning epochok száma (default: 100)'
    )
    parser.add_argument(
        '--batch', type=int, default=16,
        help='Batch méret (csökkentsd ha GPU memória hiba, default: 16)'
    )
    parser.add_argument(
        '--imgsz', type=int, default=640,
        help='Kép méret (default: 640)'
    )
    parser.add_argument(
        '--resume', default=None,
        help='Folytatás egy korábbi checkpoint-ból (.pt fájl)'
    )
    parser.add_argument(
        '--output', default='models',
        help='Kimeneti mappa a kész modellnek (default: models/)'
    )
    parser.add_argument(
        '--name', default='poker_yolo',
        help='Tréning futás neve (default: poker_yolo)'
    )
    parser.add_argument(
        '--device', default='0',
        help='GPU device (0=első GPU, cpu=CPU, default: 0)'
    )
    args = parser.parse_args()

    # Importok
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n❌ ultralytics csomag szükséges!")
        print("   pip install ultralytics")
        sys.exit(1)

    # Ellenőrzések
    if not os.path.exists(args.data):
        print(f"\n❌ data.yaml nem található: {args.data}")
        print("   Roboflow-ból exportáld YOLOv8 formátumban")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  🎯  Poker YOLO Tréning")
    print("=" * 60)
    print(f"\n  Adathalmaz:  {args.data}")
    print(f"  Alap modell: {args.model}")
    print(f"  Epochok:     {args.epochs}")
    print(f"  Batch:       {args.batch}")
    print(f"  Kép méret:   {args.imgsz}")
    print(f"  Device:      {args.device}")
    print(f"  Kimenet:     {args.output}")
    print("=" * 60)

    # ── Modell betöltés ───────────────────────────────────────────────────
    if args.resume:
        print(f"\n  Folytatás: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"\n  Alap modell letöltése: {args.model}")
        model = YOLO(args.model)

    # ── Tréning ──────────────────────────────────────────────────────────
    print("\n  Tréning indítása...\n")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        patience=20,         # early stopping: 20 epoch javulás nélkül → leáll
        save=True,
        save_period=25,       # checkpoint mentés 25 epochonként
        plots=True,           # tréning görbék mentése
        verbose=True,

        # Augmentáció (a póker képekhez optimalizálva)
        hsv_h=0.01,          # szín eltolás (minimális – a kártyaszínek fontosak!)
        hsv_s=0.3,           # szaturáció variáció
        hsv_v=0.3,           # fényesség variáció
        degrees=0.0,         # NEM forgatjuk (a kártyák mindig függőlegesek)
        translate=0.05,      # kis eltolás
        scale=0.2,           # kis méretezés
        flipud=0.0,          # NEM tükrözzük függőlegesen
        fliplr=0.0,          # NEM tükrözzük vízszintesen (szövegek fordítva lennének)
        mosaic=0.5,          # mosaic augmentáció (mérsékelt)
    )

    # ── Eredmények ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Tréning kész!")
    print("=" * 60)

    # Best modell másolása
    best_pt = os.path.join('runs', 'detect', args.name, 'weights', 'best.pt')
    if os.path.exists(best_pt):
        import shutil
        output_path = os.path.join(args.output, 'poker_yolo_best.pt')
        shutil.copy2(best_pt, output_path)
        print(f"\n  ✅ Best modell: {output_path}")

    last_pt = os.path.join('runs', 'detect', args.name, 'weights', 'last.pt')
    if os.path.exists(last_pt):
        import shutil
        output_last = os.path.join(args.output, 'poker_yolo_last.pt')
        shutil.copy2(last_pt, output_last)
        print(f"  ✅ Last modell: {output_last}")

    results_dir = os.path.join('runs', 'detect', args.name)
    print(f"  📊 Tréning eredmények: {results_dir}")

    print(f"\n  Következő lépés:")
    print(f"    python run_live.py --yolo {os.path.join(args.output, 'poker_yolo_best.pt')}")
    print("=" * 60)


def validate_model(model_path: str, data_yaml: str):
    """Betanított modell validálása a teszt adatokon."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.val(data=data_yaml, verbose=True)

    print("\nValidációs eredmények:")
    print(f"  mAP50:     {results.box.map50:.3f}")
    print(f"  mAP50-95:  {results.box.map:.3f}")

    for i, name in results.names.items():
        ap50 = results.box.ap50[i] if i < len(results.box.ap50) else 0
        print(f"  {name:20s}: AP50={ap50:.3f}")

    return results


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'validate':
        # python -m live_reader.train_yolo validate --model models/best.pt --data dataset/data.yaml
        parser = argparse.ArgumentParser()
        parser.add_argument('command')
        parser.add_argument('--model', required=True)
        parser.add_argument('--data', required=True)
        args = parser.parse_args()
        validate_model(args.model, args.data)
    else:
        main()
