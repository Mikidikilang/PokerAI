# YOLO Tréning Útmutató – BoaBet Póker Asztal Detekció

## Tartalom
1. [Áttekintés](#1-áttekintés)
2. [Screenshotok gyűjtése](#2-screenshotok-gyűjtése)
3. [Címkézés Roboflow-ban](#3-címkézés-roboflow-ban)
4. [YOLO tréning](#4-yolo-tréning)
5. [Tesztelés](#5-tesztelés)
6. [Hibaelhárítás](#6-hibaelhárítás)

---

## 1. Áttekintés

### Mit csinálunk?
Betanítunk egy YOLO objektum-detektáló modellt, ami egy screenshoton
megtalálja a póker asztal elemeit: kártyákat, pot szöveget, stackeket,
dealer gombot, akció badge-eket stb.

### Mennyi munka?
| Lépés | Idő |
|-------|-----|
| Screenshotok gyűjtése | 30-40 perc (játék közben fut) |
| Címkézés (Roboflow) | 3-5 óra (500 kép × ~30 mp/kép) |
| YOLO tréning | 30-60 perc (GPU-n automatikus) |
| **Összesen** | **~5-7 óra** |

### Mi kell hozzá?
- Python 3.10+
- NVIDIA GPU (tréninghez)
- Roboflow fiók (ingyenes, https://roboflow.com)
- BoaBet account (play money jó)

---

## 2. Screenshotok gyűjtése

### 2.1 Telepítés

```bash
pip install mss opencv-python numpy ultralytics pytesseract
```

### 2.2 Gyűjtés

```bash
# BoaBet nyitva a böngészőben → ülj le egy asztalhoz
# Majd futtasd:
python -m live_reader.collect_screenshots --output screenshots/ --interval 3
```

Ez 3 másodpercenként ment egy screenshotot. **Játssz közben** ~30 percet,
vagy ha nem akarsz játszani, **nézd a játékot** (observer mode).

### 2.3 Hány kép kell?

| Mennyiség | Minőség |
|-----------|---------|
| 300-500 | Minimális, működik de pontatlan lehet |
| 500-800 | **Ajánlott** – jó egyensúly munka és pontosság közt |
| 800-1500 | Kiváló – nagyon stabil modell |

### 2.4 Változatosság (KRITIKUS!)

A modell csak azt tanulja meg, amit lát. Gyűjts képeket:

- ✅ **Minden street-ből**: preflop (üres board), flop (3 lap), turn (4), river (5)
- ✅ **Fold-okkal**: "Fold" badge-ek a játékosoknál
- ✅ **All-in**: "All-In" badge
- ✅ **Különböző pot méretek**: $0.06-tól $50-ig
- ✅ **Különböző stack méretek**: $0.50-tól $100-ig
- ✅ **Showdown**: más játékosok kártyái is látszanak
- ✅ **Kevés játékos**: heads-up, 3-handed
- ✅ **Tele asztal**: 9 játékos
- ❌ **NE legyen** csak egyféle szituáció (pl. csak preflop)

---

## 3. Címkézés Roboflow-ban

### 3.1 Roboflow fiók létrehozása

1. Menj ide: https://roboflow.com
2. Regisztrálj (ingyenes tier bőven elég)
3. Hozz létre új projektet:
   - **Projekt neve**: `boabet-poker-detector`
   - **License**: Private
   - **Projekt típus**: **Object Detection**
   - **Annotation Group**: `poker_elements`

### 3.2 Képek feltöltése

1. Kattints **Upload** → húzd be a `screenshots/` mappa tartalmát
2. Roboflow automatikusan felismeri a PNG fájlokat
3. Kattints **Save and Continue**

### 3.3 Osztályok (Labels) létrehozása

Hozd létre ezeket az osztályokat (pontosan így, kisbetűvel):

```
card_face        – Felfordított kártya (a rank és suit is látszik)
card_back        – Lefordított kártya (hátlap)
pot_text         – A "Pot: $4.65" felirat
stack_text       – Stack méret ($10.33, $4.44 stb.)
player_name      – Játékos neve (SkyEagle, Prince10 stb.)
dealer_button    – A "D" dealer gomb
action_badge     – Akció felirat (Fold, All-In, Call, Raise stb.)
bet_amount       – A zöld mezőn lévő bet összeg ($0.4, $4.25)
```

### 3.4 Hogyan címkézz (lépésről lépésre)

Minden képen:

#### Kártyák (card_face)
- Rajzolj **bounding box**-ot **minden** felfordított kártya köré
- A box szorosan fogja körbe a kártyát
- Board lapok: egyenként, nem egyben!
- Saját lapok: egyenként
- Showdown-nál: más játékosok kártyái is

```
  ┌──────┐ ┌──────┐ ┌──────┐
  │ 8♥   │ │ 7♥   │ │ 10♣  │   ← 3 külön "card_face" box
  └──────┘ └──────┘ └──────┘
```

#### Lefordított kártyák (card_back)
- A hátlappal lefelé lévő kártyákat is jelöld meg
- Ezek jelzik, hogy a játékosnak van keze (nem foldolt)

#### Pot szöveg (pot_text)
- A "Pot: $4.65" teljes szövegre
- Egy box per kép (általában egy pot van)

```
  ┌─────────────┐
  │ Pot: $4.65  │   ← "pot_text"
  └─────────────┘
```

#### Stack méretek (stack_text)
- **Minden** játékos stack összege külön box
- Csak a szám + dollárjel: "$10.33"

```
  ┌────────┐
  │ $10.33 │   ← "stack_text"
  └────────┘
```

#### Játékos nevek (player_name)
- A név szöveg köré: "SkyEagle", "Prince10" stb.
- Fontos a HUD tracker számára

#### Dealer gomb (dealer_button)
- A kicsi "D" gomb
- Egy box per kép (csak egy button van)

#### Akció badge-ek (action_badge)
- "Fold", "All-In", "Call", "Raise", "Check" stb.
- Minden badge külön box
- A BoaBet-en ezek a szürke hátteres címkék

```
  ┌──────┐
  │ Fold │   ← "action_badge"
  └──────┘
  
  ┌────────┐
  │ All-In │   ← "action_badge"
  └────────┘
```

#### Bet összeg (bet_amount)
- Az asztal zöld mezőjén lévő számok: "$0.4", "$4.25"
- NEM a stack szám, hanem az aktuálisan betett összeg

### 3.5 Címkézési tippek

1. **Pontosság**: a box szorosan illeszkedjen az elemre, ne legyen sok üres hely
2. **Konzisztencia**: mindig ugyanazokat az osztályokat használd
3. **Ne hagyd ki**: ha egy elem látható, jelöld meg! A YOLO a hiányzó címkékből azt tanulja, hogy "ott nincs semmi"
4. **Részben takart elemek**: ha egy elem 50%+ látható, jelöld meg
5. **Sebesség**: 500 kép × ~30 mp/kép = ~4 óra. Lehet sorozatban, nem kell egyszerre

### 3.6 Roboflow Annotációs Gyorsbillentyűk

| Billentyű | Funkció |
|-----------|---------|
| `B` | Bounding box rajzolás |
| `1-9` | Osztály gyorsválasztás |
| `D` | Következő kép |
| `A` | Előző kép |
| `Ctrl+D` | Címke törlése |
| `Ctrl+C/V` | Címke másolás/beillesztés |

### 3.7 Adathalmaz felosztás

Roboflow automatikusan felosztja:
- **Train**: 70% (tréning)
- **Valid**: 20% (validáció)
- **Test**: 10% (teszt)

Ez jó alapbeállítás, ne változtasd.

### 3.8 Export

1. Kattints **Generate** → **New Version**
2. **Preprocessing**: Resize to 640×640 (vagy Auto Orient + Resize)
3. **Augmentation**: NE adj hozzá (a tréning scriptben mi állítjuk)
4. Kattints **Generate**
5. **Export Format**: **YOLOv8**
6. **Download zip** → csomagold ki a `dataset/` mappába

Az eredmény struktúra:
```
dataset/
├── data.yaml          ← osztály definíciók
├── train/
│   ├── images/        ← tréning képek
│   └── labels/        ← tréning címkék (.txt)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## 4. YOLO tréning

### 4.1 Telepítés

```bash
pip install ultralytics torch torchvision
```

### 4.2 Tréning futtatás

```bash
python -m live_reader.train_yolo --data dataset/data.yaml --epochs 100
```

**Opciók:**

```bash
# Gyorsabb, kisebb modell (kevésbé pontos):
python -m live_reader.train_yolo --data dataset/data.yaml --model yolov8n.pt --epochs 80

# Ajánlott (jó egyensúly):
python -m live_reader.train_yolo --data dataset/data.yaml --model yolov8s.pt --epochs 100

# Pontosabb, nagyobb modell (lassabb):
python -m live_reader.train_yolo --data dataset/data.yaml --model yolov8m.pt --epochs 150

# Ha kevés a GPU memória:
python -m live_reader.train_yolo --data dataset/data.yaml --batch 8

# Folytatás (ha félbeszakadt):
python -m live_reader.train_yolo --data dataset/data.yaml --resume models/poker_yolo_last.pt
```

### 4.3 Mire figyelj a tréning közben

A terminálban ezek az értékek fontosak:
- **mAP50**: a fő metrika. **0.8 felett jó**, 0.9 felett kiváló
- **Loss**: csökkennie kell. Ha stagnál → több adat kell vagy több epoch
- **box_loss**: a bounding box pontosság
- **cls_loss**: az osztályozás pontossága

### 4.4 Tréning eredmények

A tréning végén a `runs/detect/poker_yolo/` mappában:
- `weights/best.pt` → **legjobb modell** (ezt használd!)
- `weights/last.pt` → utolsó epoch
- `results.png` → tréning görbék
- `confusion_matrix.png` → melyik osztályt hol keveri
- `val_batch0_pred.png` → predikciók vizualizálva

A `models/poker_yolo_best.pt` automatikusan oda másolódik.

### 4.5 Validáció

```bash
python -m live_reader.train_yolo validate --model models/poker_yolo_best.pt --data dataset/data.yaml
```

---

## 5. Tesztelés

### 5.1 Read-only mód (nincs AI, csak OCR)

```bash
python run_live.py --yolo models/poker_yolo_best.pt --read-only --my-seat 7
```

Ez kiírja a konzolra amit lát: kártyákat, potot, stackeket.
Ellenőrizd, hogy helyesen olvassa-e!

### 5.2 Teljes mód (AI javaslatokkal)

```bash
python run_live.py --yolo models/poker_yolo_best.pt \
                   --model 9max_ppo_v4.pth \
                   --my-seat 7 \
                   --bb 0.04 --sb 0.02
```

### 5.3 Mit ellenőrizz

- [ ] A kártyákat helyesen ismeri fel?
- [ ] A pot összeg jó?
- [ ] A stackeket jól olvassa?
- [ ] A dealer gombot megtalálja?
- [ ] Észleli-e az új kéz kezdetét?
- [ ] A street váltásokat detektálja?
- [ ] Amikor te következel, ad javaslatot?

---

## 6. Hibaelhárítás

### "A YOLO nem talál semmit"
- Ellenőrizd a confidence threshold-ot: `--confidence 0.3`
- A kép méret megfelelő? (640px ajánlott)
- Elég képet címkéztél? (min 300)

### "A számok rosszul olvasódnak"
- Tesseract telepítve van? (`tesseract --version`)
- Állítsd be a path-ot: `set TESSERACT_CMD=C:\...\tesseract.exe`
- A stack/pot szöveg elég nagy? Próbálj nagyobb ablakot

### "Kártyákat nem ismeri fel"
- A YOLO detektálja a card_face-t? (--read-only módban check)
- Elég kártya címke van a tréning adatban? (min 200 card_face)
- A bounding box pontos? (szorosan illeszkedik?)

### "Lassú" (alacsony FPS)
- Használj yolov8n.pt-t (nano, gyorsabb)
- Csökkentsd az FPS-t: `--fps 1.0`
- GPU-n futtatod? (a YOLO automatikusan GPU-t használ ha van)

### "Rossz szék hozzárendelés"
- A seat_config jó? Futtasd kalibrálással
- `--my-seat` paraméter helyes?

### Roboflow-tól nem tudsz letölteni
- Alternatíva: `pip install roboflow`
  ```python
  from roboflow import Roboflow
  rf = Roboflow(api_key="YOUR_API_KEY")
  project = rf.workspace().project("boabet-poker-detector")
  dataset = project.version(1).download("yolov8")
  ```
