"""
live_reader/screen_reader.py  –  Képernyőolvasó YOLO + OCR

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITEKTÚRA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Screenshot (mss)
       │
       ▼
  YOLO Object Detection
       │  → Detektálja: kártyákat, pot szöveget, stackeket,
       │     dealer gombot, akció badge-eket, játékosneveket
       ▼
  Régió-specifikus feldolgozás:
       ├── Kártyák → CardClassifier CNN (rank + suit)
       ├── Számok (pot, stack, bet) → digit OCR
       ├── Szövegek (nevek, akciók) → Tesseract / EasyOCR
       └── Button/highlight → pozíció alapú seat mapping
       │
       ▼
  ScreenState dataclass

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HASZNÁLAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  reader = ScreenReader(
      yolo_model_path  = 'models/poker_yolo.pt',
      card_model_path  = 'models/card_classifier.pt',  # opcionális
      seat_config_path = 'configs/boabet_9max.json',
      my_seat          = 4,
  )

  state = reader.read_frame()
  # → ScreenState

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEPENDENCIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  pip install mss opencv-python ultralytics pytesseract numpy Pillow
  
  # Tesseract telepítés (Windows):
  # https://github.com/tesseract-ocr/tesseract/releases
  # Majd: set TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""

import json
import logging
import os
import re
import time
from typing import Optional

import numpy as np

logger = logging.getLogger("PokerAI.ScreenReader")

# Lazy importok – csak ha tényleg használjuk
_mss = None
_cv2 = None
_pytesseract = None


def _import_mss():
    global _mss
    if _mss is None:
        import mss as _mss_module
        _mss = _mss_module
    return _mss


def _import_cv2():
    global _cv2
    if _cv2 is None:
        import cv2 as _cv2_module
        _cv2 = _cv2_module
    return _cv2


def _import_tesseract():
    global _pytesseract
    if _pytesseract is None:
        import pytesseract as _pt
        _pytesseract = _pt
        # Windows Tesseract path
        tesseract_cmd = os.environ.get('TESSERACT_CMD')
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            _pt.pytesseract.tesseract_cmd = tesseract_cmd
    return _pytesseract


# ─────────────────────────────────────────────────────────────────────────────
# Importok (data_types ugyanabból a csomagból)
# ─────────────────────────────────────────────────────────────────────────────

from .data_types import (
    CardRead, PlayerRead, ScreenState, Detection, DetectionClass
)


# ─────────────────────────────────────────────────────────────────────────────
# Seat Config – hol vannak a székek a képernyőn
# ─────────────────────────────────────────────────────────────────────────────

class SeatConfig:
    """
    A székek pozícióját tárolja relatív koordinátákban (0.0-1.0).
    
    A BoaBet 9-max asztalán a székek nagyjából így helyezkednek el:
    
         [1]    [2]    [3]
      [0]                  [4]
         [8]    [7]    [6]
                [5]  ← mi (alul középen, tipikusan)
    
    A pontos pozíciókat a screenshotból mérjük ki,
    vagy a kalibráló eszköz generálja.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.seats = {}       # seat_index → {'x': float, 'y': float, 'r': float}
        self.board_region = {}  # {'x': ..., 'y': ..., 'w': ..., 'h': ...}
        self.pot_region = {}
        self.table_region = {}  # az asztal területe a képernyőn
        self.num_seats = 9

        if config_path and os.path.exists(config_path):
            self.load(config_path)
        else:
            self._default_boabet_9max()

    def _default_boabet_9max(self):
        """
        Alapértelmezett 9-max elrendezés a BoaBet-hez.
        Relatív koordináták (0.0-1.0) az asztal területéhez képest.
        Ezek közelítő értékek – a kalibrációs eszköz pontosítja.
        """
        self.num_seats = 9
        # (x, y) – a szék közepe relatívan
        positions = [
            (0.12, 0.45),  # Seat 0: bal közép
            (0.20, 0.22),  # Seat 1: bal felső
            (0.40, 0.15),  # Seat 2: felső közép-bal
            (0.60, 0.15),  # Seat 3: felső közép-jobb
            (0.80, 0.22),  # Seat 4: jobb felső
            (0.88, 0.45),  # Seat 5: jobb közép
            (0.75, 0.70),  # Seat 6: jobb alsó
            (0.50, 0.78),  # Seat 7: alsó közép
            (0.25, 0.70),  # Seat 8: bal alsó
        ]
        for i, (x, y) in enumerate(positions):
            self.seats[i] = {'x': x, 'y': y, 'r': 0.08}  # r = sugár a seat matchinghez

        self.board_region = {'x': 0.30, 'y': 0.38, 'w': 0.40, 'h': 0.15}
        self.pot_region = {'x': 0.35, 'y': 0.30, 'w': 0.30, 'h': 0.08}

    def load(self, path: str):
        """JSON konfigurációból betöltés."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.num_seats = data.get('num_seats', 9)
        self.seats = {int(k): v for k, v in data.get('seats', {}).items()}
        self.board_region = data.get('board_region', self.board_region)
        self.pot_region = data.get('pot_region', self.pot_region)
        self.table_region = data.get('table_region', {})
        logger.info(f"SeatConfig betöltve: {path} ({self.num_seats} seat)")

    def save(self, path: str):
        """Mentés JSON-be."""
        data = {
            'num_seats': self.num_seats,
            'seats': {str(k): v for k, v in self.seats.items()},
            'board_region': self.board_region,
            'pot_region': self.pot_region,
            'table_region': self.table_region,
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"SeatConfig mentve: {path}")

    def nearest_seat(self, x_rel: float, y_rel: float) -> int:
        """
        Megkeresi a legközelebbi széket egy (x, y) relatív koordinátához.
        A YOLO detekció center pontját adja be → melyik szék mellé esik.
        """
        best_seat = 0
        best_dist = float('inf')
        for seat_idx, pos in self.seats.items():
            dx = x_rel - pos['x']
            dy = y_rel - pos['y']
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_seat = seat_idx
        return best_seat

    def is_in_board_region(self, x_rel: float, y_rel: float) -> bool:
        """Egy pont a board (közösségi lapok) régióban van-e."""
        br = self.board_region
        return (br['x'] <= x_rel <= br['x'] + br['w'] and
                br['y'] <= y_rel <= br['y'] + br['h'])


# ─────────────────────────────────────────────────────────────────────────────
# YOLO Detector wrapper
# ─────────────────────────────────────────────────────────────────────────────

class YOLODetector:
    """
    YOLO modell wrapper a póker asztal elemeinek detektálásához.
    
    A modell betöltése és futtatása az ultralytics könyvtárral történik.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.5,
                 device: str = 'auto'):
        """
        model_path: betanított YOLO modell (.pt fájl)
        confidence_threshold: minimum confidence a detekciókhoz
        device: 'auto', 'cpu', '0' (GPU)
        """
        self._model_path = model_path
        self._confidence = confidence_threshold
        self._device = device
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            logger.info(f"YOLO modell betöltése: {self._model_path}")
            self._model = YOLO(self._model_path)
            logger.info(f"YOLO kész | osztályok: {self._model.names}")
        except ImportError:
            raise ImportError(
                "ultralytics csomag szükséges: pip install ultralytics"
            )

    def detect(self, image: np.ndarray) -> list:
        """
        YOLO detekció futtatása egy képen.
        
        image: numpy array (BGR, OpenCV formátum)
        return: list[Detection]
        """
        self._ensure_loaded()

        results = self._model.predict(
            image,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self._model.names.get(cls_id, f"class_{cls_id}")

                detections.append(Detection(
                    class_name=cls_name,
                    confidence=conf,
                    x1=int(x1), y1=int(y1),
                    x2=int(x2), y2=int(y2),
                ))

        return detections


# ─────────────────────────────────────────────────────────────────────────────
# Szám olvasó (OCR régióból)
# ─────────────────────────────────────────────────────────────────────────────

class NumberReader:
    """
    Számok kiolvasása képrégióból.
    Elsősorban: pot méret, stack méret, bet összeg.
    
    Előfeldolgozás → Tesseract (digits-only) → posztprocesszálás
    """

    @staticmethod
    def read_number(image_region: np.ndarray) -> Optional[float]:
        """
        Egy kivágott képrégióból számot olvas.
        
        image_region: BGR numpy array (a YOLO bounding box kivágata)
        return: float vagy None ha nem sikerült
        """
        cv2 = _import_cv2()
        tesseract = _import_tesseract()

        if image_region is None or image_region.size == 0:
            return None

        try:
            # 1. Szürkeárnyalat
            gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)

            # 2. Felskálázás (2x – javítja az OCR pontosságot)
            h, w = gray.shape
            if h < 30:
                scale = max(2, 40 // max(h, 1))
                gray = cv2.resize(gray, (w * scale, h * scale),
                                  interpolation=cv2.INTER_CUBIC)

            # 3. Kontraszt fokozás (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)

            # 4. Binarizálás (Otsu)
            _, binary = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 5. Inverz ha a háttér világos
            if np.mean(binary) > 127:
                binary = 255 - binary

            # 6. Tesseract – csak számjegyek
            text = tesseract.image_to_string(
                binary,
                config='--psm 7 -c tessedit_char_whitelist=0123456789.,$'
            ).strip()

            return NumberReader._parse_number(text)

        except Exception as e:
            logger.debug(f"NumberReader hiba: {e}")
            return None

    @staticmethod
    def _parse_number(text: str) -> Optional[float]:
        """
        OCR szöveg → float konverzió.
        Kezeli: "$4.65", "4,65", "$10.33", "Pot: $4.65" stb.
        """
        if not text:
            return None

        # Szűrés: csak számjegyek, pont, vessző
        cleaned = re.sub(r'[^0-9.,]', '', text)
        if not cleaned:
            return None

        # Vessző → pont (európai formátum kezelés)
        # Ha van pont ÉS vessző: a vessző ezres elválasztó
        if ',' in cleaned and '.' in cleaned:
            cleaned = cleaned.replace(',', '')
        elif ',' in cleaned:
            # Csak vessző: lehet tizedes vagy ezres
            parts = cleaned.split(',')
            if len(parts) == 2 and len(parts[1]) <= 2:
                cleaned = cleaned.replace(',', '.')  # tizedes
            else:
                cleaned = cleaned.replace(',', '')   # ezres

        try:
            value = float(cleaned)
            # Sanity check
            if 0 <= value < 1_000_000:
                return value
            return None
        except ValueError:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Szöveg olvasó (nevek, akciók)
# ─────────────────────────────────────────────────────────────────────────────

class TextReader:
    """Játékos nevek és akció szövegek olvasása."""

    # Ismert akció kulcsszavak
    KNOWN_ACTIONS = {
        "fold", "call", "check", "raise", "bet", "all-in", "allin",
        "all in", "muck", "show",
    }

    @staticmethod
    def read_text(image_region: np.ndarray) -> str:
        """Szöveg olvasása egy kivágott képrégióból."""
        cv2 = _import_cv2()
        tesseract = _import_tesseract()

        if image_region is None or image_region.size == 0:
            return ""

        try:
            gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)

            # Felskálázás
            h, w = gray.shape
            if h < 25:
                scale = max(2, 30 // max(h, 1))
                gray = cv2.resize(gray, (w * scale, h * scale),
                                  interpolation=cv2.INTER_CUBIC)

            _, binary = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if np.mean(binary) > 127:
                binary = 255 - binary

            text = tesseract.image_to_string(
                binary, config='--psm 7'
            ).strip()

            return text

        except Exception as e:
            logger.debug(f"TextReader hiba: {e}")
            return ""

    @staticmethod
    def parse_action(text: str) -> tuple:
        """
        Akció szöveg → (action_type, amount) tuple.
        
        "Fold" → ("fold", 0.0)
        "Call $4.00" → ("call", 4.0)
        "Raise $12.50" → ("raise", 12.5)
        "All-In" → ("allin", 0.0)
        """
        if not text:
            return ("", 0.0)

        lower = text.lower().strip()

        # All-in variációk
        if "all" in lower and "in" in lower:
            # Próbáljuk kiolvasni az összeget
            amount = NumberReader._parse_number(text)
            return ("allin", amount or 0.0)

        # Egyszerű akciók
        for action in ["fold", "check", "muck", "show"]:
            if action in lower:
                return (action, 0.0)

        # Akciók összeggel
        for action in ["call", "raise", "bet"]:
            if action in lower:
                amount = NumberReader._parse_number(text)
                return (action, amount or 0.0)

        return ("", 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# ScreenReader – fő osztály
# ─────────────────────────────────────────────────────────────────────────────

class ScreenReader:
    """
    A póker asztal képernyőolvasója.
    
    Kombinálja:
      - mss: gyors screenshot
      - YOLO: objektum detekció (kártyák, gombok, szöveg régiók)
      - Tesseract: szám és szöveg olvasás a detektált régiókból
    
    Használat:
        reader = ScreenReader(
            yolo_model_path='models/poker_yolo.pt',
            seat_config_path='configs/boabet_9max.json',
            my_seat=4,
        )
        
        state = reader.read_frame()
    """

    def __init__(self,
                 yolo_model_path: str,
                 seat_config_path: str = None,
                 card_model_path: str = None,
                 my_seat: int = 0,
                 window_title: str = "BoaBet",
                 yolo_confidence: float = 0.5,
                 monitor_index: int = 1,
                 capture_region: dict = None):
        """
        yolo_model_path:  betanított YOLO modell
        seat_config_path: szék pozíciók JSON
        card_model_path:  kártya klasszifikáló (opcionális, ha YOLO 52 osztályos)
        my_seat:          saját szék index
        window_title:     ablak cím (screenshot célzáshoz)
        yolo_confidence:  minimum detection confidence
        monitor_index:    melyik monitor (mss: 1 = elsődleges)
        capture_region:   fix régió: {'left': x, 'top': y, 'width': w, 'height': h}
        """
        self._my_seat = my_seat
        self._window_title = window_title
        self._monitor_index = monitor_index
        self._capture_region = capture_region

        # YOLO detektor
        self._detector = YOLODetector(
            yolo_model_path,
            confidence_threshold=yolo_confidence,
        )

        # Szék konfiguráció
        self._seat_config = SeatConfig(seat_config_path)

        # Kártya klasszifikáló (ha külön modell van)
        self._card_classifier = None
        if card_model_path:
            self._card_classifier = self._load_card_classifier(card_model_path)

        # Név cache (ne olvassa minden frame-ben)
        self._name_cache = {}       # seat → username
        self._name_cache_age = {}   # seat → utolsó frissítés timestamp
        self._NAME_CACHE_TTL = 30   # 30 másodpercig érvényes

        # Debounce: az utolsó N frame stabil értékei
        self._debounce_pot = []       # utolsó 3 pot érték
        self._debounce_stacks = {}    # seat → utolsó 3 stack érték
        self._DEBOUNCE_FRAMES = 3

        logger.info(
            f"ScreenReader inicializálva | my_seat={my_seat} | "
            f"YOLO={yolo_model_path}"
        )

    # ── Fő API ────────────────────────────────────────────────────────────────

    def read_frame(self) -> ScreenState:
        """
        Egyetlen frame leolvasása a képernyőről.
        
        Return: ScreenState – a teljes asztal állapota
        """
        start_time = time.time()

        # 1. Screenshot
        image = self._capture_screen()
        if image is None:
            return ScreenState(
                timestamp=time.time(),
                is_valid=False,
                ocr_errors=["Screenshot sikertelen"],
            )

        img_h, img_w = image.shape[:2]

        # 2. YOLO detekció
        detections = self._detector.detect(image)

        # 3. Detekciók feldolgozása
        state = ScreenState(
            timestamp=time.time(),
            is_valid=True,
            my_seat=self._my_seat,
            raw_detections=detections,
        )

        board_cards = []
        players = {i: PlayerRead(seat_index=i) for i in range(self._seat_config.num_seats)}
        pot_readings = []
        errors = []

        for det in detections:
            # Relatív koordináták (0.0-1.0)
            cx_rel = det.center[0] / img_w
            cy_rel = det.center[1] / img_h

            try:
                # ── Kártyák ──────────────────────────────────────────────
                if det.class_name == "card_face":
                    card = self._recognize_card(image, det)
                    if card:
                        if self._seat_config.is_in_board_region(cx_rel, cy_rel):
                            board_cards.append(card)
                        else:
                            seat = self._seat_config.nearest_seat(cx_rel, cy_rel)
                            players[seat].cards.append(card)

                elif det.class_name == "card_back":
                    # Lefordított kártya → a szék aktív
                    seat = self._seat_config.nearest_seat(cx_rel, cy_rel)
                    players[seat].is_active = True

                # ── Pot ──────────────────────────────────────────────────
                elif det.class_name == "pot_text":
                    region = det.crop_from(image)
                    value = NumberReader.read_number(region)
                    if value is not None:
                        pot_readings.append(value)

                # ── Stack ────────────────────────────────────────────────
                elif det.class_name == "stack_text":
                    seat = self._seat_config.nearest_seat(cx_rel, cy_rel)
                    region = det.crop_from(image)
                    value = NumberReader.read_number(region)
                    if value is not None:
                        players[seat].stack = value

                # ── Játékos név ──────────────────────────────────────────
                elif det.class_name == "player_name":
                    seat = self._seat_config.nearest_seat(cx_rel, cy_rel)
                    name = self._read_player_name(image, det, seat)
                    if name:
                        players[seat].username = name

                # ── Dealer gomb ──────────────────────────────────────────
                elif det.class_name == "dealer_button":
                    seat = self._seat_config.nearest_seat(cx_rel, cy_rel)
                    players[seat].is_dealer = True
                    state.dealer_seat = seat

                # ── Akció badge ──────────────────────────────────────────
                elif det.class_name == "action_badge":
                    seat = self._seat_config.nearest_seat(cx_rel, cy_rel)
                    region = det.crop_from(image)
                    text = TextReader.read_text(region)
                    action_type, amount = TextReader.parse_action(text)
                    players[seat].last_action_text = text
                    if action_type == "fold":
                        players[seat].is_active = False

                # ── Bet összeg (a zöld mezőn) ────────────────────────────
                elif det.class_name == "bet_amount":
                    seat = self._seat_config.nearest_seat(cx_rel, cy_rel)
                    region = det.crop_from(image)
                    value = NumberReader.read_number(region)
                    if value is not None:
                        players[seat].bet_this_round = value

                # ── Aktív jelző ──────────────────────────────────────────
                elif det.class_name == "seat_highlight":
                    seat = self._seat_config.nearest_seat(cx_rel, cy_rel)
                    players[seat].is_current_turn = True
                    state.active_player_seat = seat

                # ── Akciógombok (mi következünk) ─────────────────────────
                elif det.class_name.startswith("btn_"):
                    # Ha akciógombok láthatók → mi következünk
                    state.active_player_seat = self._my_seat
                    players[self._my_seat].is_current_turn = True

            except Exception as e:
                errors.append(f"{det.class_name}: {e}")

        # 4. Eredmények összerakása
        # Board lapok rendezése x koordináta szerint (bal→jobb)
        board_cards.sort(key=lambda c: c.bbox[0] if c.bbox else 0)
        state.board_cards = board_cards

        # Saját hole cards
        my_cards = players[self._my_seat].cards
        if len(my_cards) >= 2:
            state.my_hole_cards = my_cards[:2]

        # Pot (debounce)
        state.pot_total = self._debounce_value(
            pot_readings, self._debounce_pot
        )

        # Stackek (debounce)
        for seat_idx, player in players.items():
            if player.stack is not None:
                if seat_idx not in self._debounce_stacks:
                    self._debounce_stacks[seat_idx] = []
                stable = self._debounce_value(
                    [player.stack], self._debounce_stacks[seat_idx]
                )
                if stable is not None:
                    player.stack = stable

        # Aktív játékosok száma
        state.num_active_players = sum(
            1 for p in players.values() if p.is_active
        )

        state.players = list(players.values())
        state.ocr_errors = errors
        state.frame_ms = (time.time() - start_time) * 1000

        return state

    # ── Screenshot ────────────────────────────────────────────────────────────

    def _capture_screen(self) -> Optional[np.ndarray]:
        """Screenshot a póker asztalról."""
        mss = _import_mss()
        cv2 = _import_cv2()

        try:
            with mss.mss() as sct:
                if self._capture_region:
                    # Fix régió
                    monitor = self._capture_region
                else:
                    # Teljes monitor
                    monitor = sct.monitors[self._monitor_index]

                screenshot = sct.grab(monitor)
                # mss → numpy (BGRA → BGR)
                image = np.array(screenshot)
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                return image

        except Exception as e:
            logger.error(f"Screenshot hiba: {e}")
            return None

    # ── Kártya felismerés ─────────────────────────────────────────────────────

    def _recognize_card(self, image: np.ndarray, det: Detection) -> Optional[CardRead]:
        """
        Egy detektált kártya rank+suit felismerése.
        
        Ha a YOLO modell 52 osztályos (As, Kh, ...) → közvetlenül
        Ha a YOLO csak 'card_face'-t ad → külön classifier kell
        """
        # Opció 1: A YOLO class_name maga a kártya (pl. "As", "Kh")
        if len(det.class_name) == 2 and det.class_name[0] in 'AKQJT98765432':
            rank = det.class_name[0]
            suit = det.class_name[1].lower()
            if suit in 'shdc':
                return CardRead(
                    rank=rank, suit=suit,
                    confidence=det.confidence,
                    bbox=(det.x1, det.y1, det.x2, det.y2),
                )

        # Opció 2: Külön card classifier
        if self._card_classifier is not None:
            region = det.crop_from(image)
            return self._classify_card(region, det)

        # Opció 3: Template matching fallback (ha nincs classifier)
        # TODO: implementálni ha szükséges
        logger.debug(f"Kártya felismerés sikertelen: {det.class_name}")
        return None

    def _classify_card(self, card_image: np.ndarray, det: Detection) -> Optional[CardRead]:
        """Kártya klasszifikáló CNN használata."""
        # TODO: implementálni a CardClassifier CNN-nel
        # Ez a YOLO tréning OPCIÓ A-jához kell
        return None

    @staticmethod
    def _load_card_classifier(model_path: str):
        """Kártya klasszifikáló modell betöltése."""
        # TODO: implementálni
        logger.info(f"Card classifier: {model_path} (TODO)")
        return None

    # ── Név cache ─────────────────────────────────────────────────────────────

    def _read_player_name(self, image: np.ndarray, det: Detection,
                          seat: int) -> Optional[str]:
        """
        Játékos név olvasása cache-eléssel.
        A neveket nem kell minden frame-ben újraolvasni.
        """
        now = time.time()
        if (seat in self._name_cache and
                now - self._name_cache_age.get(seat, 0) < self._NAME_CACHE_TTL):
            return self._name_cache[seat]

        region = det.crop_from(image)
        name = TextReader.read_text(region)

        if name and len(name) >= 2:
            # Egyszerű cleanup
            name = name.strip().replace('\n', '')
            self._name_cache[seat] = name
            self._name_cache_age[seat] = now
            return name

        return self._name_cache.get(seat)

    # ── Debounce ──────────────────────────────────────────────────────────────

    def _debounce_value(self, new_values: list,
                        history: list) -> Optional[float]:
        """
        Debounce: csak akkor fogadja el az értéket, ha stabil.
        Animáció közben az OCR szemetet olvashat.

        [TASK-2 FIX] Ha new_values üres (nincs detekció az adott frame-ben),
        a history-t töröljük.  Az eredeti kód megtartotta a régi history-t,
        ami kéz vége után is visszaadta az utolsó pot/stack értéket
        (phantom detekció) -> hamis hand_over eseményt okozhatott.
        """
        if new_values:
            # Az első értéket vesszük (pot-nál általában egy van)
            value = new_values[0]
            history.append(value)
        else:
            # Nincs detekció -> az objektum eltunt a képernyon,
            # töröljük a múltat, hogy ne maradjon "phantom" érték.
            history.clear()
            return None

        # Max N értéket tárolunk
        while len(history) > self._DEBOUNCE_FRAMES:
            history.pop(0)

        if not history:
            return None

        # Ha az utolsó N frame hasonló értéket ad → stabil
        if len(history) >= 2:
            avg = sum(history) / len(history)
            tolerance = max(avg * 0.1, 0.01)  # 10% tolerancia
            if all(abs(v - avg) < tolerance for v in history[-2:]):
                return round(avg, 2)

        # Fallback: legutolsó érték
        return history[-1]

    # ── Cache reset ───────────────────────────────────────────────────────────

    def reset_caches(self):
        """Összes cache törlése (pl. asztalváltáskor)."""
        self._name_cache.clear()
        self._name_cache_age.clear()
        self._debounce_pot.clear()
        self._debounce_stacks.clear()
        logger.debug("ScreenReader cache-ek törölve")

    # ── Debug ─────────────────────────────────────────────────────────────────

    def save_debug_frame(self, image: np.ndarray, detections: list,
                         output_path: str):
        """
        Debug: kirajzolja a detekciók bounding box-ait a képre és elmenti.
        Hasznos a YOLO modell finomhangolásához.
        """
        cv2 = _import_cv2()

        debug_img = image.copy()
        colors = {
            'card_face': (0, 255, 0),
            'card_back': (128, 128, 128),
            'pot_text': (0, 200, 255),
            'stack_text': (255, 200, 0),
            'player_name': (200, 200, 200),
            'dealer_button': (0, 0, 255),
            'action_badge': (255, 0, 200),
            'bet_amount': (100, 255, 100),
            'seat_highlight': (255, 255, 0),
        }

        for det in detections:
            color = colors.get(det.class_name, (255, 255, 255))
            cv2.rectangle(debug_img, (det.x1, det.y1), (det.x2, det.y2),
                          color, 2)
            label = f"{det.class_name} {det.confidence:.0%}"
            cv2.putText(debug_img, label, (det.x1, det.y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imwrite(output_path, debug_img)
        logger.info(f"Debug frame mentve: {output_path}")
