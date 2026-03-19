"""
live_reader/data_types.py  –  Adattípusok a képernyőolvasóhoz

Minden ScreenReader kimenet és GameBridge belső állapot itt definiált.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Kártya
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CardRead:
    """Egy felismert kártya az OCR/YOLO-ból."""
    rank: str              # 'A', 'K', 'Q', 'J', 'T', '9'...'2'
    suit: str              # 's', 'h', 'd', 'c'
    confidence: float      # felismerés biztonsága (0.0-1.0)
    bbox: tuple = None     # (x1, y1, x2, y2) pixel koordináták, opcionális

    def to_equity_format(self) -> str:
        """Equity.py formátum: 'As', 'Kh', 'Td' stb."""
        return f"{self.rank}{self.suit}"

    def to_rlcard_format(self) -> str:
        """RLCard formátum: 'SA', 'HK', 'DT' stb."""
        suit_map = {'s': 'S', 'h': 'H', 'd': 'D', 'c': 'C'}
        return f"{suit_map.get(self.suit, 'S')}{self.rank}"

    def __str__(self):
        symbols = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
        return f"{self.rank}{symbols.get(self.suit, '?')}"

    def __eq__(self, other):
        if not isinstance(other, CardRead):
            return False
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))


# ─────────────────────────────────────────────────────────────────────────────
# Játékos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlayerRead:
    """Egy szék/játékos állapota egyetlen frame-ből."""
    seat_index: int                        # 0..N-1 (képernyő pozíció)
    username: Optional[str] = None         # OCR-rel olvasott név
    stack: Optional[float] = None          # megmaradt stack ($)
    is_dealer: bool = False                # dealer gomb nála van
    is_active: bool = True                 # nem foldolt / nem ül ki
    is_current_turn: bool = False          # nála van a döntés
    is_sitting_out: bool = False           # kiült
    cards: list = field(default_factory=list)  # CardRead lista (üres ha face-down)
    last_action_text: str = ""             # "Fold", "Call", "Raise $4.00", "All-In"
    bet_this_round: Optional[float] = None # aktuális utcán betett összeg

    @property
    def has_cards(self) -> bool:
        return len(self.cards) > 0

    @property
    def is_folded(self) -> bool:
        return "fold" in self.last_action_text.lower() or not self.is_active


# ─────────────────────────────────────────────────────────────────────────────
# Képernyő állapot (egyetlen frame)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScreenState:
    """
    Egyetlen frame teljes állapota.
    A ScreenReader ezt adja vissza minden read_frame() hívásra.
    """
    timestamp: float                               # time.time()
    is_valid: bool = False                         # sikeres olvasás?

    # Asztal adatok
    my_seat: int = 0                               # melyik szék a miénk
    players: list = field(default_factory=list)     # PlayerRead lista
    board_cards: list = field(default_factory=list) # CardRead lista (0/3/4/5)
    my_hole_cards: list = field(default_factory=list)  # CardRead lista (0 vagy 2)
    pot_total: Optional[float] = None              # teljes pot
    current_bet: Optional[float] = None            # legnagyobb bet az utcán

    # Kontextus
    dealer_seat: Optional[int] = None              # button pozíció
    active_player_seat: Optional[int] = None       # ki következik
    num_active_players: int = 0                    # nem foldolt játékosok

    # Diagnosztika
    frame_ms: float = 0.0                          # feldolgozási idő ms
    ocr_errors: list = field(default_factory=list)  # figyelmeztetések
    raw_detections: list = field(default_factory=list)  # nyers YOLO detekciók (debug)

    @property
    def num_board_cards(self) -> int:
        return len(self.board_cards)

    @property
    def street(self) -> int:
        """0=preflop, 1=flop, 2=turn, 3=river"""
        n = self.num_board_cards
        if n == 0: return 0
        if n == 3: return 1
        if n == 4: return 2
        return 3

    @property
    def street_name(self) -> str:
        return ['Preflop', 'Flop', 'Turn', 'River'][min(self.street, 3)]


# ─────────────────────────────────────────────────────────────────────────────
# Delta (két frame közti változás)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlayerAction:
    """Egy becsült játékos akció a delta-ból."""
    seat: int
    username: str
    action_type: str         # "fold", "call", "raise", "check", "bet", "allin"
    amount: float = 0.0      # bet/raise összeg
    abstract_action: int = 1  # 0-6 az RTAManager számára

    def __str__(self):
        if self.amount > 0:
            return f"{self.username}: {self.action_type} ${self.amount:.2f}"
        return f"{self.username}: {self.action_type}"


@dataclass
class FrameDelta:
    """Két egymást követő frame közti különbségek."""
    new_hand_detected: bool = False
    new_street: Optional[int] = None       # 1=flop, 2=turn, 3=river
    new_board_cards: list = field(default_factory=list)
    player_actions: list = field(default_factory=list)  # PlayerAction lista
    active_player_changed: bool = False
    new_active_seat: Optional[int] = None
    pot_changed: bool = False
    new_pot: Optional[float] = None
    my_turn_started: bool = False
    my_turn_ended: bool = False
    hand_over: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Állapotgép fázisok
# ─────────────────────────────────────────────────────────────────────────────

class GamePhase(Enum):
    WAITING_FOR_HAND = "waiting"
    HAND_ACTIVE      = "active"
    MY_TURN          = "my_turn"
    HAND_OVER        = "over"


# ─────────────────────────────────────────────────────────────────────────────
# YOLO detekció típusok
# ─────────────────────────────────────────────────────────────────────────────

class DetectionClass(Enum):
    """
    YOLO osztályok – ezeket kell címkézni a tréning adatokban.

    Két megközelítés a kártyákra:

    OPCIÓ A (egyszerűbb, ajánlott kezdéshez):
      Egy 'card' osztály + másodlagos CNN a rank/suit felismeréséhez
      → Kevesebb címkézés, de két lépéses felismerés

    OPCIÓ B (pontosabb, több munka):
      52 külön osztály minden kártyához (As, Kh, Td, ...)
      → Több címkézés, de egy lépésben felismer

    A többi elem mindkét opcióban ugyanaz.
    """
    # Kártyák
    CARD_FACE = "card_face"            # Felfordított kártya (rank+suit látható)
    CARD_BACK = "card_back"            # Lefordított kártya

    # Szöveg régiók
    POT_TEXT = "pot_text"              # "Pot: $4.65"
    STACK_TEXT = "stack_text"          # "$10.33" egy játékos alatt
    PLAYER_NAME = "player_name"       # "SkyEagle", "Prince10" stb.

    # Gombok és jelölők
    DEALER_BUTTON = "dealer_button"    # D gomb
    ACTION_BADGE = "action_badge"      # "Fold", "All-In", "Call" badge
    BET_AMOUNT = "bet_amount"          # "$0.4", "$4.25" – bet összeg a zöld részen

    # Akció gombok (amikor mi következünk)
    BUTTON_FOLD = "btn_fold"
    BUTTON_CALL = "btn_call"
    BUTTON_RAISE = "btn_raise"
    BUTTON_CHECK = "btn_check"
    BUTTON_ALLIN = "btn_allin"

    # Szék / játékos régió
    SEAT_HIGHLIGHT = "seat_highlight"   # aktív játékos kiemelése (timer/glow)
    EMPTY_SEAT = "empty_seat"           # üres szék


# ─────────────────────────────────────────────────────────────────────────────
# YOLO detekció eredmény
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Egyetlen YOLO detekció."""
    class_name: str        # DetectionClass név
    confidence: float      # 0.0-1.0
    x1: int                # bounding box bal felső x
    y1: int                # bounding box bal felső y
    x2: int                # bounding box jobb alsó x
    y2: int                # bounding box jobb alsó y

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def crop_from(self, image):
        """Kivágja a detektált régiót a képből (numpy array)."""
        return image[self.y1:self.y2, self.x1:self.x2]
