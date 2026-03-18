"""
inference/obs_builder.py  –  RLCard obs vektor pontos rekonstrukciója

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORRÁS: rlcard/envs/nolimitholdem.py + rlcard/games/nolimitholdem/game.py
        + rlcard/games/limitholdem/player.py + rlcard/games/base.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OBS VEKTOR STRUKTÚRA (forrás: nolimitholdem.py _extract_state()):

  state_shape = [[54] for _ in range(self.num_players)]  → MINDIG 54 dim!

  obs = np.zeros(54)
  cards = public_cards + hand          ← board + kéz EGYÜTT
  idx = [self.card2index[card] for card in cards]
  obs[idx] = 1                         ← one-hot, 0..51 dimenzió
  obs[52] = float(my_chips)            ← in_chips: BETETT összeg
  obs[53] = float(max(all_chips))      ← max betett összeg

KÁRTYA FORMÁTUM (forrás: Card.get_index() = self.suit + self.rank):
  Card('S','A').get_index() = 'SA'
  Card('H','K').get_index() = 'HK'
  Card('D','T').get_index() = 'DT'

CARD2INDEX SORREND (forrás: limitholdem/card2index.json):
  'SA'=0, 'S2'=1, ..., 'SK'=12
  'HA'=13, 'H2'=14, ..., 'HK'=25
  'DA'=26, 'D2'=27, ..., 'DK'=38
  'CA'=39, 'C2'=40, ..., 'CK'=51

CHIP ÉRTÉKEK – KRITIKUS RÉSZLET:
  my_chips  = player.in_chips     = amit a LEOSZTÁSBA TETT (betett összeg)
  all_chips = [p.in_chips ...]    = minden játékos betett összege
  stakes    = [p.remained_chips]  = megmaradt stack (raw_obs['stakes'])

  Az obs[52] és obs[53] BETETT összeget tartalmaz, NEM a maradék stacket!

  OCR-ből való rekonstrukcióhoz:
    my_chips  = mennyi chipet tettél be EDDIG ebben a leosztásban
    all_chips = minden játékos által betett összeg
    Ha csak a maradék stacket tudod (stakes), akkor:
      in_chips = init_stack - remained_chips
      (ha nem tudod az init_stack-et, közelítsd 0-val preflop elején)

KONVERZIÓS ÖSSZEFOGLALÓ:
  Mi formátumunk → rlcard formátum:
    'As' → 'SA'    (Ace of Spades)
    'Kh' → 'HK'    (King of Hearts)
    'Td' → 'DT'    (Ten of Diamonds)
    '2c' → 'C2'    (Two of Clubs)
"""

import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger("PokerAI")

# ─────────────────────────────────────────────────────────────────────────────
# CARD2INDEX – forrás: rlcard/games/limitholdem/card2index.json
# ─────────────────────────────────────────────────────────────────────────────

CARD2INDEX = {
    "SA": 0,  "S2": 1,  "S3": 2,  "S4": 3,  "S5": 4,
    "S6": 5,  "S7": 6,  "S8": 7,  "S9": 8,  "ST": 9,
    "SJ": 10, "SQ": 11, "SK": 12,
    "HA": 13, "H2": 14, "H3": 15, "H4": 16, "H5": 17,
    "H6": 18, "H7": 19, "H8": 20, "H9": 21, "HT": 22,
    "HJ": 23, "HQ": 24, "HK": 25,
    "DA": 26, "D2": 27, "D3": 28, "D4": 29, "D5": 30,
    "D6": 31, "D7": 32, "D8": 33, "D9": 34, "DT": 35,
    "DJ": 36, "DQ": 37, "DK": 38,
    "CA": 39, "C2": 40, "C3": 41, "C4": 42, "C5": 43,
    "C6": 44, "C7": 45, "C8": 46, "C9": 47, "CT": 48,
    "CJ": 49, "CQ": 50, "CK": 51,
}

OBS_SIZE = 54  # state_shape = [[54] for _ in range(num_players)] – mindig 54!

# Kártya formátum fordítótáblák (a mi formátumunk → rlcard Card.get_index())
_SUIT_TO_RLCARD = {'s': 'S', 'h': 'H', 'd': 'D', 'c': 'C'}
_RANK_TO_RLCARD = {
    '2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9',
    't':'T','j':'J','q':'Q','k':'K','a':'A',
}


# ─────────────────────────────────────────────────────────────────────────────
# Kártya konverzió
# ─────────────────────────────────────────────────────────────────────────────

def our_format_to_rlcard(card) -> Optional[str]:
    """
    A mi formátumunkból ('As') rlcard Card.get_index() formátumra ('SA').

    Forrás: Card.get_index() = self.suit + self.rank  (nagybetű Suit+Rank)

    Példák:
        'As' → 'SA'    'Kh' → 'HK'    'Td' → 'DT'    '2c' → 'C2'
        'as' → 'SA'    'KH' → 'HK'    (kis/nagybetű vegyes is OK)

    Érvénytelen → None (nem dob hibát).
    """
    if not card or not isinstance(card, str) or len(card) < 2:
        return None
    rank_char = card[0].lower()
    suit_char = card[1].lower()
    rank = _RANK_TO_RLCARD.get(rank_char)
    suit = _SUIT_TO_RLCARD.get(suit_char)
    if rank is None or suit is None:
        return None
    return suit + rank   # 'S' + 'A' = 'SA'


def card_to_obs_index(card) -> Optional[int]:
    """
    Kártya → obs vektor index (0..51).

    Elfogad mindkét formátumot:
        'As'  → 0    (a mi formátumunk)
        'SA'  → 0    (rlcard Card.get_index() formátum)

    Érvénytelen → None.
    """
    if not card or not isinstance(card, str):
        return None
    # Próbáljuk rlcard formátumként (SA, HK, ...)
    upper = card.upper()
    if upper in CARD2INDEX:
        return CARD2INDEX[upper]
    # Próbáljuk a mi formátumunkból konvertálni
    rlcard_fmt = our_format_to_rlcard(card)
    if rlcard_fmt and rlcard_fmt in CARD2INDEX:
        return CARD2INDEX[rlcard_fmt]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ObsBuilder
# ─────────────────────────────────────────────────────────────────────────────

class ObsBuilder:
    """
    RLCard NolimitHoldem obs vektor pontos rekonstrukciója.

    Az rlcard _extract_state() pontos másolata:

        obs = np.zeros(54)
        cards = public_cards + hand
        obs[[card2index[c] for c in cards]] = 1
        obs[52] = float(my_chips)       ← BETETT összeg (in_chips)
        obs[53] = float(max(all_chips)) ← max betett összeg

    FIGYELEM: my_chips és all_chips a BETETT összeget jelenti,
    NEM a megmaradt stacket! Lásd a modul docstringet.

    Használat:
        builder = ObsBuilder()

        # Preflop, mi vagyunk a BB (2 chipet tettünk be):
        obs = builder.build(
            hole_cards  = ['As', 'Kh'],
            board_cards = [],
            my_chips    = 2.0,          # betett összeg (in_chips)
            all_chips   = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],  # minden játékos betett összege
        )

        # Flop, valaki 20-at emelt, mi 20-at hívtunk (összesen 22 berakva):
        obs = builder.build(
            hole_cards  = ['As', 'Kh'],
            board_cards = ['Td', '7c', '2s'],
            my_chips    = 22.0,         # in_chips (sb 1 + flop call 20 + bb 2 - 1)
            all_chips   = [22.0, 21.0, 0.0, 20.0, 0.0, 0.0],
        )
    """

    OBS_SIZE = OBS_SIZE  # 54, minden játékosszámnál

    def __init__(self, num_players: int = 6):
        self.num_players = num_players

    def build(self,
              hole_cards:  List,
              board_cards: List,
              my_chips:    float,
              all_chips:   List[float]) -> np.ndarray:
        """
        Összerakja az rlcard-kompatibilis obs vektort.

        Paraméterek:
            hole_cards:  ['As', 'Kh']  – saját lapok (a mi formátumunk VAGY rlcard)
            board_cards: ['Td', '7c']  – board lapok (preflop: [])
            my_chips:    float  – BETETT összeg (player.in_chips, NEM remained_chips!)
            all_chips:   list   – minden játékos BETETT összege

        Visszatér: np.ndarray shape=(54,) dtype=float32
        """
        obs = np.zeros(self.OBS_SIZE, dtype=np.float32)

        # ── Lapok one-hot (public_cards + hand együtt, 0-51 dim) ─────────────
        # Az rlcard sorrendje: public_cards + hand (nem hand + public_cards!)
        all_cards = list(board_cards or []) + list(hole_cards or [])
        for card in all_cards:
            idx = card_to_obs_index(card)
            if idx is not None:
                obs[idx] = 1.0

        # ── Chip értékek (52-53 dim) ──────────────────────────────────────────
        obs[52] = float(my_chips)
        chips   = [float(c) for c in (all_chips or []) if c is not None]
        obs[53] = float(max(chips)) if chips else float(my_chips)

        return obs

    @property
    def obs_size(self) -> int:
        return self.OBS_SIZE

    def __repr__(self):
        return f"ObsBuilder(num_players={self.num_players}, obs_size={self.OBS_SIZE})"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: OCR stack → in_chips konverzió
# ─────────────────────────────────────────────────────────────────────────────

def remained_to_in_chips(remained_chips: float,
                          init_chips: float = 100.0) -> float:
    """
    Ha az OCR a MEGMARADT stack-et adja (stakes/remained_chips),
    ezt konvertálja BETETT összeggé (in_chips) az obs_builder számára.

    Képlet: in_chips = init_chips - remained_chips

    Paraméterek:
        remained_chips: megmaradt chip (OCR által látott stack méret)
        init_chips:     induló chip (default: 100, a chips_for_each config értéke)

    Megjegyzés: Ha nem tudod pontosan az init_chips-t (pl. különböző stackekkel
    ülnek le), az OCR-ből olvasott induló stack értéket használd.
    """
    return max(0.0, float(init_chips) - float(remained_chips))


# ─────────────────────────────────────────────────────────────────────────────
# Gyors helper
# ─────────────────────────────────────────────────────────────────────────────

def build_obs(hole_cards:  List,
              board_cards: List,
              my_chips:    float,
              all_chips:   List[float],
              num_players: int = 6) -> np.ndarray:
    """
    Gyors helper – ObsBuilder példányosítás nélkül.
    """
    return ObsBuilder(num_players).build(
        hole_cards=hole_cards, board_cards=board_cards,
        my_chips=my_chips, all_chips=all_chips,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def _run_sanity_check():
    print("=== ObsBuilder sanity check (rlcard forráskód alapján) ===\n")

    # 1. Kártya konverzió – Card.get_index() = suit + rank
    cases = [
        # (mi formátum, rlcard formátum, card2index index)
        ('As', 'SA',  0),   # Ace of Spades   – SA=0
        ('2s', 'S2',  1),   # Two of Spades   – S2=1
        ('Ts', 'ST',  9),   # Ten of Spades   – ST=9
        ('Ks', 'SK', 12),   # King of Spades  – SK=12
        ('Ah', 'HA', 13),   # Ace of Hearts   – HA=13
        ('Kh', 'HK', 25),   # King of Hearts  – HK=25
        ('Ad', 'DA', 26),   # Ace of Diamonds – DA=26
        ('Td', 'DT', 35),   # Ten of Diamonds – DT=35
        ('Ac', 'CA', 39),   # Ace of Clubs    – CA=39
        ('2c', 'C2', 40),   # Two of Clubs    – C2=40
        ('7c', 'C7', 45),   # Seven of Clubs  – C7=45
        ('Kc', 'CK', 51),   # King of Clubs   – CK=51
    ]
    for our_fmt, rlcard_fmt, expected_idx in cases:
        got_fmt = our_format_to_rlcard(our_fmt)
        assert got_fmt == rlcard_fmt, f"{our_fmt}→{got_fmt} (expected {rlcard_fmt})"
        got_idx = card_to_obs_index(our_fmt)
        assert got_idx == expected_idx, f"{our_fmt} idx={got_idx} (expected {expected_idx})"
        got_idx2 = card_to_obs_index(rlcard_fmt)
        assert got_idx2 == expected_idx, f"{rlcard_fmt} direct idx={got_idx2}"
    assert our_format_to_rlcard('XX') is None
    assert card_to_obs_index(None) is None
    print("OK: kártya konverzió")
    print("    As→SA=0, Kh→HK=25, Td→DT=35, 2c→C2=40, Kc→CK=51")

    # 2. obs vektor – pontos rlcard logika másolata
    builder = ObsBuilder(num_players=6)
    assert builder.obs_size == 54

    # Teszteset: preflop
    # Lapok: As(SA=0), Kh(HK=25) a kézben, Td(DT=35) a boardon
    # Az rlcard sorrendje: cards = public_cards + hand
    # → [DT, SA, HK] → idx = [35, 0, 25]
    obs = builder.build(
        hole_cards  = ['As', 'Kh'],
        board_cards = ['Td'],
        my_chips    = 2.0,      # BB-ként 2 chipet tett be
        all_chips   = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    )

    assert obs.shape == (54,) and obs.dtype == np.float32
    assert obs[0]  == 1.0, f"As(SA=0): {obs[0]}"
    assert obs[25] == 1.0, f"Kh(HK=25): {obs[25]}"
    assert obs[35] == 1.0, f"Td(DT=35): {obs[35]}"
    assert obs[:52].sum() == 3.0, f"3 lap: {obs[:52].sum()}"
    assert obs[52] == 2.0,  f"my_chips=2: {obs[52]}"
    assert obs[53] == 2.0,  f"max(all_chips)=2: {obs[53]}"
    print("OK: obs rekonstrukció (shape=54, lapok + chip értékek)")
    print(f"    As=obs[0]={obs[0]:.0f}, Kh=obs[25]={obs[25]:.0f}, Td=obs[35]={obs[35]:.0f}")
    print(f"    my_chips=obs[52]={obs[52]:.0f}, max_chips=obs[53]={obs[53]:.0f}")

    # 3. Preflop – üres board
    obs_pre = builder.build(['As','Kh'], [], 2.0, [1.0, 2.0])
    assert obs_pre[:52].sum() == 2.0
    assert obs_pre[0] == 1.0 and obs_pre[25] == 1.0
    print("OK: preflop (üres board, 2 kártyás one-hot)")

    # 4. Érvénytelen lapok silent skip
    obs_bad = builder.build(['As', 'INVALID', None, ''], [], 0.0, [0.0])
    assert obs_bad[0] == 1.0 and obs_bad[:52].sum() == 1.0
    print("OK: érvénytelen lapok silent skip")

    # 5. obs_size mindig 54 – játékosszámtól független
    for n in [2,3,4,5,6,7,8,9]:
        assert ObsBuilder(n).obs_size == 54
    print("OK: obs_size=54 minden játékosszámnál (state_shape=[[54]...])")

    # 6. remained_to_in_chips helper
    assert remained_to_in_chips(98.0, 100.0) == 2.0
    assert remained_to_in_chips(100.0, 100.0) == 0.0
    assert remained_to_in_chips(80.0, 100.0) == 20.0
    print("OK: remained_to_in_chips (megmaradt→betett konverzió)")

    # 7. build_obs helper
    obs2 = build_obs(['As','Kh'], [], 2.0, [1.0, 2.0])
    assert obs2.shape == (54,) and obs2[0] == 1.0
    print("OK: build_obs helper")

    print("\n=== Összes teszt OK! ===\n")
    print("FONTOS MEGJEGYZÉSEK:")
    print("  1. obs mérete MINDIG 54, játékosszámtól FÜGGETLEN")
    print("  2. obs[52] = in_chips = BETETT összeg (NEM megmaradt stack!)")
    print("  3. obs[53] = max(all_chips) = max betett összeg")
    print("  4. Kártya sorrend: board (public) + kéz (hand) együtt")
    print()
    print("OCR integrációhoz:")
    print("  Ha az OCR a megmaradt stacket adja (pl. 98 chipet lát),")
    print("  használd: my_chips = remained_to_in_chips(98.0, init_stack=100.0)")
    print("  → my_chips = 2.0 (amennyit betett: BB volt)")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.WARNING)
    _run_sanity_check()
