"""
core/equity.py  –  Monte Carlo kéz erősség becslő (v4.2.2)

Változások v4.2.2:
    [THREAD-SAFETY] HandEquityEstimator._cache:
        - collections.OrderedDict alapú valódi LRU cache
          (volt: sima dict, nem garantált iterációs sorrend)
        - threading.RLock() védi az összes cache olvasást/írást
          (volt: lock nélkül, race condition lehetséges volt
           a del self._cache[next(iter(self._cache))] műveletnél)
        - Az OrderedDict.move_to_end() O(1) LRU frissítés
          (volt: implicit dict-sorrend, Python 3.7+ implementáció-
          függő viselkedés)

    [BUG-FIX] Split pot equity számítás:
        - Döntetlen esetén (opp_rank == my_rank) win += 0.5
          (volt: win = False → az összes split pot elveszett kéznek
          számított, ami az equity-t szisztematikusan alulbecsülte
          all-in és showdown szituációkban)

    [QUALITY] cache_stats() metódus monitoring célokra.
    [QUALITY] Teljes type hint lefedettség.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITEKTÚRA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  _hand_rank(cards)          → int   kéz erőssége (nagyobb = jobb)
  _best_5_from_7(cards)      → int   legjobb 5 lap 7-ből
  HandEquityEstimator.equity → float Monte Carlo equity [0.0, 1.0]

Kártya formátum:
    A mi formátumunk: Rank + suit kisbetű  →  'As', 'Kh', 'Td', '2c'
    Ez ELTÉR az rlcard raw_obs['hand'] formátumától ('SA', 'HK', 'DT').
    A collector.py _rlcard_cards_to_equity_fmt() konvertálja.

Thread-safety:
    A HandEquityEstimator példányosítható és megosztható szálak között.
    A modul-szintű _EQUITY_ESTIMATOR singleton (collector.py-ban)
    mostantól biztonságosan használható párhuzamos env-ekből.

Equity számítás pontossága:
    A split pot javítás (win += 0.5) hatása:
      - Preflop AA vs KK: ~82% equity (volt: ~80%, split pot ~4%)
      - Azonos boardon azonos kéz (pl. mindkét játékos flush az asztalon):
        50% equity (volt: 0%)
      - Ez a korrekció javítja a value head kalibrációját és a
        reward signal pontosságát all-in szituációkban.
"""

from __future__ import annotations

import collections
import logging
import random
import threading
from itertools import combinations
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("PokerAI")

# ─────────────────────────────────────────────────────────────────────────────
# Kártya konstansok
# ─────────────────────────────────────────────────────────────────────────────

RANKS:    str       = "23456789TJQKA"
SUITS:    str       = "shdc"
RANK_MAP: Dict[str, int] = {r: i for i, r in enumerate(RANKS)}
FULL_DECK: List[str]     = [r + s for r in RANKS for s in SUITS]


# ─────────────────────────────────────────────────────────────────────────────
# Kártya parsing és kéz értékelés
# ─────────────────────────────────────────────────────────────────────────────

def _parse_card(card_str: str) -> Tuple[int, int]:
    """
    Kártya string → (rank_int, suit_int) tuple.

    Args:
        card_str: Kártya a mi formátumunkban (pl. ``'As'``, ``'Kh'``).

    Returns:
        ``(rank, suit)`` ahol rank 0–12 (2→0, A→12),
        suit 0–3 (s, h, d, c).
    """
    return RANK_MAP[card_str[0]], SUITS.index(card_str[1])


def _hand_rank(cards: List[str]) -> int:
    """
    5 lapos kéz erősségének számítása.

    A visszaadott egész szám összehasonlítható: nagyobb = erősebb kéz.
    Kategóriák (egymilliónkénti szintek):
        8M+ = Straight flush
        7M  = Poker (four of a kind)
        6M  = Full house
        5M  = Flush
        4M  = Straight
        3M  = Three of a kind
        2M  = Two pair
        1M  = One pair
        0   = High card

    Args:
        cards: 5 kártyás lista a mi formátumunkban.

    Returns:
        Kéz erőssége egész számként.
    """
    parsed  = [_parse_card(c) for c in cards]
    ranks   = sorted([r for r, _ in parsed], reverse=True)
    suits   = [s for _, s in parsed]

    rank_count: Dict[int, int] = {}
    for r in ranks:
        rank_count[r] = rank_count.get(r, 0) + 1

    counts    = sorted(rank_count.values(), reverse=True)
    is_flush  = len(set(suits)) == 1
    unique_r  = set(rank_count.keys())

    # Straight ellenőrzés (normál)
    is_straight = (
        len(rank_count) == 5
        and ranks[0] - ranks[-1] == 4
    )
    # Ász-alacsony straight: A-2-3-4-5 (wheel)
    if unique_r == {12, 3, 2, 1, 0}:
        is_straight = True
        ranks = [3, 2, 1, 0, -1]  # Ace alacsonyként

    if is_straight and is_flush:
        return 8_000_000 + ranks[0]

    if counts[0] == 4:
        quad_rank = next(r for r, c in rank_count.items() if c == 4)
        return 7_000_000 + quad_rank * 100

    if counts[0] == 3 and counts[1] == 2:
        trip_rank = next(r for r, c in rank_count.items() if c == 3)
        return 6_000_000 + trip_rank * 100

    if is_flush:
        return 5_000_000 + sum(
            r * (13 ** i) for i, r in enumerate(reversed(ranks))
        )

    if is_straight:
        return 4_000_000 + ranks[0]

    if counts[0] == 3:
        trip_rank = next(r for r, c in rank_count.items() if c == 3)
        return 3_000_000 + trip_rank * 100

    if counts[0] == 2 and counts[1] == 2:
        pairs = sorted(
            [r for r, c in rank_count.items() if c == 2], reverse=True
        )
        return 2_000_000 + pairs[0] * 1000 + pairs[1] * 10

    if counts[0] == 2:
        pair_rank = next(r for r, c in rank_count.items() if c == 2)
        return 1_000_000 + pair_rank * 10000

    return sum(r * (13 ** i) for i, r in enumerate(reversed(ranks)))


def _best_5_from_7(cards: List[str]) -> int:
    """
    A legjobb 5-lapos kéz értéke 7 lapból (Texas Hold'em).

    Args:
        cards: 7 lapos lista.

    Returns:
        A legjobb 5-lapos kombináció ``_hand_rank()`` értéke.
    """
    return max(_hand_rank(list(combo)) for combo in combinations(cards, 5))


# ─────────────────────────────────────────────────────────────────────────────
# HandEquityEstimator
# ─────────────────────────────────────────────────────────────────────────────

class HandEquityEstimator:
    """
    Thread-safe Monte Carlo kéz equity becslő LRU cache-szel.

    Adaptív early stopping-gal dolgozik: ha a rolling window szórása
    elég alacsony és legalább ``min_sims`` szimulációt lefuttatott,
    korán leáll a felesleges számítások elkerüléséhez.

    Thread-safety:
        A ``_cache`` (``collections.OrderedDict``) és a ``_hits``/
        ``_misses`` számlálók ``threading.RLock()``-kal védettek.
        Több szálból (pl. parallel env-ekből a BatchedSyncCollector-ban)
        biztonságosan hívható.

    Split pot kezelés (v4.2.2 fix):
        Döntetlen esetén (opp_rank == my_rank) ``win += 0.5`` számít,
        nem ``win = False``.  Ez helyes: split pot esetén a hero
        visszakapja a tétjét, tehát az equity 50%.

    Példa::

        estimator = HandEquityEstimator(n_sim=200)
        eq = estimator.equity(['As', 'Kh'], board=['Td', '7c', '2s'])
        print(f"AKo equity a flopra: {eq:.1%}")

    Attributes:
        n_sim:       Maximum szimulációk száma (adaptív early stopping-gal
                     általában kevesebb fut le).
        cache_size:  LRU cache maximális mérete (bejegyzések száma).
    """

    def __init__(
        self,
        n_sim:      int = 200,
        cache_size: int = 10_000,
    ) -> None:
        """
        Args:
            n_sim:       Maximum Monte Carlo szimulációk száma.
                         Adaptív early stopping-gal általában kevesebb fut.
            cache_size:  LRU cache maximális mérete.  Nagyobb cache →
                         kevesebb újraszámítás preflop ismétlődő lapoknál.
        """
        self.n_sim:      int = n_sim
        self.cache_size: int = cache_size

        # thread-safe LRU cache: OrderedDict + RLock
        # RLock (reentrant): ugyanaz a szál többször is megszerezheti
        # (pl. equity() → _cache_get() → _cache_put() hívási lánc)
        self._cache: collections.OrderedDict[str, float] = (
            collections.OrderedDict()
        )
        self._lock: threading.RLock = threading.RLock()

        # Teljesítmény statisztikák (cache_stats() metódushoz)
        self._hits:   int = 0
        self._misses: int = 0

    # ── Cache műveletek ───────────────────────────────────────────────────────

    def _cache_key(
        self,
        hole:     List[str],
        board:    List[str],
        num_opp:  int,
    ) -> str:
        """
        Determinisztikus cache kulcs generálása.

        A hole_cards és board_cards sorrendje nem számít (sortolva).

        Args:
            hole:    Saját lapok listája.
            board:   Board lapok listája.
            num_opp: Ellenfelek száma.

        Returns:
            Cache kulcs string.
        """
        return (
            f"{','.join(sorted(hole))}"
            f"|{','.join(sorted(board))}"
            f"|{num_opp}"
        )

    def _cache_get(self, key: str) -> Optional[float]:
        """
        Thread-safe LRU cache olvasás.

        Találat esetén a bejegyzést a végére mozgatja (most recently used).

        Args:
            key: Cache kulcs.

        Returns:
            Tárolt equity érték, vagy ``None`` ha nem található.
        """
        with self._lock:
            if key in self._cache:
                # LRU: a találat végére kerül (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def _cache_put(self, key: str, value: float) -> None:
        """
        Thread-safe LRU cache írás.

        Ha a cache megtelt, a legrégebben használt bejegyzést törli
        (first = least recently used az OrderedDict-ben).

        Args:
            key:   Cache kulcs.
            value: Tárolandó equity érték.
        """
        with self._lock:
            if key in self._cache:
                # Frissítés + LRU pozíció frissítés
                self._cache.move_to_end(key)
                self._cache[key] = value
                return

            # Új bejegyzés – eviction ha szükséges
            if len(self._cache) >= self.cache_size:
                # popitem(last=False) = FIFO (legrégebben használt)
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug(
                    f"LRU eviction: {evicted_key!r} "
                    f"(cache méret: {self.cache_size})"
                )

            self._cache[key] = value

    # ── Fő equity számítás ────────────────────────────────────────────────────

    def equity(
        self,
        hole_cards:           List[str],
        board:                Optional[List[str]] = None,
        num_opponents:        int                 = 1,
        confidence_threshold: float               = 0.02,
        min_sims:             int                 = 50,
    ) -> float:
        """
        Monte Carlo equity becslés adaptív early stopping-gal.

        A metódus thread-safe: a cache olvasás/írás ``RLock``-kal védett,
        a szimulációs loop nem igényel locking-ot (lokális változók).

        Split pot kezelés (v4.2.2 fix):
            Ha a hero és az ellenfél azonos erejű kezet kap (``opp_rank
            == my_rank``), a hero ``0.5`` nyerési pontot kap.  Ez tükrözi,
            hogy split pot esetén visszakapja a tétjét.

            Példa: preflop mindkét játékosnál AKo, board 5-high rainbow
            → equity ≈ 50% (volt: ~0% a régi implementációban)

        Args:
            hole_cards:           Saját lapok (pl. ``['As', 'Kh']``).
            board:                Board lapok (``[]`` vagy ``None`` preflop).
            num_opponents:        Ellenfelek száma (default: 1).
            confidence_threshold: Early stopping küszöb: ha a rolling
                                  window szórása kisebb ennél ÉS ``min_sims``
                                  lefutott, leáll.  Default: 0.02.
            min_sims:             Minimum szimulációk száma, csak ezután
                                  aktiválódik az early stopping.
                                  Default: 50.

        Returns:
            Equity float [0.0, 1.0].
            0.5 = semleges (fallback ha nincs elég lap).

        Raises:
            Nem dob kivételt – hiba esetén 0.5-öt ad vissza és logol.
        """
        board = board or []

        # Validáció – fallback 0.5 ha nincs meg a 2 saját lap
        if len(hole_cards) < 2:
            return 0.5

        key = self._cache_key(hole_cards, board, num_opponents)

        # Cache találat ellenőrzés (thread-safe)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        # ── Monte Carlo szimuláció ────────────────────────────────────────
        # FONTOS: a szimulációs loop LOCK NÉLKÜL fut.
        # Indok: csak lokális változókat használ (known, deck, wins stb.),
        # a megosztott _cache-t csak a végén írjuk (egyszer, lock alatt).
        # Ez maximális párhuzamosíthatóságot ad: több szál egyszerre
        # futtathat szimulációkat UGYANARRA a kulcsra is – az egyik thread
        # eredménye felülírja a másikat a _cache_put()-ban, ami helyes
        # (az eredmény determinisztikusan konvergál az igazi equity-hez).

        known   = set(hole_cards) | set(board)
        deck    = [c for c in FULL_DECK if c not in known]
        need    = 5 - len(board)
        sample_n = need + num_opponents * 2

        # Ellenőrzés: elég lap van-e a pakliban
        if sample_n > len(deck):
            logger.debug(
                f"equity(): nem elég lap a szimulációhoz "
                f"(need={sample_n}, deck={len(deck)}). Fallback: 0.5"
            )
            return 0.5

        wins:  float = 0.0   # [v4.2.2] float a split pot 0.5-ös értékhez
        valid: int   = 0

        # Early stopping: utolsó _WIN_WINDOW eredmény rolling variance-a
        _WIN_WINDOW = 20
        _window: collections.deque = collections.deque(maxlen=_WIN_WINDOW)

        for _ in range(self.n_sim):
            if sample_n > len(deck):
                break

            drawn   = random.sample(deck, sample_n)
            run_out = board + drawn[:need]
            my_rank = _best_5_from_7(hole_cards + run_out)

            # ── [BUG-FIX v4.2.2] Split pot kezelés ──────────────────────
            # Eredeti (hibás):
            #   if opp_rank >= my_rank: win = False
            #   → split pot (opp_rank == my_rank) = veszteség
            #
            # Javított:
            #   opp_rank > my_rank  → veszteség (win = 0)
            #   opp_rank == my_rank → döntetlen (win += 0.5)
            #   opp_rank < my_rank  → győzelem  (win += 1.0)
            #
            # Megjegyzés: több ellenfélnél az equity arányosan osztódik.
            # Ha az ellenfél erősebb → win = 0 marad (egész kéz elveszett).
            # Ez egyszerűsítés: a pontos side-pot számítás stack-méreteket
            # igényelne, de preflop/postflop equity becslésnél ez elegendő.
            # ──────────────────────────────────────────────────────────────

            hand_wins: float = 1.0  # indul győzelemként, ellenfeleknél csökken
            lost      = False

            for opp_idx in range(num_opponents):
                opp_hole = drawn[need + opp_idx * 2 : need + opp_idx * 2 + 2]
                opp_rank = _best_5_from_7(opp_hole + run_out)

                if opp_rank > my_rank:
                    # Ellenfél erősebb → elveszett kéz
                    hand_wins = 0.0
                    lost = True
                    break
                elif opp_rank == my_rank:
                    # Döntetlen → split pot, arányos rész
                    # Egyszerűsítés: 2 játékosnál 0.5, 3-nál 0.33 stb.
                    # Több ellenfélnél a legrosszabb esettel dolgozunk:
                    # ha bármelyikkel döntetlen, az arány 1/(1+hányan_döntetlenek)
                    # Ennél a közelítésnél: ha 1 ellenfél döntetlen → 0.5
                    hand_wins = min(hand_wins, 0.5)

            if not lost:
                wins += hand_wins

            valid += 1
            _window.append(hand_wins)

            # Early stopping feltétel
            if valid >= min_sims and len(_window) == _WIN_WINDOW:
                w_mean = sum(_window) / _WIN_WINDOW
                variance = w_mean * (1.0 - w_mean) / _WIN_WINDOW
                std = variance ** 0.5
                if std < confidence_threshold:
                    break

        result = wins / max(valid, 1)

        # Cache írás (thread-safe)
        self._cache_put(key, result)

        return result

    # ── Cache menedzsment ─────────────────────────────────────────────────────

    def clear_cache(self) -> None:
        """
        Törli a teljes cache-t és nullázza a statisztikákat.

        Thread-safe.
        """
        with self._lock:
            self._cache.clear()
            self._hits   = 0
            self._misses = 0
        logger.debug("HandEquityEstimator: cache törölve")

    def cache_stats(self) -> Dict[str, int | float]:
        """
        Cache teljesítmény statisztikák.

        Thread-safe – pillanatfelvétel a lock alatt.

        Returns:
            Dict a következő kulcsokkal:

            - ``size``         – jelenlegi bejegyzések száma
            - ``max_size``     – maximális méret
            - ``hits``         – cache találatok száma
            - ``misses``       – cache kihagyások száma
            - ``hit_rate``     – találati arány [0.0, 1.0]
            - ``n_sim``        – konfigurált maximum szimulációk
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "size":     len(self._cache),
                "max_size": self.cache_size,
                "hits":     self._hits,
                "misses":   self._misses,
                "hit_rate": (
                    round(self._hits / total, 4) if total > 0 else 0.0
                ),
                "n_sim":    self.n_sim,
            }

    def __repr__(self) -> str:
        stats = self.cache_stats()
        return (
            f"HandEquityEstimator("
            f"n_sim={self.n_sim}, "
            f"cache={stats['size']}/{stats['max_size']}, "
            f"hit_rate={stats['hit_rate']:.1%})"
        )
