"""
core/equity.py  –  Monte Carlo kéz erősség becslő
"""
import random
from typing import List, Tuple

RANKS = '23456789TJQKA'
SUITS = 'shdc'
RANK_MAP = {r: i for i, r in enumerate(RANKS)}
FULL_DECK = [r + s for r in RANKS for s in SUITS]

def _parse_card(card_str):
    return RANK_MAP[card_str[0]], SUITS.index(card_str[1])

def _hand_rank(cards):
    parsed  = [_parse_card(c) for c in cards]
    ranks   = sorted([r for r, _ in parsed], reverse=True)
    suits   = [s for _, s in parsed]
    r_count = {}
    for r in ranks: r_count[r] = r_count.get(r, 0) + 1
    counts = sorted(r_count.values(), reverse=True)
    is_flush    = len(set(suits)) == 1
    is_straight = (len(r_count) == 5 and ranks[0] - ranks[-1] == 4)
    if set(ranks) == {12, 3, 2, 1, 0}: is_straight = True; ranks = [3, 2, 1, 0, -1]
    if is_straight and is_flush: return 8_000_000 + ranks[0]
    if counts[0] == 4:
        quad_rank = [r for r, c in r_count.items() if c == 4][0]
        return 7_000_000 + quad_rank * 100
    if counts[0] == 3 and counts[1] == 2:
        trip_rank = [r for r, c in r_count.items() if c == 3][0]
        return 6_000_000 + trip_rank * 100
    if is_flush: return 5_000_000 + sum(r * (13 ** i) for i, r in enumerate(reversed(ranks)))
    if is_straight: return 4_000_000 + ranks[0]
    if counts[0] == 3:
        trip_rank = [r for r, c in r_count.items() if c == 3][0]
        return 3_000_000 + trip_rank * 100
    if counts[0] == 2 and counts[1] == 2:
        pairs = sorted([r for r, c in r_count.items() if c == 2], reverse=True)
        return 2_000_000 + pairs[0] * 1000 + pairs[1] * 10
    if counts[0] == 2:
        pair_rank = [r for r, c in r_count.items() if c == 2][0]
        return 1_000_000 + pair_rank * 10000
    return sum(r * (13 ** i) for i, r in enumerate(reversed(ranks)))

def _best_5_from_7(cards):
    from itertools import combinations
    return max(_hand_rank(list(combo)) for combo in combinations(cards, 5))

class HandEquityEstimator:
    def __init__(self, n_sim=200, cache_size=10_000):
        self.n_sim = n_sim          # maximális szimulációk száma
        self.cache_size = cache_size
        self._cache = {}

    def _cache_key(self, hole, board, num_opp):
        return f"{','.join(sorted(hole))}|{','.join(board)}|{num_opp}"

    def equity(self, hole_cards, board=None, num_opponents=1,
               confidence_threshold=0.02, min_sims=50):
        """
        Monte Carlo equity becslés adaptív early stopping-gal.

        Paraméterek:
            hole_cards:           saját lapok (pl. ['As', 'Kh'])
            board:                board lapok ([] ha preflop)
            num_opponents:        ellenfelek száma
            confidence_threshold: ha a rolling std < küszöb ÉS min_sims lefutott,
                                  early stop – felesleges szimulációk elkerülése
            min_sims:             legalább ennyi szimuláció fut (default: 50)

        Megjegyzés: self.n_sim MAXIMUM, nem fix szám.
                    A cache kulcs változatlan → visszafelé kompatibilis.
        """
        board = board or []
        key = self._cache_key(hole_cards, board, num_opponents)
        if key in self._cache:
            return self._cache[key]

        known    = set(hole_cards) | set(board)
        deck     = [c for c in FULL_DECK if c not in known]
        need     = 5 - len(board)
        sample_n = need + num_opponents * 2

        wins  = 0
        valid = 0   # érvényes szimulációk száma (sample_n <= deck)

        # Early stopping változók
        # Wilson-score helyett egyszerű rolling variance: utolsó 20 sim win/loss
        _window = []
        _WIN_WINDOW = 20  # ennyi utolsó eredményből számítjuk a stabilitást

        for sim_i in range(self.n_sim):
            if sample_n > len(deck):
                continue
            drawn   = random.sample(deck, sample_n)
            run_out = board + drawn[:need]
            my_rank = _best_5_from_7(hole_cards + run_out)
            win = True
            for o in range(num_opponents):
                opp_hole = drawn[need + o*2 : need + o*2 + 2]
                if _best_5_from_7(opp_hole + run_out) >= my_rank:
                    win = False
                    break
            wins  += int(win)
            valid += 1

            # Rolling window frissítése
            _window.append(int(win))
            if len(_window) > _WIN_WINDOW:
                _window.pop(0)

            # Early stopping: legalább min_sims lefutott ÉS a window stabil
            if valid >= min_sims and len(_window) == _WIN_WINDOW:
                w_mean = sum(_window) / _WIN_WINDOW
                # Variancia = p*(1-p)/N – ha p közel 0 vagy 1, hamar megáll
                std = (w_mean * (1.0 - w_mean) / _WIN_WINDOW) ** 0.5
                if std < confidence_threshold:
                    break

        result = wins / max(valid, 1)
        if len(self._cache) >= self.cache_size:
            del self._cache[next(iter(self._cache))]
        self._cache[key] = result
        return result
