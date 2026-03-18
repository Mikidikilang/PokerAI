"""
rta_example.py  –  RTAManager használati példa

Bemutatja:
  1. Inicializálás több modellel
  2. 6-max asztal indítása
  3. Kéz kezelés + döntés kérése
  4. Ellenfél akció rögzítése (username alapján)
  5. Asztalméret változás (pl. 6→4 fő) – modellváltás
  6. Ellenfél memória megmarad a modellváltás után
  7. Diagnosztika

Futtatás:
    python rta_example.py

MEGJEGYZÉS: valódi .pth fájlok nélkül a ModelPool betöltési hibát dob.
Ehhez a példához mock modell van beépítve – töröld a MockModelPool részt
ha éles checkpoint-okat használsz.
"""

import sys
import os
import numpy as np
import torch
import collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# Mock setup (éles kódban nem kell)
# A MockModelPool az RTAManager _pool tagját cseréli le, hogy
# valódi .pth fájlok nélkül is futtatható legyen a példa.
# ─────────────────────────────────────────────────────────────────────────────

from core.model import AdvancedPokerAI
from core.features import compute_state_size
from core.action_mapper import PokerActionMapper


def _make_mock_model(num_players: int, device: torch.device) -> tuple:
    """Véletlenszerűen inicializált modell teszteléshez."""
    # Obs méret becslése (production-ban temp env-ből mérik)
    obs_sizes = {2: 50, 3: 57, 4: 64, 5: 68, 6: 72, 7: 79, 8: 86, 9: 100}
    obs_size   = obs_sizes.get(num_players, 72)
    state_size = compute_state_size(obs_size, num_players)
    action_size = PokerActionMapper.NUM_CUSTOM_ACTIONS

    model = AdvancedPokerAI(state_size, action_size).to(device)
    model.eval()
    return model, state_size, action_size


class MockModelPool:
    """Valódi .pth fájlok nélkül működő model pool."""
    def __init__(self, device):
        self._device  = device
        self._models  = {}
        self._meta    = {}
        self._paths   = {2: 'mock', 4: 'mock', 6: 'mock', 9: 'mock'}

    def get(self, num_players):
        num_players = int(num_players)
        if num_players not in self._models:
            m, ss, as_ = _make_mock_model(num_players, self._device)
            self._models[num_players] = m
            self._meta[num_players]   = {'state_size': ss, 'action_size': as_}
            print(f"  [MockModelPool] {num_players}p modell létrehozva "
                  f"(state_size={ss})")
        return (self._models[num_players],
                self._meta[num_players]['state_size'],
                self._meta[num_players]['action_size'])

    def available_sizes(self):
        return sorted(self._paths.keys())

    def preload_all(self):
        for n in self._paths:
            self.get(n)


# ─────────────────────────────────────────────────────────────────────────────
# 1. RTAManager inicializálás
# ─────────────────────────────────────────────────────────────────────────────

from inference.rta_manager import RTAManager

print("=" * 60)
print("RTAManager – használati példa")
print("=" * 60)

# Éles kódban:
# manager = RTAManager({
#     2: '2max_ppo_v4.pth',
#     6: '6max_ppo_v4.pth',
#     9: '9max_ppo_v4.pth',
# }, device='cpu')

manager = RTAManager(
    model_paths={2: 'mock_2p.pth', 6: 'mock_6p.pth'},
    device='cpu',
    equity_sims=50,   # teszthez kevesebb szimuláció
)
# Mock pool cseréje (éles kódban nem kell!)
manager._pool = MockModelPool(torch.device('cpu'))

print("\n[1] RTAManager kész:", manager)

# ─────────────────────────────────────────────────────────────────────────────
# 2. 6-max asztal indítása
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("[2] 6-max asztal indítása")

seat_map_6p = {
    0: 'hero',
    1: 'fish99',
    2: 'reg42',
    3: 'nit_joe',
    4: 'aggro_anna',
    5: 'loose_bill',
}

manager.manage_table_change(
    num_players = 6,
    seat_map    = seat_map_6p,
    my_seat     = 0,
    button_seat = 5,
)

print("Asztali info:", manager.current_table_info())

# ─────────────────────────────────────────────────────────────────────────────
# 3. Első kéz – preflop döntés
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("[3] Első kéz – preflop")

manager.new_hand(
    my_stack   = 200.0,
    all_stacks = {
        'hero':       200.0,
        'fish99':     150.0,
        'reg42':      300.0,
        'nit_joe':    100.0,
        'aggro_anna': 250.0,
        'loose_bill': 180.0,
    },
    bb = 2.0,
    sb = 1.0,
)

# Ellenfél akciók rögzítése (preflop, előttünk léptek)
manager.record_opponent_action('fish99',     action=1, pot_size=3.0)   # call
manager.record_opponent_action('reg42',      action=4, bet_amount=8.0, pot_size=3.0)   # raise 50%
manager.record_opponent_action('nit_joe',    action=0, pot_size=11.0)  # fold
manager.record_opponent_action('aggro_anna', action=1, pot_size=11.0)  # call
manager.record_opponent_action('loose_bill', action=1, pot_size=11.0)  # call

# Obs vektor szimulálása (production-ban az rlcard adja)
obs_size_6p = 72
obs = np.random.rand(obs_size_6p).astype(np.float32)

result = manager.get_recommendation(
    obs_vector    = obs,
    legal_actions = [0, 1, 2, 3, 4, 5, 6],
    hole_cards    = ['As', 'Kh'],
    board_cards   = [],
    current_pot   = 19.0,
    call_amount   = 8.0,
)

print(f"\nAjánlás: {result['action_name']} ({result['confidence']*100:.0f}%)")
print(f"Equity:  {result['equity']*100:.0f}%")
print(f"SPR:     {result['spr']}")
print(f"Top3:    {result['top3']}")
print(f"Magyarázat: {result['explanation']}")
print(f"Ismert ellenfelek: {result['known_players']}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Saját akció + flop
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("[4] Saját akció + flop")

manager.record_my_action(action=1, bet_amount=8.0, pot_size=19.0)  # call
manager.new_street(1)  # flop

manager.record_opponent_action(
    'aggro_anna', action=4, bet_amount=15.0, pot_size=40.0,
    context={'is_cbet_opp': True}
)

obs_flop = np.random.rand(obs_size_6p).astype(np.float32)
result_flop = manager.get_recommendation(
    obs_vector    = obs_flop,
    legal_actions = [0, 1, 4, 5, 6],
    hole_cards    = ['As', 'Kh'],
    board_cards   = ['Ad', '7c', '2s'],
    current_pot   = 55.0,
    call_amount   = 15.0,
)

print(f"\nFlop ajánlás: {result_flop['action_name']} ({result_flop['confidence']*100:.0f}%)")
print(f"Equity top páron: {result_flop['equity']*100:.0f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Asztalméret változás: 6→4 fő (két játékos kiült)
#    A GlobalPlayerTracker NEM resetelődik – fish99 és reg42 statjai megmaradnak
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("[5] Asztalméret változás: 6→4 fő (modellváltás)")

# fish99 és loose_bill kiült
seat_map_4p = {
    0: 'hero',
    1: 'reg42',
    2: 'nit_joe',
    3: 'aggro_anna',
}

manager.manage_table_change(
    num_players = 4,
    seat_map    = seat_map_4p,
    my_seat     = 0,
    button_seat = 3,
)

print("Asztali info (4p):", manager.current_table_info())

# Ellenőrzés: az ismert ellenfelek száma változatlan
print(f"Ismert ellenfelek (4p modell után is): {len(manager._global_tracker)}")

# reg42 statjai megmaradtak
print("\nreg42 statjai (megmaradtak a modellváltás után):")
print(manager.player_stats('reg42'))

# ─────────────────────────────────────────────────────────────────────────────
# 6. Döntés a 4p modellel
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("[6] Döntés a 4p modellel")

manager.new_hand(
    my_stack   = 195.0,
    all_stacks = {
        'hero':       195.0,
        'reg42':      285.0,
        'nit_joe':    95.0,
        'aggro_anna': 240.0,
    },
    bb = 2.0,
    sb = 1.0,
)

obs_size_4p = 64
obs_4p = np.random.rand(obs_size_4p).astype(np.float32)

result_4p = manager.get_recommendation(
    obs_vector    = obs_4p,
    legal_actions = [0, 1, 3, 4, 6],
    hole_cards    = ['Qh', 'Jd'],
    board_cards   = [],
    current_pot   = 3.0,
    call_amount   = 2.0,
)

print(f"\n4p Ajánlás: {result_4p['action_name']} ({result_4p['confidence']*100:.0f}%)")
print(f"Ismert ellenfelek: {result_4p['known_players']}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Diagnosztika – összes ismert játékos statjai
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 60)
print("[7] Összes ismert játékos statjai")

for p in manager.all_player_stats():
    if 'error' not in p:
        print(f"  {p['username']:15s} | "
              f"VPIP={p['VPIP']:>4s} PFR={p['PFR']:>4s} "
              f"AF={p['AF']:>4s} 3bet={p['3bet%']:>4s} | "
              f"hands={p['hands_seen']}")

print("\n" + "=" * 60)
print("Teszt sikeres!")
