"""
rta_example.py  –  RTAManager v4 használati példa

Bemutatja:
  1. Inicializálás SQLite adatbázissal
  2. 6-max asztal indítása + batch preload
  3. Kéz kezelés + döntés kérése
  4. Ellenfél akció rögzítése (username alapján, SQLite-ba ment)
  5. Asztalméret változás 6→4 fő – modellváltás, DB megmarad
  6. Ellenfél memória megmarad a modellváltás után
  7. Diagnosztika + DB info
  8. JSON → SQLite migráció (ha régi adatbázisod van)

Futtatás:
    python rta_example.py

MEGJEGYZÉS: valódi .pth fájlok nélkül MockModelPool fut.
"""

import sys
import os
import tempfile
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# Mock setup (éles kódban nem kell – valódi .pth fájlokat tölt be)
# ─────────────────────────────────────────────────────────────────────────────

from core.model import AdvancedPokerAI
from core.features import compute_state_size
from core.action_mapper import PokerActionMapper


def _make_mock_model(num_players: int, device: torch.device) -> tuple:
    obs_size    = 54  # ObsBuilder.OBS_SIZE – rlcard nolimitholdem mindig 54
    state_size  = compute_state_size(obs_size, num_players)
    action_size = PokerActionMapper.NUM_CUSTOM_ACTIONS
    model = AdvancedPokerAI(state_size, action_size).to(device)
    model.eval()
    return model, state_size, action_size


class MockModelPool:
    def __init__(self, device):
        self._device  = device
        self._models  = {}
        self._meta    = {}
        self._paths   = {2: 'mock', 4: 'mock', 6: 'mock', 9: 'mock'}

    def get(self, num_players):
        n = int(num_players)
        if n not in self._models:
            m, ss, as_ = _make_mock_model(n, self._device)
            self._models[n] = m
            self._meta[n]   = {'state_size': ss, 'action_size': as_}
            print(f"  [MockModelPool] {n}p modell | state_size={ss}")
        return self._models[n], self._meta[n]['state_size'], self._meta[n]['action_size']

    def available_sizes(self): return sorted(self._paths.keys())
    def preload_all(self):
        for n in self._paths: self.get(n)


# ─────────────────────────────────────────────────────────────────────────────
# Példa futtatása
# ─────────────────────────────────────────────────────────────────────────────

from inference.rta_manager import RTAManager

print("=" * 65)
print("RTAManager v4 – SQLite ellenfél adatbázis példa")
print("=" * 65)

# Ideiglenes DB fájl a példához
db_fd, db_path = tempfile.mkstemp(suffix='.db')
os.close(db_fd)
os.unlink(db_path)

try:
    # ── 1. Inicializálás ──────────────────────────────────────────────────────
    print("\n[1] Inicializálás")

    manager = RTAManager(
        model_paths   = {2: 'mock_2p.pth', 4: 'mock_4p.pth', 6: 'mock_6p.pth'},
        db_path       = db_path,     # SQLite adatbázis – automatikusan létrejön
        device        = 'cpu',
        equity_sims   = 50,          # teszthez kevesebb
        tracker_memory= 1000,
    )
    manager._pool = MockModelPool(torch.device('cpu'))  # mock csere
    print(f"  RTAManager: {manager}")
    print(f"  DB info: {manager.db_info()}")

    # ── 2. 6-max asztal ───────────────────────────────────────────────────────
    print("\n[2] 6-max asztal indítása")

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
    print(f"  Asztal: {manager.current_table_info()['seat_order']}")

    # ── 3. Első kéz ───────────────────────────────────────────────────────────
    print("\n[3] Első kéz – preflop")

    manager.new_hand(
        my_stack   = 200.0,
        all_stacks = {
            'hero': 200.0, 'fish99': 150.0, 'reg42': 300.0,
            'nit_joe': 100.0, 'aggro_anna': 250.0, 'loose_bill': 180.0,
        },
        bb=2.0, sb=1.0,
    )

    # Ellenfél akciók (preflop, előttünk léptek)
    manager.record_opponent_action('fish99',     action=1, pot_size=3.0)
    manager.record_opponent_action('reg42',      action=4, bet_amount=8.0, pot_size=3.0)
    manager.record_opponent_action('nit_joe',    action=0, pot_size=11.0)
    manager.record_opponent_action('aggro_anna', action=1, pot_size=11.0)
    manager.record_opponent_action('loose_bill', action=1, pot_size=11.0)

    result = manager.get_recommendation(
        legal_actions = [0,1,2,3,4,5,6],
        hole_cards    = ['As','Kh'],
        board_cards   = [],
        current_pot   = 19.0,
        call_amount   = 8.0,
    )

    print(f"  Ajánlás: {result['action_name']} ({result['confidence']*100:.0f}%)")
    print(f"  Equity:  {result['equity']*100:.0f}%  SPR: {result['spr']}")
    print(f"  Top3:    {result['top3']}")
    print(f"  DB játékosok: {result['db_players']}")

    manager.record_my_action(action=result['action'], bet_amount=8.0, pot_size=19.0)

    # ── 4. Flop ───────────────────────────────────────────────────────────────
    print("\n[4] Flop")
    manager.new_street(1)
    manager.record_opponent_action(
        'aggro_anna', action=4, bet_amount=15.0, pot_size=40.0,
        context={'is_cbet_opp': True}
    )
    result_flop = manager.get_recommendation(
        legal_actions = [0,1,4,5,6],
        hole_cards    = ['As','Kh'],
        board_cards   = ['Ad','7c','2s'],
        current_pot   = 55.0,
        call_amount   = 15.0,
    )
    print(f"  Flop ajánlás: {result_flop['action_name']} "
          f"({result_flop['confidence']*100:.0f}%) | "
          f"Equity: {result_flop['equity']*100:.0f}%")

    # ── 5. Asztalméret változás: 6→4 fő ──────────────────────────────────────
    print("\n[5] Asztalméret változás: 6→4 fő (fish99 és loose_bill kiült)")

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

    print(f"  4p modell aktív | DB: {manager.db_info()['total_players']} játékos")

    # ── 6. Ellenfél memória megmarad ──────────────────────────────────────────
    print("\n[6] Ellenfél memória ellenőrzés")

    # aggro_anna statjai megmaradtak (session közben rögzítettük)
    s = manager.player_stats('aggro_anna')
    print(f"  aggro_anna: VPIP={s['VPIP']} PFR={s['PFR']} "
          f"rolling={s['rolling_hands']} lifetime={s['lifetime_hands']}")

    # reg42 statjai is megvannak
    s2 = manager.player_stats('reg42')
    print(f"  reg42:      VPIP={s2['VPIP']} PFR={s2['PFR']} "
          f"rolling={s2['rolling_hands']} lifetime={s2['lifetime_hands']}")

    # ── 7. Második kéz a 4p modellel ─────────────────────────────────────────
    print("\n[7] Második kéz – 4p modell")

    manager.new_hand(
        my_stack   = 195.0,
        all_stacks = {
            'hero': 195.0, 'reg42': 285.0,
            'nit_joe': 95.0, 'aggro_anna': 240.0,
        },
        bb=2.0, sb=1.0,
    )

    result_4p = manager.get_recommendation(
        legal_actions = [0,1,3,4,6],
        hole_cards    = ['Qh','Jd'],
        board_cards   = [],
        current_pot   = 3.0,
        call_amount   = 2.0,
    )
    print(f"  Ajánlás: {result_4p['action_name']} "
          f"({result_4p['confidence']*100:.0f}%)")

    # ── 8. DB diagnosztika ────────────────────────────────────────────────────
    print("\n[8] Adatbázis diagnosztika")
    db_info = manager.db_info()
    print(f"  Mód:          {db_info['mode']}")
    print(f"  DB fájl:      {db_info['db_path']}")
    print(f"  Összes ismert játékos: {db_info['total_players']}")
    print(f"  Cache (aktív): {db_info['cached']}")
    print(f"  Dirty (nem flush-olt): {db_info['dirty']}")
    print(f"  DB méret:     {db_info['db_size_mb']*1024:.1f} KB")

    print("\n[9] Top játékosok lifetime kézszám szerint:")
    for p in manager.top_players(n=5):
        print(f"  {p['username']:15s} | {p['lifetime_hands']:4d} kéz | "
              f"VPIP={p['lt_VPIP']} PFR={p['lt_PFR']}")

    # Context manager nélkül: manuális flush
    manager.flush_tracker()
    print(f"\n  Flush után DB méret: {manager.db_info()['db_size_mb']*1024:.1f} KB")

    # ── 9. JSON → SQLite migráció demo ───────────────────────────────────────
    print("\n[10] JSON → SQLite migráció (ha régi adatbázisod van):")
    print("  from core.opponent_tracker import GlobalPlayerTracker")
    print("  tracker = GlobalPlayerTracker.migrate_from_json(")
    print("      'players.json', 'players.db'")
    print("  )")
    print("  # Ezután csak a players.db fájlt add meg db_path-nak")

    print("\n" + "=" * 65)
    print("Teszt sikeres!")

finally:
    # Takarítás
    for ext in ['', '-wal', '-shm']:
        f = db_path + ext
        if os.path.exists(f):
            os.unlink(f)
