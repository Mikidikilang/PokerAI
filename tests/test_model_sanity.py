#!/usr/bin/env python3
"""
test_model_sanity.py  v3.3 SPRINT3  –  Poker AI Vizsgáztató Központ

SPRINT 1 VÁLTOZÁSOK (2026-03-21):
  FIX-1: make_scenarios – 72o facing raise elvárások javítása
          expected_good: {0,1} → {0}  (call is -EV HU-ban)
          expected_bad:  {5,6} → {2,3,4,5,6}  (minden raise rossz)
          severity:      WARNING → CRITICAL
  FIX-2: run_position_test – pozíciós teszt izolálása
          A BTN és BB szituációk most azonos pénzügyi kontextust
          használnak (my_chips, all_chips, pot, call_amount egyenlő),
          csak a button_pos és my_player_id változik.
          Ez kiszűri a pot-odds / call-amount konfundáló hatását,
          és tisztán a pozíció-enkódolás hatását méri.

SPRINT 2 VÁLTOZÁSOK (2026-03-21):
  NEW-1: run_value_calibration_test() – Value head alapkalibráció
          A query_model() már visszaadta a value becsléstszám minden
          szituációnál, de soha nem volt validálva. Ez az ÚJ teszt
          elméleti határokon ellenőrzi a value head kimeneteit:
            AA preflop:    value > 0  (pozitív EV)
            72o vs raise:  value < 0  (negatív EV)
            Nut flush:     value > 0  (erős kéz, pozitív EV)
            Top set:       value > 0
            Air river bet: value < 0.5 (nem profitábilis call)
          Az rlcard BB-egységes value skálán dolgozik, ahol a semleges
          körülbelül 0. A tesztek nem igényelnek rlcard-ot.
  NEW-2: run_exploit_test() bővítése
          Régi: 1 spot Nit ellen, 1 spot Maniac ellen, 1 adaptáció teszt
          Új:   3 spot Nit ellen (blöff frekvencia mérés több boardon),
                2 spot Maniac ellen (trap, check-raise),
                1 spot Calling Station ellen (value bet frekvencia),
                szigorúbb adaptáció delta küszöb: ≥10% (volt: +5%)
                összesített exploit score: X/7
  NEW-3: Grade formula felülvizsgálata
          Régi: csak failed count + degeneration → grade
          Új:   penalty pont rendszer – minden súlyos hiányosság +1 penalty:
            +1 ha situational failed > 0  (CRITICAL hibák)
            +1 ha draw fold% > 30%
            +1 ha avg entropy > 1.3
            +1 ha pozíció-tudatos < 2/5  (pozíció-vak modell)
            +1 ha exploit score < 3/7
            +1 ha value kalibráció < 3/5
            +1 per degeneration flag
          Grade skála:
            0 penalty: 🟢 JÓ
            1 penalty: 🟡 ELFOGADHATÓ
            2 penalty: 🟠 PROBLÉMÁS
            3+ penalty: 🔴 KOMOLY HIBÁK

SPRINT 3 VÁLTOZÁSOK (2026-03-21):
  NEW-4: Bootstrap konfidencia-intervallum a winrate teszthez
          A run_winrate_test() eddig csak pontbecslést adott (BB/100).
          2000 kézből a véletlen ingadozás ±10-15 BB/100 is lehet, ezért
          egy pontbecslés statisztikailag nem megbízható.
          Új: _bootstrap_ci(hand_payoffs, bb, n_boot=1000) függvény,
          amely 1000 bootstrap újramintavételezéssel 95%-os CI-t számol.
          Kimenet: "BB/100: +42.3 [95% CI: +28.1, +56.4] ✅ szignifikáns"
          A play_hands() most per-kéz payoff listát is visszaad.
          Grade nincs változtatva (winrate opcionális), de ha fut:
            ha CI alsó határa > 0 → statisztikailag szignifikáns nyerő
  NEW-5: ScenarioGenerator osztály + run_scenario_generator_test()
          Programmatikus szituáció-generátor 3 dimenzióban:
          (a) Stack depth sweep: top pair kéz 4 stack mélységben
              (10BB / 20BB / 50BB / 100BB) – felismeri-e a push/fold zónát?
          (b) Board texture sweep: KcJd top pair 4 különböző textúrán
              (dry / two-tone / monotone / paired) – csökkenti-e a bet%-ot
              veszélyesebb textúrán?
          (c) River-specifikus szituációk (4 spot): value bet, bluff-catch,
              missed flush, nut river – célzottan a 20M modell ismert
              ⚠ gyengeségét teszteli ('Air facing river pot bet')
          Mindhárom dimenzió informális (nem penalty), de logolva van.
          A run_single_model összefoglalójában megjelenik.

TESZTEK:
  1.  Szituációs (14 eset: preflop, postflop, 3bet, short stack, board texture)
  2.  Pozíciós tudatosság (BTN vs BB, 5 kéz) [SPRINT1 JAVÍTVA]
  3.  HUD exploitáció (7 spot: nit×3, maniac×2, calling station, adaptáció) [SPRINT2]
  4.  Draw equity awareness (8 szituáció, out-kezelés)
  5.  Bet sizing analízis (Spearman korreláció kézerő vs raise méret)
  6.  Konzisztencia / entropy (döntési magabiztosság)
  7.  Value head kalibráció (5+1 szituáció, elméleti határok) [SPRINT2]
  8.  Scenario Generator (stack depth / board texture / river) [SPRINT3 ÚJ]
  9.  Poker statok (VPIP, PFR, AF, 3-bet%, C-bet%)
  10. BB/100 winrate + bootstrap 95% CI [SPRINT3, --winrate]
  11. Modell összehasonlítás [--compare A.pth B.pth ...]

Használat:
    python test_model_sanity.py 2max_ppo_v4.pth
    python test_model_sanity.py 2max_ppo_v4.pth --hands 5000 --verbose
    python test_model_sanity.py 2max_ppo_v4.pth --winrate --winrate-hands 5000
    python test_model_sanity.py 2max_ppo_v4.pth --out-dir ModellNaplo/2max_ppo_v4_4M
    python test_model_sanity.py --compare 2max_5M.pth 2max_10M.pth 2max_20M.pth
"""

import sys, os, argparse, collections, random, json, math, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np
from core.model import AdvancedPokerAI
from core.action_mapper import PokerActionMapper
from core.features import (ActionHistoryEncoder, build_state_tensor,
                            compute_state_size, ACTION_HISTORY_LEN)
from core.opponent_tracker import OpponentHUDTracker
from core.equity import HandEquityEstimator
from inference.obs_builder import ObsBuilder
from utils.checkpoint_utils import safe_load_checkpoint

RANKS = 'AKQJT98765432'
SUITS = 'shdc'
ALL_CARDS = [r + s for r in RANKS for s in SUITS]
# ═══════════════════════════════════════════════════════════════════════════════
# [TASK-16] PER-PLAYER-COUNT KONFIGURÁCIÓK
# ─────────────────────────────────────────────────────────────────────────────
# Minden asztalméretre (2..9) külön kalibrált stat-targetok és scenario
# elvárások.  Az összes player-count-specifikus érték itt van definiálva,
# nem szétszórva a függvénytörzsekben.
# ═══════════════════════════════════════════════════════════════════════════════

def _table_category(np_: int) -> str:
    """Visszaad egy olvasható kategória-nevet az asztalméretre."""
    if np_ == 2:           return "HU"
    elif np_ == 3:         return "3-max"
    elif np_ in (4, 5):    return f"{np_}-max (short-handed)"
    elif np_ == 6:         return "6-max"
    else:                  return f"{np_}-max (full ring)"


# ── VPIP / PFR / AF / 3-bet / C-bet célsávok ─────────────────────────────────
# Forrás: PokerTracker / GTO-solver irodalom + élő mikro-stakes minták.
# Formátum: (lo, hi) zárt intervallum.  Ha a modell ezen kívülre esik: ⚠
#
# | cat       | VPIP      | PFR       | AF        | 3-bet     | C-bet     |
# |-----------|-----------|-----------|-----------|-----------|-----------|
# | HU        | 55–90     | 35–75     | 1.5–5.0   | 25–55     | 50–85     |
# | 3-max     | 42–68     | 30–55     | 1.5–4.5   | 12–28     | 55–82     |
# | 4-5-max   | 28–48     | 20–40     | 1.5–4.0   | 8–20      | 55–80     |
# | 6-max     | 22–38     | 16–30     | 1.5–3.5   | 6–15      | 55–78     |
# | 7-9 ring  | 14–28     | 10–22     | 1.3–3.5   | 4–12      | 55–78     |

TABLE_STAT_TARGETS = {
    #           vpip       pfr        af         tbet       cbet
    2:  dict(vpip=(55,90), pfr=(35,75), af=(1.5,5.0), tbet=(25,55), cbet=(50,85)),
    3:  dict(vpip=(42,68), pfr=(30,55), af=(1.5,4.5), tbet=(12,28), cbet=(55,82)),
    4:  dict(vpip=(28,48), pfr=(20,40), af=(1.5,4.0), tbet=( 8,20), cbet=(55,80)),
    5:  dict(vpip=(28,48), pfr=(20,40), af=(1.5,4.0), tbet=( 8,20), cbet=(55,80)),
    6:  dict(vpip=(22,38), pfr=(16,30), af=(1.5,3.5), tbet=( 6,15), cbet=(55,78)),
    7:  dict(vpip=(14,28), pfr=(10,22), af=(1.3,3.5), tbet=( 4,12), cbet=(55,78)),
    8:  dict(vpip=(14,28), pfr=(10,22), af=(1.3,3.5), tbet=( 4,12), cbet=(55,78)),
    9:  dict(vpip=(14,28), pfr=(10,22), af=(1.3,3.5), tbet=( 4,12), cbet=(55,78)),
}

def _stat_ok(val, lo, hi):
    return lo <= val <= hi




# ═══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogger:
    def __init__(self, model_path, num_players, n_hands, out_dir="logs"):
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        self.log_path = os.path.join(
            out_dir, f"test_{model_name}_{num_players}p_{n_hands}h_{ts}.log"
        )
        self._file = open(self.log_path, 'w', encoding='utf-8')
        self.results = {'timestamp': ts, 'model': model_path,
                        'num_players': num_players, 'n_hands': n_hands,
                        'scenarios': [], 'summary': {}}

    def log(self, text, console=True):
        self._file.write(text + '\n')
        if console: print(text)

    def close(self):
        json_path = self.log_path.replace('.log', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        self._file.close()
        return self.log_path, json_path


# ═══════════════════════════════════════════════════════════════════════════════
# SEGÉD: Spearman korreláció (scipy nélkül)
# ═══════════════════════════════════════════════════════════════════════════════

def _rank_data(data):
    indexed = sorted(enumerate(data), key=lambda x: x[1])
    ranks = [0.0] * len(data)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j+1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks

def _spearman(x, y):
    if len(x) < 3: return 0.0
    rx, ry = _rank_data(x), _rank_data(y)
    n = len(x)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    denom = n * (n * n - 1)
    return 1.0 - (6.0 * d_sq) / denom if denom else 0.0

def _entropy(probs):
    return -sum(p * math.log(p + 1e-10) for p in probs if p > 0)


def _bootstrap_ci(hand_payoffs, bb, n_boot=1000, alpha=0.05):
    """
    Bootstrap 95% konfidencia-intervallum a BB/100 winrate-hez.

    SPRINT 3 – NEW-4: Statisztikai robosztusság.

    Motiváció: 2000 kézből a pontbecslés (BB/100) akár ±10-15 BB/100
    véletlen szórást hordoz. Egy +20 BB/100 pontbecslés a CI alapján
    lehet [+5, +35] – nyerő –, vagy [-5, +45] – nem szignifikáns.
    Csak a CI alsó határa dönti el, valóban nyerő-e a modell.

    Algoritmus: n_boot=1000 bootstrap resample, mindegyikből BB/100,
    majd a [alpha/2, 1-alpha/2] percentilisek a CI határok.
    Reprodukálható: rng seed=42.

    Paraméterek:
      hand_payoffs: list[float] – egyes kezek chip nyereménye
      bb:           float – big blind mérete (skálázáshoz)
      n_boot:       int   – bootstrap iterációk száma (default: 1000)
      alpha:        float – szignifikancia szint (default: 0.05 → 95% CI)

    Visszatér: (mean_bb100, ci_low, ci_high)
      Ha kevés az adat (<30 kéz): (None, None, None)
    """
    n = len(hand_payoffs)
    if n < 30:
        return None, None, None
    arr = np.array(hand_payoffs, dtype=np.float64)
    rng = np.random.default_rng(seed=42)
    # Vektorizált bootstrap: (n_boot, n) mintavétel egyszerre
    indices  = rng.integers(0, n, size=(n_boot, n))
    boot_bb100 = arr[indices].mean(axis=1) / bb * 100
    boot_bb100.sort()
    lo_i = max(0, int(n_boot * alpha / 2))
    hi_i = min(n_boot - 1, int(n_boot * (1.0 - alpha / 2)))
    return float(arr.mean() / bb * 100), float(boot_bb100[lo_i]), float(boot_bb100[hi_i])


# ═══════════════════════════════════════════════════════════════════════════════
# MODELL LEKÉRDEZÉS
# ═══════════════════════════════════════════════════════════════════════════════

def query_model(model, obs_builder, tracker, history_encoder, scenario,
                num_players, device, equity_est=None, action_history=None):
    hole = scenario['hole_cards']
    board = scenario.get('board_cards', [])
    obs = obs_builder.build(hole_cards=hole, board_cards=board,
                            my_chips=scenario.get('my_chips', 2.0),
                            all_chips=scenario.get('all_chips', [2.0]*num_players))
    state = {
        'obs': obs,
        'raw_obs': {
            'my_chips': scenario.get('my_chips', 2.0),
            'all_chips': scenario.get('all_chips', [2.0]*num_players),
            'pot': scenario.get('pot', 3.0),
            'public_cards': board, 'hand': hole,
            'button': scenario.get('button_pos', 0),
            'call_amount': scenario.get('call_amount', 0.0),
        },
    }
    if equity_est is None: equity_est = HandEquityEstimator(n_sim=200)
    try: equity = equity_est.equity(hole, board, num_opponents=max(num_players-1, 1))
    except: equity = 0.5
    if action_history is None:
        action_history = collections.deque(maxlen=ACTION_HISTORY_LEN)
    state_t = build_state_tensor(
        state, tracker, action_history, history_encoder, num_players,
        my_player_id=scenario.get('my_player_id', 0),
        bb=scenario.get('bb', 2.0), sb=scenario.get('sb', 1.0),
        initial_stack=scenario.get('stack', 100.0),
        street=scenario.get('street', 0), equity=equity)
    with torch.no_grad():
        probs_t, value, _ = model.forward(state_t.to(device),
                                           scenario.get('legal_actions', [0,1,2,3,4,5,6]))
    probs = probs_t.squeeze(0).cpu().numpy()
    return probs, int(np.argmax(probs)), float(value.squeeze().cpu()), equity


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SZITUÁCIÓS TESZTEK
# ═══════════════════════════════════════════════════════════════════════════════

def make_scenarios(np_):
    """
    Szituációs tesztek – [TASK-16] per-player-count elvárásokkal.

    Struktúra:
      A) UNIVERZÁLIS (minden asztalméretnél azonos elvárások) – 10 eset
         Főleg postflop és alapvető preflop döntések, ahol a játékosszám
         nem változtatja meg a helyes választ.

      B) PREFLOP – player-count-adjusted (4 eset)
         T8o facing 4-bet, 8BB 22 push/fold, BB preflop steal,
         QQ facing 3-bet:  más elvárás HU / 6-max / full ring esetén.

      C) PLAYER-COUNT-SPECIFIKUS (3-4 eset per kategória)
         HU:       BTN open széles range, 3-bet AJo, BB steal defense
         Short:    squeeze ATs, CO cold call fold, UTG range
         6-max:    BTN open T9s, UTG fold K2o, BB vs BTN defend
         Full ring: UTG fold 87s, EP fold vs UTG raise, UTG open QQ

    Severity:
      CRITICAL → failed esetén grade penalty +1
      WARNING  → csak log-olva, nem büntetett
    """
    bb=2.0; sb=1.0; stk=100.0; S=[]

    def sc(**kw):
        kw.setdefault('bb', bb)
        kw.setdefault('sb', sb)
        kw.setdefault('stack', stk)
        kw.setdefault('legal_actions', [0,1,2,3,4,5,6])
        kw.setdefault('all_chips', [kw.get('my_chips', 2.0)] * np_)
        S.append(kw)

    # ── A) UNIVERZÁLIS SZITUÁCIÓK ────────────────────────────────────────────
    # Ezek minden asztalméretnél ugyanolyan döntést igényelnek.

    sc(name='AA preflop – raise',
       hole_cards=['As','Ah'], board_cards=[], street=0,
       my_chips=2.0, pot=3.0, call_amount=0.0,
       expected_good={2,3,4,5,6}, expected_bad={0},
       severity='CRITICAL', category='preflop',
       description='AA préflop: raise mindig helyes (VPIP + PFR).')

    sc(name='KK vs open – call/raise',
       hole_cards=['Ks','Kh'], board_cards=[], street=0,
       my_chips=2.0, all_chips=[6.0,2.0]+[0.0]*(np_-2),
       pot=9.0, call_amount=4.0,
       expected_good={1,2,3,4,5,6}, expected_bad={0},
       severity='CRITICAL', category='preflop',
       description='KK vs open: soha nem fold, call vagy re-raise.')

    sc(name='AKs preflop – call/raise',
       hole_cards=['As','Ks'], board_cards=[], street=0,
       my_chips=1.0, all_chips=[1.0,2.0]+[0.0]*(np_-2),
       pot=3.0, call_amount=1.0,
       expected_good={1,2,3,4,5,6}, expected_bad={0},
       severity='CRITICAL', category='preflop',
       description='AKs: top 3% kéz, call vagy raise – fold nem opció.')

    sc(name='72o facing raise – fold',
       hole_cards=['7s','2h'], board_cards=[], street=0,
       my_chips=2.0, all_chips=[6.0,2.0]+[0.0]*(np_-2),
       pot=9.0, call_amount=4.0,
       expected_good={0}, expected_bad={2,3,4,5,6},
       severity='CRITICAL', category='preflop',
       description='72o vs raise: fold MINDEN asztalméretnél (pot odds ~30%, equity ≤38%).')

    sc(name='Nut flush flop – bet',
       hole_cards=['As','Ks'], board_cards=['Qs','7s','3s'],
       street=1, my_chips=10.0,
       all_chips=[10.0,10.0]+[0.0]*(np_-2),
       pot=20.0, call_amount=0.0, stack=90.0,
       expected_good={2,3,4,5,6}, expected_bad={0},
       severity='CRITICAL', category='postflop',
       description='Nut flush: bet/raise, ne slow play multiway sem.')

    sc(name='Top set dry flop – bet',
       hole_cards=['Ah','Ad'], board_cards=['Ac','7d','2s'],
       street=1, my_chips=8.0,
       all_chips=[8.0,8.0]+[0.0]*(np_-2),
       pot=16.0, call_amount=0.0, stack=92.0,
       expected_good={1,2,3,4,5,6}, expected_bad={0},
       severity='CRITICAL', category='postflop',
       description='Top set: bet/raise vagy check (slow play); fold soha.')

    sc(name='Combo draw (15 out) flop – semi-bluff',
       hole_cards=['Jh','Th'], board_cards=['9h','8d','2h'],
       street=1, my_chips=6.0,
       all_chips=[12.0,6.0]+[0.0]*(np_-2),
       pot=18.0, call_amount=6.0, stack=94.0,
       expected_good={1,4,5,6}, expected_bad={0},
       severity='CRITICAL', category='semi_bluff',
       description='15 out combo draw: call vagy raise; fold -EV (33% equity).')

    sc(name='TPTK monotone board – passive',
       hole_cards=['As','Kc'], board_cards=['Jh','9h','8h'],
       street=1, my_chips=10.0,
       all_chips=[30.0,10.0]+[0.0]*(np_-2),
       pot=40.0, call_amount=20.0, stack=90.0,
       expected_good={0,1}, expected_bad={5,6},
       severity='WARNING', category='board_texture',
       description='TPTK 3-flush boardon óriási bet ellen: fold/call OK, raise HIBA.')

    sc(name='20:1 pot odds – call',
       hole_cards=['5h','4d'], board_cards=['As','Ks','Qd','7c'],
       street=2, my_chips=20.0,
       all_chips=[22.0,20.0]+[0.0]*(np_-2),
       pot=40.0, call_amount=2.0, stack=80.0,
       expected_good={1,2,3,4,5,6}, expected_bad={0},
       severity='WARNING', category='pot_odds',
       description='20:1 pot odds: matematikailag call (szinte bármilyen kézzel).')

    sc(name='Air facing river pot bet – fold',
       hole_cards=['9h','8h'], board_cards=['As','Kd','3c','5s','Jd'],
       street=3, my_chips=30.0,
       all_chips=[60.0,30.0]+[0.0]*(np_-2),
       pot=60.0, call_amount=30.0, stack=70.0,
       expected_good={0}, expected_bad={5,6},
       severity='WARNING', category='pot_odds',
       description='Air a riveren pot-bet ellen: fold (9 high, ~8% equity).')

    # ── B) PREFLOP – player-count-adjusted ──────────────────────────────────
    # Ezek elvárásai np_-től függően változnak.

    # T8o facing 4-bet:
    #   HU:      pot odds ~40% vs ~38% equity → marginal, call/fold ok
    #   3-5 max: fold inkább (range-disadvantage)
    #   6+ max:  fold CRITICAL (cold 4-bet range megköveteli)
    if np_ == 2:
        t8o_good = {0, 1};  t8o_bad = {5, 6};  t8o_sev = 'WARNING'
        t8o_desc = 'HU: T8o vs 4-bet – pot odds marginális call; raise HIBA.'
    elif np_ <= 5:
        t8o_good = {0};     t8o_bad = {4,5,6};  t8o_sev = 'WARNING'
        t8o_desc = f'{np_}p: T8o vs 4-bet – fold preferált (range disadvantage).'
    else:
        t8o_good = {0};     t8o_bad = {3,4,5,6}; t8o_sev = 'CRITICAL'
        t8o_desc = f'{np_}p: T8o vs 4-bet – fold KÖTELEZŐ (cold 4-bet range).'
    sc(name='T8o facing 4-bet',
       hole_cards=['Th','8d'], board_cards=[], street=0,
       my_chips=18.0, all_chips=[45.0,18.0]+[0.0]*(np_-2),
       pot=64.0, call_amount=27.0,
       expected_good=t8o_good, expected_bad=t8o_bad,
       severity=t8o_sev, category='3bet', description=t8o_desc)

    # QQ facing 3-bet:
    #   HU/short: call vagy 4-bet
    #   Full ring vs EP 3-bet: QQ marginal, fold/call/raise mind ok
    if np_ <= 5:
        qq_good = {1,4,5,6}; qq_bad = {0}; qq_sev = 'CRITICAL'
    else:
        qq_good = {0,1,4,5,6}; qq_bad = {2,3}; qq_sev = 'WARNING'
    sc(name='QQ facing 3-bet',
       hole_cards=['Qs','Qh'], board_cards=[], street=0,
       my_chips=6.0, all_chips=[18.0,6.0]+[0.0]*(np_-2),
       pot=25.0, call_amount=12.0,
       expected_good=qq_good, expected_bad=qq_bad,
       severity=qq_sev, category='3bet',
       description=f'{np_}p QQ vs 3-bet: {"call/4-bet (fold HIBA)" if np_<=5 else "fold/call/4-bet (min-raise HIBA)"}')

    # 10BB A8o push/fold:
    #   Universal: push inkább, de severity és description különbözik
    if np_ == 2:
        a8o_good = {5,6};  a8o_bad = {0};  a8o_sev = 'WARNING'
        a8o_desc = 'HU 10BB A8o BTN: push (top 30% HU push range).'
    elif np_ <= 5:
        a8o_good = {5,6};  a8o_bad = {0};  a8o_sev = 'WARNING'
        a8o_desc = f'{np_}p 10BB A8o BTN: push (BTN/CO pozícióban jó shove).'
    else:
        a8o_good = {0,5,6}; a8o_bad = {2,3}; a8o_sev = 'WARNING'
        a8o_desc = f'{np_}p 10BB A8o BTN: push/fold; min-raise HIBA (bloating).'
    sc(name='10BB A8o BTN push/fold',
       hole_cards=['Ah','8d'], board_cards=[], street=0,
       my_chips=1.0, all_chips=[1.0,2.0]+[0.0]*(np_-2),
       pot=3.0, call_amount=1.0, stack=20.0,
       button_pos=0, my_player_id=0,
       expected_good=a8o_good, expected_bad=a8o_bad,
       severity=a8o_sev, category='short_stack', description=a8o_desc)

    # 8BB 22 push/fold:
    #   HU/BTN short: push
    #   Full ring UTG: 22 is tighter, fold/push both ok
    if np_ <= 5:
        p22_good = {5,6};    p22_bad = {2,3}; p22_sev = 'WARNING'
    else:
        p22_good = {0,5,6};  p22_bad = {2,3}; p22_sev = 'WARNING'
    sc(name='8BB 22 BTN push/fold',
       hole_cards=['2s','2h'], board_cards=[], street=0,
       my_chips=1.0, all_chips=[1.0,2.0]+[0.0]*(np_-2),
       pot=3.0, call_amount=1.0, stack=16.0,
       button_pos=0, my_player_id=0,
       expected_good=p22_good, expected_bad=p22_bad,
       severity=p22_sev, category='short_stack',
       description=f'{np_}p 8BB 22 BTN: {"push (2:1 chip EV)" if np_<=5 else "fold/push ok (full ring tight)"}.')

    # ── C) PLAYER-COUNT-SPECIFIKUS SZITUÁCIÓK ───────────────────────────────

    if np_ == 2:
        # ── HU-SPECIFIKUS ──
        # BTN open 83s: HU-ban a button 70%+ range-t nyit
        sc(name='[HU] BTN open 83s',
           hole_cards=['8h','3h'], board_cards=[], street=0,
           my_chips=1.0, all_chips=[1.0,2.0]+[0.0]*(np_-2),
           pot=3.0, call_amount=1.0, stack=100.0,
           button_pos=0, my_player_id=0,
           expected_good={2,3,4,5,6}, expected_bad={0},
           severity='WARNING', category='hu_specific',
           description='HU BTN: 83s standard nyitás (HU push/fold top ~70%).')

        # 3-bet AJo vs BTN open: HU-ban az AJo erős 3-bet kéz
        sc(name='[HU] 3-bet AJo vs BTN',
           hole_cards=['Ac','Jd'], board_cards=[], street=0,
           my_chips=2.0, all_chips=[6.0,2.0]+[0.0]*(np_-2),
           pot=9.0, call_amount=4.0,
           expected_good={1,4,5,6}, expected_bad={0},
           severity='WARNING', category='hu_specific',
           description='HU BB: AJo vs BTN open – 3-bet vagy call; fold HIBA.')

        # BB K2o vs min-open: HU BB defends very wide
        sc(name='[HU] BB defend K2o vs min-open',
           hole_cards=['Kd','2c'], board_cards=[], street=0,
           my_chips=2.0, all_chips=[4.0,2.0]+[0.0]*(np_-2),
           pot=6.0, call_amount=2.0,
           expected_good={1,2,3}, expected_bad={0},
           severity='WARNING', category='hu_specific',
           description='HU BB: K2o vs 2BB min-open – defend (pot odds 33%, equity >35%).')

    elif np_ <= 5:
        # ── SHORT-HANDED (3-5max) SPECIFIKUS ──
        # Squeeze ATs BB: BTN open + CO call → squeeze vagy fold, ne call
        sc(name=f'[{np_}p] Squeeze ATs BB',
           hole_cards=['Ah','Th'], board_cards=[], street=0,
           my_chips=2.0, all_chips=[6.0,4.0,2.0]+[0.0]*(np_-3),
           pot=12.0, call_amount=4.0,
           expected_good={0,5,6}, expected_bad={2,3},
           severity='WARNING', category='short_specific',
           description=f'{np_}p BB ATs vs BTN open + call: squeeze (5-6x) vagy fold; cold call -EV.')

        # CO cold call T9s vs BTN open: short-handed-ben fold preferált
        sc(name=f'[{np_}p] CO fold T9s vs BTN',
           hole_cards=['Td','9d'], board_cards=[], street=0,
           my_chips=2.0, all_chips=[6.0,2.0]+[0.0]*(np_-2),
           pot=9.0, call_amount=4.0,
           expected_good={0,5,6}, expected_bad={},
           severity='WARNING', category='short_specific',
           description=f'{np_}p CO T9s vs BTN 3BB open: fold vagy squeeze (cold call marginal).')

        # UTG open QQ: nyitni kell
        sc(name=f'[{np_}p] UTG open QQ',
           hole_cards=['Qs','Qh'], board_cards=[], street=0,
           my_chips=1.0, all_chips=[1.0,2.0]+[0.0]*(np_-2),
           pot=3.0, call_amount=1.0, stack=100.0,
           button_pos=np_-1, my_player_id=0,
           expected_good={2,3,4,5,6}, expected_bad={0},
           severity='CRITICAL', category='short_specific',
           description=f'{np_}p UTG QQ: raise mindig (fold CRITICAL HIBA).')

    elif np_ == 6:
        # ── 6-MAX SPECIFIKUS ──
        # BTN open T9s: 6-max BTN ~55% open range, T9s benne van
        sc(name='[6max] BTN open T9s',
           hole_cards=['Ts','9s'], board_cards=[], street=0,
           my_chips=1.0, all_chips=[1.0,2.0]+[0.0]*(np_-2),
           pot=3.0, call_amount=1.0, stack=100.0,
           button_pos=0, my_player_id=0,
           expected_good={2,3,4,5,6}, expected_bad={0},
           severity='WARNING', category='6max_specific',
           description='6-max BTN T9s: standard nyitás (BTN ~55% open range).')

        # UTG fold K2o: 6-max UTG range ~16%, K2o kívül esik
        sc(name='[6max] UTG fold K2o',
           hole_cards=['Kd','2c'], board_cards=[], street=0,
           my_chips=1.0, all_chips=[1.0,2.0]+[0.0]*(np_-2),
           pot=3.0, call_amount=1.0, stack=100.0,
           button_pos=np_-1, my_player_id=0,
           expected_good={0}, expected_bad={3,4,5,6},
           severity='CRITICAL', category='6max_specific',
           description='6-max UTG K2o: fold (UTG range ~16%, K2o kívülre esik).')

        # BB defend K7o vs BTN min-open: 6-max BB defends ~50%
        sc(name='[6max] BB defend K7o vs BTN',
           hole_cards=['Kh','7d'], board_cards=[], street=0,
           my_chips=2.0, all_chips=[4.0,2.0]+[0.0]*(np_-2),
           pot=6.0, call_amount=2.0,
           expected_good={1,2,3}, expected_bad={},
           severity='WARNING', category='6max_specific',
           description='6-max BB K7o vs BTN 2BB open: call (BB defense ~50%; fold too tight).')

    else:
        # ── FULL RING (7-9 max) SPECIFIKUS ──
        # UTG fold 87s: full ring UTG ~12%, 87s kívülre esik
        sc(name=f'[{np_}p] UTG fold 87s',
           hole_cards=['8h','7h'], board_cards=[], street=0,
           my_chips=1.0, all_chips=[1.0,2.0]+[0.0]*(np_-2),
           pot=3.0, call_amount=1.0, stack=100.0,
           button_pos=np_-1, my_player_id=0,
           expected_good={0}, expected_bad={3,4,5,6},
           severity='CRITICAL', category='ring_specific',
           description=f'{np_}p UTG 87s: fold (UTG range ~12%; 87s junk kéznek számít).')

        # EP fold KJo vs UTG open: full ring-ben EP vs UTG raise KJo fold
        sc(name=f'[{np_}p] EP fold KJo vs UTG',
           hole_cards=['Kd','Jc'], board_cards=[], street=0,
           my_chips=2.0, all_chips=[6.0,2.0]+[0.0]*(np_-2),
           pot=9.0, call_amount=4.0,
           expected_good={0,5,6}, expected_bad={2,3},
           severity='WARNING', category='ring_specific',
           description=f'{np_}p EP KJo vs UTG open: fold vagy 4-bet; flat call -EV (dominated).')

        # UTG open QQ: full ring-ben is nyitni kell
        sc(name=f'[{np_}p] UTG open QQ',
           hole_cards=['Qs','Qh'], board_cards=[], street=0,
           my_chips=1.0, all_chips=[1.0,2.0]+[0.0]*(np_-2),
           pot=3.0, call_amount=1.0, stack=100.0,
           button_pos=np_-1, my_player_id=0,
           expected_good={2,3,4,5,6}, expected_bad={0},
           severity='CRITICAL', category='ring_specific',
           description=f'{np_}p UTG QQ: raise (fold CRITICAL HIBA full ring-ben is).')

    return S


def run_scenarios(model, ob, tr, he, np_, dev, ee, tl, verbose):
    scenarios = make_scenarios(np_); mp = PokerActionMapper()
    passed=0; failed=0; warnings=0
    tl.log(f"\n{'─'*65}")
    tl.log(f"  SZITUÁCIÓS TESZTEK ({len(scenarios)} eset)")
    tl.log(f"{'─'*65}\n")
    for sc in scenarios:
        probs, best, value, equity = query_model(model, ob, tr, he, sc, np_, dev, ee)
        is_good = best in sc['expected_good']
        is_bad = best in sc.get('expected_bad', set())
        sev = sc.get('severity','WARNING')
        if is_bad:
            if sev=='CRITICAL': st='❌ FAIL'; failed+=1
            else: st='⚠ WARN'; warnings+=1
        elif is_good: st='✅ PASS'; passed+=1
        else: st='⚠ WARN'; warnings+=1
        si = np.argsort(probs)[::-1]
        t3 = " | ".join(f"{mp.action_name(int(i))} {probs[i]*100:.0f}%" for i in si[:3])
        tl.log(f"  {st}  {sc['name']}")
        tl.log(f"       {mp.action_name(best)} ({probs[best]*100:.0f}%) | Top3: {t3}")
        tl.log(f"       Eq:{equity*100:.0f}% Val:{value:.3f} [{sc.get('category','')}]")
        if is_bad:
            desc = sc.get('description', '')
            tl.log(f"       ⚠ {desc}")
        if verbose: tl.log(f"       {[f'{p:.3f}' for p in probs]}")
        tl.results['scenarios'].append({'name':sc['name'],'status':st,
            'action':mp.action_name(best),'confidence':float(probs[best])})
    tl.log(f"\n  Eredmény: {passed} ✅  {warnings} ⚠  {failed} ❌")
    return passed, warnings, failed


# ═══════════════════════════════════════════════════════════════════════════════
# 2. POZÍCIÓS TUDATOSSÁG
# ═══════════════════════════════════════════════════════════════════════════════

def run_position_test(model, ob, tr, he, np_, dev, ee, tl):
    """
    SPRINT 1 – FIX-2: Pozíciós teszt izolálása.

    PROBLÉMA AZ EREDETI KÓDBAN:
      A BTN és BB szituációk eltérő my_chips, all_chips, pot és call_amount
      értékeket használtak, ami a pozíció hatását más változókkal keverte össze.
      Például BTN: call_amount=1.0, pot=3.0  vs  BB: call_amount=4.0, pot=8.0.
      Ez azt jelenti, hogy a pot odds is különbözik, nem csak a pozíció.

    JAVÍTÁS:
      Mindkét szituáció pontosan ugyanazt a pénzügyi kontextust kapja
      (my_chips, all_chips, pot, call_amount azonos).
      Egyedül a button_pos és my_player_id különbözik.

      Kontextus: hero mindig 2BB-t tett be, az ellenfél 6BB-re emelt (3BB open),
      a pot 8BB, és 4BB-t kell callolni. Ebből a döntési pontból a BTN
      (aki pozícióban van a flop utáni utcákon is) viszonylagosan agresszívebbnek
      kell lennie, mint a BB (aki OOP lesz postflop).

      A teszt kérdése: "azonos pot odds és kéz mellett a pozíció-enkódolás
      egyedül befolyásolja-e a döntést?"

    ÉRTÉKELÉS:
      position_aware = BTN agresszívabb (magasabb raise%) VAGY BTN kevesebbet fold.
      Ha a modell a pozíció-enkódolást nem olvassa, mindkét esetben ugyanúgy dönt.
    """
    tl.log(f"\n{'─'*65}")
    tl.log(f"  POZÍCIÓS TUDATOSSÁG TESZT (izolált, fix pénzügyi kontextus)")
    tl.log(f"{'─'*65}\n")
    tl.log(f"  Kontextus: hero 2BB betét, ellenfél 6BB open, pot=8BB, call=4BB")
    tl.log(f"  Változó:   csak button_pos (BTN=0 vs BB=1)\n")

    bb=2.0; results=[]

    # Azonos pénzügyi alap – CSAK button_pos különbözik
    # Mindkét esetben: hero posted 2BB, ellenfél raised to 6BB → call 4BB more
    shared_ctx = dict(
        board_cards=[], street=0,
        my_chips=2.0,
        all_chips=[6.0, 2.0] + [0.0] * (np_ - 2),
        pot=8.0, call_amount=4.0,
        bb=bb, sb=1.0, stack=100.0,
        legal_actions=[0, 1, 2, 3, 4, 5, 6],
    )

    for name, cards in [
        ('K9o', ['Kd', '9h']),
        ('A5o', ['Ac', '5d']),
        ('QTo', ['Qs', 'Th']),
        ('J8s', ['Jh', '8h']),
        ('T7s', ['Ts', '7s']),
    ]:
        btn_sc = {**shared_ctx, 'hole_cards': cards, 'button_pos': 0, 'my_player_id': 0}
        bb_sc  = {**shared_ctx, 'hole_cards': cards, 'button_pos': 1, 'my_player_id': 0}

        pb,  _, _, _ = query_model(model, ob, tr, he, btn_sc, np_, dev, ee)
        pbb, _, _, _ = query_model(model, ob, tr, he, bb_sc,  np_, dev, ee)

        btn_raise_pct = sum(pb[a]  for a in range(2, 7))
        bb_raise_pct  = sum(pbb[a] for a in range(2, 7))
        btn_fold_pct  = pb[0]
        bb_fold_pct   = pbb[0]

        # Position-aware: BTN-ről legalább 3%-kal agresszívabb VAGY kevesebbet fold
        # A 3% küszöb kiszűri a numerikus zajt a float arithmetikából
        position_aware = (btn_raise_pct > bb_raise_pct + 0.03) or \
                         (btn_fold_pct  < bb_fold_pct  - 0.03)

        delta = btn_raise_pct - bb_raise_pct
        tl.log(
            f"  {'✅' if position_aware else '⚠'} {name}: "
            f"BTN raise%={btn_raise_pct*100:.0f} "
            f"BB raise%={bb_raise_pct*100:.0f} "
            f"(Δ={delta*100:+.0f}%)"
        )
        results.append({
            'hand': name,
            'position_aware': position_aware,
            'btn_aggr': float(btn_raise_pct),
            'bb_aggr':  float(bb_raise_pct),
            'delta':    float(delta),
        })

    n = sum(1 for r in results if r['position_aware'])
    avg_delta = sum(r['delta'] for r in results) / len(results)
    tl.log(f"\n  Pozíció-tudatos: {n}/{len(results)} "
           f"({'✅ JÓ' if n >= 3 else '⚠ GYENGE'})")
    tl.log(f"  Átlagos BTN–BB agresszió delta: {avg_delta*100:+.1f}%")
    if avg_delta < 0:
        tl.log(f"  ⚠ FIGYELEM: A modell BB-ből agresszívabb mint BTN-ből – "
               f"pozíció-enkódolás valószínűleg nem hat a döntésre.")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HUD EXPLOITÁCIÓ  (SPRINT 2 – bővítve)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_nit_tracker(np_):
    """Nit profil: 90% fold-to-cbet, ritkán bet postflop."""
    tr = OpponentHUDTracker(np_)
    for _ in range(50): tr.record_postflop_action(1, 0, facing_cbet=True)
    for _ in range(5):  tr.record_postflop_action(1, 1, facing_cbet=True)
    # Preflop: nagyon szűk – csak raise, semmi call
    for _ in range(10): tr.record_preflop_action(1, 2)
    for _ in range(40): tr.record_preflop_action(1, 0)
    return tr

def _build_maniac_tracker(np_):
    """Maniac profil: 80+ preflop raise, 40+ postflop bet."""
    tr = OpponentHUDTracker(np_)
    for _ in range(80): tr.record_preflop_action(1, 2)
    for _ in range(40): tr.record_postflop_action(1, 4)
    return tr

def _build_calling_station_tracker(np_):
    """Calling station: soha nem foldol, soha nem raise-el cbet ellen."""
    tr = OpponentHUDTracker(np_)
    for _ in range(60): tr.record_postflop_action(1, 1, facing_cbet=True)
    for _ in range(5):  tr.record_postflop_action(1, 0, facing_cbet=True)
    for _ in range(50): tr.record_preflop_action(1, 1)  # mindig call
    return tr

def run_exploit_test(model, ob, he, np_, dev, ee, tl):
    """
    SPRINT 2 – Bővített HUD exploitáció teszt (7 spot).

    RÉGI (3 spot): egyetlen board Nit ellen, egyetlen board Maniac ellen,
                   1 adaptáció delta mérés (küszöb: +5%).
    ÚJ (7 spot):
      Nit×3:           blöff frekvencia 3 különböző boardon/kontextusban
      Maniac×2:        trap + check-raise jellegű döntés
      Calling Station: value bet frekvencia (erős kézzel tegyünk-e nagyot?)
      Adaptáció delta: Nit vs Neutral küszöb ≥10% (volt: +5%)

    Értékelés: X/7 passed → exploit_score
    """
    tl.log(f"\n{'─'*65}")
    tl.log(f"  HUD EXPLOITÁCIÓ TESZT (SPRINT2: 7 spot)")
    tl.log(f"{'─'*65}\n")
    mp = PokerActionMapper()
    R  = []

    nit  = _build_nit_tracker(np_)
    man  = _build_maniac_tracker(np_)
    call = _build_calling_station_tracker(np_)
    neu  = OpponentHUDTracker(np_)  # semleges (üres) baseline

    # ── NIT BLÖFF SZITUÁCIÓK (3 spot) ────────────────────────────────────────
    # Közös: air/blöff kéz → ha a modell olvassa a Nit profilt, többet blöfföl
    tl.log(f"  [Nit ellen – blöff frekvencia, 3 spot]")

    nit_spots = [
        # (név, hole, board, street, pot, call_amount, stack)
        ('Nit vs air K72r',     ['9d','4c'], ['Ks','7d','2c'], 1, 12.0, 0.0, 94.0),
        ('Nit vs air A-high turn', ['8h','6d'], ['As','Td','3c','2h'], 2, 18.0, 0.0, 82.0),
        ('Nit vs air low flop', ['Jd','9s'], ['6h','4c','2d'], 1, 10.0, 0.0, 90.0),
    ]

    nit_bet_pcts = []
    neu_bet_pcts = []
    for name, hole, board, street, pot, call_amt, stack in nit_spots:
        sc = {
            'hole_cards': hole, 'board_cards': board, 'street': street,
            'my_chips': pot/2, 'all_chips': [pot/2]*2 + [0.0]*(np_-2),
            'pot': pot, 'call_amount': call_amt,
            'bb': 2.0, 'sb': 1.0, 'stack': stack,
            'legal_actions': [0,1,2,3,4,5,6],
        }
        pn,  bn,  _, _ = query_model(model, ob, nit, he, sc, np_, dev, ee)
        pnu, bnu, _, _ = query_model(model, ob, neu, he, sc, np_, dev, ee)
        nit_bet = sum(pn[a]  for a in range(2,7)) * 100
        neu_bet = sum(pnu[a] for a in range(2,7)) * 100
        nit_bet_pcts.append(nit_bet)
        neu_bet_pcts.append(neu_bet)
        # Pass: Nit ellen raise-el (blöfföl) VAGY bet% ≥ 30%
        g = (bn >= 2) or (nit_bet >= 30)
        R.append({'test': name, 'passed': g, 'profile': 'nit'})
        tl.log(f"  {'✅' if g else '⚠'} {name}: {mp.action_name(bn)} "
               f"bet%={nit_bet:.0f}% (neutral={neu_bet:.0f}%)")

    # ── MANIAC TRAP SZITUÁCIÓK (2 spot) ──────────────────────────────────────
    tl.log(f"\n  [Maniac ellen – trap / nem fold]")

    # Spot 1: AA-val ne foldjunk maniac ellen (már az eredeti is)
    sc_man1 = {
        'hole_cards': ['As','Ah'], 'board_cards': ['Kd','7c','3s'], 'street': 1,
        'my_chips': 12.0, 'all_chips': [12.0,12.0]+[0.0]*(np_-2),
        'pot': 24.0, 'call_amount': 0.0,
        'bb': 2.0, 'sb': 1.0, 'stack': 88.0, 'legal_actions': [0,1,2,3,4,5,6],
    }
    _, bm1, _, _ = query_model(model, ob, man, he, sc_man1, np_, dev, ee)
    g_man1 = (bm1 != 0)  # ne dobjuk el a kezet
    R.append({'test': 'Maniac AA trap', 'passed': g_man1, 'profile': 'maniac'})
    tl.log(f"  {'✅' if g_man1 else '❌'} Maniac AA trap: AA → {mp.action_name(bm1)} "
           f"(fold=FAIL)")

    # Spot 2: Erős kézzel maniac ellen ne csak checkelj – bet/raise elvárt
    sc_man2 = {
        'hole_cards': ['Kh','Ks'], 'board_cards': ['Kd','8c','3s'], 'street': 1,
        'my_chips': 8.0, 'all_chips': [8.0,8.0]+[0.0]*(np_-2),
        'pot': 16.0, 'call_amount': 0.0,
        'bb': 2.0, 'sb': 1.0, 'stack': 92.0, 'legal_actions': [0,1,2,3,4,5,6],
    }
    pm2, bm2, _, _ = query_model(model, ob, man, he, sc_man2, np_, dev, ee)
    man_bet2 = sum(pm2[a] for a in range(2,7)) * 100
    # Maniac ellen set-tel bet% elvárt ≥ 50% (ne trap-eljen túlzottan)
    g_man2 = (man_bet2 >= 50) or (bm2 >= 2)
    R.append({'test': 'Maniac KKK bet', 'passed': g_man2, 'profile': 'maniac'})
    tl.log(f"  {'✅' if g_man2 else '⚠'} Maniac KKK bet: {mp.action_name(bm2)} "
           f"bet%={man_bet2:.0f}%")

    # ── CALLING STATION VALUE BET (1 spot) ───────────────────────────────────
    tl.log(f"\n  [Calling Station ellen – value bet méret]")

    sc_cs = {
        'hole_cards': ['Ac','Ah'], 'board_cards': ['As','7h','2d'], 'street': 1,
        'my_chips': 6.0, 'all_chips': [6.0,6.0]+[0.0]*(np_-2),
        'pot': 12.0, 'call_amount': 0.0,
        'bb': 2.0, 'sb': 1.0, 'stack': 94.0, 'legal_actions': [0,1,2,3,4,5,6],
    }
    pcs, bcs, _, _ = query_model(model, ob, call, he, sc_cs, np_, dev, ee)
    cs_bet = sum(pcs[a] for a in range(2,7)) * 100
    # Calling Station ellen trips-szel bet% ≥ 60% elvárt (ő úgyis callol)
    g_cs = (cs_bet >= 60) or (bcs >= 2)
    R.append({'test': 'CallingStation value', 'passed': g_cs, 'profile': 'calling_station'})
    tl.log(f"  {'✅' if g_cs else '⚠'} CallingStation AAA value: "
           f"{mp.action_name(bcs)} bet%={cs_bet:.0f}%")

    # ── ADAPTÁCIÓ DELTA (Nit vs Neutral, szigorúbb küszöb) ───────────────────
    tl.log(f"\n  [Adaptáció delta – Nit vs Neutral, küszöb ≥10%]")

    # Átlagos Nit blöff% az első spotból (legtisztább referencia)
    avg_nit_bet = sum(nit_bet_pcts) / len(nit_bet_pcts)
    avg_neu_bet = sum(neu_bet_pcts) / len(neu_bet_pcts)
    delta = avg_nit_bet - avg_neu_bet
    # SPRINT2: szigorúbb küszöb volt +5% → most +10%
    g_adapt = delta >= 10.0
    R.append({'test': 'HUD adaptáció (≥10%)', 'passed': g_adapt, 'profile': 'delta'})
    tl.log(f"  {'✅' if g_adapt else '⚠'} Adaptáció delta: "
           f"nit_avg={avg_nit_bet:.0f}% vs neutral_avg={avg_neu_bet:.0f}% "
           f"(Δ={delta:+.0f}%, küszöb=+10%)")

    # ── ÖSSZESÍTÉS ────────────────────────────────────────────────────────────
    total   = len(R)
    passed  = sum(1 for r in R if r['passed'])
    tl.log(f"\n  Exploit összesítő: {passed}/{total}")
    if passed >= 5: tl.log(f"  ✅ HUD exploitáció rendben")
    elif passed >= 3: tl.log(f"  ⚠ Részleges HUD exploitáció")
    else: tl.log(f"  ❌ HUD-vakság – a modell nem reagál az ellenfél profiljára")

    return R


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DRAW EQUITY AWARENESS (v3a)
# ═══════════════════════════════════════════════════════════════════════════════

def run_draw_test(model, ob, tr, he, np_, dev, ee, tl):
    tl.log(f"\n{'─'*65}")
    tl.log(f"  DRAW EQUITY AWARENESS TESZT")
    tl.log(f"{'─'*65}\n")
    mp=PokerActionMapper(); R=[]; fd=0; dt=0
    base={'my_chips':8.0,'all_chips':[14.0,8.0]+[0.0]*(np_-2),'pot':22.0,
          'call_amount':6.0,'bb':2.0,'sb':1.0,'stack':92.0,'legal_actions':[0,1,2,3,4,5,6]}
    draws=[
        ('Flush draw 9out',['Ah','5h'],['Kh','9d','3h'],1,9,{1,2,3,4,5,6},{0},'CRITICAL'),
        ('OESD 8out',['Jd','Tc'],['9s','8h','2d'],1,8,{1,2,3,4,5,6},{0},'CRITICAL'),
        ('Combo 15out AGGRO',['Qh','Jh'],['Th','9d','2h'],1,15,{4,5,6},{0},'CRITICAL'),
        ('Gutshot 4out',['6s','5s'],['9d','7h','2c'],1,4,{0,1},set(),'INFO'),
        ('TP+flush draw',['As','Ks'],['Ac','7s','3s'],1,12,{2,3,4,5,6},{0},'CRITICAL'),
        ('Flush MISSED river',['Ah','5h'],['Kh','9d','3h','Jc','2s'],3,0,{0,1},set(),'INFO'),
        ('OESD turn 8out',['Jd','Tc'],['9s','8h','2d','Ks'],2,8,{1,2,3,4,5,6},{0},'WARNING'),
        ('Board paired FD',['6h','5h'],['Kh','Kd','3h'],1,7,{0,1},{5,6},'WARNING'),
    ]
    passed=0
    for nm,hole,board,st,outs,eg,eb,sev in draws:
        sc={**base,'name':nm,'hole_cards':hole,'board_cards':board,'street':st}
        probs,best,_,eq=query_model(model,ob,tr,he,sc,np_,dev,ee)
        ig=best in eg; ib=best in eb
        if ib: s='❌' if sev=='CRITICAL' else '⚠'
        elif ig: s='✅'; passed+=1
        else: s='⚠'
        if outs>=8: dt+=1; (fd:=fd+1) if best==0 else None
        ap=sum(probs[a] for a in range(2,7))*100
        tl.log(f"  {s} {nm}: {mp.action_name(best)} eq={eq*100:.0f}% aggr={ap:.0f}%")
        R.append({'name':nm,'outs':outs,'equity':eq,'fold':best==0,'aggression':ap})
    dfp=fd/max(dt,1)*100
    tl.log(f"\n  Draw awareness: {passed}/{len(draws)} OK | "
           f"Erős draw fold: {fd}/{dt} ({dfp:.0f}%)")
    if dfp>30: tl.log(f"  ❌ Túl sok draw fold")
    elif dfp>10: tl.log(f"  ⚠ Néhány draw fold")
    else: tl.log(f"  ✅ Draw kezelés OK")
    eqs=[r['equity'] for r in R if r['outs']>0]
    ags=[r['aggression'] for r in R if r['outs']>0]
    corr=_spearman(eqs,ags) if len(eqs)>=3 else 0.0
    tl.log(f"  Equity↔Agresszió r={corr:.2f}")
    return {'passed':passed,'total':len(draws),'draw_fold_pct':dfp,'eq_aggr_corr':corr,'details':R}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BET SIZING ANALÍZIS (v3a)
# ═══════════════════════════════════════════════════════════════════════════════

def run_sizing_test(model, ob, tr, he, np_, dev, ee, tl):
    tl.log(f"\n{'─'*65}")
    tl.log(f"  BET SIZING ANALÍZIS")
    tl.log(f"{'─'*65}\n")
    mp=PokerActionMapper()

    tl.log(f"  ── Preflop BTN open ──")
    pf_hands=[('72o',['7s','2h'],1),('T5o',['Td','5c'],2),('K8o',['Kd','8h'],3),
              ('QJs',['Qh','Jh'],4),('TT',['Ts','Th'],5),('AQo',['Ac','Qd'],6),
              ('KK',['Ks','Kh'],7),('AA',['As','Ah'],8)]
    pf_r=[]
    for nm,cards,st in pf_hands:
        sc={'hole_cards':cards,'board_cards':[],'street':0,'my_chips':1.0,
            'all_chips':[1.0,2.0]+[0.0]*(np_-2),'pot':3.0,'call_amount':1.0,
            'bb':2.0,'sb':1.0,'stack':100.0,'button_pos':0,'my_player_id':0,
            'legal_actions':[0,1,2,3,4,5,6]}
        probs,best,_,eq=query_model(model,ob,tr,he,sc,np_,dev,ee)
        wt=sum(probs[a]*a for a in range(7))
        tl.log(f"    {nm:5s} str={st}: {mp.action_name(best):18s} tier={wt:.1f} eq={eq*100:.0f}%")
        pf_r.append({'hand':nm,'strength':st,'weighted_tier':wt})
    pc=_spearman([r['strength'] for r in pf_r],[r['weighted_tier'] for r in pf_r])
    tl.log(f"\n  Preflop sizing r={pc:.2f} ({'✅' if pc>0.5 else '⚠' if pc>0.2 else '❌'})")

    tl.log(f"\n  ── Postflop bet sizing (Kd 7d 2c) ──")
    pp_hands=[('Air 94o',['9d','4c'],1),('Mid pair',['7s','6s'],2),
              ('Top pair',['Kc','Jd'],3),('Overpair',['As','Ah'],4),
              ('Set',['7h','7c'],5),('Top set',['Ks','Kh'],6)]
    pp_r=[]
    for nm,cards,st in pp_hands:
        sc={'hole_cards':cards,'board_cards':['Kd','7d','2c'],'street':1,
            'my_chips':6.0,'all_chips':[6.0,6.0]+[0.0]*(np_-2),'pot':12.0,
            'call_amount':0.0,'bb':2.0,'sb':1.0,'stack':94.0,
            'legal_actions':[0,1,2,3,4,5,6]}
        probs,best,_,eq=query_model(model,ob,tr,he,sc,np_,dev,ee)
        wt=sum(probs[a]*a for a in range(7))
        tl.log(f"    {nm:12s} str={st}: {mp.action_name(best):18s} tier={wt:.1f} eq={eq*100:.0f}%")
        pp_r.append({'hand':nm,'strength':st,'weighted_tier':wt})
    ppc=_spearman([r['strength'] for r in pp_r],[r['weighted_tier'] for r in pp_r])
    tl.log(f"\n  Postflop sizing r={ppc:.2f} ({'✅' if ppc>0.5 else '⚠' if ppc>0.2 else '❌'})")
    avg=(pc+ppc)/2
    tl.log(f"  Átlag sizing r={avg:.2f}")
    return {'preflop_corr':pc,'postflop_corr':ppc,'avg_corr':avg}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. KONZISZTENCIA / ENTROPY (ÚJ v3b)
# ═══════════════════════════════════════════════════════════════════════════════

def run_consistency_test(model, ob, tr, he, np_, dev, ee, tl):
    tl.log(f"\n{'─'*65}")
    tl.log(f"  KONZISZTENCIA / ENTROPY TESZT")
    tl.log(f"{'─'*65}\n")
    mp=PokerActionMapper()

    cases=[
        ('AA preflop (KELL alacsony)',['As','Ah'],[],0,
         {'my_chips':2.0,'all_chips':[2.0]*np_,'pot':3.0,'call_amount':0.0},
         'low'),
        ('72o facing raise (elfogadható magas)',['7s','2h'],[],0,
         {'my_chips':2.0,'all_chips':[6.0,2.0]+[0.0]*(np_-2),'pot':9.0,'call_amount':4.0},
         'any'),
        ('Nut flush flop (KELL alacsony)',['As','Ks'],['Qs','7s','3s'],1,
         {'my_chips':10.0,'all_chips':[10.0,10.0]+[0.0]*(np_-2),'pot':20.0,'call_amount':0.0,'stack':90.0},
         'low'),
        ('Marginális kéz (OK ha magas)',['Kd','9h'],[],0,
         {'my_chips':1.0,'all_chips':[1.0,2.0]+[0.0]*(np_-2),'pot':3.0,'call_amount':1.0},
         'any'),
        ('Top set (KELL alacsony)',['Ah','Ad'],['Ac','7d','2s'],1,
         {'my_chips':8.0,'all_chips':[8.0,8.0]+[0.0]*(np_-2),'pot':16.0,'call_amount':0.0,'stack':92.0},
         'low'),
    ]

    results=[]
    for name,hole,board,street,ctx,expect in cases:
        sc={'hole_cards':hole,'board_cards':board,'street':street,
            'bb':2.0,'sb':1.0,'stack':ctx.get('stack',100.0),
            'legal_actions':[0,1,2,3,4,5,6],**ctx}
        probs,best,_,eq=query_model(model,ob,tr,he,sc,np_,dev,ee)
        ent=_entropy(probs)
        conf=float(probs[best])
        ok = True
        if expect=='low' and ent>1.2: ok=False
        tl.log(f"  {'✅' if ok else '⚠'} {name}")
        tl.log(f"     {mp.action_name(best)} ({conf*100:.0f}%) | "
               f"Entropy={ent:.2f} {'(alacsony ✅)' if ent<0.8 else '(közepes)' if ent<1.2 else '(magas ⚠)'}")
        results.append({'name':name,'entropy':ent,'confidence':conf,'ok':ok})

    avg_ent=np.mean([r['entropy'] for r in results])
    low_ok=sum(1 for r in results if r['ok'])
    tl.log(f"\n  Átlag entropy: {avg_ent:.2f} | Magabiztos: {low_ok}/{len(results)}")
    if avg_ent<0.8: tl.log(f"  ✅ Jó döntési magabiztosság")
    elif avg_ent<1.3: tl.log(f"  ⚠ Közepes magabiztosság")
    else: tl.log(f"  ❌ Bizonytalan döntések – több tréning kell")
    return {'avg_entropy':float(avg_ent),'confident_count':low_ok,
            'total':len(results),'details':results}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. VALUE HEAD KALIBRÁCIÓ  (SPRINT 2 – ÚJ)
# ═══════════════════════════════════════════════════════════════════════════════

def run_value_calibration_test(model, ob, tr, he, np_, dev, ee, tl):
    """
    SPRINT 2 – ÚJ TESZT: Value head alapkalibráció.

    A query_model() minden szituációnál visszaad egy `value` float-ot
    (a critic head kimenete), de az eredeti kód soha nem validálta ezt.

    Ez a teszt elméleti határok ellen ellenőrzi a value becsléseket.
    Az rlcard BB-egységes reward skálán dolgozik:
      value ≈ 0   →  semleges EV (fold/check break-even)
      value > 0   →  pozitív várható érték
      value < 0   →  negatív várható érték

    5 szituáció, minden esetben egy irányítélt elvárás:
      AA preflop (raise opportunity):   value > 0      erős kéz, értékesíthető
      72o facing raise:                 value < 0      negatív EV, el kell dobni
      Nut flush flop (can bet):         value > 0      erős kéz, értékesíthető
      Top set dry flop:                 value > 0
      Air facing river pot bet:         value < 0.5    nem profitábilis call

    Fontos korlát: a value head a TRÉNING eloszlásából tanul, ezért
    az abszolút értékek nem feltétlenül kalibráltak BB-re. A teszt
    csak az EV IRÁNYÁT ellenőrzi (előjel / relatív sorrend), nem az
    abszolút magnitudót.
    """
    tl.log(f"\n{'─'*65}")
    tl.log(f"  VALUE HEAD KALIBRÁCIÓ TESZT (SPRINT2 – ÚJ)")
    tl.log(f"{'─'*65}\n")
    tl.log(f"  Értékelés: az EV iránya (előjel/relatív) helyes-e?")
    tl.log(f"  Skála: value>0 = pozitív EV, value<0 = negatív EV\n")

    cases = [
        # (név, hole, board, street, ctx_dict, elvárás_fn, elvárás_leírás)
        (
            'AA preflop (EV > 0)',
            ['As','Ah'], [], 0,
            {'my_chips':2.0, 'all_chips':[2.0]*np_, 'pot':3.0, 'call_amount':0.0},
            lambda v: v > 0,
            'value > 0',
        ),
        (
            '72o vs raise (EV < 0)',
            ['7s','2h'], [], 0,
            {'my_chips':2.0, 'all_chips':[6.0,2.0]+[0.0]*(np_-2),
             'pot':9.0, 'call_amount':4.0},
            lambda v: v < 0,
            'value < 0',
        ),
        (
            'Nut flush flop (EV > 0)',
            ['As','Ks'], ['Qs','7s','3s'], 1,
            {'my_chips':10.0, 'all_chips':[10.0,10.0]+[0.0]*(np_-2),
             'pot':20.0, 'call_amount':0.0, 'stack':90.0},
            lambda v: v > 0,
            'value > 0',
        ),
        (
            'Top set dry flop (EV > 0)',
            ['Ah','Ad'], ['Ac','7d','2s'], 1,
            {'my_chips':8.0, 'all_chips':[8.0,8.0]+[0.0]*(np_-2),
             'pot':16.0, 'call_amount':0.0, 'stack':92.0},
            lambda v: v > 0,
            'value > 0',
        ),
        (
            'Air river pot-bet (EV < 0.5)',
            ['9h','8h'], ['As','Kd','3c','5s','Jd'], 3,
            {'my_chips':30.0, 'all_chips':[60.0,30.0]+[0.0]*(np_-2),
             'pot':60.0, 'call_amount':30.0, 'stack':70.0},
            lambda v: v < 0.5,
            'value < 0.5',
        ),
    ]

    results = []
    for name, hole, board, street, ctx, check_fn, expect_str in cases:
        sc = {
            'hole_cards': hole, 'board_cards': board, 'street': street,
            'bb': 2.0, 'sb': 1.0, 'stack': ctx.get('stack', 100.0),
            'legal_actions': [0,1,2,3,4,5,6], **ctx,
        }
        _, best, value, equity = query_model(model, ob, tr, he, sc, np_, dev, ee)
        ok = check_fn(value)
        tl.log(
            f"  {'✅' if ok else '⚠'} {name}"
            f"\n     value={value:+.4f}  elvárás: [{expect_str}]"
            f"  eq={equity*100:.0f}%"
        )
        results.append({'name': name, 'value': value, 'ok': ok,
                        'expect': expect_str, 'equity': equity})

    passed = sum(1 for r in results if r['ok'])
    total  = len(results)

    # Konzisztencia ellenőrzés: erős kéz value-ja > gyenge kéz value-ja?
    aa_val  = next((r['value'] for r in results if 'AA'  in r['name']), None)
    o72_val = next((r['value'] for r in results if '72o' in r['name']), None)
    if aa_val is not None and o72_val is not None:
        rank_ok = aa_val > o72_val
        tl.log(f"\n  Relatív sorrend: AA({aa_val:+.3f}) > 72o({o72_val:+.3f}) "
               f"→ {'✅ helyes' if rank_ok else '⚠ fordított!'}")
        results.append({'name': 'AA > 72o sorrend', 'value': aa_val - o72_val,
                        'ok': rank_ok, 'expect': 'aa_val > 72o_val', 'equity': 0})
        if rank_ok: passed += 1
        total += 1

    tl.log(f"\n  Value kalibráció: {passed}/{total}")
    if passed == total:
        tl.log(f"  ✅ Value head EV iránya helyes minden esetben")
    elif passed >= total * 0.6:
        tl.log(f"  ⚠ Value head részben kalibrált ({passed}/{total})")
    else:
        tl.log(f"  ❌ Value head nincs kalibrálva – a critic nem tanult EV-t")

    return {'passed': passed, 'total': total, 'details': results}


# ═══════════════════════════════════════════════════════════════════════════════
# 9. BB/100 WINRATE + BOOTSTRAP CI  (SPRINT3 – bővítve)
# ═══════════════════════════════════════════════════════════════════════════════

def run_winrate_test(model, np_, device, n_hands, tl):
    """
    SPRINT 3 – NEW-4: Bootstrap konfidencia-intervallum hozzáadása.

    RÉGI: pontbecslés (BB/100), nincs statisztikai megbízhatóság mérve.
    ÚJ:   _bootstrap_ci() 1000 újramintavételezéssel 95% CI-t számol.
          A play_hands() most per-kéz payoff listát is visszaad.

    Értékelés:
      ha CI alsó határa > 0  → ✅ statisztikailag szignifikáns nyerő
      ha CI átfedi a 0-t     → ⚠ nem meggyőző (több kéz kell)
      ha CI felső határa < 0 → ❌ szignifikánsan vesztes
    """
    tl.log(f"\n{'─'*65}")
    tl.log(f"  BB/100 WINRATE TESZT + Bootstrap 95% CI ({n_hands} kéz)")
    tl.log(f"{'─'*65}\n")

    try:
        import rlcard
    except ImportError:
        tl.log(f"  ⚠ rlcard nem elérhető – winrate teszt kihagyva")
        return {'self_play': None, 'vs_random': None, 'error': 'rlcard not installed'}

    from core.features import detect_street, ActionHistoryEncoder
    mapper = PokerActionMapper()
    bb = 2.0; sb = 1.0

    def play_hands(model_p0, model_p1, n, is_random_p1=False):
        """
        Lejátszik n kezet, visszaad BB/100-at, db-t, összeg-et
        és per-kéz payoff listát (a bootstrap CI-hoz).
        """
        env = rlcard.make('no-limit-holdem', config={'game_num_players': np_})
        tracker    = OpponentHUDTracker(np_)
        he_loc     = ActionHistoryEncoder(np_, PokerActionMapper.NUM_CUSTOM_ACTIONS)
        equity_est = HandEquityEstimator(n_sim=100)
        total_payoff = 0.0
        completed    = 0
        hand_payoffs = []   # ÚJ: per-kéz payoffok a bootstraphoz

        for hand_idx in range(n):
            try:
                state, player_id = env.reset()
            except:
                continue
            ah         = collections.deque(maxlen=ACTION_HISTORY_LEN)
            steps      = 0
            model_seat = hand_idx % 2

            while not env.is_over() and steps < 200:
                steps += 1
                raw_legal = state.get('legal_actions', [1])
                abs_legal = mapper.get_abstract_legal_actions(raw_legal)

                if is_random_p1 and player_id != model_seat:
                    aa = random.choice(abs_legal)
                else:
                    try:
                        hand     = env.game.players[player_id].hand
                        hole_eq  = [f"{c.get_index()[1]}{c.get_index()[0].lower()}"
                                    for c in hand]
                        board_eq = [f"{c.get_index()[1]}{c.get_index()[0].lower()}"
                                    for c in env.game.public_cards]
                        eq = equity_est.equity(hole_eq, board_eq,
                                               num_opponents=max(np_-1, 1))
                    except:
                        eq = 0.5

                    street   = detect_street(state)
                    obs      = np.array(state['obs'], dtype=np.float32)
                    st_dict  = {'obs': obs, 'raw_obs': state.get('raw_obs', {})}
                    state_t  = build_state_tensor(
                        st_dict, tracker, ah, he_loc, np_,
                        my_player_id=player_id, bb=bb, sb=sb,
                        initial_stack=100.0, street=street, equity=eq)

                    act_model = model_p0 if player_id == model_seat else model_p1
                    with torch.no_grad():
                        probs_t, _, _ = act_model.forward(state_t.to(device), abs_legal)
                    probs = probs_t.squeeze(0).cpu().numpy()
                    aa    = int(np.argmax(probs))

                ea = mapper.get_env_action(aa, raw_legal)
                ah.append((player_id, aa, 0.0))
                try:
                    state, player_id = env.step(ea)
                except:
                    break

            try:
                payoffs       = env.get_payoffs()
                payout        = float(payoffs[model_seat])
                total_payoff += payout
                hand_payoffs.append(payout)   # ÚJ
                completed    += 1
            except:
                pass

        bb100 = (total_payoff / max(completed, 1)) / bb * 100
        return bb100, completed, total_payoff, hand_payoffs   # ÚJ: +hand_payoffs

    t0 = time.time()

    # ── SELF-PLAY ─────────────────────────────────────────────────────────────
    tl.log(f"  Self-play ({n_hands} kéz)...", console=True)
    sp_bb100, sp_n, sp_total, sp_payoffs = play_hands(model, model, n_hands,
                                                       is_random_p1=False)
    sp_mean, sp_lo, sp_hi = _bootstrap_ci(sp_payoffs, bb)

    tl.log(f"  Self-play BB/100: {sp_bb100:+.1f} ({sp_n} kéz)")
    if sp_mean is not None:
        sig = sp_lo > 0 or sp_hi < 0
        tl.log(f"  Bootstrap 95% CI: [{sp_lo:+.1f}, {sp_hi:+.1f}]"
               + ("  ⚠ aszimmetria szignifikáns" if sig else "  ✅ szimmetria OK"))
    if abs(sp_bb100) < 10:
        tl.log(f"  ✅ Self-play közel 0 – szimmetrikus (elvárt)")
    else:
        tl.log(f"  ⚠ Self-play aszimmetria: {sp_bb100:+.1f} BB/100")

    # ── VS RANDOM ─────────────────────────────────────────────────────────────
    tl.log(f"\n  Vs random ({n_hands} kéz)...", console=True)
    rnd_bb100, rnd_n, rnd_total, rnd_payoffs = play_hands(model, model, n_hands,
                                                           is_random_p1=True)
    rnd_mean, rnd_lo, rnd_hi = _bootstrap_ci(rnd_payoffs, bb)
    elapsed = time.time() - t0

    tl.log(f"  Vs random BB/100: {rnd_bb100:+.1f} ({rnd_n} kéz)")
    if rnd_mean is not None:
        if rnd_lo > 0:
            sig_str = "✅ szignifikánsan nyerő"
        elif rnd_hi < 0:
            sig_str = "❌ szignifikánsan vesztes"
        else:
            sig_str = "⚠ nem meggyőző (CI átfedi 0-t, több kéz kell)"
        tl.log(f"  Bootstrap 95% CI: [{rnd_lo:+.1f}, {rnd_hi:+.1f}]  {sig_str}")

    if rnd_bb100 > 30:
        tl.log(f"  ✅ Jól veri a random ellenfelet")
    elif rnd_bb100 > 0:
        tl.log(f"  ⚠ Nyerő, de nem eléggé ({rnd_bb100:+.1f})")
    else:
        tl.log(f"  ❌ Random ellen sem nyerő – komoly probléma!")
    tl.log(f"\n  Winrate teszt idő: {elapsed:.1f}s")

    return {
        'self_play_bb100':   sp_bb100,
        'self_play_ci_low':  sp_lo,
        'self_play_ci_high': sp_hi,
        'vs_random_bb100':   rnd_bb100,
        'vs_random_ci_low':  rnd_lo,
        'vs_random_ci_high': rnd_hi,
        'vs_random_significant': (rnd_lo is not None and rnd_lo > 0),
        'hands_played': sp_n,
        'elapsed':      elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SCENARIO GENERATOR  (SPRINT3 – ÚJ)
# ═══════════════════════════════════════════════════════════════════════════════

class ScenarioGenerator:
    """
    SPRINT 3 – NEW-5: Programmatikus szituáció-generátor.

    A hardcoded 14 szituáció jó alap, de nem terjed ki az összes
    stratégiailag fontos dimenzióra. Ez az osztály automatikusan generál
    szituáció-dikteket 3 dimenzióban:

    (a) stack_depth_sweep(): azonos kéz 4 stack mélységben
        Kérdés: felismeri-e a modell a push/fold zónát (≤20BB)?
        Elvárás: 10BB-nél szignifikánsan különbözik a döntés vs 100BB-nél.

    (b) board_texture_sweep(): KcJd top pair 4 textúrán
        Kérdés: csökkenti-e a bet%-ot veszélyesebb boardon?
        Elvárás: bet% csökken dry→two-tone→monotone sorrendben.

    (c) river_spots(): 4 river-specifikus szituáció
        Célzottan a 20M modell ismert gyengeségét teszteli.
        ('Air facing river pot bet' ⚠ WARN → river döntések általánosan)
    """

    @staticmethod
    def stack_depth_sweep(np_):
        """
        Top pair (KsJd on Kh7c2d flop) 4 stack mélységben.
        Minden esetben ugyanaz a pot/call arány (~33% pot c-bet facing),
        csak a stack mérete változik.
        """
        bb  = 2.0; sb = 1.0
        pot = bb * 3       # ~3BB pot a flopra
        call_amt = pot * 0.33
        my_chips = bb + call_amt

        scenarios = []
        for sbb in [10, 20, 50, 100]:
            stack = max(0.1, sbb * bb - my_chips)
            scenarios.append({
                'name':        f'Top pair {sbb}BB stack',
                'stack_bb':    sbb,
                'hole_cards':  ['Ks', 'Jd'],
                'board_cards': ['Kh', '7c', '2d'],
                'street':      1,
                'my_chips':    my_chips,
                'all_chips':   [my_chips] * 2 + [0.0] * (np_ - 2),
                'pot':         pot,
                'call_amount': call_amt,
                'bb': bb, 'sb': sb, 'stack': stack,
                'legal_actions': [0, 1, 2, 3, 4, 5, 6],
            })
        return scenarios

    @staticmethod
    def board_texture_sweep(np_):
        """
        KcJd (top pair good kicker) 4 különböző board textúrán.
        Kontextus: min-raise elleni döntés flopon.
        """
        boards = [
            ('Dry K72r',     ['Ks', '7d', '2c'],  'dry'),
            ('Two-tone K72', ['Ks', '7h', '2h'],  'two_tone'),
            ('Monotone K98', ['Kh', '9h', '8h'],  'monotone'),
            ('Paired KK7',   ['Ks', 'Kd', '7c'],  'paired'),
        ]
        scenarios = []
        for name, board, texture in boards:
            scenarios.append({
                'name':        f'KcJd – {name}',
                'texture':     texture,
                'hole_cards':  ['Kc', 'Jd'],
                'board_cards': board,
                'street':      1,
                'my_chips':    6.0,
                'all_chips':   [6.0, 6.0] + [0.0] * (np_ - 2),
                'pot':         12.0,
                'call_amount': 0.0,
                'bb': 2.0, 'sb': 1.0, 'stack': 94.0,
                'legal_actions': [0, 1, 2, 3, 4, 5, 6],
            })
        return scenarios

    @staticmethod
    def river_spots(np_):
        """
        4 river-specifikus szituáció.
        A 20M modell 'Air facing river pot bet' ⚠ WARN eredményére
        reagálva ezek célzottan a river döntéshozatalt tesztelik:
          1. Value bet (erős kézzel nyomjon ki értéket)
          2. Bluff-catch fold (gyengével dobjon el pot-size bet-et)
          3. Missed flush (dobjon el vagy overbet-blöfföl)
          4. Nut river (maximális értékelés trips+)
        """
        return [
            {
                'name':          'River value bet (2-pair, check to)',
                'expected_good': {2, 3, 4, 5, 6},
                'expected_bad':  {0},
                'severity':      'WARNING',
                'category':      'river_value',
                'hole_cards':    ['Ah', 'Kh'],
                'board_cards':   ['Ac', 'Kd', '7c', '3s', 'Jd'],
                'street': 3,
                'my_chips': 20.0,
                'all_chips': [20.0, 20.0] + [0.0] * (np_ - 2),
                'pot': 40.0, 'call_amount': 0.0,
                'bb': 2.0, 'sb': 1.0, 'stack': 80.0,
                'legal_actions': [0, 1, 2, 3, 4, 5, 6],
            },
            {
                'name':          'River bluff-catch fold (Q-high, pot-bet)',
                'expected_good': {0},
                'expected_bad':  {5, 6},
                'severity':      'WARNING',
                'category':      'river_bluff_catch',
                'hole_cards':    ['Qh', 'Jd'],
                'board_cards':   ['Ac', 'Kd', '3c', '5s', 'Td'],
                'street': 3,
                'my_chips': 30.0,
                'all_chips': [60.0, 30.0] + [0.0] * (np_ - 2),
                'pot': 60.0, 'call_amount': 30.0,
                'bb': 2.0, 'sb': 1.0, 'stack': 70.0,
                'legal_actions': [0, 1, 2, 3, 4, 5, 6],
            },
            {
                'name':          'Missed flush river (fold or overbet-bluff)',
                'expected_good': {0, 6},
                'expected_bad':  {3, 4},
                'severity':      'WARNING',
                'category':      'river_missed_draw',
                'hole_cards':    ['Ah', '5h'],
                'board_cards':   ['Kh', '9d', '3h', 'Jc', '2s'],
                'street': 3,
                'my_chips': 12.0,
                'all_chips': [12.0, 12.0] + [0.0] * (np_ - 2),
                'pot': 24.0, 'call_amount': 0.0,
                'bb': 2.0, 'sb': 1.0, 'stack': 88.0,
                'legal_actions': [0, 1, 2, 3, 4, 5, 6],
            },
            {
                'name':          'Nut river trips (max value)',
                'expected_good': {4, 5, 6},
                'expected_bad':  {0},
                'severity':      'WARNING',
                'category':      'river_nut',
                'hole_cards':    ['As', 'Ad'],
                'board_cards':   ['Ah', 'Kd', '7c', '3s', '2d'],
                'street': 3,
                'my_chips': 25.0,
                'all_chips': [25.0, 25.0] + [0.0] * (np_ - 2),
                'pot': 50.0, 'call_amount': 0.0,
                'bb': 2.0, 'sb': 1.0, 'stack': 75.0,
                'legal_actions': [0, 1, 2, 3, 4, 5, 6],
            },
        ]


def run_scenario_generator_test(model, ob, tr, he, np_, dev, ee, tl):
    """
    SPRINT 3 – NEW-5: ScenarioGenerator futtatása és kiértékelése.

    A 3 dimenzió mindegyike informális (nem épül bele a penalty-be),
    de részletes logot és összesítő metrikát ad.
    """
    tl.log(f"\n{'─'*65}")
    tl.log(f"  SCENARIO GENERATOR TESZT (SPRINT3 – ÚJ)")
    tl.log(f"{'─'*65}\n")

    mp  = PokerActionMapper()
    gen = ScenarioGenerator()

    # ── (a) STACK DEPTH SWEEP ────────────────────────────────────────────────
    tl.log(f"  [a] Stack depth sweep – Top pair KsJd, 4 stack mélység\n")
    stack_scenarios = gen.stack_depth_sweep(np_)
    stack_results   = []
    for sc in stack_scenarios:
        probs, best, val, eq = query_model(model, ob, tr, he, sc, np_, dev, ee)
        raise_pct = sum(probs[a] for a in range(2, 7)) * 100
        allin_pct = probs[6] * 100
        wt        = sum(probs[a] * a for a in range(7))
        tl.log(
            f"  {sc['stack_bb']:3d}BB  →  {mp.action_name(best):18s} "
            f"raise%={raise_pct:.0f}%  allin%={allin_pct:.0f}%  "
            f"tier={wt:.2f}"
        )
        stack_results.append({
            'stack_bb': sc['stack_bb'],
            'best': best, 'raise_pct': raise_pct,
            'allin_pct': allin_pct, 'weighted_tier': wt,
        })

    # Diagnózis: 10BB-nél lényegesen több all-in mint 100BB-nél?
    r10  = next(r for r in stack_results if r['stack_bb'] == 10)
    r100 = next(r for r in stack_results if r['stack_bb'] == 100)
    stack_aware = r10['allin_pct'] > r100['allin_pct'] + 10.0
    tl.log(f"\n  Stack-tudatosság: 10BB allin%={r10['allin_pct']:.0f}% vs "
           f"100BB allin%={r100['allin_pct']:.0f}%  "
           f"→ {'✅ felismeri a short-stack zónát' if stack_aware else '⚠ azonos döntés (stack-vak?)'}")

    # ── (b) BOARD TEXTURE SWEEP ──────────────────────────────────────────────
    tl.log(f"\n  [b] Board texture sweep – KcJd top pair, 4 textúra\n")
    texture_scenarios = gen.board_texture_sweep(np_)
    texture_results   = []
    for sc in texture_scenarios:
        probs, best, val, eq = query_model(model, ob, tr, he, sc, np_, dev, ee)
        bet_pct = sum(probs[a] for a in range(2, 7)) * 100
        tl.log(
            f"  {sc['texture']:10s}  →  {mp.action_name(best):18s} "
            f"bet%={bet_pct:.0f}%  eq={eq*100:.0f}%"
        )
        texture_results.append({
            'texture': sc['texture'], 'name': sc['name'],
            'best': best, 'bet_pct': bet_pct,
        })

    # Diagnózis: dry-nél nagyobb bet% mint monotone-nál?
    dry_bet      = next(r['bet_pct'] for r in texture_results if r['texture'] == 'dry')
    monotone_bet = next(r['bet_pct'] for r in texture_results if r['texture'] == 'monotone')
    texture_aware = dry_bet > monotone_bet + 5.0
    tl.log(f"\n  Texture-tudatosság: dry={dry_bet:.0f}% vs monotone={monotone_bet:.0f}%  "
           f"→ {'✅ veszélyesebb boardon visszafog' if texture_aware else '⚠ azonos bet% (texture-vak?)'}")

    # ── (c) RIVER SPOTS ──────────────────────────────────────────────────────
    tl.log(f"\n  [c] River-specifikus szituációk (4 spot)\n")
    river_scenarios = gen.river_spots(np_)
    river_results   = []
    river_passed    = 0
    for sc in river_scenarios:
        probs, best, val, eq = query_model(model, ob, tr, he, sc, np_, dev, ee)
        is_good = best in sc['expected_good']
        is_bad  = best in sc.get('expected_bad', set())
        if is_bad:   st = '⚠'
        elif is_good: st = '✅'; river_passed += 1
        else:         st = '⚠'
        tl.log(f"  {st} {sc['name']}")
        tl.log(f"     → {mp.action_name(best)} ({probs[best]*100:.0f}%)  "
               f"val={val:+.3f}  eq={eq*100:.0f}%")
        river_results.append({
            'name': sc['name'], 'best': best,
            'passed': is_good and not is_bad,
            'value': val, 'category': sc['category'],
        })

    tl.log(f"\n  River szituációk: {river_passed}/{len(river_scenarios)}")
    if river_passed == len(river_scenarios):
        tl.log(f"  ✅ River döntések rendben")
    elif river_passed >= 2:
        tl.log(f"  ⚠ River döntések részben helyes ({river_passed}/4)")
    else:
        tl.log(f"  ❌ River döntések gyengék – tréning fókusz javasolt")

    return {
        'stack_aware':    stack_aware,
        'texture_aware':  texture_aware,
        'river_passed':   river_passed,
        'river_total':    len(river_scenarios),
        'stack_results':  stack_results,
        'texture_results': texture_results,
        'river_results':  river_results,
    }




def run_poker_stats(model, ob, tr, he, np_, dev, ee, n_hands, tl):
    """
    VPIP / PFR / AF / 3-bet% / C-bet% mérés és kiértékelés.

    [TASK-16] Teljesen átdolgozva:
      1. TABLE_STAT_TARGETS alapján per-player-count célsávok
      2. Multiway szituációk (3+ játékosnál a 'coldcall' és 'multiway_post'
         típusú kezek hozzáadva) – realisztikusabb stat-gyűjtés
      3. Részletesebb kimeneti log (ideális sáv, eltérés, kategória)
    """
    tl.log(f"\n{'─'*65}")
    cat = _table_category(np_)
    tl.log(f"  POKER STATISZTIKÁK – {cat} ({n_hands} random kéz)")
    tl.log(f"{'─'*65}\n")

    mp = PokerActionMapper()
    bb = 2.0; sb = 1.0
    ac = collections.Counter()

    # Stat számlálók
    vo=0; va=0   # VPIP: opportunities / voluntary
    po=0; pa=0   # PFR: open opportunities / raises
    to=0; ta=0   # 3-bet: facing-open opportunities / 3-bets
    pb=0; pc=0   # AF: postflop bets / calls
    co=0; ca=0   # C-bet: cbet opportunities / cbets
    np2=0; fp=0  # Premium fold counter

    # Szituáció típusok – np_-től függő eloszlás
    # HU: csak open/facing_raise/facing_3bet/postflop
    # 3-5max: + coldcall (CO vs BTN open)
    # 6+: + coldcall + multiway_post
    if np_ == 2:
        sit_pool = ['open']*4 + ['facing_raise']*3 + ['facing_3bet'] + ['postflop']*2
    elif np_ <= 5:
        sit_pool = ['open']*3 + ['facing_raise']*2 + ['facing_3bet'] + ['coldcall'] + ['postflop']*3
    else:
        sit_pool = ['open']*2 + ['facing_raise']*2 + ['facing_3bet'] + ['coldcall'] + ['multiway_post'] + ['postflop']*3

    for _ in range(n_hands):
        deck = list(ALL_CARDS); random.shuffle(deck)
        hole = [deck[0], deck[1]]
        sit  = random.choice(sit_pool)

        if sit == 'open':
            # Hero on BTN/SB open
            sc = {
                'hole_cards': hole, 'board_cards': [], 'street': 0,
                'my_chips': 1.0,
                'all_chips': [1.0, 2.0] + [0.0]*(np_-2),
                'pot': 3.0, 'call_amount': 1.0,
                'bb': bb, 'sb': sb, 'stack': 100.0,
                'button_pos': 0, 'my_player_id': 0,
                'legal_actions': [0,1,2,3,4,5,6],
            }
            _, best, _, _ = query_model(model, ob, tr, he, sc, np_, dev, ee)
            vo += 1; po += 1
            if best >= 1: va += 1
            if best >= 2: pa += 1

        elif sit == 'facing_raise':
            # Hero BB facing BTN open
            sc = {
                'hole_cards': hole, 'board_cards': [], 'street': 0,
                'my_chips': 2.0,
                'all_chips': [6.0, 2.0] + [0.0]*(np_-2),
                'pot': 8.0, 'call_amount': 4.0,
                'bb': bb, 'sb': sb, 'stack': 100.0,
                'button_pos': 1, 'my_player_id': 0,
                'legal_actions': [0,1,2,3,4,5,6],
            }
            _, best, _, _ = query_model(model, ob, tr, he, sc, np_, dev, ee)
            vo += 1; to += 1
            if best >= 1: va += 1
            if best >= 2: ta += 1

        elif sit == 'facing_3bet':
            # Hero BTN facing 3-bet from BB
            sc = {
                'hole_cards': hole, 'board_cards': [], 'street': 0,
                'my_chips': 6.0,
                'all_chips': [18.0, 6.0] + [0.0]*(np_-2),
                'pot': 24.0, 'call_amount': 12.0,
                'bb': bb, 'sb': sb, 'stack': 100.0,
                'button_pos': 0, 'my_player_id': 0,
                'legal_actions': [0,1,2,3,4,5,6],
            }
            _, best, _, _ = query_model(model, ob, tr, he, sc, np_, dev, ee)
            vo += 1
            if best >= 1: va += 1

        elif sit == 'coldcall':
            # CO facing BTN open (cold call scenario – 3-5max és 6+max)
            co_chips = random.choice([2.0, 4.0])  # limp or raise
            sc = {
                'hole_cards': hole, 'board_cards': [], 'street': 0,
                'my_chips': co_chips,
                'all_chips': [6.0, co_chips, 2.0] + [0.0]*(np_-3),
                'pot': 6.0 + co_chips + 2.0, 'call_amount': 6.0 - co_chips,
                'bb': bb, 'sb': sb, 'stack': 100.0,
                'button_pos': 0, 'my_player_id': 2,
                'legal_actions': [0,1,2,3,4,5,6],
            }
            _, best, _, _ = query_model(model, ob, tr, he, sc, np_, dev, ee)
            vo += 1
            if best >= 1: va += 1

        elif sit == 'multiway_post':
            # Multiway flop (BTN + CO + BB): realisztikus 6+ játékosnál
            nb = random.choice([3])
            board = deck[2:5]
            mc  = random.uniform(4, 15)
            oc1 = random.uniform(4, 15)
            oc2 = random.uniform(0, mc)
            call = random.uniform(0, 10) if random.random() > 0.4 else 0.0
            sc = {
                'hole_cards': hole, 'board_cards': board, 'street': 1,
                'my_chips': mc,
                'all_chips': [oc1, oc2, mc] + [0.0]*(np_-3),
                'pot': oc1 + oc2 + mc, 'call_amount': call,
                'bb': bb, 'sb': sb, 'stack': 100.0 - mc,
                'legal_actions': [0,1,2,3,4,5,6],
            }
            _, best, _, _ = query_model(model, ob, tr, he, sc, np_, dev, ee)
            if best >= 2: pb += 1
            elif best == 1 and call > 0: pc += 1
            if call < 0.01: co += 1; ca += (1 if best >= 2 else 0)

        else:  # postflop
            nb  = random.choice([3,4,5]); board = deck[2:2+nb]
            st  = 1 if nb == 3 else (2 if nb == 4 else 3)
            mc  = random.uniform(5, 30); oc = random.uniform(5, 30)
            call = random.uniform(0, 15) if random.random() > 0.4 else 0.0
            sc = {
                'hole_cards': hole, 'board_cards': board, 'street': st,
                'my_chips': mc,
                'all_chips': [oc, mc] + [0.0]*(np_-2),
                'pot': mc + oc, 'call_amount': call,
                'bb': bb, 'sb': sb, 'stack': 100.0 - mc,
                'legal_actions': [0,1,2,3,4,5,6],
            }
            _, best, _, _ = query_model(model, ob, tr, he, sc, np_, dev, ee)
            if best >= 2: pb += 1
            elif best == 1 and call > 0: pc += 1
            if call < 0.01: co += 1; ca += (1 if best >= 2 else 0)

        ac[best] += 1

        # Prémium fold tracker
        r1, r2 = hole[0][0], hole[1][0]
        if ((r1 == r2 and r1 in 'AKQJ')
                or ('A' in (r1, r2) and set((r1, r2)) & set('KQJ'))):
            np2 += 1
            if best == 0: fp += 1

    total = sum(ac.values())
    tl.log(f"  Akció eloszlás ({total} kéz):")
    for a in range(7):
        c = ac.get(a, 0); p = c / max(total, 1) * 100
        tl.log(f"    {mp.action_name(a):18s}: {c:5d} ({p:5.1f}%) {'█'*int(p/2)}")

    vpip = va / max(vo, 1) * 100
    pfr  = pa / max(po, 1) * 100
    af   = pb / max(pc, 1)
    tbet = ta / max(to, 1) * 100
    cbet = ca / max(co, 1) * 100

    # ── Per-player-count célsávok ──────────────────────────────────────────
    tgt = TABLE_STAT_TARGETS[min(np_, 9)]
    vlo, vhi = tgt['vpip'];  vok  = _stat_ok(vpip, vlo, vhi)
    plo, phi = tgt['pfr'];   pok  = _stat_ok(pfr,  plo, phi)
    alo, ahi = tgt['af'];    aok  = _stat_ok(af,   alo, ahi)
    tlo, thi = tgt['tbet'];  tok  = _stat_ok(tbet, tlo, thi)
    clo, chi = tgt['cbet'];  cok  = _stat_ok(cbet, clo, chi)

    tl.log(f"\n  ── Statisztikák [{cat}] ──")
    tl.log(f"  {'Mutató':<10} {'Érték':>7}   {'Ideális sáv':>13}   Státusz")
    tl.log(f"  {'─'*50}")
    tl.log(f"  {'VPIP':<10} {vpip:>6.1f}%   [{vlo:3d}–{vhi:3d}%]       {'✅' if vok else '⚠'}")
    tl.log(f"  {'PFR':<10} {pfr:>6.1f}%   [{plo:3d}–{phi:3d}%]       {'✅' if pok else '⚠'}")
    tl.log(f"  {'AF':<10} {af:>6.2f}    [{alo:.1f}–{ahi:.1f}]        {'✅' if aok else '⚠'}")
    tl.log(f"  {'3-bet%':<10} {tbet:>6.1f}%   [{tlo:3d}–{thi:3d}%]       {'✅' if tok else '⚠'}")
    tl.log(f"  {'C-bet%':<10} {cbet:>6.1f}%   [{clo:3d}–{chi:3d}%]       {'✅' if cok else '⚠'}")

    if np2 > 0:
        fpp = fp / np2 * 100
        tl.log(f"\n  Prémium kéz fold: {fp}/{np2} ({fpp:.0f}%) "
               f"{'✅' if fpp<=5 else '⚠' if fpp<=15 else '❌'}")

    degen = []
    fpct   = ac.get(0,0) / max(total,1) * 100
    aipct  = ac.get(6,0) / max(total,1) * 100
    mx     = max(ac.values()) / max(total,1) * 100 if ac else 0
    if mx > 80:   degen.append(f"❌ Degenerált: {mp.action_name(max(ac,key=ac.get))} {mx:.0f}%")
    if fpct > 70: degen.append(f"❌ Passzív: fold {fpct:.0f}%")
    if aipct > 40: degen.append(f"❌ AllIn spam: {aipct:.0f}%")
    for d in degen: tl.log(f"  {d}")

    return {
        'vpip': vpip, 'pfr': pfr, 'af': af,
        'three_bet': tbet, 'cbet': cbet,
        'fold_pct': fpct, 'allin_pct': aipct,
        'premium_fold': fp / max(np2,1) * 100,
        'degeneration': degen,
        'vpip_ok': vok, 'pfr_ok': pok, 'af_ok': aok,
        'tbet_ok': tok, 'cbet_ok': cok,
    }


def compare_models(model_paths, np_, device, n_hands, seed, do_winrate, wr_hands):
    print(f"\n{'='*75}")
    print(f"  🔄  MODELL ÖSSZEHASONLÍTÁS ({len(model_paths)} modell)")
    print(f"{'='*75}\n")

    all_results = []
    for path in model_paths:
        print(f"  ── {os.path.basename(path)} ──")
        random.seed(seed); np.random.seed(seed)
        r = run_single_model(path, np_, device, n_hands, seed,
                             verbose=False, do_winrate=do_winrate,
                             wr_hands=wr_hands, quiet=True)
        if r: all_results.append(r)

    if len(all_results) < 2:
        print("  ⚠ Legalább 2 modell kell az összehasonlításhoz")
        return

    print(f"\n{'='*75}")
    print(f"  ÖSSZEHASONLÍTÁS")
    print(f"{'='*75}\n")

    header = f"  {'Metrika':<22s}"
    for r in all_results:
        name = os.path.basename(r['model'])[:15]
        header += f" | {name:>15s}"
    print(header)
    print(f"  {'─'*22}" + "─┼─".join(['─'*15]*len(all_results)))

    rows = [
        ('Epizódok',        lambda r: f"{r.get('episodes',0):>12,}"),
        ('Szituáció PASS',  lambda r: f"{r['s_passed']}/{r['s_passed']+r['s_warn']+r['s_fail']}"),
        ('Pozíció tudatos', lambda r: f"{r['pos_aware']}/{r['pos_total']}"),
        ('Pozíció delta',   lambda r: f"{r.get('pos_avg_delta',0)*100:+.1f}%"),
        ('Exploit sikeres', lambda r: f"{r['exp_pass']}/{r['exp_total']}"),
        ('Draw awareness',  lambda r: f"{r['draw_passed']}/{r['draw_total']}"),
        ('Draw fold%',      lambda r: f"{r['draw_fold_pct']:.0f}%"),
        ('Sizing preflop r',lambda r: f"{r['sizing_pf']:.2f}"),
        ('Sizing postflop r',lambda r:f"{r['sizing_pp']:.2f}"),
        ('Entropy átlag',   lambda r: f"{r['avg_entropy']:.2f}"),
        ('Value calib',     lambda r: f"{r.get('valcal_passed','?')}/{r.get('valcal_total','?')}"),
        ('River spots',     lambda r: f"{r.get('scgen_river','?')}/4"),
        ('Stack aware',     lambda r: '✅' if r.get('scgen_stack') else '⚠'),
        ('Texture aware',   lambda r: '✅' if r.get('scgen_texture') else '⚠'),
        ('Penalty összeg',  lambda r: str(r.get('penalty','?'))),
        ('VPIP',            lambda r: f"{r['vpip']:.0f}%"),
        ('PFR',             lambda r: f"{r['pfr']:.0f}%"),
        ('AF',              lambda r: f"{r['af']:.2f}"),
        ('Prémium fold%',   lambda r: f"{r['prem_fold']:.0f}%"),
        ('Értékelés',       lambda r: r['grade']),
    ]
    if do_winrate:
        rows.append(('BB/100 self',   lambda r: f"{r.get('sp_bb100',0):+.1f}"))
        rows.append(('  CI self',     lambda r:
            (f"[{r['sp_ci_low']:+.1f},{r['sp_ci_high']:+.1f}]"
             if r.get('sp_ci_low') is not None else '–')))
        rows.append(('BB/100 random', lambda r: f"{r.get('rnd_bb100',0):+.1f}"))
        rows.append(('  CI random',   lambda r:
            (f"[{r['rnd_ci_low']:+.1f},{r['rnd_ci_high']:+.1f}]"
             if r.get('rnd_ci_low') is not None else '–')))
        rows.append(('  Szignifikáns',lambda r:
            '✅ igen' if r.get('rnd_significant') else '⚠ nem'))

    for label, fn in rows:
        line = f"  {label:<22s}"
        for r in all_results:
            try: val = fn(r)
            except: val = 'n/a'
            line += f" | {val:>15s}"
        print(line)

    print(f"\n{'='*75}\n")


def run_single_model(model_path, np_, device_str, n_hands, seed,
                     verbose=False, do_winrate=False, wr_hands=2000,
                     quiet=False, out_dir="logs"):
    """Egyetlen modell teljes tesztelése, visszaadja az összesített dict-et."""
    device = torch.device(device_str)
    try:
        ck = safe_load_checkpoint(model_path, map_location=device)
    except Exception as e:
        print(f"  ❌ Nem tölthető: {model_path}: {e}")
        return None
    if not (isinstance(ck, dict) and 'state_dict' in ck):
        print(f"  ❌ Érvénytelen: {model_path}"); return None

    state_size=ck.get('state_size',475); action_size=ck.get('action_size',7)
    episodes=ck.get('episodes_trained',0)

    if np_ is None:
        np_ = None
        for n in range(2,10):
            if compute_state_size(54,n)==state_size: np_=n; break
    if np_ is None:
        print(f"  ⚠ state_size={state_size}"); return None

    tl = TestLogger(model_path, np_, n_hands, out_dir=out_dir)
    if quiet:
        tl.log = lambda text, console=False: tl._file.write(text+'\n')

    tl.log(f"\n{'='*65}")
    cat_label = _table_category(np_)
    tl.log(f"  🧪  POKER AI VIZSGÁZTATÓ v4.0 [TASK-16: {cat_label}]")
    tl.log(f"{'='*65}")
    tl.log(f"  Modell: {model_path} ({episodes:,} ep)")
    tl.log(f"  {np_}p | {n_hands:,} kéz | seed={seed}")

    model = AdvancedPokerAI(state_size=state_size, action_size=action_size).to(device)
    model.load_state_dict(ck['state_dict']); model.eval()
    ob=ObsBuilder(np_); tr=OpponentHUDTracker(np_)
    he=ActionHistoryEncoder(np_,action_size); ee=HandEquityEstimator(n_sim=200)

    p,w,f  = run_scenarios(model,ob,tr,he,np_,device,ee,tl,verbose)
    pos    = run_position_test(model,ob,tr,he,np_,device,ee,tl)
    exp    = run_exploit_test(model,ob,he,np_,device,ee,tl)
    draw   = run_draw_test(model,ob,tr,he,np_,device,ee,tl)
    sizing = run_sizing_test(model,ob,tr,he,np_,device,ee,tl)
    consist= run_consistency_test(model,ob,tr,he,np_,device,ee,tl)
    valcal = run_value_calibration_test(model,ob,tr,he,np_,device,ee,tl)
    scgen  = run_scenario_generator_test(model,ob,tr,he,np_,device,ee,tl)
    stats  = run_poker_stats(model,ob,tr,he,np_,device,ee,n_hands,tl)

    wr = {}
    if do_winrate:
        wr = run_winrate_test(model,np_,device,wr_hands,tl)

    # ── SPRINT 2: Penalty-alapú grade formula ────────────────────────────────
    # Minden súlyos hiányosság +1 penalty pontot kap.
    # A végső grade a penalty összegén alapszik.
    #
    # RÉGI logika: grade = f(failed_count + degeneration)
    # ÚJ logika:   minden dimenzió bekerül egy-egy penalty-ként:
    #
    #   +1 ha situational CRITICAL fail > 0      (❌ szituáció)
    #   +1 ha draw fold% > 30%                   (draw-kezelés hiba)
    #   +1 ha avg entropy > 1.3                  (döntési bizonytalanság)
    #   +1 ha pozíció-tudatos < 2/5              (pozíció-vak)
    #   +1 ha exploit score < 3/7                (HUD-vakság)
    #   +1 ha value kalibráció < 3 passed        (critic nem tanult EV-t)
    #   +1 per degeneration flag                 (VPIP/PFR/AF szélsőség)
    #   +1 ha VPIP/PFR/3bet mindhárom rossz        ([TASK-16] per-player-count)
    #
    # Grade skála:
    #   0 penalty → 🟢 JÓ
    #   1 penalty → 🟡 ELFOGADHATÓ
    #   2 penalty → 🟠 PROBLÉMÁS
    #   3+ penalty → 🔴 KOMOLY HIBÁK

    pa           = sum(1 for r in pos if r['position_aware'])
    pos_avg_delta= sum(r['delta'] for r in pos) / len(pos) if pos else 0.0
    ep           = sum(1 for r in exp if r['passed'])
    exp_total    = len(exp)

    penalty = 0
    penalty_log = []

    if f > 0:
        penalty += 1
        penalty_log.append(f"szituáció FAIL={f}")
    if draw['draw_fold_pct'] > 30:
        penalty += 1
        penalty_log.append(f"draw fold%={draw['draw_fold_pct']:.0f}>30")
    if consist['avg_entropy'] > 1.3:
        penalty += 1
        penalty_log.append(f"entropy={consist['avg_entropy']:.2f}>1.3")
    if pa < 2:
        penalty += 1
        penalty_log.append(f"pozíció-tudatos={pa}/5<2")
    if ep < 3:
        penalty += 1
        penalty_log.append(f"exploit={ep}/{exp_total}<3")
    if valcal['passed'] < 3:
        penalty += 1
        penalty_log.append(f"value_cal={valcal['passed']}/{valcal['total']}<3")
    for degen in stats.get('degeneration', []):
        penalty += 1
        penalty_log.append(f"degen:{degen[:20]}")

    # [TASK-16] Per-player-count stat check: VPIP + PFR + 3-bet mind rossz → penalty
    stat_failures = sum([
        0 if stats.get('vpip_ok', True) else 1,
        0 if stats.get('pfr_ok',  True) else 1,
        0 if stats.get('tbet_ok', True) else 1,
    ])
    if stat_failures >= 3:
        penalty += 1
        penalty_log.append(f"stats[{_table_category(np_)}]: VPIP+PFR+3bet mind rossz")

    if   penalty == 0: g = '🟢 JÓ'
    elif penalty == 1: g = '🟡 ELFOGADHATÓ'
    elif penalty == 2: g = '🟠 PROBLÉMÁS'
    else:              g = '🔴 KOMOLY HIBÁK'

    tl.log(f"\n{'='*65}")
    tl.log(f"  ÖSSZEFOGLALÓ")
    tl.log(f"{'='*65}")
    tl.log(f"  Szituációs:     {p} ✅  {w} ⚠  {f} ❌")
    tl.log(f"  Pozíció:        {pa}/{len(pos)} (avg delta: {pos_avg_delta*100:+.1f}%)")
    tl.log(f"  Exploit:        {ep}/{exp_total}")
    tl.log(f"  Draw awareness: {draw['passed']}/{draw['total']} (fold%={draw['draw_fold_pct']:.0f})")
    tl.log(f"  Sizing:         pf r={sizing['preflop_corr']:.2f} pp r={sizing['postflop_corr']:.2f}")
    tl.log(f"  Entropy:        {consist['avg_entropy']:.2f}")
    tl.log(f"  Value calib:    {valcal['passed']}/{valcal['total']}")
    tl.log(f"  ScenGen river:  {scgen['river_passed']}/{scgen['river_total']} "
           f"| stack={'✅' if scgen['stack_aware'] else '⚠'} "
           f"| texture={'✅' if scgen['texture_aware'] else '⚠'}")
    tl.log(f"  Stat [{_table_category(np_)}]:  "
           f"VPIP:{stats['vpip']:.0f}%{'✅' if stats.get('vpip_ok') else '⚠'} "
           f"PFR:{stats['pfr']:.0f}%{'✅' if stats.get('pfr_ok') else '⚠'} "
           f"AF:{stats['af']:.2f}{'✅' if stats.get('af_ok') else '⚠'} "
           f"3b:{stats['three_bet']:.0f}%{'✅' if stats.get('tbet_ok') else '⚠'} "
           f"Cb:{stats['cbet']:.0f}%{'✅' if stats.get('cbet_ok') else '⚠'}")
    if wr:
        sp_ci  = wr.get('self_play_ci_low')
        rnd_ci = wr.get('vs_random_ci_low')
        sp_str  = (f"[{wr['self_play_ci_low']:+.1f}, {wr['self_play_ci_high']:+.1f}]"
                   if sp_ci is not None else "–")
        rnd_str = (f"[{wr['vs_random_ci_low']:+.1f}, {wr['vs_random_ci_high']:+.1f}]"
                   if rnd_ci is not None else "–")
        tl.log(f"  BB/100 self:    {wr.get('self_play_bb100',0):+.1f}  CI:{sp_str}")
        tl.log(f"  BB/100 random:  {wr.get('vs_random_bb100',0):+.1f}  CI:{rnd_str}")
    tl.log(f"\n  Penalty összeg: {penalty}"
           + (f"  [{', '.join(penalty_log)}]" if penalty_log else ""))
    tl.log(f"  ÉRTÉKELÉS: {g}  ({episodes:,} ep)")

    tl.results['summary'] = {
        'grade': g, 'passed': p, 'warnings': w, 'failed': f,
        'penalty': penalty, 'penalty_reasons': penalty_log,
    }
    lp, jp = tl.close()
    if not quiet:
        print(f"\n  📄 Log:  {lp}")
        print(f"  📊 JSON: {jp}")
        print(f"{'='*65}\n")

    return {
        'model':model_path, 'episodes':episodes, 'grade':g,
        's_passed':p, 's_warn':w, 's_fail':f,
        'pos_aware':pa, 'pos_total':len(pos), 'pos_avg_delta': pos_avg_delta,
        'exp_pass':ep, 'exp_total':exp_total,
        'draw_passed':draw['passed'], 'draw_total':draw['total'],
        'draw_fold_pct':draw['draw_fold_pct'],
        'sizing_pf':sizing['preflop_corr'], 'sizing_pp':sizing['postflop_corr'],
        'avg_entropy':consist['avg_entropy'],
        'valcal_passed':valcal['passed'], 'valcal_total':valcal['total'],
        'penalty':penalty,
        'scgen_river':   scgen['river_passed'],
        'scgen_stack':   scgen['stack_aware'],
        'scgen_texture': scgen['texture_aware'],
        'vpip':stats['vpip'], 'pfr':stats['pfr'], 'af':stats['af'],
        'prem_fold':stats['premium_fold'],
        'sp_bb100':   wr.get('self_play_bb100', 0),
        'sp_ci_low':  wr.get('self_play_ci_low'),
        'sp_ci_high': wr.get('self_play_ci_high'),
        'rnd_bb100':   wr.get('vs_random_bb100', 0),
        'rnd_ci_low':  wr.get('vs_random_ci_low'),
        'rnd_ci_high': wr.get('vs_random_ci_high'),
        'rnd_significant': wr.get('vs_random_significant', False),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Poker AI Vizsgáztató v3.3 SPRINT3')
    parser.add_argument('model_path', nargs='?', default=None, help='.pth modell fájl')
    parser.add_argument('--num-players', type=int, default=None)
    parser.add_argument('--hands', type=int, default=1000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--winrate', action='store_true',
                        help='BB/100 mérés (lassú)')
    parser.add_argument('--winrate-hands', type=int, default=2000)
    parser.add_argument('--compare', nargs='+', metavar='MODEL',
                        help='Több modell összehasonlítása')
    parser.add_argument('--out-dir', default='logs',
                        help='Log és JSON mentési mappa (default: logs/). '
                             'Automatikus tesztelésnél a runner állítja be.')
    args = parser.parse_args()

    if args.compare:
        compare_models(args.compare, args.num_players, args.device,
                       args.hands, args.seed, args.winrate, args.winrate_hands)
        return

    if args.model_path is None:
        parser.print_help(); return

    random.seed(args.seed); np.random.seed(args.seed)
    run_single_model(args.model_path, args.num_players, args.device,
                     args.hands, args.seed, args.verbose,
                     args.winrate, args.winrate_hands, quiet=False,
                     out_dir=args.out_dir)


if __name__ == '__main__':
    main()
