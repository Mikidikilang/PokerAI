#!/usr/bin/env python3
"""
test_model_sanity.py  v3 FINAL –  Poker AI Vizsgáztató Központ

TESZTEK:
  1. Szituációs (15 eset: preflop, postflop, 3bet, short stack, board texture)
  2. Pozíciós tudatosság (BTN vs BB, 5 kéz)
  3. HUD exploitáció (nit blöff, maniac trap, adaptáció)
  4. Draw equity awareness (8 szituáció, out-kezelés)
  5. Bet sizing analízis (Spearman korreláció kézerő vs raise méret)
  6. Konzisztencia / entropy (döntési magabiztosság)
  7. Poker statok (VPIP, PFR, AF, 3-bet%, C-bet%)
  8. BB/100 winrate (self-play + random ellenfél) [--winrate]
  9. Modell összehasonlítás [--compare A.pth B.pth ...]

Használat:
    python test_model_sanity.py 2max_ppo_v4.pth
    python test_model_sanity.py 2max_ppo_v4.pth --hands 5000 --verbose
    python test_model_sanity.py 2max_ppo_v4.pth --winrate --winrate-hands 5000
    python test_model_sanity.py 2max_ppo_v4.pth --out-dir ModellNaplo/2max_ppo_v4_4M
    python test_model_sanity.py --compare 2max_5M.pth 2max_10M.pth 2max_20M.pth
"""

import sys, os, argparse, collections, random, json, math, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, numpy as np
from core.model import AdvancedPokerAI
from core.action_mapper import PokerActionMapper
from core.features import (ActionHistoryEncoder, build_state_tensor,
                            compute_state_size, ACTION_HISTORY_LEN)
from core.opponent_tracker import OpponentHUDTracker
from core.equity import HandEquityEstimator
from inference.obs_builder import ObsBuilder

RANKS = 'AKQJT98765432'
SUITS = 'shdc'
ALL_CARDS = [r + s for r in RANKS for s in SUITS]


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
    bb=2.0; sb=1.0; stk=100.0; S=[]
    def sc(**kw):
        kw.setdefault('bb',bb); kw.setdefault('sb',sb); kw.setdefault('stack',stk)
        kw.setdefault('legal_actions',[0,1,2,3,4,5,6])
        kw.setdefault('all_chips',[kw.get('my_chips',2.0)]*np_)
        S.append(kw)
    sc(name='AA preflop – raise',hole_cards=['As','Ah'],board_cards=[],street=0,
       my_chips=2.0,pot=3.0,call_amount=0.0,expected_good={2,3,4,5,6},
       expected_bad={0},severity='CRITICAL',category='preflop')
    sc(name='KK vs open',hole_cards=['Ks','Kh'],board_cards=[],street=0,
       my_chips=2.0,all_chips=[6.0,2.0]+[0.0]*(np_-2),pot=9.0,call_amount=4.0,
       expected_good={1,2,3,4,5,6},expected_bad={0},severity='CRITICAL',category='preflop')
    sc(name='AKs preflop',hole_cards=['As','Ks'],board_cards=[],street=0,
       my_chips=1.0,all_chips=[1.0,2.0]+[0.0]*(np_-2),pot=3.0,call_amount=1.0,
       expected_good={1,2,3,4,5,6},expected_bad={0},severity='CRITICAL',category='preflop')
    sc(name='72o facing raise',hole_cards=['7s','2h'],board_cards=[],street=0,
       my_chips=2.0,all_chips=[6.0,2.0]+[0.0]*(np_-2),pot=9.0,call_amount=4.0,
       expected_good={0,1},expected_bad={5,6},severity='WARNING',category='preflop')
    sc(name='QQ facing 3-bet',hole_cards=['Qs','Qh'],board_cards=[],street=0,
       my_chips=6.0,all_chips=[18.0,6.0]+[0.0]*(np_-2),pot=25.0,call_amount=12.0,
       expected_good={1,4,5,6},expected_bad={0},severity='CRITICAL',category='3bet')
    sc(name='T8o facing 4-bet',hole_cards=['Th','8d'],board_cards=[],street=0,
       my_chips=18.0,all_chips=[45.0,18.0]+[0.0]*(np_-2),pot=64.0,call_amount=27.0,
       expected_good={0},expected_bad={5,6},severity='WARNING',category='3bet')
    sc(name='Nut flush flop',hole_cards=['As','Ks'],board_cards=['Qs','7s','3s'],
       street=1,my_chips=10.0,all_chips=[10.0,10.0]+[0.0]*(np_-2),
       pot=20.0,call_amount=0.0,stack=90.0,
       expected_good={2,3,4,5,6},expected_bad={0},severity='CRITICAL',category='postflop')
    sc(name='Top set dry flop',hole_cards=['Ah','Ad'],board_cards=['Ac','7d','2s'],
       street=1,my_chips=8.0,all_chips=[8.0,8.0]+[0.0]*(np_-2),
       pot=16.0,call_amount=0.0,stack=92.0,
       expected_good={1,2,3,4,5,6},expected_bad={0},severity='CRITICAL',category='postflop')
    sc(name='Combo draw (15 out) flop',hole_cards=['Jh','Th'],
       board_cards=['9h','8d','2h'],street=1,my_chips=6.0,
       all_chips=[12.0,6.0]+[0.0]*(np_-2),pot=18.0,call_amount=6.0,stack=94.0,
       expected_good={1,4,5,6},expected_bad={0},severity='CRITICAL',category='semi_bluff')
    sc(name='TPTK monotone board',hole_cards=['As','Kc'],
       board_cards=['Jh','9h','8h'],street=1,my_chips=10.0,
       all_chips=[30.0,10.0]+[0.0]*(np_-2),pot=40.0,call_amount=20.0,stack=90.0,
       expected_good={0,1},expected_bad={5,6},severity='WARNING',category='board_texture')
    sc(name='10BB A8o BTN shove',hole_cards=['Ah','8d'],board_cards=[],street=0,
       my_chips=1.0,all_chips=[1.0,2.0]+[0.0]*(np_-2),pot=3.0,call_amount=1.0,
       stack=20.0,button_pos=0,my_player_id=0,
       expected_good={5,6},expected_bad={0},severity='WARNING',category='short_stack')
    sc(name='8BB 22 push/fold',hole_cards=['2s','2h'],board_cards=[],street=0,
       my_chips=1.0,all_chips=[1.0,2.0]+[0.0]*(np_-2),pot=3.0,call_amount=1.0,
       stack=16.0,button_pos=0,my_player_id=0,
       expected_good={0,5,6},expected_bad={2,3},severity='WARNING',category='short_stack')
    sc(name='20:1 pot odds',hole_cards=['5h','4d'],
       board_cards=['As','Ks','Qd','7c'],street=2,my_chips=20.0,
       all_chips=[22.0,20.0]+[0.0]*(np_-2),pot=40.0,call_amount=2.0,stack=80.0,
       expected_good={1,2,3,4,5,6},expected_bad={0},severity='WARNING',category='pot_odds')
    sc(name='Air facing river pot bet',hole_cards=['9h','8h'],
       board_cards=['As','Kd','3c','5s','Jd'],street=3,my_chips=30.0,
       all_chips=[60.0,30.0]+[0.0]*(np_-2),pot=60.0,call_amount=30.0,stack=70.0,
       expected_good={0},expected_bad={5,6},severity='WARNING',category='pot_odds')
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
        if is_bad: tl.log(f"       ⚠ {sc.get('description','')}")
        if verbose: tl.log(f"       {[f'{p:.3f}' for p in probs]}")
        tl.results['scenarios'].append({'name':sc['name'],'status':st,
            'action':mp.action_name(best),'confidence':float(probs[best])})
    tl.log(f"\n  Eredmény: {passed} ✅  {warnings} ⚠  {failed} ❌")
    return passed, warnings, failed


# ═══════════════════════════════════════════════════════════════════════════════
# 2. POZÍCIÓS TUDATOSSÁG
# ═══════════════════════════════════════════════════════════════════════════════

def run_position_test(model, ob, tr, he, np_, dev, ee, tl):
    tl.log(f"\n{'─'*65}")
    tl.log(f"  POZÍCIÓS TUDATOSSÁG TESZT")
    tl.log(f"{'─'*65}\n")
    bb=2.0; results=[]
    for name, cards in [('K9o',['Kd','9h']),('A5o',['Ac','5d']),('QTo',['Qs','Th']),
                         ('J8s',['Jh','8h']),('T7s',['Ts','7s'])]:
        btn={'hole_cards':cards,'board_cards':[],'street':0,'my_chips':1.0,
             'all_chips':[1.0,2.0]+[0.0]*(np_-2),'pot':3.0,'call_amount':1.0,
             'bb':bb,'sb':1.0,'stack':100.0,'button_pos':0,'my_player_id':0,
             'legal_actions':[0,1,2,3,4,5,6]}
        bbs={'hole_cards':cards,'board_cards':[],'street':0,'my_chips':2.0,
             'all_chips':[6.0,2.0]+[0.0]*(np_-2),'pot':8.0,'call_amount':4.0,
             'bb':bb,'sb':1.0,'stack':100.0,'button_pos':1,'my_player_id':0,
             'legal_actions':[0,1,2,3,4,5,6]}
        pb,_,_,_=query_model(model,ob,tr,he,btn,np_,dev,ee)
        pbb,_,_,_=query_model(model,ob,tr,he,bbs,np_,dev,ee)
        ba=sum(pb[a] for a in range(2,7)); bba=sum(pbb[a] for a in range(2,7))
        aw=ba>bba or pb[0]<pbb[0]
        tl.log(f"  {'✅' if aw else '⚠'} {name}: BTN raise%={ba*100:.0f} BB raise%={bba*100:.0f}")
        results.append({'hand':name,'position_aware':aw,'btn_aggr':float(ba),'bb_aggr':float(bba)})
    n=sum(1 for r in results if r['position_aware'])
    tl.log(f"\n  Pozíció-tudatos: {n}/{len(results)} ({'✅ JÓ' if n>=3 else '⚠ GYENGE'})")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HUD EXPLOITÁCIÓ
# ═══════════════════════════════════════════════════════════════════════════════

def run_exploit_test(model, ob, he, np_, dev, ee, tl):
    tl.log(f"\n{'─'*65}")
    tl.log(f"  HUD EXPLOITÁCIÓ TESZT")
    tl.log(f"{'─'*65}\n")
    mp=PokerActionMapper(); R=[]
    nit=OpponentHUDTracker(np_)
    for _ in range(50): nit.record_postflop_action(1,0,facing_cbet=True)
    for _ in range(5):  nit.record_postflop_action(1,1,facing_cbet=True)
    sc={'hole_cards':['9d','4c'],'board_cards':['Ks','7d','2c'],'street':1,
        'my_chips':6.0,'all_chips':[6.0,6.0]+[0.0]*(np_-2),'pot':12.0,
        'call_amount':0.0,'bb':2.0,'sb':1.0,'stack':94.0,'legal_actions':[0,1,2,3,4,5,6]}
    pn,bn,_,_=query_model(model,ob,nit,he,sc,np_,dev,ee)
    bn_pct=sum(pn[a] for a in range(2,7))*100; g1=bn>=2
    R.append({'test':'Nit blöff','passed':g1})
    tl.log(f"  {'✅' if g1 else '⚠'} Nit blöff: {mp.action_name(bn)} bet%={bn_pct:.0f}%")

    man=OpponentHUDTracker(np_)
    for _ in range(80): man.record_preflop_action(1,2)
    for _ in range(40): man.record_postflop_action(1,4)
    sc2={'hole_cards':['As','Ah'],'board_cards':['Kd','7c','3s'],'street':1,
         'my_chips':12.0,'all_chips':[12.0,12.0]+[0.0]*(np_-2),'pot':24.0,
         'call_amount':0.0,'bb':2.0,'sb':1.0,'stack':88.0,'legal_actions':[0,1,2,3,4,5,6]}
    _,bm,_,_=query_model(model,ob,man,he,sc2,np_,dev,ee); g2=bm!=0
    R.append({'test':'Maniac trap','passed':g2})
    tl.log(f"  {'✅' if g2 else '❌'} Maniac trap: AA → {mp.action_name(bm)}")

    neu=OpponentHUDTracker(np_)
    pneu,_,_,_=query_model(model,ob,neu,he,sc,np_,dev,ee)
    neu_pct=sum(pneu[a] for a in range(2,7))*100; g3=bn_pct>neu_pct+5
    R.append({'test':'HUD adaptáció','passed':g3})
    tl.log(f"  {'✅' if g3 else '⚠'} Adaptáció: nit={bn_pct:.0f}% vs semleges={neu_pct:.0f}%")
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
# 7. BB/100 WINRATE (ÚJ v3b) – opcionális --winrate flag
# ═══════════════════════════════════════════════════════════════════════════════

def run_winrate_test(model, np_, device, n_hands, tl):
    tl.log(f"\n{'─'*65}")
    tl.log(f"  BB/100 WINRATE TESZT ({n_hands} kéz)")
    tl.log(f"{'─'*65}\n")

    try:
        import rlcard
    except ImportError:
        tl.log(f"  ⚠ rlcard nem elérhető – winrate teszt kihagyva")
        return {'self_play':None,'vs_random':None,'error':'rlcard not installed'}

    from core.features import detect_street, ActionHistoryEncoder
    mapper = PokerActionMapper()
    bb=2.0; sb=1.0

    def play_hands(model_p0, model_p1, n, is_random_p1=False):
        env = rlcard.make('no-limit-holdem', config={'game_num_players': np_})
        tracker = OpponentHUDTracker(np_)
        he = ActionHistoryEncoder(np_, PokerActionMapper.NUM_CUSTOM_ACTIONS)
        equity_est = HandEquityEstimator(n_sim=100)
        total_payoff = 0.0
        completed = 0

        for hand_idx in range(n):
            try:
                state, player_id = env.reset()
            except:
                continue
            ah = collections.deque(maxlen=ACTION_HISTORY_LEN)
            steps = 0
            model_seat = hand_idx % 2

            while not env.is_over() and steps < 200:
                steps += 1
                raw_legal = state.get('legal_actions', [1])
                abs_legal = mapper.get_abstract_legal_actions(raw_legal)

                if is_random_p1 and player_id != model_seat:
                    aa = random.choice(abs_legal)
                else:
                    try:
                        hand = env.game.players[player_id].hand
                        hole_eq = [f"{c.get_index()[1]}{c.get_index()[0].lower()}" for c in hand]
                        board_eq = [f"{c.get_index()[1]}{c.get_index()[0].lower()}" for c in env.game.public_cards]
                        eq = equity_est.equity(hole_eq, board_eq, num_opponents=max(np_-1,1))
                    except:
                        eq = 0.5

                    street = detect_street(state)
                    obs = np.array(state['obs'], dtype=np.float32)
                    st_dict = {'obs': obs, 'raw_obs': state.get('raw_obs', {})}
                    state_t = build_state_tensor(
                        st_dict, tracker, ah, he, np_,
                        my_player_id=player_id, bb=bb, sb=sb,
                        initial_stack=100.0, street=street, equity=eq)

                    act_model = model_p0 if player_id == model_seat else model_p1
                    with torch.no_grad():
                        probs_t, _, _ = act_model.forward(state_t.to(device), abs_legal)
                    probs = probs_t.squeeze(0).cpu().numpy()
                    aa = int(np.argmax(probs))

                ea = mapper.get_env_action(aa, raw_legal)
                ah.append((player_id, aa, 0.0))
                try:
                    state, player_id = env.step(ea)
                except:
                    break

            try:
                payoffs = env.get_payoffs()
                total_payoff += float(payoffs[model_seat])
                completed += 1
            except:
                pass

        bb100 = (total_payoff / max(completed, 1)) / bb * 100
        return bb100, completed, total_payoff

    t0 = time.time()

    tl.log(f"  Self-play ({n_hands} kéz)...", console=True)
    sp_bb100, sp_n, sp_total = play_hands(model, model, n_hands, is_random_p1=False)
    tl.log(f"  Self-play BB/100: {sp_bb100:+.1f} ({sp_n} kéz, total={sp_total:+.1f})")
    if abs(sp_bb100) < 10:
        tl.log(f"  ✅ Self-play közel 0 – szimmetrikus (elvárt)")
    else:
        tl.log(f"  ⚠ Self-play aszimmetria: {sp_bb100:+.1f} BB/100")

    tl.log(f"\n  Vs random ({n_hands} kéz)...", console=True)
    rnd_bb100, rnd_n, rnd_total = play_hands(model, model, n_hands, is_random_p1=True)
    elapsed = time.time() - t0
    tl.log(f"  Vs random BB/100: {rnd_bb100:+.1f} ({rnd_n} kéz, total={rnd_total:+.1f})")
    if rnd_bb100 > 30:
        tl.log(f"  ✅ Jól veri a random ellenfelet")
    elif rnd_bb100 > 0:
        tl.log(f"  ⚠ Nyerő, de nem eléggé ({rnd_bb100:+.1f})")
    else:
        tl.log(f"  ❌ Random ellen sem nyerő – komoly probléma!")
    tl.log(f"\n  Winrate teszt idő: {elapsed:.1f}s")

    return {'self_play_bb100':sp_bb100,'vs_random_bb100':rnd_bb100,
            'hands_played':sp_n,'elapsed':elapsed}


# ═══════════════════════════════════════════════════════════════════════════════
# 8. POKER STATISZTIKÁK
# ═══════════════════════════════════════════════════════════════════════════════

def run_poker_stats(model, ob, tr, he, np_, dev, ee, n_hands, tl):
    tl.log(f"\n{'─'*65}")
    tl.log(f"  POKER STATISZTIKÁK ({n_hands} random kéz)")
    tl.log(f"{'─'*65}\n")
    mp=PokerActionMapper(); bb=2.0; sb=1.0
    ac=collections.Counter()
    vo=0;va=0;po=0;pa_=0;pb=0;pc_=0;co=0;ca_=0;to=0;ta=0;fp=0;np2=0

    for _ in range(n_hands):
        deck=list(ALL_CARDS); random.shuffle(deck); hole=[deck[0],deck[1]]
        sit=random.choice(['open','facing_raise','postflop','facing_3bet'])
        if sit=='open':
            sc={'hole_cards':hole,'board_cards':[],'street':0,'my_chips':1.0,
                'all_chips':[1.0,2.0]+[0.0]*(np_-2),'pot':3.0,'call_amount':1.0,
                'bb':bb,'sb':sb,'stack':100.0,'button_pos':0,'my_player_id':0,
                'legal_actions':[0,1,2,3,4,5,6]}
            _,best,_,_=query_model(model,ob,tr,he,sc,np_,dev,ee)
            vo+=1;po+=1
            if best>=1:va+=1
            if best>=2:pa_+=1
        elif sit=='facing_raise':
            sc={'hole_cards':hole,'board_cards':[],'street':0,'my_chips':2.0,
                'all_chips':[6.0,2.0]+[0.0]*(np_-2),'pot':8.0,'call_amount':4.0,
                'bb':bb,'sb':sb,'stack':100.0,'button_pos':1,'my_player_id':0,
                'legal_actions':[0,1,2,3,4,5,6]}
            _,best,_,_=query_model(model,ob,tr,he,sc,np_,dev,ee)
            vo+=1;to+=1
            if best>=1:va+=1
            if best>=2:ta+=1
        elif sit=='facing_3bet':
            sc={'hole_cards':hole,'board_cards':[],'street':0,'my_chips':6.0,
                'all_chips':[18.0,6.0]+[0.0]*(np_-2),'pot':24.0,'call_amount':12.0,
                'bb':bb,'sb':sb,'stack':100.0,'button_pos':0,'my_player_id':0,
                'legal_actions':[0,1,2,3,4,5,6]}
            _,best,_,_=query_model(model,ob,tr,he,sc,np_,dev,ee)
            vo+=1
            if best>=1:va+=1
        else:
            nb=random.choice([3,4,5]);board=deck[2:2+nb]
            st=1 if nb==3 else(2 if nb==4 else 3)
            mc=random.uniform(5,30);oc=random.uniform(5,30)
            call=random.uniform(0,15) if random.random()>0.4 else 0.0
            sc={'hole_cards':hole,'board_cards':board,'street':st,'my_chips':mc,
                'all_chips':[oc,mc]+[0.0]*(np_-2),'pot':mc+oc,'call_amount':call,
                'bb':bb,'sb':sb,'stack':100.0-mc,'legal_actions':[0,1,2,3,4,5,6]}
            _,best,_,_=query_model(model,ob,tr,he,sc,np_,dev,ee)
            if best>=2:pb+=1
            elif best==1 and call>0:pc_+=1
            if call<0.01:co+=1;ca_+=(1 if best>=2 else 0)
        ac[best]+=1
        r1,r2=hole[0][0],hole[1][0]
        if(r1==r2 and r1 in'AKQJ')or('A'in(r1,r2)and set((r1,r2))&set('KQJ')):
            np2+=1;fp+=(1 if best==0 else 0)

    total=sum(ac.values())
    tl.log(f"  Akció eloszlás ({total} kéz):")
    for a in range(7):
        c=ac.get(a,0);p=c/max(total,1)*100
        tl.log(f"    {mp.action_name(a):18s}: {c:5d} ({p:5.1f}%) {'█'*int(p/2)}")
    vpip=va/max(vo,1)*100;pfr=pa_/max(po,1)*100
    af=pb/max(pc_,1);tbet=ta/max(to,1)*100;cbet=ca_/max(co,1)*100
    tl.log(f"\n  VPIP:{vpip:5.1f}% PFR:{pfr:5.1f}% AF:{af:5.2f} 3bet:{tbet:5.1f}% Cbet:{cbet:5.1f}%")
    if np_==2:
        vok=55<=vpip<=90;pok=35<=pfr<=75;aok=1.5<=af<=5.0;cok=50<=cbet<=85
        tl.log(f"  HU ideális: VPIP 55-90 | PFR 35-75 | AF 1.5-5 | Cbet 50-85")
    else:
        vok=18<=vpip<=35;pok=15<=pfr<=30;aok=1.5<=af<=4.0;cok=55<=cbet<=80
        tl.log(f"  {np_}-max: VPIP 18-35 | PFR 15-30 | AF 1.5-4 | Cbet 55-80")
    tl.log(f"  VPIP{'✅'if vok else'⚠'} PFR{'✅'if pok else'⚠'} "
           f"AF{'✅'if aok else'⚠'} Cbet{'✅'if cok else'⚠'}")
    if np2>0:
        fpp=fp/np2*100
        tl.log(f"  Prémium fold: {fp}/{np2} ({fpp:.0f}%) "
               f"{'✅'if fpp<=5 else'⚠'if fpp<=15 else'❌'}")
    degen=[]
    fpct=ac.get(0,0)/max(total,1)*100;aipct=ac.get(6,0)/max(total,1)*100
    mx=max(ac.values())/max(total,1)*100 if ac else 0
    if mx>80:degen.append(f"❌ Degenerált:{mp.action_name(max(ac,key=ac.get))} {mx:.0f}%")
    if fpct>70:degen.append(f"❌ Passzív:{fpct:.0f}%")
    if aipct>40:degen.append(f"❌ AllIn spam:{aipct:.0f}%")
    for d in degen:tl.log(f"  {d}")
    return{'vpip':vpip,'pfr':pfr,'af':af,'three_bet':tbet,'cbet':cbet,
           'fold_pct':fpct,'allin_pct':aipct,
           'premium_fold':fp/max(np2,1)*100,'degeneration':degen,
           'vpip_ok':vok,'pfr_ok':pok,'af_ok':aok}


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MODELL ÖSSZEHASONLÍTÁS (--compare)
# ═══════════════════════════════════════════════════════════════════════════════

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
        ('Epizódok', lambda r: f"{r.get('episodes',0):>12,}"),
        ('Szituáció PASS', lambda r: f"{r['s_passed']}/{r['s_passed']+r['s_warn']+r['s_fail']}"),
        ('Pozíció tudatos', lambda r: f"{r['pos_aware']}/{r['pos_total']}"),
        ('Exploit sikeres', lambda r: f"{r['exp_pass']}/{r['exp_total']}"),
        ('Draw awareness', lambda r: f"{r['draw_passed']}/{r['draw_total']}"),
        ('Draw fold%', lambda r: f"{r['draw_fold_pct']:.0f}%"),
        ('Sizing preflop r', lambda r: f"{r['sizing_pf']:.2f}"),
        ('Sizing postflop r', lambda r: f"{r['sizing_pp']:.2f}"),
        ('Entropy átlag', lambda r: f"{r['avg_entropy']:.2f}"),
        ('VPIP', lambda r: f"{r['vpip']:.0f}%"),
        ('PFR', lambda r: f"{r['pfr']:.0f}%"),
        ('AF', lambda r: f"{r['af']:.2f}"),
        ('Prémium fold%', lambda r: f"{r['prem_fold']:.0f}%"),
        ('Értékelés', lambda r: r['grade']),
    ]
    if do_winrate:
        rows.append(('BB/100 self-play', lambda r: f"{r.get('sp_bb100',0):+.1f}"))
        rows.append(('BB/100 vs random', lambda r: f"{r.get('rnd_bb100',0):+.1f}"))

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
        ck = torch.load(model_path, map_location=device, weights_only=False)
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
    tl.log(f"  🧪  POKER AI VIZSGÁZTATÓ v3 FINAL")
    tl.log(f"{'='*65}")
    tl.log(f"  Modell: {model_path} ({episodes:,} ep)")
    tl.log(f"  {np_}p | {n_hands:,} kéz | seed={seed}")

    model = AdvancedPokerAI(state_size=state_size, action_size=action_size).to(device)
    model.load_state_dict(ck['state_dict']); model.eval()
    ob=ObsBuilder(np_); tr=OpponentHUDTracker(np_)
    he=ActionHistoryEncoder(np_,action_size); ee=HandEquityEstimator(n_sim=200)

    p,w,f = run_scenarios(model,ob,tr,he,np_,device,ee,tl,verbose)
    pos = run_position_test(model,ob,tr,he,np_,device,ee,tl)
    exp = run_exploit_test(model,ob,he,np_,device,ee,tl)
    draw = run_draw_test(model,ob,tr,he,np_,device,ee,tl)
    sizing = run_sizing_test(model,ob,tr,he,np_,device,ee,tl)
    consist = run_consistency_test(model,ob,tr,he,np_,device,ee,tl)
    stats = run_poker_stats(model,ob,tr,he,np_,device,ee,n_hands,tl)

    wr = {}
    if do_winrate:
        wr = run_winrate_test(model,np_,device,wr_hands,tl)

    ti = f + len(stats.get('degeneration',[]))
    if draw['draw_fold_pct']>30: ti+=1
    if ti==0 and w<=3: g='🟢 JÓ'
    elif ti==0: g='🟡 ELFOGADHATÓ'
    elif ti<=2: g='🟠 PROBLÉMÁS'
    else: g='🔴 KOMOLY HIBÁK'

    pa=sum(1 for r in pos if r['position_aware'])
    ep=sum(1 for r in exp if r['passed'])

    tl.log(f"\n{'='*65}")
    tl.log(f"  ÖSSZEFOGLALÓ")
    tl.log(f"{'='*65}")
    tl.log(f"  Szituációs:     {p} ✅  {w} ⚠  {f} ❌")
    tl.log(f"  Pozíció:        {pa}/{len(pos)}")
    tl.log(f"  Exploit:        {ep}/{len(exp)}")
    tl.log(f"  Draw awareness: {draw['passed']}/{draw['total']} (fold%={draw['draw_fold_pct']:.0f})")
    tl.log(f"  Sizing:         pf r={sizing['preflop_corr']:.2f} pp r={sizing['postflop_corr']:.2f}")
    tl.log(f"  Entropy:        {consist['avg_entropy']:.2f}")
    tl.log(f"  VPIP:{stats['vpip']:.0f}% PFR:{stats['pfr']:.0f}% AF:{stats['af']:.2f}")
    if wr:
        tl.log(f"  BB/100 self:    {wr.get('self_play_bb100',0):+.1f}")
        tl.log(f"  BB/100 random:  {wr.get('vs_random_bb100',0):+.1f}")
    tl.log(f"\n  ÉRTÉKELÉS: {g}  ({episodes:,} ep)")

    tl.results['summary'] = {'grade':g,'passed':p,'warnings':w,'failed':f}
    lp, jp = tl.close()
    if not quiet:
        print(f"\n  📄 Log:  {lp}")
        print(f"  📊 JSON: {jp}")
        print(f"{'='*65}\n")

    return {
        'model':model_path, 'episodes':episodes, 'grade':g,
        's_passed':p, 's_warn':w, 's_fail':f,
        'pos_aware':pa, 'pos_total':len(pos),
        'exp_pass':ep, 'exp_total':len(exp),
        'draw_passed':draw['passed'], 'draw_total':draw['total'],
        'draw_fold_pct':draw['draw_fold_pct'],
        'sizing_pf':sizing['preflop_corr'], 'sizing_pp':sizing['postflop_corr'],
        'avg_entropy':consist['avg_entropy'],
        'vpip':stats['vpip'], 'pfr':stats['pfr'], 'af':stats['af'],
        'prem_fold':stats['premium_fold'],
        'sp_bb100':wr.get('self_play_bb100',0),
        'rnd_bb100':wr.get('vs_random_bb100',0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Poker AI Vizsgáztató v3 FINAL')
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

    # ── Compare mód ──────────────────────────────────────────────────────
    if args.compare:
        compare_models(args.compare, args.num_players, args.device,
                       args.hands, args.seed, args.winrate, args.winrate_hands)
        return

    # ── Egyetlen modell ──────────────────────────────────────────────────
    if args.model_path is None:
        parser.print_help(); return

    random.seed(args.seed); np.random.seed(args.seed)
    run_single_model(args.model_path, args.num_players, args.device,
                     args.hands, args.seed, args.verbose,
                     args.winrate, args.winrate_hands, quiet=False,
                     out_dir=args.out_dir)


if __name__ == '__main__':
    main()
