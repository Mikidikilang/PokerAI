#!/usr/bin/env python3
"""
tests/test_equity_v422.py  –  HandEquityEstimator v4.2.2 verifikáció

Futtatás:  python tests/test_equity_v422.py
Eredmény:  25/25 ✅
"""
import sys, os, collections, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.equity import HandEquityEstimator

PASS="✅"; FAIL="❌"; results=[]
def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"  {status}  {name}"
    if detail: msg += f"  ({detail})"
    print(msg); results.append(condition)

print("\n── 1. Split pot fix ──────────────────────────────────────────")
est = HandEquityEstimator(n_sim=2000, cache_size=100)
eq_royal = est.equity(["2c","3c"], board=["Ac","Kc","Qc","Jc","Tc"], num_opponents=1)
check("Royal flush board → split ~50%", 0.45<=eq_royal<=0.55, f"equity={eq_royal:.3f}")
eq_broadway = est.equity(["2s","2h"], board=["As","Kc","Qh","Jd","Tc"], num_opponents=1)
check("Broadway straight board, trash hole → split ~50%", 0.45<=eq_broadway<=0.55, f"equity={eq_broadway:.3f}")
eq_ako = est.equity(["As","Kh"], board=[], num_opponents=1)
check("AKo vs véletlenszerű ellenfél (küszöb: 0.55–0.72)", 0.55<=eq_ako<=0.72, f"equity={eq_ako:.3f}")

print("\n── 2. Ismert equity értékek ──────────────────────────────────")
est2 = HandEquityEstimator(n_sim=1000, cache_size=100)
eq_aa = est2.equity(["As","Ah"], board=[], num_opponents=1)
check("AA preflop HU equity > 78%", eq_aa>0.78, f"equity={eq_aa:.3f}")
eq_72 = est2.equity(["7s","2h"], board=[], num_opponents=1)
check("72o preflop HU equity < 45%", eq_72<0.45, f"equity={eq_72:.3f}")
check("AA equity > 72o equity", eq_aa>eq_72, f"AA={eq_aa:.3f} > 72o={eq_72:.3f}")
eq_nf = est2.equity(["As","Ks"], board=["Qs","7s","3s"], num_opponents=1)
check("Nut flush flop equity > 75%", eq_nf>0.75, f"equity={eq_nf:.3f}")
eq_ts = est2.equity(["Ah","Ad"], board=["Ac","7d","2s"], num_opponents=1)
check("Top set (AAA) flop equity > 75%", eq_ts>0.75, f"equity={eq_ts:.3f}")
check("Nut flush és top set mindkettő > 75%", eq_nf>0.75 and eq_ts>0.75, f"flush={eq_nf:.3f}, set={eq_ts:.3f}")

print("\n── 3. Thread-safety (8 párhuzamos szál) ─────────────────────")
est_t = HandEquityEstimator(n_sim=200, cache_size=50)
errors=[]; thread_results={}
def worker(tid, hole, board):
    try:
        for _ in range(20):
            eq = est_t.equity(hole, board, num_opponents=1)
            if not (0.0<=eq<=1.0): errors.append(f"Thread {tid}: {eq}")
        thread_results[tid]=True
    except Exception as exc:
        errors.append(f"Thread {tid}: {exc}"); thread_results[tid]=False
hands=[(["As","Ah"],[]),(["Ks","Kh"],[]),(["As","Ah"],[]),(["Ks","Kh"],[]),(["Qs","Qh"],["Ac","7d","2s"]),(["Js","Jh"],["Ac","7d","2s"]),(["As","Kh"],[]),(["7s","2h"],[])]
threads=[threading.Thread(target=worker, args=(i,hands[i][0],hands[i][1])) for i in range(8)]
t0=time.time()
for t in threads: t.start()
for t in threads: t.join()
elapsed=time.time()-t0
check("8 párhuzamos szál: nincs race condition", len(errors)==0, f"{elapsed:.2f}s")
if errors: [print(f"    → {e}") for e in errors[:3]]
check("8 párhuzamos szál: mindegyik lefutott", all(thread_results.values()), f"{sum(thread_results.values())}/8")

print("\n── 4. LRU cache méret korlát ─────────────────────────────────")
CACHE_LIMIT=10
est_lru=HandEquityEstimator(n_sim=50, cache_size=CACHE_LIMIT)
pairs=[("A","K"),("A","Q"),("A","J"),("A","T"),("K","Q"),("K","J"),("K","T"),("Q","J"),("Q","T"),("J","T"),("A","9"),("A","8"),("A","7"),("K","9"),("Q","9")]
for r1,r2 in pairs: est_lru.equity([f"{r1}s",f"{r2}h"], board=[], num_opponents=1)
s=est_lru.cache_stats()
check(f"Cache méret ≤ {CACHE_LIMIT}", s["size"]<=CACHE_LIMIT, f"size={s['size']}, max={CACHE_LIMIT}")
check(f"Cache méret pontosan == {CACHE_LIMIT}", s["size"]==CACHE_LIMIT, f"size={s['size']}")

print("\n── 5. Cache hit rate ─────────────────────────────────────────")
est_hr=HandEquityEstimator(n_sim=200, cache_size=100)
est_hr.clear_cache()
_=est_hr.equity(["As","Ah"], board=[], num_opponents=1)
s1=est_hr.cache_stats()
check("Első hívás: cache miss", s1["hits"]==0 and s1["misses"]==1, f"hits={s1['hits']}, misses={s1['misses']}")
_=est_hr.equity(["As","Ah"], board=[], num_opponents=1)
s2=est_hr.cache_stats()
check("Második hívás: cache hit", s2["hits"]==1 and s2["misses"]==1, f"hits={s2['hits']}")
check("Hit rate pontosan 50%", s2["hit_rate"]==0.5, f"hit_rate={s2['hit_rate']}")
_=est_hr.equity(["As","Ah"], board=[], num_opponents=1)
_=est_hr.equity(["Ah","As"], board=[], num_opponents=1)
s3=est_hr.cache_stats()
check("Sorrend-független kulcs: ['As','Ah'] == ['Ah','As']", s3["hits"]==3, f"hits={s3['hits']} (elvárt: 3)")
est_hr.clear_cache()
s4=est_hr.cache_stats()
check("clear_cache() után: size=0, hits=0, misses=0", s4["size"]==0 and s4["hits"]==0 and s4["misses"]==0, f"size={s4['size']}, hits={s4['hits']}, misses={s4['misses']}")

print("\n── 6. API backward compatibility ────────────────────────────")
est_bc=HandEquityEstimator(n_sim=100, cache_size=1000)
check("equity(hole_cards) – board default None", 0.0<=est_bc.equity(["As","Kh"])<=1.0)
check("equity(hole_cards, []) – üres board", 0.0<=est_bc.equity(["As","Kh"],[])<=1.0)
check("equity(..., num_opponents=2)", 0.0<=est_bc.equity(["As","Kh"],["Ac","7d"],num_opponents=2)<=1.0)
check("equity(..., confidence_threshold, min_sims)", 0.0<=est_bc.equity(["As","Kh"],["Ac","7d","2s"],confidence_threshold=0.01,min_sims=30)<=1.0)
check("Hiányos hole_cards (< 2 lap) → fallback 0.5", est_bc.equity(["As"])==0.5)
check("repr() tartalmaz HandEquityEstimator nevet", "HandEquityEstimator" in repr(est_bc))
check("cache_stats() kulcsai helyesek", all(k in est_bc.cache_stats() for k in ("size","max_size","hits","misses","hit_rate","n_sim")))

passed=sum(results); total=len(results)
print(f"\n{'='*60}\n  Eredmény: {passed}/{total} teszt sikeres")
print(f"  {'✅ Minden teszt átment!' if passed==total else f'❌ {total-passed} sikertelen'}\n{'='*60}\n")
sys.exit(0 if passed==total else 1)
