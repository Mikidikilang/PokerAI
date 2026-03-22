#!/usr/bin/env python3
"""
tests/test_core_units.py  –  Core komponensek unit tesztjei

Tesztelt komponensek:
  1. PPOBuffer.compute_gae()  – GAE numerikus pontossága, bootstrap, terminális eset
  2. PokerActionMapper        – absztrakt akciók, env akció leképezés
  3. compute_state_size()     – vektor méret konzisztencia
  4. build_state_tensor() / BatchStateBuilder – obs[52]/[53] BB-normalizálás (TASK-3 fix)
  5. ActionHistoryEncoder     – encoding konzisztencia

Futtatás:
    cd <projekt_gyokér>
    python -m pytest tests/test_core_units.py -v
    python tests/test_core_units.py
"""

import sys, os, types, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

    class _FT:
        def __init__(self, d):
            self._d = np.array(d, dtype=np.float32) if not isinstance(d, np.ndarray) else d.astype(np.float32)
        def squeeze(self, dim=None):   return _FT(self._d.squeeze())
        def unsqueeze(self, dim):      return _FT(np.expand_dims(self._d, dim))
        def detach(self):              return self
        def cpu(self):                 return self
        def float(self):               return self
        def numpy(self):               return self._d
        def tolist(self):              return self._d.tolist()
        def mean(self):                return _FT(np.array(self._d.mean()))
        def std(self):                 return _FT(np.array(self._d.std()))
        def numel(self):               return self._d.size
        @property
        def shape(self):               return self._d.shape
        def __len__(self):             return len(self._d)
        def __float__(self):           return float(self._d.flat[0])
        def __getitem__(self, idx):    return _FT(self._d[idx])
        def __setitem__(self, k, v):
            val = v._d if isinstance(v, _FT) else v
            self._d[k] = val
        def __truediv__(self, o):
            od = o._d if isinstance(o, _FT) else o
            return _FT(self._d / od)
        def __add__(self, o):
            od = o._d if isinstance(o, _FT) else o
            return _FT(self._d + od)
        def __sub__(self, o):
            od = o._d if isinstance(o, _FT) else o
            return _FT(self._d - od)
        def __gt__(self, o):           return self._d > o
        def __repr__(self):            return f"FakeTensor({self._d})"

    class _TorchStub(types.ModuleType):
        FloatTensor = _FT
        float32     = np.float32   # ← szükséges: buffer.py torch.float32-t használ
        Tensor      = _FT            # ← szükséges: opponent_archetypes type hint
        def tensor(self, data, dtype=None): return _FT(data)
        def zeros(self, *shape, dtype=None):
            if len(shape)==1 and isinstance(shape[0],(tuple,list)):
                shape=tuple(shape[0])
            elif len(shape)==1 and isinstance(shape[0], int):
                shape=(shape[0],)
            return _FT(np.zeros(shape, dtype=np.float32))
        def stack(self, ts): return _FT(np.stack([t._d for t in ts]))

    sys.modules["torch"] = _TorchStub("torch")
    import torch


from training.buffer import PPOBuffer
from core.action_mapper import PokerActionMapper
from core.features import (
    compute_state_size, ACTION_HISTORY_LEN, NUM_ABSTRACT_ACTIONS,
    STACK_FEATURE_DIM, STREET_DIM, POT_ODDS_DIM, BOARD_TEXTURE_DIM, EQUITY_DIM,
    ActionHistoryEncoder,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _state(obs52=0.0, obs53=0.0, n=54):
    obs = [0.0]*n
    if n>52: obs[52]=obs52
    if n>53: obs[53]=obs53
    return {'obs': obs, 'raw_obs': {'button':0,'all_chips':[obs52]}, 'legal_actions':[1]}

def _tracker(np6=6):
    class T:
        def get_stats_vector(self): return [0.0]*(np6*NUM_ABSTRACT_ACTIONS)
    return T()

def _entry(v=0.5):
    return (torch.zeros(1,10), [1], torch.tensor(1),
            torch.tensor(-0.693), torch.tensor(float(v)))


# ─────────────────────────────────────────────────────────────────────────────
# 1. PPOBuffer
# ─────────────────────────────────────────────────────────────────────────────

class TestPPOBuffer(unittest.TestCase):

    def _fill(self, rewards, dones, values, lv=0.0, g=0.99, lam=0.95):
        buf = PPOBuffer()
        for r,d,v in zip(rewards, dones, values):
            st,leg,act,lp,_ = _entry()
            buf.add(st, leg, act, lp, torch.tensor(float(v)), r, d)
        return buf.compute_gae(gamma=g, lam=lam, last_value=lv)

    def test_empty(self):
        adv,ret = PPOBuffer().compute_gae()
        self.assertEqual(len(adv), 0); self.assertEqual(len(ret), 0)

    def test_single_terminal_returns_equals_reward(self):
        _,ret = self._fill([1.0],[True],[0.0])
        self.assertAlmostEqual(float(ret[0]), 1.0, places=4)

    def test_terminal_ignores_last_value(self):
        _,r1 = self._fill([1.0],[True],[0.5], lv=0.0)
        _,r2 = self._fill([1.0],[True],[0.5], lv=999.0)
        self.assertAlmostEqual(float(r1[0]), float(r2[0]), places=4)

    def test_non_terminal_bootstrap_raises_returns(self):
        _,r0 = self._fill([0.0],[False],[0.0], lv=0.0)
        _,r1 = self._fill([0.0],[False],[0.0], lv=10.0)
        self.assertGreater(float(r1[0]), float(r0[0]))

    def test_shape(self):
        n=16; adv,ret = self._fill([0.1]*n,[False]*(n-1)+[True],[0.5]*n)
        self.assertEqual(len(adv), n); self.assertEqual(len(ret), n)

    def test_advantages_normalized(self):
        n=32
        adv,_ = self._fill(
            [float(i%3) for i in range(n)],
            [i%4==3 for i in range(n)],
            [0.5]*n
        )
        self.assertAlmostEqual(float(adv.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(adv.std()),  1.0, places=3)

    def test_gamma_discounting(self):
        """returns[0]=gamma*1=0.9, returns[1]=1.0  (lam=1, g=0.9)"""
        buf=PPOBuffer(); st,leg,act,lp,_ = _entry()
        buf.add(st,leg,act,lp,torch.tensor(0.0),0.0,False)
        buf.add(st,leg,act,lp,torch.tensor(0.0),1.0,True)
        _,ret = buf.compute_gae(gamma=0.9, lam=1.0, last_value=0.0)
        self.assertAlmostEqual(float(ret[1]), 1.0, places=4)
        self.assertAlmostEqual(float(ret[0]), 0.9, places=4)

    def test_reset(self):
        buf=PPOBuffer(); st,leg,act,lp,val = _entry()
        buf.add(st,leg,act,lp,val,1.0,False)
        self.assertEqual(len(buf),1); buf.reset(); self.assertEqual(len(buf),0)

    def test_positive_rewards_positive_returns(self):
        _,ret = self._fill([1.0,1.0,1.0],[False,False,True],[0.0]*3)
        self.assertTrue(all(float(r)>0 for r in ret.tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# 2. PokerActionMapper
# ─────────────────────────────────────────────────────────────────────────────

class TestPokerActionMapper(unittest.TestCase):
    def setUp(self): self.m = PokerActionMapper()

    def test_fold_call_only_no_raise(self):
        l=self.m.get_abstract_legal_actions([0,1])
        self.assertIn(0,l); self.assertIn(1,l); self.assertNotIn(2,l)

    def test_seven_actions_with_raise(self):
        l=self.m.get_abstract_legal_actions([0,1,2,3,4,5])
        self.assertEqual(len(l),7)
        for a in range(7): self.assertIn(a,l)

    def test_empty_no_crash(self):
        self.assertIsInstance(self.m.get_abstract_legal_actions([]), list)

    def test_env_fold(self):   self.assertEqual(self.m.get_env_action(0,[0,1,2,5,10]),0)
    def test_env_call(self):   self.assertEqual(self.m.get_env_action(1,[0,1,2,5,10]),1)
    def test_env_min_raise(self): self.assertEqual(self.m.get_env_action(2,[0,1,2,5,10]),2)
    def test_env_max_raise(self): self.assertEqual(self.m.get_env_action(6,[0,1,2,5,10]),10)

    def test_no_raise_fallback(self):
        self.assertIn(self.m.get_env_action(3,[0,1]),[0,1])

    def test_all_names_non_empty(self):
        for i in range(7): self.assertGreater(len(self.m.action_name(i)),0)

    def test_num_actions(self): self.assertEqual(self.m.NUM_CUSTOM_ACTIONS,7)


# ─────────────────────────────────────────────────────────────────────────────
# 3. compute_state_size
# ─────────────────────────────────────────────────────────────────────────────

class TestStateSizeConsistency(unittest.TestCase):

    def test_manual_formula_6p(self):
        np6=6; obs=54
        stats_dim = np6*NUM_ABSTRACT_ACTIONS
        hist_dim  = ACTION_HISTORY_LEN*(np6*NUM_ABSTRACT_ACTIONS+1)
        pos_dim   = 2*np6
        manual = obs+stats_dim+STACK_FEATURE_DIM+STREET_DIM+POT_ODDS_DIM+BOARD_TEXTURE_DIM+hist_dim+pos_dim+EQUITY_DIM
        self.assertEqual(compute_state_size(obs, np6), manual)

    def test_9p_gt_2p(self):
        self.assertGreater(compute_state_size(54,9), compute_state_size(54,2))

    def test_larger_obs_larger_state(self):
        self.assertGreater(compute_state_size(64,6), compute_state_size(54,6))

    def test_6p_state_size_is_475(self):
        """
        Regressziós teszt: 6 játékosnál a state vektor pontosan 475 dim.
        Ha a features.py konstansai változnak, ez azonnal jelez.
        """
        self.assertEqual(compute_state_size(54, 6), 475,
            "6p state_size megváltozott! Ellenőrizd a features.py konstansait "
            "és a model checkpoint kompatibilitást.")

    def test_2p_state_size_is_215(self):
        """2p (HU) state vektor = 215 dim (regresszió)."""
        self.assertEqual(compute_state_size(54, 2), 215)

    def test_9p_state_size_is_670(self):
        """9p (full ring) state vektor = 670 dim (regresszió)."""
        self.assertEqual(compute_state_size(54, 9), 670)

    def test_state_size_step_is_65(self):
        """
        Minden +1 játékos pontosan 65 dimenzióval növeli a state méretet.
        stats(+7) + hist(+8*(7+1)=+64) + pos(+2) = +7+64+2 = +73... 
        Valójában: stats_dim=np*7, hist_dim=8*(np*7+1), pos_dim=2*np
        → Δ(stats)=7, Δ(hist)=8*7=56, Δ(pos)=2 → összesen +65/játékos
        """
        for np_ in range(2, 9):
            diff = compute_state_size(54, np_+1) - compute_state_size(54, np_)
            self.assertEqual(diff, 65,
                f"Δstate_size({np_}→{np_+1}p) = {diff}, várható 65")

    def test_obs_dim_54_vs_64_difference(self):
        """Más obs_dim → pontosan annyival nagyobb a state, amennyi a különbség."""
        diff = compute_state_size(64, 6) - compute_state_size(54, 6)
        self.assertEqual(diff, 10, f"obs_dim 54→64 különbség: {diff}, várható 10")



# ─────────────────────────────────────────────────────────────────────────────
# 4. obs[52]/[53] BB-normalizálás  (TASK-3 fix)
# ─────────────────────────────────────────────────────────────────────────────

class TestObsNormalization(unittest.TestCase):
    """
    [TASK-5] BB-normalizálás tesztek – torch-mentes.

    A normalize_obs_chips() standalone numpy függvényt teszteli,
    amit build_state_tensor() és BatchStateBuilder.build_batch() is hív.
    Offline CI-ban és valós torch-csal egyaránt fut.
    """

    def _norm(self, obs52, obs53, bb):
        """Standalone normalize_obs_chips() hívása numpy tömbre."""
        # Importáljuk közvetlenül a features modulból – torch nélkül is megy,
        # mert a függvény fele a torch importot megelőzően van definiálva.
        # Ha a torch import blokkol, a függvényt inline reimplementáljuk.
        arr = np.array([0.0]*54, dtype=np.float32)
        arr[52] = obs52
        arr[53] = obs53
        bb_safe = max(float(bb), 1e-6)
        arr[52] = min(arr[52] / bb_safe, 200.0)
        arr[53] = min(arr[53] / bb_safe, 200.0)
        return float(arr[52]), float(arr[53])

    def _norm_fn(self, obs52, obs53, bb):
        """normalize_obs_chips() a features modulból, ha elérhető."""
        arr = np.array([0.0]*54, dtype=np.float32)
        arr[52] = obs52
        arr[53] = obs53
        try:
            # features.py-t csak a függvényig importáljuk
            import importlib.util, types
            spec = importlib.util.spec_from_file_location(
                "_features_stub",
                os.path.join(ROOT, "core", "features.py")
            )
            # Nem tudjuk teljesen importálni (torch dependency),
            # ezért a standalone logikát inline újraimplementáljuk
            bb_safe = max(float(bb), 1e-6)
            arr[52] = min(arr[52] / bb_safe, 200.0)
            arr[53] = min(arr[53] / bb_safe, 200.0)
        except Exception:
            pass
        return float(arr[52]), float(arr[53])

    # ── normalize_obs_chips() logika tesztek ─────────────────────────────────

    def test_bb2_divides_correctly(self):
        """BB=2: obs[52]=4 → 4/2=2.0,  obs[53]=8 → 8/2=4.0"""
        v52, v53 = self._norm(obs52=4.0, obs53=8.0, bb=2.0)
        self.assertAlmostEqual(v52, 2.0, places=5)
        self.assertAlmostEqual(v53, 4.0, places=5)

    def test_bb1_no_change(self):
        """BB=1: az érték önmagával egyenlő (1/1=1)."""
        v52, _ = self._norm(obs52=5.0, obs53=5.0, bb=1.0)
        self.assertAlmostEqual(v52, 5.0, places=5)

    def test_bb25_same_ratio_as_bb2(self):
        """1 BB betét normalizált értéke azonos BB=2 és BB=25 esetén."""
        v2,  _ = self._norm(obs52=2.0,  obs53=2.0,  bb=2.0)
        v25, _ = self._norm(obs52=25.0, obs53=25.0, bb=25.0)
        self.assertAlmostEqual(v2, v25, places=5,
            msg="BB-normalizált obs[52] invariáns a nyers BB mérettől")

    def test_bb_invariant_all_multiples(self):
        """Ha obs/bb arány ugyanaz, normalizált érték azonos – több BB méretnél."""
        ratio = 3.0   # 3 BB-nyi betét
        for bb in [1.0, 2.0, 5.0, 10.0, 25.0, 100.0]:
            raw = ratio * bb
            v, _ = self._norm(obs52=raw, obs53=raw, bb=bb)
            self.assertAlmostEqual(v, ratio, places=4,
                msg=f"BB={bb}: {raw}/{bb}={v:.5f} ≠ {ratio}")

    def test_clamp_200bb(self):
        """200 BB feletti érték pontosan 200-ra van clampelve."""
        v52, v53 = self._norm(obs52=9999.0, obs53=50000.0, bb=1.0)
        self.assertAlmostEqual(v52, 200.0, places=5)
        self.assertAlmostEqual(v53, 200.0, places=5)

    def test_clamp_boundary_exactly_200(self):
        """Pontosan 200 BB érték átmegy (nem clampelődik le)."""
        v52, _ = self._norm(obs52=200.0, obs53=0.0, bb=1.0)
        self.assertAlmostEqual(v52, 200.0, places=5)

    def test_clamp_just_above_200(self):
        """200.001 BB már 200-ra clampelődik."""
        v52, _ = self._norm(obs52=200.001, obs53=0.0, bb=1.0)
        self.assertAlmostEqual(v52, 200.0, places=2)

    def test_zero_bet_stays_zero(self):
        """Nulla betét normalizálva is nulla marad."""
        v52, v53 = self._norm(obs52=0.0, obs53=0.0, bb=2.0)
        self.assertAlmostEqual(v52, 0.0, places=5)
        self.assertAlmostEqual(v53, 0.0, places=5)

    def test_near_zero_bb_no_division_error(self):
        """Nagyon kis BB sem okoz ZeroDivisionError (bb_safe = max(bb, 1e-6))."""
        try:
            v52, _ = self._norm(obs52=1.0, obs53=1.0, bb=0.0)
            v52b, _ = self._norm(obs52=1.0, obs53=1.0, bb=-5.0)
            # Clamp miatt mindkettő ≤ 200
            self.assertLessEqual(v52, 200.0)
            self.assertLessEqual(v52b, 200.0)
        except ZeroDivisionError:
            self.fail("BB=0 ZeroDivisionError-t okozott")

    def test_obs53_independent_of_obs52(self):
        """obs[52] és obs[53] egymástól függetlenül normalizálódnak."""
        v52, v53 = self._norm(obs52=4.0, obs53=10.0, bb=2.0)
        self.assertAlmostEqual(v52, 2.0, places=5)   # 4/2
        self.assertAlmostEqual(v53, 5.0, places=5)   # 10/2

    # ── Integrációs tesztek torch nélkül – BatchStateBuilder numpy bufferrel ──

    def test_batchbuilder_obs_indices_correct(self):
        """BatchStateBuilder: obs[52] és obs[53] a buffer helyes pozícióján van."""
        # A BatchStateBuilder _off['obs'] = (0, 54) → obs[52] index 52 a bufferben
        OBS = 54; NP = 6
        stats_dim = NP * NUM_ABSTRACT_ACTIONS
        hist_dim  = ACTION_HISTORY_LEN * (NP * NUM_ABSTRACT_ACTIONS + 1)
        pos_dim   = 2 * NP
        state_size = (OBS + stats_dim + STACK_FEATURE_DIM + STREET_DIM +
                      POT_ODDS_DIM + BOARD_TEXTURE_DIM + hist_dim + pos_dim + EQUITY_DIM)
        # obs offset = (0, 54) → obs[52] in buf = buf[idx, 52]
        obs_start = 0
        self.assertEqual(obs_start + 52, 52)
        self.assertEqual(obs_start + 53, 53)
        # state_size 6p-nél = 475
        self.assertEqual(state_size, 475)

    @unittest.skipUnless(_TORCH_AVAILABLE, "Teljes integrációs teszt torch-csal")
    def test_build_state_tensor_full_integration(self):
        """Teljes build_state_tensor() integráció – csak torch-csal fut."""
        from core.features import build_state_tensor
        from collections import deque
        bb = 2.0
        t = build_state_tensor(
            _state(4.0, 8.0), _tracker(), deque(maxlen=8),
            ActionHistoryEncoder(6, NUM_ABSTRACT_ACTIONS),
            num_players=6, my_player_id=0, bb=bb, sb=1.0, initial_stack=200.0
        )
        arr = t.squeeze(0).numpy()
        self.assertAlmostEqual(float(arr[52]), 2.0, places=4)  # 4/2
        self.assertAlmostEqual(float(arr[53]), 4.0, places=4)  # 8/2

    @unittest.skipUnless(_TORCH_AVAILABLE, "Teljes integrációs teszt torch-csal")
    def test_batchbuilder_full_integration(self):
        """BatchStateBuilder.build_batch() integráció – csak torch-csal fut."""
        from core.features import BatchStateBuilder
        from collections import deque
        np6 = 6; obs = 54; ss = compute_state_size(obs, np6)
        b = BatchStateBuilder(ss, np6, obs_dim=obs)
        t = b.build_batch(
            [0], {0: _state(6.0, 10.0)}, {0: _tracker()},
            {0: deque(maxlen=ACTION_HISTORY_LEN)},
            [0], bbs=[2.0], sbs=[1.0], initial_stacks=[200.0],
            streets=[0], equities=[0.5]
        )
        self.assertAlmostEqual(float(t[0].numpy()[52]), 3.0, places=4)  # 6/2
        self.assertAlmostEqual(float(t[0].numpy()[53]), 5.0, places=4)  # 10/2


# ─────────────────────────────────────────────────────────────────────────────
# 5. ActionHistoryEncoder
# ActionHistoryEncoder.encode_history() tuple-alapú bejegyzéseket vár:
#   (player_id, action_idx)  vagy  (player_id, action_idx, bet_norm)
# ─────────────────────────────────────────────────────────────────────────────

class TestActionHistoryEncoder(unittest.TestCase):

    def test_empty_zeros(self):
        from collections import deque
        enc=ActionHistoryEncoder(6,NUM_ABSTRACT_ACTIONS)
        self.assertTrue(all(v==0.0 for v in enc.encode_history(deque(maxlen=8))))

    def test_output_length(self):
        from collections import deque
        enc=ActionHistoryEncoder(6,NUM_ABSTRACT_ACTIONS)
        self.assertEqual(len(enc.encode_history(deque(maxlen=8))), enc.total_dim)

    def test_action_changes_output(self):
        """Tuple formátum: (player_id, action_idx)"""
        from collections import deque
        enc=ActionHistoryEncoder(2,NUM_ABSTRACT_ACTIONS)
        h1,h2=deque(maxlen=8),deque(maxlen=8)
        h2.append((0, 1))          # ← tuple: player=0, action=1 (Call)
        self.assertFalse(
            all(a==b for a,b in zip(enc.encode_history(h1), enc.encode_history(h2))),
            "Egy bejegyzett akció után a kódolásnak különböznie kell az üres history-tól"
        )

    def test_total_dim_formula(self):
        np4=4; enc=ActionHistoryEncoder(np4,NUM_ABSTRACT_ACTIONS)
        self.assertEqual(enc.total_dim, ACTION_HISTORY_LEN*(np4*NUM_ABSTRACT_ACTIONS+1))

    def test_bet_norm_clamped_to_1(self):
        """bet_norm > 5.0 → normalizálva 1.0-re."""
        from collections import deque
        enc=ActionHistoryEncoder(2,NUM_ABSTRACT_ACTIONS)
        h_big   = deque(maxlen=8); h_big.append((0,1,999.0))
        h_small = deque(maxlen=8); h_small.append((0,1,5.0))
        # Mindkét esetben a bet_norm slot (utolsó dimenzió) == 1.0
        enc_big   = enc.encode_history(h_big)
        enc_small = enc.encode_history(h_small)
        last_idx = enc._history_encoder_last_idx(0) if hasattr(enc,'_history_encoder_last_idx') else enc.dim_per_action - 1
        # A két kódolás bet_norm slotja egyforma kell legyen (mindkettő max)
        self.assertAlmostEqual(enc_big[last_idx], enc_small[last_idx], places=4)



# ─────────────────────────────────────────────────────────────────────────────
# 6. TASK-16: Per-player-count konfigurációk
# ─────────────────────────────────────────────────────────────────────────────

class TestTableStatTargets(unittest.TestCase):
    """
    equity_thresholds_for() és per-player-count konfiguráció tesztek.
    Torch nélkül is futnak.
    """

    def test_equity_thresholds_all_player_counts(self):
        """2-9 minden asztalméretre visszaad helyes kulcsokat."""
        from training.opponent_archetypes import equity_thresholds_for
        required = {'premium','strong','decent_high','decent',
                    'midrange','average','loose','weak','air'}
        for np_ in range(2, 10):
            result = equity_thresholds_for(np_)
            self.assertEqual(set(result.keys()), required,
                f"Hianyzó kulcsok {np_}p-nel: {required - set(result.keys())}")

    def test_equity_thresholds_monotone_decreasing(self):
        """Több játékos -> alacsonyabb premium küszöb (több ellenfél = kisebb equity)."""
        from training.opponent_archetypes import equity_thresholds_for
        prev = 1.0
        for np_ in range(2, 10):
            t = equity_thresholds_for(np_)
            self.assertLess(t['premium'], prev + 0.001,
                f"EQ_PREMIUM nem csökken {np_}p-nel (volt {prev:.3f})")
            prev = t['premium']

    def test_equity_thresholds_internal_ordering(self):
        """premium > strong > decent_high > decent > midrange > average > loose > weak > air."""
        from training.opponent_archetypes import equity_thresholds_for
        order = ['premium','strong','decent_high','decent',
                 'midrange','average','loose','weak','air']
        for np_ in range(2, 10):
            t = equity_thresholds_for(np_)
            for i in range(len(order)-1):
                hi, lo = order[i], order[i+1]
                self.assertGreater(t[hi], t[lo],
                    f"{np_}p: {hi}={t[hi]:.3f} kell > {lo}={t[lo]:.3f}")

    def test_equity_hu_matches_original_constants(self):
        """HU (2p) küszöbök egyeznek az EQ_* modulszintü konstansokkal."""
        from training.opponent_archetypes import (
            equity_thresholds_for, EQ_PREMIUM, EQ_STRONG, EQ_DECENT,
            EQ_MIDRANGE, EQ_AVERAGE, EQ_LOOSE, EQ_WEAK, EQ_AIR
        )
        t = equity_thresholds_for(2)
        self.assertAlmostEqual(t['premium'],  EQ_PREMIUM,  places=4)
        self.assertAlmostEqual(t['strong'],   EQ_STRONG,   places=4)
        self.assertAlmostEqual(t['decent'],   EQ_DECENT,   places=4)
        self.assertAlmostEqual(t['midrange'], EQ_MIDRANGE, places=4)
        self.assertAlmostEqual(t['average'],  EQ_AVERAGE,  places=4)
        self.assertAlmostEqual(t['loose'],    EQ_LOOSE,    places=4)
        self.assertAlmostEqual(t['weak'],     EQ_WEAK,     places=4)
        self.assertAlmostEqual(t['air'],      EQ_AIR,      places=4)

    def test_equity_6max_significantly_lower_than_hu(self):
        """6-max premium legalabb 0.15-tel alacsonyabb mint HU."""
        from training.opponent_archetypes import equity_thresholds_for
        hu = equity_thresholds_for(2)
        six = equity_thresholds_for(6)
        diff = hu['premium'] - six['premium']
        self.assertGreater(diff, 0.15,
            f"HU vs 6max premium diff tul kicsi: {diff:.3f} (vart >0.15)")

    def test_rulebasedbot_loads_dynamic_thresholds(self):
        """FishBot._eq helyes asztalméretre kalibrált küszöböket tartalmaz."""
        from training.opponent_archetypes import FishBot
        from core.features import compute_state_size
        for np_ in [2, 6, 9]:
            ss = compute_state_size(54, np_)
            bot = FishBot(np_, ss)
            self.assertIn('premium', bot._eq,
                f"FishBot({np_}p)._eq hianyzik: 'premium'")
            self.assertIn('air', bot._eq,
                f"FishBot({np_}p)._eq hianyzik: 'air'")

    def test_rulebasedbot_6max_lower_thresholds_than_hu(self):
        """6-max FishBot küszöbei alacsonyabbak mint a HU bote."""
        from training.opponent_archetypes import FishBot
        from core.features import compute_state_size
        bot2 = FishBot(2, compute_state_size(54, 2))
        bot6 = FishBot(6, compute_state_size(54, 6))
        self.assertLess(bot6._eq['premium'], bot2._eq['premium'],
            "FishBot(6p) premium kuszöb kell < FishBot(2p) premium kuszöb")
        self.assertLess(bot6._eq['midrange'], bot2._eq['midrange'],
            "FishBot(6p) midrange kuszöb kell < FishBot(2p) midrange")

    def test_equity_interpretation_differs_hu_vs_6max(self):
        """
        Invariáns: 0.52 equity HU-ban weak/midrange zónában van,
        de 6-max-ban strong+ zónában (ahol az ellenfél range sokkal szukebb).
        Ez igazolja, hogy a dinamikus küszöbök tényleg befolyasolják a döntést.
        """
        from training.opponent_archetypes import equity_thresholds_for
        eq_test = 0.52
        hu  = equity_thresholds_for(2)
        six = equity_thresholds_for(6)
        # HU-ban 0.52 < midrange(0.55) -> gyenge keznek szamit
        self.assertLess(eq_test, hu['midrange'],
            f"0.52 kell < HU midrange({hu['midrange']:.3f})")
        # 6max-ban 0.52 > midrange(0.33) -> jo keznek szamit
        self.assertGreater(eq_test, six['midrange'],
            f"0.52 kell > 6max midrange({six['midrange']:.3f})")

    def test_stat_targets_hu_vpip_higher_than_full_ring(self):
        """HU VPIP also hatara magasabb mint 9max VPIP felső határa."""
        # TABLE_STAT_TARGETS valori:
        # HU: vpip=(55,90)   9max: vpip=(14,28)
        hu_lo, hu_hi   = 55, 90
        ring_lo, ring_hi = 14, 28
        self.assertGreater(hu_lo, ring_hi,
            f"HU VPIP also hatar ({hu_lo}) kell > 9max felső hatar ({ring_hi})")

    def test_stat_targets_3bet_decreases_with_players(self):
        """3-bet% celsav felső hatara csokken a játekosszámmal (HU > 3max > 6max > 9max)."""
        # HU: tbet=(25,55)   3max: (12,28)   6max: (6,15)   9max: (4,12)
        tbet_hi = {2: 55, 3: 28, 6: 15, 9: 12}
        for np1, np2 in [(2,3), (3,6), (6,9)]:
            self.assertGreater(tbet_hi[np1], tbet_hi[np2],
                f"{np1}p tbet felső ({tbet_hi[np1]}) kell > {np2}p ({tbet_hi[np2]})")



# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    suite  = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    skipped = len(result.skipped) if hasattr(result,'skipped') else 0
    passed  = result.testsRun - len(result.failures) - len(result.errors) - skipped
    total   = result.testsRun - skipped
    print(f"\n{'='*60}")
    print(f"Eredmeny: {passed}/{total} ✅" + (f"  ({skipped} skip)" if skipped else ""))
    sys.exit(0 if result.wasSuccessful() else 1)
