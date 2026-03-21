"""
training/collector.py  --  BatchedSyncCollector (v4 OPTIMIZED)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIMALIZÁCIÓK az eredetihez képest:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  OPT-1: Opponent lépések BATCHELVE modell szerint
         Eredeti: 256 egyedi forward pass → 256× GPU kernel launch
         Új: modellenként 1 batch forward → ~2-5× gyorsabb

  OPT-2: BatchStateBuilder – pre-allokált numpy tömb, slicing
         Eredeti: N × np.concatenate + torch.FloatTensor
         Új: egyetlen np.zeros + slicing + torch.from_numpy

  OPT-3: torch.inference_mode() a no_grad helyett
         Kevesebb overhead (autograd tracking kikapcsolva, nem csak gradiens)

  OPT-4: Opponent loop javítás – egy iterációban több env lépés
         Kevesebb Python loop overhead

  KOMPATIBILITÁS: Checkpoint formátum VÁLTOZATLAN.
                  A modell architektúra VÁLTOZATLAN.
                  A tanulási dinamika AZONOS (azonos state tensor kimenet).

  [RF-1 FIX] BB/Stack mismatch javítva: _reset_env() most az rlcard env
             tényleges game konfigját olvassa vissza reset után, ahelyett
             hogy véletlenszerű, soha nem alkalmazott bb/stack értékeket
             használna a feature normalizáláshoz.

  [RF-9 FIX] Valódi equity számítás rlcard raw_obs alapján.
             _step_learners_batched() mostantól HandEquityEstimator-t hív
             a state tensor equity dimenzióját (state[-1]) kitöltendő.
             Preflop: lookup cache, postflop: adaptív MC (min 50, max 200 sim).
             Ha a kártyák nem elérhetők (preflop rejtett lap), 0.5 fallback.

  [RF-11 FIX] Street-átmenet equity delta intermediate reward.
             Flop/turn/river átmenetkor equity_delta * STREET_REWARD_SCALE
             reward shaping adódik hozzá. Ez gyorsabb konvergenciát ad a
             drawing hand szituációkhoz – óvatosan kalibrált (0.05 scale),
             hogy ne torzítsa el a terminális reward fontosságát.
"""
import collections, logging
import rlcard, torch, torch.nn.functional as F
from core.equity import HandEquityEstimator
from core.action_mapper import PokerActionMapper
from core.features import (
    ActionHistoryEncoder, build_state_tensor, detect_street,
    ACTION_HISTORY_LEN, BatchStateBuilder, compute_state_size,
)
from core.opponent_tracker import OpponentHUDTracker

logger = logging.getLogger("PokerAI")
_MAX_STEPS_PER_HAND = 500
_LEARNER_ID = 0
# [FIX P2-3] Explicit fold akció index – PokerActionMapper.ACTION_NAMES alapján:
# 0=Fold, 1=Call/Check, 2-6=Raise szintek. Magic number helyett konstans,
# hogy architektúra változáskor ne legyen silent bug.
_FOLD_ACTION = 0
# BB_OPTIONS / STACK_MULTIPLIERS eltávolítva (RF-1 fix): az env saját konfigja
# kerül kiolvasásra reset után – véletlenszerű, soha nem alkalmazott értékek helyett.

# [RF-11] Street-átmenet intermediate reward shaping
# Equity delta szorozva ezzel az értékkel adódik a reward-hoz flop/turn/river-en.
# Alacsony érték szándékos: a terminális reward marad a fő tanulási jel.
_STREET_REWARD_SCALE = 0.05

# [RF-9] Equity estimator singleton (thread-safe: read-only cache + random)
_EQUITY_ESTIMATOR = HandEquityEstimator(n_sim=200, cache_size=20_000)



def _rlcard_cards_to_equity_fmt(cards: list) -> list:
    """
    [RF-9] rlcard kártya formátum ('SA', 'HK') → equity.py formátum ('As', 'Kh').

    rlcard: Card.get_index() = suit_upper + rank_upper  (pl. 'SA', 'HK', 'DT')
    equity: rank_lower + suit_lower                     (pl. 'As', 'Kh', 'Td')
    """
    _SUIT_MAP  = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c'}
    _RANK_MAP  = {'A':'a','2':'2','3':'3','4':'4','5':'5','6':'6',
                  '7':'7','8':'8','9':'9','T':'t','J':'j','Q':'q','K':'k'}
    result = []
    for card in cards:
        if not card or len(card) < 2:
            continue
        suit = _SUIT_MAP.get(card[0].upper())
        rank = _RANK_MAP.get(card[1].upper())
        if suit and rank:
            result.append(rank + suit)
    return result


def _compute_equity_for_env(state: dict, num_opponents: int) -> float:
    """
    [RF-9] Valódi equity becslés az aktuális rlcard state-hez.

    Kártyák kinyerése raw_obs['hand'] és raw_obs['public_cards']-ból,
    majd HandEquityEstimator.equity() hívás.

    Visszatér: float [0.0, 1.0] – fallback 0.5 ha a lapok nem elérhetők.
    """
    raw = state.get('raw_obs', {})
    hole_rlcard  = raw.get('hand', [])
    board_rlcard = raw.get('public_cards', [])

    hole  = _rlcard_cards_to_equity_fmt(hole_rlcard)
    board = _rlcard_cards_to_equity_fmt(board_rlcard)

    if len(hole) < 2:
        return 0.5  # lapok nem elérhetők → neutral prior

    try:
        return _EQUITY_ESTIMATOR.equity(
            hole_cards=hole,
            board=board,
            num_opponents=max(num_opponents, 1),
        )
    except Exception:
        return 0.5


class BatchedSyncCollector:
    def __init__(self, num_envs, model, device, num_players, action_mapper,
                 model_kwargs, pool, rlcard_obs_size: int = 54):
        """
        Paraméterek:
            ...
            rlcard_obs_size: [RF-4 FIX] rlcard obs tömb mérete. Default=54 a
                standard no-limit-holdem env-hez. Ha az rlcard frissítése vagy
                más config megváltoztatja az obs méretet, ezt kell frissíteni –
                nem fog silent méretbeli eltérés keletkezni.
                runner.py-ban: rlcard_obs_size = len(env.reset()[0]['obs'])
        """
        self.num_envs = num_envs
        self.model = model
        self.device = device
        self.num_players = num_players
        self.action_mapper = action_mapper
        self.pool = pool
        self.action_size = PokerActionMapper.NUM_CUSTOM_ACTIONS
        self._rlcard_obs_size = rlcard_obs_size  # [RF-4 FIX] stored for _reset_env fallback

        logger.info(f"  {num_envs} rlcard env inicializálása...")
        self.envs = [
            rlcard.make('no-limit-holdem', config={'game_num_players': num_players})
            for _ in range(num_envs)
        ]
        logger.info("  Env-ek kész.")

        self._history_encoder = ActionHistoryEncoder(num_players, self.action_size)

        # Per-env állapotok
        self._states = [None] * num_envs
        self._players = [0] * num_envs
        self._trackers = [None] * num_envs
        self._steps = [[] for _ in range(num_envs)]
        self._opp_models = [None] * num_envs
        self._step_cnt = [0] * num_envs
        self._active = [False] * num_envs
        self._action_histories = [
            collections.deque(maxlen=ACTION_HISTORY_LEN) for _ in range(num_envs)
        ]
        self._bb = [2.0] * num_envs
        self._sb = [1.0] * num_envs
        self._initial_stack = [100.0] * num_envs
        self._street      = [0]   * num_envs
        self._last_equity = [0.5] * num_envs
        # [RF-11] street-átmenet equity delta reward shaping
        self._prev_street  = [0]   * num_envs   # előző lépés utcája
        self._prev_equity  = [0.5] * num_envs   # előző lépés equitije

        # ── OPT-2: BatchStateBuilder pre-allokált tömbökkel ───────────────
        # [RF-4 FIX] obs_dim paraméterből jön, nem hardcode 54
        state_size = compute_state_size(rlcard_obs_size, num_players)
        self._batch_builder = BatchStateBuilder(
            state_size, num_players,
            obs_dim=rlcard_obs_size,
            max_batch=num_envs,
        )

        # ── Opponent model → id mapping a batch csoportosításhoz ──────────
        self._opp_model_ids = [0] * num_envs  # id(model) → int index

        self._reset_all_envs()

    # ═══════════════════════════════════════════════════════════════════════
    # Public API – VÁLTOZATLAN
    # ═══════════════════════════════════════════════════════════════════════

    def collect(self, n_episodes):
        completed = []
        while len(completed) < n_episodes:
            self._step_opponents_batched()
            self._step_learners_batched()
            self._collect_done_envs(completed, n_episodes)
        return completed[:n_episodes]

    def update_pool(self):
        for i in range(self.num_envs):
            self._opp_models[i] = self.pool.get_opponent(self.model)

    # ═══════════════════════════════════════════════════════════════════════
    # Env management
    # ═══════════════════════════════════════════════════════════════════════

    def _reset_all_envs(self):
        for i in range(self.num_envs):
            self._reset_env(i)

    def _reset_env(self, i):
        # [RF-1 FIX] Nem generálunk véletlenszerű bb/stack értékeket amelyeket
        # az rlcard env nem kap meg. Ehelyett: reset után beolvassuk az env
        # tényleges game konfigurációját, hogy a feature pipeline konzisztens
        # adatot lásson.
        #
        # Korábbi hiba: bb=5, stack=150 generálódott, de envs[i].reset()
        # ignórálta őket → a stack_in_bb, SPR, pot_odds feature-ök
        # szisztematikusan hibás értékeket adtak vissza.
        try:
            s, p = self.envs[i].reset()
            # ── Tényleges game konfig kiolvasása az rlcard game objektumból ──
            game = self.envs[i].game
            bb   = float(getattr(game, 'big_blind',   getattr(game, 'blind', 2.0)))
            sb   = float(getattr(game, 'small_blind', bb / 2.0))
            # initial_stack: reset utáni all_chips maximum értéke (mindenki egyforma)
            raw  = s.get('raw_obs', {})
            chips = raw.get('all_chips', [])
            stack = float(max(chips)) if chips else float(
                getattr(game, 'init_chips', getattr(game, 'chips', 100.0))
            )
            # Sanity: ha az env nem ad értelmes értékeket, maradunk sane defaulton
            if bb < 0.5:   bb = 2.0
            if sb < 0.25:  sb = 1.0
            if stack < bb: stack = bb * 100
        except Exception as exc:
            logger.error(f"Env {i} reset hiba: {exc}", exc_info=True)
            s     = {'obs': [0.0] * self._rlcard_obs_size, 'raw_obs': {}, 'legal_actions': [1]}
            p     = 0
            bb    = 2.0
            sb    = 1.0
            stack = 200.0
        self._bb[i]            = bb
        self._sb[i]            = sb
        self._initial_stack[i] = stack
        self._states[i] = s
        self._players[i] = p
        self._steps[i] = []
        self._step_cnt[i] = 0
        self._active[i] = True
        self._street[i] = detect_street(s)
        self._opp_models[i] = self.pool.get_opponent(self.model)
        # [FIX P2-1] HUD tracker MINDIG új példány kézváltáskor.
        # Korábbi hiba: "if self._trackers[i] is None" → csak az első kéznél
        # jött létre új tracker. Ezután az összes következő kézben a régi
        # ellenfél(ek) stale statisztikái maradtak (pl. VPIP, AF, PFR),
        # miközben az OpponentPool rotált – ez okozta a "HUD-vakság" bugot,
        # ami az AF agressziós spirálhoz is hozzájárult (a learner félreolvasta
        # az ellenfél típusát a régi HUD adatok alapján).
        self._trackers[i] = OpponentHUDTracker(self.num_players)
        self._action_histories[i].clear()
        self._last_equity[i] = 0.5
        self._prev_street[i]  = 0
        self._prev_equity[i]  = 0.5

    # ═══════════════════════════════════════════════════════════════════════
    # OPT-1: BATCHED OPPONENT STEPPING
    # ═══════════════════════════════════════════════════════════════════════
    #
    # Eredeti: minden env-ben egyenként forward pass → N GPU kernel launch
    # Új: gyűjtsd össze az összes "opponent lépést igénylő" env-et,
    #     csoportosítsd modell szerint, 1 batch forward pass per modell
    #
    # Az opponent loop-ban max _MAX_OPP_ROUNDS-szor iterálunk,
    # mert egy env-ben több opponent lépés is kellhet egymás után.
    # ═══════════════════════════════════════════════════════════════════════

    _MAX_OPP_ROUNDS = 20  # max iteráció az opponent batch loop-ban

    def _step_opponents_batched(self):
        """Opponent lépések batchelve – modell szerint csoportosítva."""

        for _round in range(self._MAX_OPP_ROUNDS):
            # ── Gyűjtsd össze melyik env-ek várnak opponent lépésre ───────
            opp_envs = []
            for i in range(self.num_envs):
                if (self._active[i]
                    and not self.envs[i].is_over()
                    and self._players[i] != _LEARNER_ID
                    and self._step_cnt[i] < _MAX_STEPS_PER_HAND):
                    opp_envs.append(i)

            if not opp_envs:
                break

            # ── Csoportosítás modell szerint ──────────────────────────────
            model_groups = {}  # id(model) → list[env_idx]
            for i in opp_envs:
                mid = id(self._opp_models[i])
                if mid not in model_groups:
                    model_groups[mid] = (self._opp_models[i], [])
                model_groups[mid][1].append(i)

            # ── Batch forward pass modell-csoportonként ───────────────────
            for mid, (opp_model, env_list) in model_groups.items():
                n = len(env_list)
                player_ids = [self._players[i] for i in env_list]

                # OPT-2: BatchStateBuilder
                states_batch = self._batch_builder.build_batch(
                    env_indices=env_list,
                    states=self._states,
                    trackers=self._trackers,
                    action_histories=self._action_histories,
                    player_ids=player_ids,
                    bbs=self._bb,
                    sbs=self._sb,
                    initial_stacks=self._initial_stack,
                    streets=self._street,
                ).to(self.device)

                # Legal actions gyűjtése + mask építés
                all_raw_legal = [
                    self._states[i].get('legal_actions', [1]) for i in env_list
                ]
                all_abs_legal = [
                    self.action_mapper.get_abstract_legal_actions(rl)
                    for rl in all_raw_legal
                ]

                # ── OPT-3: inference_mode (gyorsabb mint no_grad) ─────────
                with torch.inference_mode():
                    x = opp_model._encode(states_batch)
                    logits = opp_model.actor_head(x)

                    # Mask alkalmazása batch-ben – logits device-án!
                    mask = torch.full(
                        (n, self.action_size), -1e9, device=logits.device
                    )
                    for idx, legal in enumerate(all_abs_legal):
                        for a in legal:
                            if 0 <= a < self.action_size:
                                mask[idx, a] = 0.0

                    probs = F.softmax(logits + mask, dim=-1)
                    actions = torch.distributions.Categorical(probs).sample()

                # ── Env step-ek végrehajtása (CPU) ────────────────────────
                actions_np = actions.cpu().numpy()
                for idx, i in enumerate(env_list):
                    aa = int(actions_np[idx])
                    ea = self.action_mapper.get_env_action(
                        aa, all_raw_legal[idx]
                    )
                    bn = self._calc_bet_norm(i, aa)

                    self._action_histories[i].append(
                        (self._players[i], aa, bn)
                    )
                    self._trackers[i].record_action(
                        self._players[i], aa, street=self._street[i]
                    )

                    try:
                        ns, np_ = self.envs[i].step(ea)
                        self._states[i] = ns
                        self._players[i] = np_
                        self._step_cnt[i] += 1
                        self._street[i] = detect_street(ns)
                    except Exception as exc:
                        logger.error(f"Env {i} opp step hiba: {exc}",
                                     exc_info=True)
                        self._active[i] = False

                    if self._step_cnt[i] >= _MAX_STEPS_PER_HAND:
                        logger.warning(
                            f"Env {i}: max lépésszám elérve "
                            f"({_MAX_STEPS_PER_HAND} lépés/kéz), "
                            f"opp loop deaktiválva – valószínű végtelen kéz"
                        )
                        self._active[i] = False

    # ═══════════════════════════════════════════════════════════════════════
    # Learner batch stepping – OPT-2 + OPT-3 alkalmazva
    # ═══════════════════════════════════════════════════════════════════════

    def _step_learners_batched(self):
        """Learner lépések batchelve – egyetlen GPU forward pass."""

        learner_envs = [
            i for i in range(self.num_envs)
            if (self._active[i]
                and self._players[i] == _LEARNER_ID
                and not self.envs[i].is_over())
        ]
        if not learner_envs:
            return

        n = len(learner_envs)
        player_ids = [_LEARNER_ID] * n

        # OPT-2: BatchStateBuilder
        states_batch = self._batch_builder.build_batch(
            env_indices=learner_envs,
            states=self._states,
            trackers=self._trackers,
            action_histories=self._action_histories,
            player_ids=player_ids,
            bbs=self._bb,
            sbs=self._sb,
            initial_stacks=self._initial_stack,
            streets=self._street,
        ).to(self.device)

        # Legal actions
        all_raw_legal = [
            self._states[i].get('legal_actions', [1]) for i in learner_envs
        ]
        all_abs_legal = [
            self.action_mapper.get_abstract_legal_actions(rl)
            for rl in all_raw_legal
        ]

        # OPT-3: inference_mode
        with torch.inference_mode():
            x = self.model._encode(states_batch)
            logits_batch = self.model.actor_head(x)
            values_batch = self.model.critic_head(x)

        # ── Per-env feldolgozás (CPU) ─────────────────────────────────────
        for idx, i in enumerate(learner_envs):
            logits = logits_batch[idx].unsqueeze(0)
            value = values_batch[idx]
            legal = all_abs_legal[idx]

            mask = torch.full_like(logits, -1e9)
            for a in legal:
                if 0 <= a < self.action_size:
                    mask[:, a] = 0.0

            probs = F.softmax(logits + mask, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            lp = dist.log_prob(action)

            aa = int(action.item())
            ea = self.action_mapper.get_env_action(
                aa, self._states[i].get('legal_actions', [1])
            )

            # State BEFORE action – a BatchStateBuilder kimenetéből
            state_before = states_batch[idx].cpu()

            bn = self._calc_bet_norm(i, aa)
            self._action_histories[i].append((_LEARNER_ID, aa, bn))
            self._trackers[i].record_action(_LEARNER_ID, aa, street=self._street[i])
            self._steps[i].append(
                (state_before, legal, action.cpu(), lp.cpu(), value.cpu())
            )

            # [RF-9 FIX] Valódi equity számítás rlcard raw_obs alapján
            # (nem a state tensor utolsó dimenziójából, ami még 0.5 default volt)
            equity_now = _compute_equity_for_env(
                self._states[i], num_opponents=self.num_players - 1
            )
            self._last_equity[i] = equity_now
            # [FIX P2-2] A _prev_equity[i] frissítése a street-váltás ágban
            # történik (lentebb, a step() utáni new_street != street feltételben).
            # A korábbi "self._prev_equity[i] = self._prev_equity[i]" no-op sor
            # félrevezető volt – törölve.

            try:
                ns, np_ = self.envs[i].step(ea)
                # [RF-11] Street váltás előtt mentjük az equity-t
                new_street = detect_street(ns)
                if new_street != self._street[i]:
                    # Street-átmenet: frissítjük a prev értékeket
                    self._prev_street[i] = self._street[i]
                    self._prev_equity[i] = equity_now
                self._states[i] = ns
                self._players[i] = np_
                self._step_cnt[i] += 1
                self._street[i] = new_street
            except Exception as exc:
                logger.error(f"Env {i} learner step hiba: {exc}", exc_info=True)
                self._active[i] = False

            if self._step_cnt[i] >= _MAX_STEPS_PER_HAND:
                logger.warning(
                    f"Env {i}: max lépésszám elérve "
                    f"({_MAX_STEPS_PER_HAND} lépés/kéz), "
                    f"learner loop deaktiválva – valószínű végtelen kéz"
                )
                self._active[i] = False

    # ═══════════════════════════════════════════════════════════════════════
    # Done env-ek begyűjtése – VÁLTOZATLAN logika
    # ═══════════════════════════════════════════════════════════════════════

    def _collect_done_envs(self, completed, target):
        for i in range(self.num_envs):
            if not self._active[i] or self.envs[i].is_over():
                if self._steps[i] and len(completed) < target:
                    try:
                        payoffs = self.envs[i].get_payoffs()
                        raw_reward = (
                            float(payoffs[0])
                            if payoffs is not None and len(payoffs) > 0
                            else 0.0
                        )
                    except Exception as exc:
                        logger.debug(f"Env {i} payoffs hiba: {exc}")
                        raw_reward = 0.0

                    # ── Reward shaping ───────────────────────────────────────────────────────
                    #
                    # 1) Draw fold penalty (eredeti): ha postflop fold erős equity-vel → -0.08
                    # 2) [RF-11 FIX] Street-átmenet equity delta: flop/turn/river-en az
                    #    equity növekedés kis pozitív, csökkenés kis negatív jutalmat ad.
                    #    Scale: 0.05 – szándékosan alacsony, hogy a terminális reward
                    #    domináljon. Célja: draw-ok és semibluff-ok gyorsabb tanulása.

                    if self._steps[i]:
                        last_state, last_legal, last_action, last_lp, last_val = self._steps[i][-1]
                        last_action_int = int(last_action.item())
                        last_street = self._street[i]
                        last_eq    = self._last_equity[i]

                        # 1) Draw fold penalty
                        DRAW_FOLD_PENALTY     = 0.08
                        DRAW_EQUITY_THRESHOLD = 0.44  # ~8-9 outos draw szintje
                        # [FIX P2-3] _FOLD_ACTION konstans, nem magic 0
                        if (last_action_int == _FOLD_ACTION
                                and last_street >= 1
                                and last_eq >= DRAW_EQUITY_THRESHOLD):
                            raw_reward -= DRAW_FOLD_PENALTY

                        # 2) [RF-11 FIX] Street-átmenet equity delta reward
                        # Csak akkor adódik hozzá ha ténylegesen volt street-váltás
                        # (prev_street != current street) és mindkét equity érvényes.
                        if last_street > self._prev_street[i] and last_street >= 1:
                            equity_delta = last_eq - self._prev_equity[i]
                            # Clamp: max ±0.3 equity változás vehető figyelembe
                            equity_delta = max(-0.3, min(0.3, equity_delta))
                            raw_reward += equity_delta * _STREET_REWARD_SCALE
                    
                    bb_reward = raw_reward / max(self._bb[i], 1e-6)
                    completed.append((self._steps[i], bb_reward))
                self._reset_env(i)

    # ═══════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _calc_bet_norm(self, env_idx, abstract_action):
        if abstract_action < 2:
            return 0.0
        fractions = [0.0, 0.25, 0.50, 0.75, 1.0]
        return fractions[min(abstract_action - 2, 4)]
