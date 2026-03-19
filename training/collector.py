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
"""
import collections, logging, random
import rlcard, torch, torch.nn.functional as F
from core.action_mapper import PokerActionMapper
from core.features import (
    ActionHistoryEncoder, build_state_tensor, detect_street,
    ACTION_HISTORY_LEN, BatchStateBuilder, compute_state_size,
)
from core.opponent_tracker import OpponentHUDTracker

logger = logging.getLogger("PokerAI")
_MAX_STEPS_PER_HAND = 500
_LEARNER_ID = 0
BB_OPTIONS = [1, 2, 5, 10, 25]
STACK_MULTIPLIERS = [20, 30, 40, 60, 80, 100, 150, 200]


class BatchedSyncCollector:
    def __init__(self, num_envs, model, device, num_players, action_mapper,
                 model_kwargs, pool):
        self.num_envs = num_envs
        self.model = model
        self.device = device
        self.num_players = num_players
        self.action_mapper = action_mapper
        self.pool = pool
        self.action_size = PokerActionMapper.NUM_CUSTOM_ACTIONS

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
        self._street = [0] * num_envs

        # ── OPT-2: BatchStateBuilder pre-allokált tömbökkel ───────────────
        state_size = compute_state_size(54, num_players)
        self._batch_builder = BatchStateBuilder(state_size, num_players, max_batch=num_envs)

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
        bb = random.choice(BB_OPTIONS)
        sb = bb // 2
        stack = bb * random.choice(STACK_MULTIPLIERS)
        self._bb[i] = float(bb)
        self._sb[i] = float(sb)
        self._initial_stack[i] = float(stack)
        try:
            s, p = self.envs[i].reset()
        except Exception as exc:
            logger.error(f"Env {i} reset hiba: {exc}", exc_info=True)
            s = {'obs': [0.0] * 54, 'raw_obs': {}, 'legal_actions': [1]}
            p = 0
        self._states[i] = s
        self._players[i] = p
        self._steps[i] = []
        self._step_cnt[i] = 0
        self._active[i] = True
        self._street[i] = detect_street(s)
        self._opp_models[i] = self.pool.get_opponent(self.model)
        if self._trackers[i] is None:
            self._trackers[i] = OpponentHUDTracker(self.num_players)
        self._action_histories[i].clear()

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

            try:
                ns, np_ = self.envs[i].step(ea)
                self._states[i] = ns
                self._players[i] = np_
                self._step_cnt[i] += 1
                self._street[i] = detect_street(ns)
            except Exception as exc:
                logger.error(f"Env {i} learner step hiba: {exc}", exc_info=True)
                self._active[i] = False

            if self._step_cnt[i] >= _MAX_STEPS_PER_HAND:
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
