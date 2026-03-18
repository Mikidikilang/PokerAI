"""
training/collector.py  –  BatchedSyncCollector (v4)

v4 javítások és újítások:
  BUG-1 JAVÍTVA: egyetlen _encode() call learner batch-nél
  BUG-2 JAVÍTVA: state_before_action mentése (nem utána!)
  BUG-3 JAVÍTVA: csak build_state_tensor() használva (no placeholder)

  ÚJ: per-env BB/SB/stack randomizálás reset-kor
  ÚJ: bet_size_norm az action history-ban
  ÚJ: street tracking per env
  ÚJ: OpponentHUDTracker record_action() street-kontextussal
"""

import collections
import logging
import random

import rlcard
import torch
import torch.nn.functional as F

from core.action_mapper import PokerActionMapper
from core.features import (
    ActionHistoryEncoder,
    build_state_tensor,
    detect_street,
    ACTION_HISTORY_LEN,
)
from core.opponent_tracker import OpponentHUDTracker

logger = logging.getLogger("PokerAI")

_MAX_STEPS_PER_HAND = 500
_LEARNER_ID         = 0   # a learner mindig player_id=0

# BB/SB és stack randomizálás (NOTES_MASTER.txt §13)
BB_OPTIONS        = [1, 2, 5, 10, 25]
STACK_MULTIPLIERS = [20, 30, 40, 60, 80, 100, 150, 200]


class BatchedSyncCollector:
    """
    Párhuzamos rlcard env gyűjtő.

    num_envs=256 env fut egyszerre; az ellenfél lépések CPU-n,
    a learner lépések GPU batch-ben futnak.
    """

    def __init__(self, num_envs, model, device, num_players,
                 action_mapper: PokerActionMapper,
                 model_kwargs: dict, pool):

        self.num_envs      = num_envs
        self.model         = model
        self.device        = device
        self.num_players   = num_players
        self.action_mapper = action_mapper
        self.pool          = pool
        self.action_size   = PokerActionMapper.NUM_CUSTOM_ACTIONS

        logger.info(f"  {num_envs} rlcard env inicializálása...")
        self.envs = [
            rlcard.make('no-limit-holdem',
                        config={'game_num_players': num_players})
            for _ in range(num_envs)
        ]
        logger.info("  Env-ek kész.")

        # Shared encoder (stateless, lehet megosztani)
        self._history_encoder = ActionHistoryEncoder(num_players, self.action_size)

        # Per-env state
        self._states          = [None]  * num_envs
        self._players         = [0]     * num_envs
        self._trackers        = [None]  * num_envs
        self._steps           = [[]    for _ in range(num_envs)]
        self._opp_models      = [None]  * num_envs
        self._step_cnt        = [0]     * num_envs
        self._active          = [False] * num_envs
        self._action_histories= [
            collections.deque(maxlen=ACTION_HISTORY_LEN)
            for _ in range(num_envs)
        ]
        self._bb              = [2.0]   * num_envs
        self._sb              = [1.0]   * num_envs
        self._initial_stack   = [100.0] * num_envs
        self._street          = [0]     * num_envs  # preflop

        self._reset_all_envs()

    # ── Publikus API ──────────────────────────────────────────────────────────

    def collect(self, n_episodes: int) -> list:
        """
        Összegyűjt n_episodes kész epizódot.
        Visszatér: list of (steps, learner_reward_in_bb)
        """
        completed = []
        while len(completed) < n_episodes:
            self._step_opponents()
            self._step_learners_batched()
            self._collect_done_envs(completed, n_episodes)
        return completed[:n_episodes]

    def update_pool(self):
        for i in range(self.num_envs):
            self._opp_models[i] = self.pool.get_opponent(self.model)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_all_envs(self):
        for i in range(self.num_envs):
            self._reset_env(i)

    def _reset_env(self, i: int):
        # v4: BB/SB/stack randomizálás
        bb    = random.choice(BB_OPTIONS)
        sb    = bb // 2
        stack = bb * random.choice(STACK_MULTIPLIERS)

        self._bb[i]            = float(bb)
        self._sb[i]            = float(sb)
        self._initial_stack[i] = float(stack)

        try:
            s, p = self.envs[i].reset()
        except Exception as exc:
            logger.error(f"Env {i} reset hiba: {exc}", exc_info=True)
            s = {'obs': [0.0] * 72, 'raw_obs': {}, 'legal_actions': [1]}
            p = 0

        self._states[i]    = s
        self._players[i]   = p
        self._steps[i]     = []
        self._step_cnt[i]  = 0
        self._active[i]    = True
        self._street[i]    = detect_street(s)
        self._opp_models[i]= self.pool.get_opponent(self.model)

        # Tracker: reset (új kéz → új statisztikák NEM törlődnek, csak history)
        if self._trackers[i] is None:
            self._trackers[i] = OpponentHUDTracker(self.num_players)
        # Action history törlése
        self._action_histories[i].clear()

    # ── Opponent lépések ──────────────────────────────────────────────────────

    def _step_opponents(self):
        for i in range(self.num_envs):
            if not self._active[i]:
                continue

            while True:
                current_player = self._players[i]
                if current_player == _LEARNER_ID or self.envs[i].is_over():
                    break

                opp_model = self._opp_models[i]
                state_t   = build_state_tensor(
                    self._states[i],
                    self._trackers[i],
                    self._action_histories[i],
                    self._history_encoder,
                    self.num_players,
                    my_player_id  = current_player,
                    bb            = self._bb[i],
                    sb            = self._sb[i],
                    initial_stack = self._initial_stack[i],
                    street        = self._street[i],
                )

                raw_legal  = self._states[i].get('legal_actions', [1])
                abs_legal  = self.action_mapper.get_abstract_legal_actions(raw_legal)

                with torch.no_grad():
                    action, _, _, _, _ = opp_model.get_action(
                        state_t, abs_legal, deterministic=False
                    )
                abstract_action = int(action.item())
                env_action      = self.action_mapper.get_env_action(
                    abstract_action, raw_legal
                )

                # Bet size normálás history-hoz
                bet_norm = self._calc_bet_norm(i, abstract_action)

                self._action_histories[i].append(
                    (current_player, abstract_action, bet_norm)
                )
                self._trackers[i].record_action(
                    current_player, abstract_action,
                    street=self._street[i]
                )

                try:
                    next_s, next_p = self.envs[i].step(env_action)
                    self._states[i]  = next_s
                    self._players[i] = next_p
                    self._step_cnt[i]+= 1
                    self._street[i]   = detect_street(next_s)
                except Exception as exc:
                    logger.error(f"Env {i} opp step hiba: {exc}", exc_info=True)
                    self._active[i] = False
                    break

                if self._step_cnt[i] >= _MAX_STEPS_PER_HAND:
                    self._active[i] = False
                    break

    # ── Learner batch inference ───────────────────────────────────────────────

    def _step_learners_batched(self):
        learner_envs = [
            i for i in range(self.num_envs)
            if (self._active[i]
                and self._players[i] == _LEARNER_ID
                and not self.envs[i].is_over())
        ]
        if not learner_envs:
            return

        states_list = []
        legal_list  = []
        for i in learner_envs:
            state_t = build_state_tensor(
                self._states[i],
                self._trackers[i],
                self._action_histories[i],
                self._history_encoder,
                self.num_players,
                my_player_id  = _LEARNER_ID,
                bb            = self._bb[i],
                sb            = self._sb[i],
                initial_stack = self._initial_stack[i],
                street        = self._street[i],
            )
            states_list.append(state_t)
            raw = self._states[i].get('legal_actions', [1])
            legal_list.append(
                self.action_mapper.get_abstract_legal_actions(raw)
            )

        states_batch = torch.cat(states_list, dim=0).to(self.device)

        # BUG-1 JAVÍTVA: egyetlen _encode() call!
        with torch.no_grad():
            x            = self.model._encode(states_batch)
            logits_batch = self.model.actor_head(x)
            values_batch = self.model.critic_head(x)

        for idx, i in enumerate(learner_envs):
            logits = logits_batch[idx].unsqueeze(0)
            value  = values_batch[idx]
            legal  = legal_list[idx]

            mask = torch.full_like(logits, -1e9)
            for a in legal:
                if 0 <= a < self.action_size:
                    mask[:, a] = 0.0

            probs  = F.softmax(logits + mask, dim=-1)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()
            lp     = dist.log_prob(action)

            abstract_action = int(action.item())
            env_action      = self.action_mapper.get_env_action(
                abstract_action,
                self._states[i].get('legal_actions', [1])
            )

            # BUG-2 JAVÍTVA: state_before mentése
            state_before = states_list[idx].squeeze(0).cpu()

            bet_norm = self._calc_bet_norm(i, abstract_action)
            self._action_histories[i].append(
                (_LEARNER_ID, abstract_action, bet_norm)
            )
            self._trackers[i].record_action(
                _LEARNER_ID, abstract_action,
                street=self._street[i]
            )

            self._steps[i].append((
                state_before,
                legal,
                action.cpu(),
                lp.cpu(),
                value.cpu(),
            ))

            try:
                next_s, next_p = self.envs[i].step(env_action)
                self._states[i]  = next_s
                self._players[i] = next_p
                self._step_cnt[i]+= 1
                self._street[i]   = detect_street(next_s)
            except Exception as exc:
                logger.error(f"Env {i} learner step hiba: {exc}", exc_info=True)
                self._active[i] = False

            if self._step_cnt[i] >= _MAX_STEPS_PER_HAND:
                self._active[i] = False

    # ── Done env-ek kezelése ──────────────────────────────────────────────────

    def _collect_done_envs(self, completed: list, target: int):
        for i in range(self.num_envs):
            if not self._active[i] or self.envs[i].is_over():
                if self._steps[i] and len(completed) < target:
                    try:
                        payoffs = self.envs[i].get_payoffs()
                        # BUG-4 JAVÍTVA: get_payoffs() numpy arrayt ad vissza.
                        # `if payoffs` → ValueError (ambiguous truth value numpy array).
                        # Helyes: len() check, nem bool cast!
                        if payoffs is not None and len(payoffs) > 0:
                            raw_reward = float(payoffs[0])
                        else:
                            raw_reward = 0.0
                    except Exception as exc:
                        logger.debug(f"Env {i} payoffs hiba: {exc}")
                        raw_reward = 0.0
                    bb_reward = raw_reward / max(self._bb[i], 1e-6)
                    completed.append((self._steps[i], bb_reward))
                self._reset_env(i)

    # ── Segédmetódusok ────────────────────────────────────────────────────────

    def _calc_bet_norm(self, env_idx: int, abstract_action: int) -> float:
        """Raise esetén pot%-os bet méret becslése (0.0 ha nem raise)."""
        if abstract_action < 2:
            return 0.0
        # Raise tier → fraction (NOTES_MASTER.txt §3)
        fractions = [0.0, 0.25, 0.50, 0.75, 1.0]
        tier = abstract_action - 2
        return fractions[min(tier, 4)]
