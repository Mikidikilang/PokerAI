"""
training/collector.py  --  BatchedSyncCollector (v4.3 CONFIG-FULL)

[CONFIG-FULL v4.3] Változások:
  - reward_cfg dict paraméterből jön (nem hard-coded konstans)
  - max_steps_per_hand paraméterből jön
  - equity_estimator paraméterből jön (n_sim, cache_size cfg-ből)
  - Új per-step reward shaping: all-in penalty, fold bonus, stack-blindness
  - _episode_penalty[i] tömb: per-epizód büntetés/bónusz accumulator

Reward shaping összefoglalása (mind a config.json-ban kapcsolható):
  draw_fold_penalty      – postflop fold erős equity-vel → bünteti (meglévő)
  street_reward_scale    – street-átmenet equity delta jutalma (meglévő)
  allin_penalty          – all-in gyenge equity-vel → bünteti (ÚJ)
  fold_bonus             – fold gyenge lapokkal → jutalmazza (ÚJ)
  stack_blindness_penalty – short-stacknél min-raise → bünteti (ÚJ)

[RF-3b bootstrap változatlan]
"""
import collections
import logging
from typing import Dict, List, Optional, Tuple

import rlcard
import torch
import torch.nn.functional as F

from core.equity import HandEquityEstimator
from core.action_mapper import PokerActionMapper
from core.features import (
    ActionHistoryEncoder,
    build_state_tensor,
    detect_street,
    ACTION_HISTORY_LEN,
    BatchStateBuilder,
    compute_state_size,
)
from core.opponent_tracker import OpponentHUDTracker

logger = logging.getLogger("PokerAI")

_LEARNER_ID   = 0
_FOLD_ACTION  = 0
_CALL_ACTION  = 1
_ALLIN_ACTION = 6            # abstract action index (PokerActionMapper szerint)
# Raise action-ök amelyek NEM all-in: ezeket bünteti a stack-blindness
_RAISE_NOT_ALLIN = frozenset([2, 3, 4, 5])

# ── Default reward cfg – SOHA ne módosítsd itt ──────────────────────────────
# A runner.py TrainingConfig-ból összeszedi és átadja.
_DEFAULT_REWARD_CFG = {
    'draw_fold_penalty':              0.08,
    'draw_equity_threshold':          0.44,
    'street_reward_scale':            0.05,
    'allin_penalty_enabled':          False,
    'allin_penalty_equity_threshold': 0.45,
    'allin_penalty_amount':           0.15,
    'fold_bonus_enabled':             False,
    'fold_bonus_equity_threshold':    0.38,
    'fold_bonus_amount':              0.05,
    'stack_blindness_penalty_enabled': False,
    'stack_blindness_bb_threshold':    15.0,
    'stack_blindness_penalty_amount':  0.10,
}


def _rlcard_cards_to_equity_fmt(cards: list) -> list:
    """rlcard kártya formátum ('SA') → equity.py formátum ('As')."""
    _SUIT_MAP = {'S': 's', 'H': 'h', 'D': 'd', 'C': 'c'}
    _RANK_MAP = {
        'A': 'a', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6',
        '7': '7', '8': '8', '9': '9', 'T': 't', 'J': 'j', 'Q': 'q', 'K': 'k',
    }
    result = []
    for card in cards:
        if not card or len(card) < 2:
            continue
        suit = _SUIT_MAP.get(card[0].upper())
        rank = _RANK_MAP.get(card[1].upper())
        if suit and rank:
            result.append(rank + suit)
    return result


class BatchedSyncCollector:
    """
    Szinkron, batch-elt tapasztalatgyűjtő.

    [CONFIG-FULL v4.3]
      reward_cfg:          reward shaping paraméterek dict-je (cfg-ből jön)
      max_steps_per_hand:  cfg.max_steps_per_hand (nem hard-coded)
      equity_estimator:    cfg.equity_n_sim / equity_cache_size (nem hard-coded)
    """

    def __init__(
        self,
        num_envs: int,
        model,
        device: torch.device,
        num_players: int,
        action_mapper: PokerActionMapper,
        model_kwargs: dict,
        pool,
        rlcard_obs_size: int = 54,
        max_steps_per_hand: int = 500,
        reward_cfg: Optional[dict] = None,
        equity_n_sim: int = 100,
        equity_min_sims: int = 30,
        equity_cache_size: int = 100_000,
    ) -> None:
        self.num_envs            = num_envs
        self.model               = model
        self.device              = device
        self.num_players         = num_players
        self.action_mapper       = action_mapper
        self.pool                = pool
        self.action_size         = PokerActionMapper.NUM_CUSTOM_ACTIONS
        self._rlcard_obs_size    = rlcard_obs_size
        self._max_steps_per_hand = max_steps_per_hand

        # Reward shaping konfig (default-okra fallback ha nem adták meg)
        self._reward_cfg = {**_DEFAULT_REWARD_CFG, **(reward_cfg or {})}

        # Equity estimator – cfg értékek alapján (nem singleton, nem hard-coded)
        self._equity_estimator = HandEquityEstimator(
            n_sim=equity_n_sim,
            cache_size=equity_cache_size,
        )
        logger.info(
            f"  Equity estimator: n_sim={equity_n_sim}, "
            f"cache={equity_cache_size:,}"
        )

        # Reward shaping logolása
        rcfg = self._reward_cfg
        logger.info(
            f"  Reward shaping:\n"
            f"    draw_fold_penalty={rcfg['draw_fold_penalty']} "
            f"(threshold={rcfg['draw_equity_threshold']})\n"
            f"    allin_penalty={rcfg['allin_penalty_enabled']} "
            f"amt={rcfg['allin_penalty_amount']} "
            f"eq<{rcfg['allin_penalty_equity_threshold']}\n"
            f"    fold_bonus={rcfg['fold_bonus_enabled']} "
            f"amt={rcfg['fold_bonus_amount']} "
            f"eq<{rcfg['fold_bonus_equity_threshold']}\n"
            f"    stack_blindness={rcfg['stack_blindness_penalty_enabled']} "
            f"amt={rcfg['stack_blindness_penalty_amount']} "
            f"bb<={rcfg['stack_blindness_bb_threshold']}"
        )

        logger.info(f"  {num_envs} rlcard env inicializálása...")
        self.envs = [
            rlcard.make(
                'no-limit-holdem',
                config={'game_num_players': num_players},
            )
            for _ in range(num_envs)
        ]
        logger.info("  Env-ek kész.")

        self._history_encoder = ActionHistoryEncoder(num_players, self.action_size)

        # Per-env állapotok
        self._states          = [None]  * num_envs
        self._players         = [0]     * num_envs
        self._trackers        = [None]  * num_envs
        self._steps           = [[]     for _ in range(num_envs)]
        self._opp_models      = [None]  * num_envs
        self._step_cnt        = [0]     * num_envs
        self._active          = [False] * num_envs
        self._action_histories = [
            collections.deque(maxlen=ACTION_HISTORY_LEN)
            for _ in range(num_envs)
        ]
        self._bb             = [2.0]   * num_envs
        self._sb             = [1.0]   * num_envs
        self._initial_stack  = [100.0] * num_envs
        self._street         = [0]     * num_envs
        self._last_equity    = [0.5]   * num_envs
        self._prev_street    = [0]     * num_envs
        self._prev_equity    = [0.5]   * num_envs

        # [CONFIG-FULL v4.3] Per-epizód penalty accumulator
        # Ide gyűlik a per-step reward shaping büntetés/bónusz összege.
        # _collect_done_envs()-ban adódik a terminal reward-hoz.
        self._episode_penalty = [0.0] * num_envs

        # [RF-3b] Bootstrap state tárolás
        self._bootstrap_next_raw:   Optional[dict] = None
        self._bootstrap_next_legal: List[int]      = [1]
        self._bootstrap_next_equity: float         = 0.5
        self._bootstrap_bb:          float         = 2.0
        self._bootstrap_sb:          float         = 1.0
        self._bootstrap_stack:       float         = 100.0
        self._bootstrap_env_idx:     int           = 0

        # BatchStateBuilder (pre-allokált)
        state_size = compute_state_size(rlcard_obs_size, num_players)
        self._batch_builder = BatchStateBuilder(
            state_size, num_players,
            obs_dim=rlcard_obs_size,
            max_batch=num_envs,
        )

        self._reset_all_envs()

    # ── Publikus API ─────────────────────────────────────────────────────────

    def collect(self, n_episodes: int) -> list:
        """Gyűjt n_episodes lezárt epizódot."""
        completed = []
        while len(completed) < n_episodes:
            self._step_opponents_batched()
            self._step_learners_batched()
            self._collect_done_envs(completed, n_episodes)
        return completed[:n_episodes]

    def get_bootstrap_value(self, model, device: torch.device) -> float:
        """[RF-3b] V(s_{T+1}) az utolsó nem-terminális learner lépés UTÁNI állapothoz."""
        if self._bootstrap_next_raw is None:
            return 0.0
        try:
            dummy_tracker = OpponentHUDTracker(self.num_players)
            state_t = build_state_tensor(
                state          = self._bootstrap_next_raw,
                tracker        = dummy_tracker,
                action_history = self._action_histories[self._bootstrap_env_idx],
                history_encoder= self._history_encoder,
                num_players    = self.num_players,
                my_player_id   = _LEARNER_ID,
                bb             = self._bootstrap_bb,
                sb             = self._bootstrap_sb,
                initial_stack  = self._bootstrap_stack,
                street         = detect_street(self._bootstrap_next_raw),
                equity         = self._bootstrap_next_equity,
            )
            with torch.inference_mode():
                _, v, _ = model.forward(state_t.to(device), self._bootstrap_next_legal)
            return float(v.item())
        except Exception as exc:
            logger.debug(f"Bootstrap value számítási hiba (fallback 0.0): {exc}")
            return 0.0

    def update_pool(self) -> None:
        for i in range(self.num_envs):
            self._opp_models[i] = self.pool.get_opponent(self.model)

    # ── Env management ───────────────────────────────────────────────────────

    def _reset_all_envs(self) -> None:
        for i in range(self.num_envs):
            self._reset_env(i)

    def _reset_env(self, i: int) -> None:
        try:
            s, p  = self.envs[i].reset()
            game  = self.envs[i].game
            bb    = float(getattr(game, 'big_blind', getattr(game, 'blind', 2.0)))
            sb    = float(getattr(game, 'small_blind', bb / 2.0))
            raw   = s.get('raw_obs', {})
            chips = raw.get('all_chips', [])
            stack = float(max(chips)) if chips else float(
                getattr(game, 'init_chips', getattr(game, 'chips', 100.0))
            )
            if bb    < 0.5:   bb    = 2.0
            if sb    < 0.25:  sb    = 1.0
            if stack < bb:    stack = bb * 100
        except Exception as exc:
            logger.error(f"Env {i} reset hiba: {exc}", exc_info=True)
            s     = {'obs': [0.0] * self._rlcard_obs_size, 'raw_obs': {}, 'legal_actions': [1]}
            p, bb, sb, stack = 0, 2.0, 1.0, 200.0

        self._bb[i]            = bb
        self._sb[i]            = sb
        self._initial_stack[i] = stack
        self._states[i]        = s
        self._players[i]       = p
        self._steps[i]         = []
        self._step_cnt[i]      = 0
        self._active[i]        = True
        self._street[i]        = detect_street(s)
        self._opp_models[i]    = self.pool.get_opponent(self.model)
        self._trackers[i]      = OpponentHUDTracker(self.num_players)
        self._action_histories[i].clear()
        self._last_equity[i]   = 0.5
        self._prev_street[i]   = 0
        self._prev_equity[i]   = 0.5
        self._episode_penalty[i] = 0.0  # [CONFIG-FULL v4.3] reset accumulator

    # ── Opponent stepping ─────────────────────────────────────────────────────

    _MAX_OPP_ROUNDS = 20

    def _step_opponents_batched(self) -> None:
        for _round in range(self._MAX_OPP_ROUNDS):
            opp_envs = [
                i for i in range(self.num_envs)
                if (
                    self._active[i]
                    and not self.envs[i].is_over()
                    and self._players[i] != _LEARNER_ID
                    and self._step_cnt[i] < self._max_steps_per_hand
                )
            ]
            if not opp_envs:
                break

            model_groups: Dict[int, Tuple] = {}
            for i in opp_envs:
                mid = id(self._opp_models[i])
                if mid not in model_groups:
                    model_groups[mid] = (self._opp_models[i], [])
                model_groups[mid][1].append(i)

            for mid, (opp_model, env_list) in model_groups.items():
                n          = len(env_list)
                player_ids = [self._players[i] for i in env_list]

                states_batch = self._batch_builder.build_batch(
                    env_indices     = env_list,
                    states          = self._states,
                    trackers        = self._trackers,
                    action_histories= self._action_histories,
                    player_ids      = player_ids,
                    bbs             = self._bb,
                    sbs             = self._sb,
                    initial_stacks  = self._initial_stack,
                    streets         = self._street,
                ).to(self.device)

                all_raw_legal = [self._states[i].get('legal_actions', [1]) for i in env_list]
                all_abs_legal = [self.action_mapper.get_abstract_legal_actions(rl) for rl in all_raw_legal]

                with torch.inference_mode():
                    x      = opp_model._encode(states_batch)
                    logits = opp_model.actor_head(x)
                    mask   = torch.full((n, self.action_size), -1e9, device=logits.device)
                    for idx, legal in enumerate(all_abs_legal):
                        for a in legal:
                            if 0 <= a < self.action_size:
                                mask[idx, a] = 0.0
                    probs   = F.softmax(logits + mask, dim=-1)
                    actions = torch.distributions.Categorical(probs).sample()

                actions_np = actions.cpu().numpy()
                for idx, i in enumerate(env_list):
                    aa = int(actions_np[idx])
                    ea = self.action_mapper.get_env_action(aa, all_raw_legal[idx])
                    bn = self._calc_bet_norm(i, aa)
                    self._action_histories[i].append((self._players[i], aa, bn))
                    self._trackers[i].record_action(self._players[i], aa, street=self._street[i])
                    try:
                        ns, np_ = self.envs[i].step(ea)
                        self._states[i]  = ns
                        self._players[i] = np_
                        self._step_cnt[i] += 1
                        self._street[i]  = detect_street(ns)
                    except Exception as exc:
                        logger.error(f"Env {i} opp step hiba: {exc}", exc_info=True)
                        self._active[i] = False

                    if self._step_cnt[i] >= self._max_steps_per_hand:
                        logger.warning(f"Env {i}: max lépésszám ({self._max_steps_per_hand})")
                        self._active[i] = False

    # ── Learner stepping ──────────────────────────────────────────────────────

    def _step_learners_batched(self) -> None:
        """
        Learner lépések – egyetlen GPU forward pass.

        [CONFIG-FULL v4.3] Per-step reward shaping:
          Az equity kiszámítása UTÁN, env.step() ELŐTT alkalmazzuk a
          per-step büntetéseket / bónuszokat. Ezek a _episode_penalty[i]
          accumulator-ba kerülnek, és a terminal reward-hoz adódnak hozzá.
        """
        learner_envs = [
            i for i in range(self.num_envs)
            if (
                self._active[i]
                and self._players[i] == _LEARNER_ID
                and not self.envs[i].is_over()
            )
        ]
        if not learner_envs:
            return

        n          = len(learner_envs)
        player_ids = [_LEARNER_ID] * n

        states_batch = self._batch_builder.build_batch(
            env_indices     = learner_envs,
            states          = self._states,
            trackers        = self._trackers,
            action_histories= self._action_histories,
            player_ids      = player_ids,
            bbs             = self._bb,
            sbs             = self._sb,
            initial_stacks  = self._initial_stack,
            streets         = self._street,
        ).to(self.device)

        all_raw_legal = [self._states[i].get('legal_actions', [1]) for i in learner_envs]
        all_abs_legal = [self.action_mapper.get_abstract_legal_actions(rl) for rl in all_raw_legal]

        with torch.inference_mode():
            x            = self.model._encode(states_batch)
            logits_batch = self.model.actor_head(x)
            values_batch = self.model.critic_head(x)

        for idx, i in enumerate(learner_envs):
            logits = logits_batch[idx].unsqueeze(0)
            value  = values_batch[idx]
            legal  = all_abs_legal[idx]

            mask = torch.full_like(logits, -1e9)
            for a in legal:
                if 0 <= a < self.action_size:
                    mask[:, a] = 0.0

            probs  = F.softmax(logits + mask, dim=-1)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()
            lp     = dist.log_prob(action)

            aa           = int(action.item())
            ea           = self.action_mapper.get_env_action(
                aa, self._states[i].get('legal_actions', [1])
            )
            state_before = states_batch[idx].cpu()
            bn           = self._calc_bet_norm(i, aa)

            self._action_histories[i].append((_LEARNER_ID, aa, bn))
            self._trackers[i].record_action(_LEARNER_ID, aa, street=self._street[i])
            self._steps[i].append(
                (state_before, legal, action.cpu(), lp.cpu(), value.cpu())
            )

            # RF-9: valódi equity (a döntés ELŐTTI állapothoz)
            equity_now           = self._compute_equity(self._states[i])
            self._last_equity[i] = equity_now

            # ── [CONFIG-FULL v4.3] Per-step reward shaping ─────────────
            self._apply_step_penalties(i, aa, equity_now)

            try:
                ns, np_ = self.envs[i].step(ea)
                new_street = detect_street(ns)

                # [RF-3b] Bootstrap state frissítése KÖZVETLENÜL az env.step() után
                if (
                    not self.envs[i].is_over()
                    and self._step_cnt[i] + 1 < self._max_steps_per_hand
                ):
                    self._bootstrap_next_raw    = ns
                    self._bootstrap_next_legal  = ns.get('legal_actions', [1])
                    self._bootstrap_next_equity = equity_now
                    self._bootstrap_bb          = self._bb[i]
                    self._bootstrap_sb          = self._sb[i]
                    self._bootstrap_stack       = self._initial_stack[i]
                    self._bootstrap_env_idx     = i

                # RF-11: street delta reward
                if new_street != self._street[i]:
                    self._prev_street[i] = self._street[i]
                    self._prev_equity[i] = equity_now

                self._states[i]   = ns
                self._players[i]  = np_
                self._step_cnt[i] += 1
                self._street[i]   = new_street

            except Exception as exc:
                logger.error(f"Env {i} learner step hiba: {exc}", exc_info=True)
                self._active[i] = False

            if self._step_cnt[i] >= self._max_steps_per_hand:
                logger.warning(f"Env {i}: max lépésszám ({self._max_steps_per_hand})")
                self._active[i] = False

    def _apply_step_penalties(self, env_idx: int, action: int, equity: float) -> None:
        """
        [CONFIG-FULL v4.3] Per-step reward shaping alkalmazása.

        Minden bekapcsolt rule-t ellenőrzi és a _episode_penalty accumulator-ba
        írja az összeget. Terminal rewardhoz adódik hozzá a _collect_done_envs()-ban.

        Args:
            env_idx: Env index
            action:  Absztrakt akció (0–6)
            equity:  MC equity a döntés ELŐTTI állapothoz (0.0–1.0)
        """
        rcfg  = self._reward_cfg
        i     = env_idx
        street = self._street[i]

        # ── All-in spam büntetés ────────────────────────────────────────
        # Ha a modell all-innel megy alacsony equity-vel → azonnali büntetés
        if (
            rcfg['allin_penalty_enabled']
            and action == _ALLIN_ACTION
            and equity < rcfg['allin_penalty_equity_threshold']
        ):
            self._episode_penalty[i] -= rcfg['allin_penalty_amount']
            logger.debug(
                f"  [penalty] env={i} all-in spam: "
                f"eq={equity:.2f} < {rcfg['allin_penalty_equity_threshold']:.2f} "
                f"→ -{rcfg['allin_penalty_amount']:.2f} BB"
            )

        # ── Fold bónusz gyenge lapokhoz ─────────────────────────────────
        # Ha fold-ol gyenge lapokkal → jutalmazza (direkt 72o fix)
        if (
            rcfg['fold_bonus_enabled']
            and action == _FOLD_ACTION
            and equity < rcfg['fold_bonus_equity_threshold']
        ):
            self._episode_penalty[i] += rcfg['fold_bonus_amount']
            logger.debug(
                f"  [bonus]   env={i} fold weak hand: "
                f"eq={equity:.2f} < {rcfg['fold_bonus_equity_threshold']:.2f} "
                f"→ +{rcfg['fold_bonus_amount']:.2f} BB"
            )

        # ── Stack-blindness büntetés ────────────────────────────────────
        # Short stacknél a min-raise rossz döntés (push/fold kell)
        if (
            rcfg['stack_blindness_penalty_enabled']
            and action in _RAISE_NOT_ALLIN
        ):
            stack_bb = self._initial_stack[i] / max(self._bb[i], 1e-6)
            if stack_bb <= rcfg['stack_blindness_bb_threshold']:
                self._episode_penalty[i] -= rcfg['stack_blindness_penalty_amount']
                logger.debug(
                    f"  [penalty] env={i} stack-blindness: "
                    f"stack={stack_bb:.1f}BB <= {rcfg['stack_blindness_bb_threshold']} "
                    f"action={action} (raise not all-in) "
                    f"→ -{rcfg['stack_blindness_penalty_amount']:.2f} BB"
                )

    # ── Done env-ek begyűjtése ─────────────────────────────────────────────

    def _collect_done_envs(self, completed: list, target: int) -> None:
        """
        Kész epizódok kinyerése, reward shaping és env reset.
        """
        for i in range(self.num_envs):
            if not self._active[i] or self.envs[i].is_over():
                if self._steps[i] and len(completed) < target:
                    try:
                        payoffs    = self.envs[i].get_payoffs()
                        raw_reward = (
                            float(payoffs[0])
                            if payoffs is not None and len(payoffs) > 0
                            else 0.0
                        )
                    except Exception as exc:
                        logger.debug(f"Env {i} payoffs hiba: {exc}")
                        raw_reward = 0.0

                    rcfg = self._reward_cfg

                    # ── Meglévő terminal reward shaping ─────────────────
                    if self._steps[i]:
                        last_action_int = int(self._steps[i][-1][2].item())
                        last_street     = self._street[i]
                        last_eq         = self._last_equity[i]

                        # Draw fold penalty (meglévő)
                        if (
                            last_action_int == _FOLD_ACTION
                            and last_street >= 1
                            and last_eq >= rcfg['draw_equity_threshold']
                        ):
                            raw_reward -= rcfg['draw_fold_penalty']

                        # Street delta reward (RF-11)
                        if last_street > self._prev_street[i] and last_street >= 1:
                            equity_delta = last_eq - self._prev_equity[i]
                            equity_delta = max(-0.3, min(0.3, equity_delta))
                            raw_reward  += equity_delta * rcfg['street_reward_scale']

                    # ── [CONFIG-FULL v4.3] Per-step penalty hozzáadása ──
                    # Az _apply_step_penalties() által felhalmozott összeget
                    # BB-re normalizálva adjuk hozzá.
                    raw_reward += self._episode_penalty[i]

                    bb_reward = raw_reward / max(self._bb[i], 1e-6)
                    completed.append((self._steps[i], bb_reward))

                self._reset_env(i)  # penalty accumulator nullázódik itt

    # ── Equity számítás ──────────────────────────────────────────────────────

    def _compute_equity(self, state: dict) -> float:
        """Valódi equity becslés az instance equity estimatorral."""
        raw          = state.get('raw_obs', {})
        hole_rlcard  = raw.get('hand', [])
        board_rlcard = raw.get('public_cards', [])
        hole         = _rlcard_cards_to_equity_fmt(hole_rlcard)
        board        = _rlcard_cards_to_equity_fmt(board_rlcard)
        if len(hole) < 2:
            return 0.5
        try:
            return self._equity_estimator.equity(
                hole_cards=hole,
                board=board,
                num_opponents=max(self.num_players - 1, 1),
            )
        except Exception:
            return 0.5

    # ── Segédek ──────────────────────────────────────────────────────────────

    def _calc_bet_norm(self, env_idx: int, abstract_action: int) -> float:
        if abstract_action < 2:
            return 0.0
        fractions = [0.0, 0.25, 0.50, 0.75, 1.0]
        return fractions[min(abstract_action - 2, 4)]
