"""
training/collector.py  --  BatchedSyncCollector (v4.2.2-BOOTSTRAP)

Változások v4.2.2-BOOTSTRAP:
    [RF-3b FIX] last_value bootstrap: helyes s_{T+1} state.

    Az eredeti kód (runner.py ~260. sor) a buffer.states[-1]-et használta
    a GAE bootstrap-hoz, ami az UTOLSÓ LEARNER LÉPÉS ELŐTTI állapot (s_T).
    A GAE képlethez V(s_{T+1}) kell – az utolsó env.step() UTÁNI state.

    Javítás:
        _step_learners_batched()-ban, sikeres env.step() után, ha az env
        NEM fejezte be a kezet (is_over() == False), eltároljuk:
          self._bootstrap_next_raw   = ns           (raw state dict)
          self._bootstrap_next_legal = ns legal      (legal actions)
          self._bootstrap_env_idx    = i             (melyik env)

        A collect() visszatérése után a runner.py a
        get_bootstrap_value(model, device) metódust hívja, amely
        felépíti a teljes state tensort (BatchStateBuilder-rel) és
        kiszámolja V(s_{T+1})-et.

    Miért kell eltárolni STEP-kor és nem collect() visszatérésekor?
        A _collect_done_envs() _reset_env()-t hív a lezárt kezekre,
        ami felülírja self._states[i]-t az új kéz initial state-jével.
        Ha ott olvasnánk, azt a post-reset state-et kapnánk, nem s_{T+1}-et.

    Equity a bootstrap state-hez:
        A bootstrap state-hez NEM számítunk új equity-t (felesleges overhead,
        a value head self-normalizált a tréning során). A legutolsó eltárolt
        self._last_equity[i] értéket használjuk – ez s_T equity-je, ami
        s_{T+1}-hez közel van (egy lépés különbség).
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
_MAX_STEPS_PER_HAND = 500
_LEARNER_ID         = 0
_FOLD_ACTION        = 0
_STREET_REWARD_SCALE = 0.05

# Equity estimator singleton (thread-safe: RLock + OrderedDict LRU)
_EQUITY_ESTIMATOR = HandEquityEstimator(n_sim=200, cache_size=20_000)


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


def _compute_equity_for_env(state: dict, num_opponents: int) -> float:
    """Valódi equity becslés az aktuális rlcard state-hez."""
    raw          = state.get('raw_obs', {})
    hole_rlcard  = raw.get('hand', [])
    board_rlcard = raw.get('public_cards', [])
    hole         = _rlcard_cards_to_equity_fmt(hole_rlcard)
    board        = _rlcard_cards_to_equity_fmt(board_rlcard)
    if len(hole) < 2:
        return 0.5
    try:
        return _EQUITY_ESTIMATOR.equity(
            hole_cards=hole,
            board=board,
            num_opponents=max(num_opponents, 1),
        )
    except Exception:
        return 0.5


class BatchedSyncCollector:
    """
    Szinkron, batch-elt tapasztalatgyűjtő.

    Változások:
        [RF-3b] Bootstrap támogatás: get_bootstrap_value() metódus
                a helyes s_{T+1} → V(s_{T+1}) számításhoz.
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
    ) -> None:
        self.num_envs     = num_envs
        self.model        = model
        self.device       = device
        self.num_players  = num_players
        self.action_mapper = action_mapper
        self.pool         = pool
        self.action_size  = PokerActionMapper.NUM_CUSTOM_ACTIONS
        self._rlcard_obs_size = rlcard_obs_size

        logger.info(f"  {num_envs} rlcard env inicializálása...")
        self.envs = [
            rlcard.make(
                'no-limit-holdem',
                config={'game_num_players': num_players},
            )
            for _ in range(num_envs)
        ]
        logger.info("  Env-ek kész.")

        self._history_encoder = ActionHistoryEncoder(
            num_players, self.action_size
        )

        # Per-env állapotok
        self._states          = [None] * num_envs
        self._players         = [0]    * num_envs
        self._trackers        = [None] * num_envs
        self._steps           = [[] for _ in range(num_envs)]
        self._opp_models      = [None] * num_envs
        self._step_cnt        = [0]    * num_envs
        self._active          = [False] * num_envs
        self._action_histories = [
            collections.deque(maxlen=ACTION_HISTORY_LEN)
            for _ in range(num_envs)
        ]
        self._bb             = [2.0]  * num_envs
        self._sb             = [1.0]  * num_envs
        self._initial_stack  = [100.0] * num_envs
        self._street         = [0]    * num_envs
        self._last_equity    = [0.5]  * num_envs
        self._prev_street    = [0]    * num_envs
        self._prev_equity    = [0.5]  * num_envs

        # ── [RF-3b] Bootstrap state tárolás ──────────────────────────────
        # Az utolsó nem-terminális learner lépés UTÁNI state és legal actions.
        # Feltöltve _step_learners_batched()-ban, olvasva get_bootstrap_value()-ban.
        # Közvetlenül env.step() után tárolódik, MIELŐTT _collect_done_envs()
        # esetleg _reset_env()-t hívna (ami felülírná self._states[i]-t).
        self._bootstrap_next_raw:   Optional[dict] = None
        self._bootstrap_next_legal: List[int]       = [1]
        self._bootstrap_next_equity: float          = 0.5
        self._bootstrap_bb:          float          = 2.0
        self._bootstrap_sb:          float          = 1.0
        self._bootstrap_stack:       float          = 100.0
        self._bootstrap_env_idx:     int            = 0

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
        """
        Gyűjt n_episodes lezárt epizódot.

        [RF-3b] A bootstrap state minden learner lépésnél frissül;
        a collect() visszatérése után get_bootstrap_value() a runner
        számára elérhetővé teszi V(s_{T+1})-et.
        """
        completed = []
        while len(completed) < n_episodes:
            self._step_opponents_batched()
            self._step_learners_batched()
            self._collect_done_envs(completed, n_episodes)
        return completed[:n_episodes]

    def get_bootstrap_value(self, model, device: torch.device) -> float:
        """
        Kiszámolja V(s_{T+1})-et az utolsó nem-terminális learner lépés
        UTÁNI állapothoz.

        [RF-3b FIX] Ez váltja ki a runner.py-ban lévő hibás bootstrapot,
        ahol ``buffer.states[-1]`` (pre-action state, s_T) volt használva
        V(s_{T+1}) helyett.

        A visszaadott érték a ``compute_gae(last_value=...)`` paraméterének
        adandó át.

        Returns:
            V(s_{T+1}) float, vagy 0.0 ha nincs érvényes bootstrap state
            (pl. az összes gyűjtött epizód terminálisan zárult).
        """
        if self._bootstrap_next_raw is None:
            return 0.0

        try:
            # Egyetlen state tensor felépítése az eltárolt next state-ből.
            # Új OpponentHUDTracker (semleges prior): a bootstrap value
            # estimate csak hozzávetőleges kell legyen, és nem kell a
            # tréninghez pontosan illeszkedő HUD stat.
            dummy_tracker = OpponentHUDTracker(self.num_players)

            state_t = build_state_tensor(
                state         = self._bootstrap_next_raw,
                tracker       = dummy_tracker,
                action_history= self._action_histories[self._bootstrap_env_idx],
                history_encoder = self._history_encoder,
                num_players   = self.num_players,
                my_player_id  = _LEARNER_ID,
                bb            = self._bootstrap_bb,
                sb            = self._bootstrap_sb,
                initial_stack = self._bootstrap_stack,
                street        = detect_street(self._bootstrap_next_raw),
                equity        = self._bootstrap_next_equity,
            )

            with torch.inference_mode():
                _, v, _ = model.forward(
                    state_t.to(device),
                    self._bootstrap_next_legal,
                )
            return float(v.item())

        except Exception as exc:
            logger.debug(
                f"Bootstrap value számítási hiba (fallback 0.0): {exc}"
            )
            return 0.0

    def update_pool(self) -> None:
        """Ellenfél modellek frissítése a pool-ból."""
        for i in range(self.num_envs):
            self._opp_models[i] = self.pool.get_opponent(self.model)

    # ── Env management ───────────────────────────────────────────────────────

    def _reset_all_envs(self) -> None:
        for i in range(self.num_envs):
            self._reset_env(i)

    def _reset_env(self, i: int) -> None:
        """
        Reset egy env-et, és olvassa vissza az rlcard game tényleges
        bb/stack konfigurációját (RF-1 fix: nincs hamis randomizálás).
        """
        try:
            s, p = self.envs[i].reset()
            game  = self.envs[i].game
            bb    = float(getattr(game, 'big_blind',
                          getattr(game, 'blind', 2.0)))
            sb    = float(getattr(game, 'small_blind', bb / 2.0))
            raw   = s.get('raw_obs', {})
            chips = raw.get('all_chips', [])
            stack = float(max(chips)) if chips else float(
                getattr(game, 'init_chips',
                        getattr(game, 'chips', 100.0))
            )
            if bb < 0.5:   bb = 2.0
            if sb < 0.25:  sb = 1.0
            if stack < bb: stack = bb * 100
        except Exception as exc:
            logger.error(f"Env {i} reset hiba: {exc}", exc_info=True)
            s     = {
                'obs': [0.0] * self._rlcard_obs_size,
                'raw_obs': {},
                'legal_actions': [1],
            }
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
        # P2-1 FIX: HUD tracker mindig új kézváltáskor
        self._trackers[i]      = OpponentHUDTracker(self.num_players)
        self._action_histories[i].clear()
        self._last_equity[i]   = 0.5
        self._prev_street[i]   = 0
        self._prev_equity[i]   = 0.5

    # ── Opponent stepping (batch) ─────────────────────────────────────────────

    _MAX_OPP_ROUNDS = 20

    def _step_opponents_batched(self) -> None:
        """Ellenfél lépések – modell szerint batch-elve."""
        for _round in range(self._MAX_OPP_ROUNDS):
            opp_envs = [
                i for i in range(self.num_envs)
                if (
                    self._active[i]
                    and not self.envs[i].is_over()
                    and self._players[i] != _LEARNER_ID
                    and self._step_cnt[i] < _MAX_STEPS_PER_HAND
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

                all_raw_legal = [
                    self._states[i].get('legal_actions', [1])
                    for i in env_list
                ]
                all_abs_legal = [
                    self.action_mapper.get_abstract_legal_actions(rl)
                    for rl in all_raw_legal
                ]

                with torch.inference_mode():
                    x      = opp_model._encode(states_batch)
                    logits = opp_model.actor_head(x)
                    mask   = torch.full(
                        (n, self.action_size), -1e9, device=logits.device
                    )
                    for idx, legal in enumerate(all_abs_legal):
                        for a in legal:
                            if 0 <= a < self.action_size:
                                mask[idx, a] = 0.0
                    probs   = F.softmax(logits + mask, dim=-1)
                    actions = torch.distributions.Categorical(probs).sample()

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
                        self._states[i]  = ns
                        self._players[i] = np_
                        self._step_cnt[i] += 1
                        self._street[i]  = detect_street(ns)
                    except Exception as exc:
                        logger.error(
                            f"Env {i} opp step hiba: {exc}", exc_info=True
                        )
                        self._active[i] = False

                    if self._step_cnt[i] >= _MAX_STEPS_PER_HAND:
                        logger.warning(
                            f"Env {i}: max lépésszám ({_MAX_STEPS_PER_HAND})"
                        )
                        self._active[i] = False

    # ── Learner stepping (batch) ──────────────────────────────────────────────

    def _step_learners_batched(self) -> None:
        """
        Learner lépések – egyetlen GPU forward pass.

        [RF-3b] Bootstrap state frissítése: minden sikeres, nem-terminális
        env.step() után eltároljuk a next state-et (ns) a bootstrap számára.
        Ez az utolsó ilyen lépés értéke marad meg, mert a GAE csak az utolsó
        buffer bejegyzés s_{T+1}-jét igényli.
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

        all_raw_legal = [
            self._states[i].get('legal_actions', [1])
            for i in learner_envs
        ]
        all_abs_legal = [
            self.action_mapper.get_abstract_legal_actions(rl)
            for rl in all_raw_legal
        ]

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
            self._trackers[i].record_action(
                _LEARNER_ID, aa, street=self._street[i]
            )
            self._steps[i].append(
                (state_before, legal, action.cpu(), lp.cpu(), value.cpu())
            )

            # RF-9: valódi equity
            equity_now            = _compute_equity_for_env(
                self._states[i], num_opponents=self.num_players - 1
            )
            self._last_equity[i]  = equity_now

            try:
                ns, np_ = self.envs[i].step(ea)
                new_street = detect_street(ns)

                # ── [RF-3b] Bootstrap state frissítése ────────────────────
                # KRITIKUS: ezt KÖZVETLENÜL az env.step() után tároljuk el,
                # MIELŐTT _collect_done_envs() esetleg _reset_env()-t hívna.
                # Ha az env nem fejezte be a kezét, ez az s_{T+1} state
                # – pontosan ami a GAE last_value bootstrap-hoz kell.
                if not self.envs[i].is_over() and self._step_cnt[i] + 1 < _MAX_STEPS_PER_HAND:
                    self._bootstrap_next_raw    = ns
                    self._bootstrap_next_legal  = ns.get('legal_actions', [1])
                    self._bootstrap_next_equity = equity_now  # közeli közelítés
                    self._bootstrap_bb          = self._bb[i]
                    self._bootstrap_sb          = self._sb[i]
                    self._bootstrap_stack       = self._initial_stack[i]
                    self._bootstrap_env_idx     = i

                # RF-11: street delta reward
                if new_street != self._street[i]:
                    self._prev_street[i]  = self._street[i]
                    self._prev_equity[i]  = equity_now

                self._states[i]   = ns
                self._players[i]  = np_
                self._step_cnt[i] += 1
                self._street[i]   = new_street

            except Exception as exc:
                logger.error(
                    f"Env {i} learner step hiba: {exc}", exc_info=True
                )
                self._active[i] = False

            if self._step_cnt[i] >= _MAX_STEPS_PER_HAND:
                logger.warning(
                    f"Env {i}: max lépésszám ({_MAX_STEPS_PER_HAND})"
                )
                self._active[i] = False

    # ── Done env-ek begyűjtése ────────────────────────────────────────────────

    def _collect_done_envs(
        self,
        completed: list,
        target: int,
    ) -> None:
        """
        Kész epizódok kinyerése, reward shaping és env reset.

        [RF-3b] A bootstrap state eltárolása a _step_learners_batched()-ban
        történt, tehát itt a _reset_env() hívás nem érinti azt.
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

                    # Reward shaping
                    if self._steps[i]:
                        last_action_int = int(
                            self._steps[i][-1][2].item()
                        )
                        last_street = self._street[i]
                        last_eq     = self._last_equity[i]

                        # Draw fold penalty
                        DRAW_FOLD_PENALTY     = 0.08
                        DRAW_EQUITY_THRESHOLD = 0.44
                        if (
                            last_action_int == _FOLD_ACTION
                            and last_street >= 1
                            and last_eq >= DRAW_EQUITY_THRESHOLD
                        ):
                            raw_reward -= DRAW_FOLD_PENALTY

                        # Street delta reward (RF-11)
                        if last_street > self._prev_street[i] and last_street >= 1:
                            equity_delta = last_eq - self._prev_equity[i]
                            equity_delta = max(-0.3, min(0.3, equity_delta))
                            raw_reward  += equity_delta * _STREET_REWARD_SCALE

                    bb_reward = raw_reward / max(self._bb[i], 1e-6)
                    completed.append((self._steps[i], bb_reward))
                self._reset_env(i)

    # ── Segédek ──────────────────────────────────────────────────────────────

    def _calc_bet_norm(self, env_idx: int, abstract_action: int) -> float:
        """Bet méret normalizálás az action history számára."""
        if abstract_action < 2:
            return 0.0
        fractions = [0.0, 0.25, 0.50, 0.75, 1.0]
        return fractions[min(abstract_action - 2, 4)]
