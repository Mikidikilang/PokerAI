"""
training/runner.py  –  Tréning session és menü rendszer (v4)

v4 változások:
  - STATE_SIZE kalkuláció compute_state_size()-szal
  - BB config mentése a checkpointba
  - trainer optimizer state is mentődik
  - reward_norm state is mentődik
"""

import os
import sys
import time
import logging

import rlcard
import torch

from core.model import AdvancedPokerAI
from core.action_mapper import PokerActionMapper
from core.features import compute_state_size, ACTION_HISTORY_LEN
from core.opponent_tracker import NUM_HUD_STATS

from .buffer import PPOBuffer
from .trainer import PPOTrainer
from .normalizer import RunningMeanStd
from .opponent_pool import OpponentPool
from .collector import BatchedSyncCollector, BB_OPTIONS, STACK_MULTIPLIERS

logger = logging.getLogger("PokerAI")

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False

# ─────────────────────────────────────────────────────────────────────────────
# Hiperparaméterek
# ─────────────────────────────────────────────────────────────────────────────
NUM_ENVS            = 256
BUFFER_COLLECT_SIZE = 1024
HIDDEN_SIZE         = 512
LEARNING_RATE       = 3e-4


def _get_model_info(filename: str) -> str:
    if not os.path.exists(filename):
        return "[ Üres / Új modell ]"
    try:
        ck  = torch.load(filename, map_location='cpu', weights_only=False)
        if isinstance(ck, dict) and 'episodes_trained' in ck:
            eps = ck['episodes_trained']
            t   = ck.get('time_spent', 0.0)
            alg = ck.get('algorithm', 'PPO')
            ss  = ck.get('state_size', '?')
            spd = (f"~{(t/eps)*100_000/60:.1f} perc/100k"
                   if eps > 0 and t > 0 else "n/a")
            return f"[ {eps:,} ep | {alg} | state={ss} | {spd} ]"
        return "[ Régi formátum ]"
    except Exception:
        return "[ Hiba a fájl olvasásakor ]"


# ─────────────────────────────────────────────────────────────────────────────
# Menü
# ─────────────────────────────────────────────────────────────────────────────

def menu_system():
    print("\n" + "=" * 70)
    print("  POKER AI v4  –  PPO + Self-Play")
    print("=" * 70)
    print("\n  v4 feature-ök:")
    print("    • BB/SB explicit + M-ratio + stack depth kategória")
    print("    • Street one-hot (preflop/flop/turn/river)")
    print("    • Pot odds + call amount + facing bet kontextus")
    print("    • Board texture (paired/monotone/connectedness/...)")
    print("    • Bet sizing az action history-ban")
    print("    • Valódi HUD statisztikák (VPIP/PFR/AF/3bet%/...)")
    print("    • Monte Carlo kéz equity")
    print("    • BB-ben mért reward (stakes-független tanulás)")
    print("    ⚠ INKOMPATIBILIS v3.x checkpointokkal!")
    print("\n  Játékosszám:")
    print("    2 = Heads-Up  |  6 = 6-max  |  9 = Full ring  |  A = összes")
    print("=" * 70)

    choice = input("\n  Választás [2/6/9/A]: ").strip().upper()

    if choice == 'A':
        return 'ALL', None

    try:
        num_players = int(choice)
        if not (2 <= num_players <= 9):
            raise ValueError
    except ValueError:
        print("  Hibás bemenet → 6 max.")
        num_players = 6

    default_name = f"{num_players}max_ppo_v4.pth"
    info = _get_model_info(default_name)
    print(f"\n  Fájl: {default_name}  {info}")
    custom = input("  Más név? (ENTER = alapértelmezett): ").strip()
    filename = custom if custom else default_name
    if not filename.endswith('.pth'):
        filename += '.pth'

    print(f"  → {filename}")
    print("=" * 70 + "\n")
    return num_players, filename


# ─────────────────────────────────────────────────────────────────────────────
# Fő tréning session
# ─────────────────────────────────────────────────────────────────────────────

def run_training_session(num_players: int, filename: str,
                          episodes_to_run: int):
    # Per-session log fájl a logs/ mappában
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_setup import setup_logging
    setup_logging(log_file="training.log", num_players=num_players)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── STATE_SIZE kalkuláció ────────────────────────────────────────────────
    temp_env = rlcard.make('no-limit-holdem',
                           config={'game_num_players': num_players})
    temp_state, _ = temp_env.reset()
    rlcard_obs_size = len(temp_state['obs'])
    del temp_env

    STATE_SIZE  = compute_state_size(rlcard_obs_size, num_players)
    ACTION_SIZE = PokerActionMapper.NUM_CUSTOM_ACTIONS

    logger.info(f"STATE_SIZE kalkuláció ({num_players}p):")
    logger.info(f"  rlcard obs:     {rlcard_obs_size}")
    logger.info(f"  HUD stats:      {num_players * NUM_HUD_STATS}")
    logger.info(f"  stack+blind:    8")
    logger.info(f"  street:         4")
    logger.info(f"  pot odds:       4")
    logger.info(f"  board texture:  6")
    logger.info(f"  action hist:    {ACTION_HISTORY_LEN * (num_players * ACTION_SIZE + 1)}")
    logger.info(f"  position:       {2 * num_players}")
    logger.info(f"  equity:         1")
    logger.info(f"  TOTAL:          {STATE_SIZE}")

    # ── Model és komponensek ─────────────────────────────────────────────────
    model_kwargs = {
        'state_size':  STATE_SIZE,
        'action_size': ACTION_SIZE,
        'hidden_size': HIDDEN_SIZE,
    }

    learner       = AdvancedPokerAI(**model_kwargs).to(device)
    buffer        = PPOBuffer()
    trainer       = PPOTrainer(learner, lr=LEARNING_RATE, device=device)
    pool          = OpponentPool(AdvancedPokerAI, model_kwargs)
    action_mapper = PokerActionMapper()
    reward_norm   = RunningMeanStd()

    # ── TensorBoard ──────────────────────────────────────────────────────────
    writer = None
    if _TB:
        log_dir = f"runs/{num_players}max_v4_{int(time.time())}"
        writer  = SummaryWriter(log_dir)
        logger.info(f"TensorBoard: {log_dir}")

    # ── Checkpoint betöltés ──────────────────────────────────────────────────
    start_episode    = 0
    total_time_spent = 0.0

    if os.path.exists(filename):
        logger.info(f"Checkpoint betöltése: {filename}")
        try:
            ck = torch.load(filename, map_location=device, weights_only=False)
            if isinstance(ck, dict) and 'state_dict' in ck:
                model_dict  = learner.state_dict()
                pretrained  = {
                    k: v for k, v in ck['state_dict'].items()
                    if k in model_dict and v.size() == model_dict[k].size()
                }
                n_loaded = len(pretrained)
                n_total  = len(model_dict)
                if n_loaded < n_total:
                    logger.warning(
                        f"Részleges betöltés: {n_loaded}/{n_total} layer "
                        f"(input_proj méretváltozás v3.x→v4 esetén normális)"
                    )
                model_dict.update(pretrained)
                learner.load_state_dict(model_dict)

                start_episode    = ck.get('episodes_trained', 0)
                total_time_spent = ck.get('time_spent', 0.0)

                if 'trainer' in ck:
                    trainer.load_state_dict(ck['trainer'])
                if 'reward_norm' in ck:
                    reward_norm.load_state_dict(ck['reward_norm'])

            else:
                learner.load_state_dict(ck, strict=False)
            logger.info(f"Folytatás: {start_episode:,}. epizódtól.")
        except Exception as exc:
            logger.error(f"Checkpoint betöltési hiba: {exc}", exc_info=True)

    pool.snapshot(learner)

    # ── Collector ────────────────────────────────────────────────────────────
    logger.info(f"BatchedSyncCollector inicializálása ({NUM_ENVS} env)...")
    collector = BatchedSyncCollector(
        num_envs      = NUM_ENVS,
        model         = learner,
        device        = device,
        num_players   = num_players,
        action_mapper = action_mapper,
        model_kwargs  = model_kwargs,
        pool          = pool,
    )

    # ── Fő tréning loop ──────────────────────────────────────────────────────
    target_episodes = start_episode + episodes_to_run
    total_collected = start_episode
    session_start   = time.time()
    metrics: dict   = {}

    logger.info("=" * 70)
    logger.info(f"Tréning indul | {num_players}p | PPO + Self-Play v4")
    logger.info(
        f"Cél: {target_episodes:,} | State: {STATE_SIZE} | "
        f"Device: {device} | Envs: {NUM_ENVS}"
    )
    logger.info("=" * 70)

    while total_collected < target_episodes:
        to_collect = min(BUFFER_COLLECT_SIZE,
                         target_episodes - total_collected)

        # Pool snapshot
        prev_snap = total_collected // OpponentPool.SNAPSHOT_INTERVAL
        next_snap = (total_collected + to_collect) // OpponentPool.SNAPSHOT_INTERVAL
        if next_snap > prev_snap and total_collected > start_episode:
            pool.snapshot(learner)
            collector.update_pool()
            logger.debug(f"Pool snapshot ({len(pool)} verzió)")

        # Gyűjtés
        learner.eval()
        try:
            all_episodes = collector.collect(to_collect)
        except Exception as exc:
            logger.error(f"Gyűjtési hiba: {exc}", exc_info=True)
            continue
        learner.train()

        if not all_episodes:
            logger.warning("Üres gyűjtési batch")
            continue

        # Buffer feltöltése
        for steps, bb_reward in all_episodes:
            reward_norm.update(bb_reward)
            norm_r  = reward_norm.normalize(bb_reward)
            n_steps = len(steps)
            for i, (s, la, a, lp, v) in enumerate(steps):
                buffer.add(
                    state         = s.unsqueeze(0),
                    legal_actions = la,
                    action        = a,
                    log_prob      = lp,
                    value         = v,
                    reward        = norm_r if i == n_steps - 1 else 0.0,
                    episode_end   = (i == n_steps - 1),
                )

        collected_this_batch  = len(all_episodes)
        total_collected      += collected_this_batch

        # PPO update
        try:
            metrics = trainer.update(buffer)
        except Exception as exc:
            logger.error(f"PPO update hiba: {exc}", exc_info=True)
            buffer.reset()
            metrics = {}

        # TensorBoard
        if writer and metrics:
            for k, mv in metrics.items():
                writer.add_scalar(f'Loss/{k}', mv, total_collected)
            writer.add_scalar('Reward/mean_bb', reward_norm.mean,      total_collected)
            writer.add_scalar('Reward/std_bb',  reward_norm.var**0.5,  total_collected)
            writer.add_scalar('Pool/size',       len(pool),             total_collected)
            writer.add_scalar('Train/lr',
                              trainer.optimizer.param_groups[0]['lr'],  total_collected)

        # 1000 epizódonként: log + checkpoint
        prev_1k = (total_collected - collected_this_batch) // 1_000
        curr_1k = total_collected // 1_000
        if curr_1k > prev_1k:
            elapsed     = time.time() - session_start
            done_so_far = total_collected - start_episode
            remaining   = target_episodes - total_collected
            eps_sec     = done_so_far / max(elapsed, 1e-6)
            eta_str     = time.strftime("%H:%M:%S",
                                        time.gmtime(remaining / max(eps_sec, 1e-6)))
            ela_str     = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            lr          = trainer.optimizer.param_groups[0]['lr']

            logger.info(
                f"Ep {total_collected:>8,}/{target_episodes:,} | "
                f"Eltelt {ela_str} | ETA {eta_str} | LR {lr:.2e} | "
                f"Actor {metrics.get('actor',   float('nan')):+.4f} | "
                f"Critic {metrics.get('critic', float('nan')):.4f} | "
                f"Ent {metrics.get('entropy',   float('nan')):.4f} | "
                f"Pool {len(pool):2d}"
            )

            _save_checkpoint(
                filename, learner, trainer, reward_norm,
                total_collected,
                total_time_spent + elapsed,
                STATE_SIZE, ACTION_SIZE
            )

    if writer:
        writer.close()

    elapsed = time.time() - session_start
    _save_checkpoint(
        filename, learner, trainer, reward_norm,
        target_episodes,
        total_time_spent + elapsed,
        STATE_SIZE, ACTION_SIZE
    )
    logger.info(f"KÉSZ  →  {filename}  ({target_episodes:,} epizód)")


def _save_checkpoint(filename, model, trainer, reward_norm,
                     episodes, time_spent, state_size, action_size):
    try:
        torch.save({
            'state_dict':       model.state_dict(),
            'trainer':          trainer.state_dict(),
            'reward_norm':      reward_norm.state_dict(),
            'episodes_trained': episodes,
            'time_spent':       time_spent,
            'algorithm':        'PPO_SelfPlay_v4',
            'state_size':       state_size,
            'action_size':      action_size,
            'bb_options':       BB_OPTIONS,
            'stack_multipliers':STACK_MULTIPLIERS,
        }, filename)
        logger.debug(f"Checkpoint mentve: {filename}")
    except Exception as exc:
        logger.error(f"Checkpoint mentési hiba: {exc}", exc_info=True)
