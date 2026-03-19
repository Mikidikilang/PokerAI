"""
training/runner.py  --  Tréning session és menü rendszer (v4 OPTIMIZED)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIMALIZÁCIÓK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  OPT-5: NUM_ENVS 256→512 – jobb GPU batch kihasználtság
  OPT-6: BUFFER_COLLECT_SIZE 1024→2048 – hatékonyabb PPO update
  OPT-7: torch.compile(mode="reduce-overhead") ha elérhető

  CHECKPOINT FORMÁTUM: VÁLTOZATLAN – régi .pth fájlok betölthetők.
"""
import os, sys, time, logging
import rlcard, torch
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

# ── OPT-5 + OPT-6: Nagyobb batch méretek ────────────────────────────────────
NUM_ENVS            = 512     # Volt: 256 → GPU jobban kihasználva
BUFFER_COLLECT_SIZE = 2048    # Volt: 1024 → jobb gradiens becslés
HIDDEN_SIZE         = 512
LEARNING_RATE       = 3e-4


def _get_model_info(filename):
    if not os.path.exists(filename):
        return "[ Üres / Új modell ]"
    try:
        ck = torch.load(filename, map_location='cpu', weights_only=False)
        if isinstance(ck, dict) and 'episodes_trained' in ck:
            eps = ck['episodes_trained']
            t = ck.get('time_spent', 0.0)
            alg = ck.get('algorithm', 'PPO')
            ss = ck.get('state_size', '?')
            spd = (f"~{(t/eps)*100_000/60:.1f} perc/100k"
                   if eps > 0 and t > 0 else "n/a")
            return f"[ {eps:,} ep | {alg} | state={ss} | {spd} ]"
        return "[ Régi formátum ]"
    except Exception:
        return "[ Hiba a fájl olvasásakor ]"


def menu_system():
    print("\n" + "=" * 70)
    print("  POKER AI v4  --  PPO + Self-Play (OPTIMIZED)")
    print("=" * 70)
    print("\n  Játékosszám:")
    print("    2 = Heads-Up  |  6 = 6-max  |  9 = Full ring  |  A = összes")
    print(f"\n  Env-ek: {NUM_ENVS} | Buffer: {BUFFER_COLLECT_SIZE} | PPO epochs: 4")
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


def _try_compile(model, device):
    """
    OPT-7: torch.compile ha elérhető (PyTorch 2.0+).
    Fallback: eredeti modell ha compile nem megy.
    """
    if not hasattr(torch, 'compile'):
        logger.info("torch.compile nem elérhető (PyTorch <2.0)")
        return model

    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        # Warmup – egy dummy forward, hogy a compile lefusson
        logger.info("torch.compile(mode='reduce-overhead') aktív – warmup...")
        dummy_state_size = model.input_proj.in_features
        dummy = torch.randn(1, dummy_state_size, device=device)
        with torch.inference_mode():
            compiled._encode(dummy)
        logger.info("torch.compile warmup kész")
        return compiled
    except Exception as exc:
        logger.warning(f"torch.compile sikertelen, fallback eredeti modellre: {exc}")
        return model


def run_training_session(num_players, filename, episodes_to_run):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_setup import setup_logging
    setup_logging(log_file="training.log", num_players=num_players)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    temp_env = rlcard.make('no-limit-holdem',
                            config={'game_num_players': num_players})
    temp_state, _ = temp_env.reset()
    rlcard_obs_size = len(temp_state['obs'])
    del temp_env
    STATE_SIZE = compute_state_size(rlcard_obs_size, num_players)
    ACTION_SIZE = PokerActionMapper.NUM_CUSTOM_ACTIONS

    logger.info(f"STATE_SIZE ({num_players}p): rlcard_obs={rlcard_obs_size} "
                f"TOTAL={STATE_SIZE}")

    model_kwargs = {
        'state_size': STATE_SIZE,
        'action_size': ACTION_SIZE,
        'hidden_size': HIDDEN_SIZE,
    }
    learner = AdvancedPokerAI(**model_kwargs).to(device)
    buffer = PPOBuffer()
    trainer = PPOTrainer(learner, lr=LEARNING_RATE, device=device)
    pool = OpponentPool(
        AdvancedPokerAI, 
        model_kwargs, 
        device=device,
        bot_ratio=0.1,  # 10% bot arány kezdésnek
        bot_types=['fish', 'nit', 'calling_station', 'lag'],
        num_players=num_players,
        state_size=STATE_SIZE
    )
    action_mapper = PokerActionMapper()
    reward_norm = RunningMeanStd()

    writer = None
    if _TB:
        log_dir = f"runs/{num_players}max_v4_{int(time.time())}"
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        logger.info(f"TensorBoard: {log_dir}")

    start_episode = 0
    total_time_spent = 0.0
    if os.path.exists(filename):
        logger.info(f"Checkpoint betöltése: {filename}")
        try:
            ck = torch.load(filename, map_location=device, weights_only=False)
            if isinstance(ck, dict) and 'state_dict' in ck:
                model_dict = learner.state_dict()
                pretrained = {
                    k: v for k, v in ck['state_dict'].items()
                    if k in model_dict and v.size() == model_dict[k].size()
                }
                n_loaded = len(pretrained)
                n_total = len(model_dict)
                if n_loaded < n_total:
                    logger.warning(
                        f"Részleges betöltés: {n_loaded}/{n_total} layer"
                    )
                model_dict.update(pretrained)
                learner.load_state_dict(model_dict)
                start_episode = ck.get('episodes_trained', 0)
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

    # ── OPT-7: torch.compile ─────────────────────────────────────────────
    learner = _try_compile(learner, device)

    pool.snapshot(learner)

    logger.info(f"BatchedSyncCollector inicializálása ({NUM_ENVS} env)...")
    collector = BatchedSyncCollector(
        num_envs=NUM_ENVS,
        model=learner,
        device=device,
        num_players=num_players,
        action_mapper=action_mapper,
        model_kwargs=model_kwargs,
        pool=pool,
    )

    target_episodes = start_episode + episodes_to_run
    total_collected = start_episode
    session_start = time.time()
    metrics = {}

    logger.info("=" * 70)
    logger.info(f"Tréning indul | {num_players}p | PPO + Self-Play v4 OPTIMIZED")
    logger.info(f"Cél: {target_episodes:,} | State: {STATE_SIZE} | "
                f"Device: {device} | Envs: {NUM_ENVS} | "
                f"Buffer: {BUFFER_COLLECT_SIZE} | PPO epochs: 4")
    logger.info("=" * 70)

    while total_collected < target_episodes:
        to_collect = min(BUFFER_COLLECT_SIZE, target_episodes - total_collected)

        # Snapshot az opponent pool-ba
        prev_snap = total_collected // OpponentPool.SNAPSHOT_INTERVAL
        next_snap = (total_collected + to_collect) // OpponentPool.SNAPSHOT_INTERVAL
        if next_snap > prev_snap and total_collected > start_episode:
            pool.snapshot(learner)
            collector.update_pool()

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

        for steps, bb_reward in all_episodes:
            reward_norm.update(bb_reward)
            norm_r = reward_norm.normalize(bb_reward)
            n_steps = len(steps)
            for i, (s, la, a, lp, v) in enumerate(steps):
                buffer.add(
                    state=s.unsqueeze(0),
                    legal_actions=la,
                    action=a,
                    log_prob=lp,
                    value=v,
                    reward=norm_r if i == n_steps - 1 else 0.0,
                    episode_end=(i == n_steps - 1),
                )

        collected_this_batch = len(all_episodes)
        total_collected += collected_this_batch

        try:
            metrics = trainer.update(buffer)
        except Exception as exc:
            logger.error(f"PPO update hiba: {exc}", exc_info=True)
            buffer.reset()
            metrics = {}

        if writer and metrics:
            for k, mv in metrics.items():
                writer.add_scalar(f'Loss/{k}', mv, total_collected)
            writer.add_scalar('Reward/mean_bb', reward_norm.mean, total_collected)
            writer.add_scalar('Reward/std_bb', reward_norm.var ** 0.5, total_collected)

        prev_1k = (total_collected - collected_this_batch) // 1_000
        curr_1k = total_collected // 1_000
        if curr_1k > prev_1k:
            elapsed = time.time() - session_start
            done_so_far = total_collected - start_episode
            remaining = target_episodes - total_collected
            eps_sec = done_so_far / max(elapsed, 1e-6)
            eta_str = time.strftime(
                "%H:%M:%S", time.gmtime(remaining / max(eps_sec, 1e-6))
            )
            ela_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            lr = trainer.optimizer.param_groups[0]['lr']

            logger.info(
                f"Ep {total_collected:>8,}/{target_episodes:,} | "
                f"Eltelt {ela_str} | ETA {eta_str} | LR {lr:.2e} | "
                f"Actor {metrics.get('actor', float('nan')):+.4f} | "
                f"Critic {metrics.get('critic', float('nan')):.4f} | "
                f"Ent {metrics.get('entropy', float('nan')):.4f} | "
                f"Pool {len(pool):2d}"
            )

            _save_checkpoint(
                filename, learner, trainer, reward_norm,
                total_collected, total_time_spent + elapsed,
                STATE_SIZE, ACTION_SIZE,
            )

    if writer:
        writer.close()
    elapsed = time.time() - session_start
    _save_checkpoint(
        filename, learner, trainer, reward_norm,
        target_episodes, total_time_spent + elapsed,
        STATE_SIZE, ACTION_SIZE,
    )
    logger.info(f"KÉSZ  →  {filename}  ({target_episodes:,} epizód)")


def _save_checkpoint(filename, model, trainer, reward_norm,
                      episodes, time_spent, state_size, action_size):
    try:
        # Ha compiled modell, az eredeti state_dict-et mentjük
        if hasattr(model, '_orig_mod'):
            sd = model._orig_mod.state_dict()
        else:
            sd = model.state_dict()

        torch.save({
            'state_dict': sd,
            'trainer': trainer.state_dict(),
            'reward_norm': reward_norm.state_dict(),
            'episodes_trained': episodes,
            'time_spent': time_spent,
            'algorithm': 'PPO_SelfPlay_v4',
            'state_size': state_size,
            'action_size': action_size,
            'bb_options': BB_OPTIONS,
            'stack_multipliers': STACK_MULTIPLIERS,
        }, filename)
        logger.debug(f"Checkpoint mentve: {filename}")
    except Exception as exc:
        logger.error(f"Checkpoint mentési hiba: {exc}", exc_info=True)
