"""
training/runner.py  --  Tréning session és menü rendszer (v4 OPTIMIZED)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIMALIZÁCIÓK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  OPT-5: NUM_ENVS 256→512 – jobb GPU batch kihasználtság
  OPT-6: BUFFER_COLLECT_SIZE 1024→2048 – hatékonyabb PPO update
  OPT-7: torch.compile(mode="reduce-overhead") ha elérhető

MÉRFÖLDKŐ RENDSZER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  MILESTONE_INTERVAL epizódonként (default: 2_000_000) a runner:
    1. Elmenti a modellt: ModellNaplo/{modellnev}_{N}M/{modellnev}_{N}M.pth
    2. Elindítja a test_model_sanity.py-t subprocessként – a tréning
       erre az időre szünetel (így a teszt megkapja a GPU erőforrásokat)
    3. A teszt eredménye (.log + .json) ugyanabba a mappába kerül

  CHECKPOINT FORMÁTUM: VÁLTOZATLAN – régi .pth fájlok betölthetők.
"""
import os, sys, time, logging, subprocess
import rlcard, torch
from core.model import AdvancedPokerAI
from core.action_mapper import PokerActionMapper
from core.features import compute_state_size, ACTION_HISTORY_LEN
from core.opponent_tracker import NUM_HUD_STATS
from .buffer import PPOBuffer
from .trainer import PPOTrainer
from .normalizer import RunningMeanStd
from .opponent_pool import OpponentPool
from .collector import BatchedSyncCollector
from utils.checkpoint_utils import safe_load_checkpoint
from config import TrainingConfig

logger = logging.getLogger("PokerAI")

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False

# ── Konfiguráció – defaults; override: train_session(cfg=TrainingConfig(...)) ─
# [ARCH FIX] A konstansok a TrainingConfig dataclass-ba kerültek (config.py).
# A modul-szintű változók visszafelé kompatibilitásból maradnak,
# de a tényleges értékek a cfg objektumból jönnek train_session()-ban.
_DEFAULT_CFG        = TrainingConfig()
NUM_ENVS            = _DEFAULT_CFG.num_envs
BUFFER_COLLECT_SIZE = _DEFAULT_CFG.buffer_collect_size
HIDDEN_SIZE         = _DEFAULT_CFG.hidden_size
LEARNING_RATE       = _DEFAULT_CFG.learning_rate
MILESTONE_INTERVAL  = _DEFAULT_CFG.milestone_interval
MILESTONE_HANDS     = 2000
MILESTONE_DIR_ROOT  = _DEFAULT_CFG.milestone_dir_root


def _get_model_info(filename):
    if not os.path.exists(filename):
        return "[ Üres / Új modell ]"
    try:
        ck = safe_load_checkpoint(filename, map_location='cpu')
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
    print(f"\n  Env-ek: {NUM_ENVS} | Buffer: {BUFFER_COLLECT_SIZE} | PPO epochs: 8")
    print(f"  Mérföldkő: minden {MILESTONE_INTERVAL:,} ep → mentés + auto-teszt")
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


def _save_checkpoint(filename, model, trainer, reward_norm,
                     episodes, time_spent, state_size, action_size,
                     num_players=2, rlcard_obs_size=54):
    """Checkpoint mentése. Kezeli a torch.compile _orig_mod prefixet."""
    try:
        if hasattr(model, '_orig_mod'):
            sd = model._orig_mod.state_dict()
        else:
            sd = model.state_dict()

        torch.save({
            'state_dict':      sd,
            'trainer':         trainer.state_dict(),
            'reward_norm':     reward_norm.state_dict(),
            'episodes_trained': episodes,
            'time_spent':      time_spent,
            'algorithm':       'PPO_SelfPlay_v4',
            'state_size':      state_size,
            'action_size':     action_size,
            # [RF-8] GRU architektúra paraméterei – betöltéshez szükségesek
            'num_players':     num_players,
            'rlcard_obs_size': rlcard_obs_size,
        }, filename)
        logger.debug(f"Checkpoint mentve: {filename}")
    except Exception as exc:
        logger.error(f"Checkpoint mentési hiba: {exc}", exc_info=True)


def _run_milestone(filename, model, trainer, reward_norm,
                   episodes, time_spent, state_size, action_size,
                   num_players, milestone_episodes, rlcard_obs_size=54):
    """
    Mérföldkő elérése: elmenti a modellt egy dedikált mappába,
    majd elindítja a test_model_sanity.py-t subprocessként.

    A tréning a teszt végéig SZÜNETEL – ez szándékos, így a teszt
    megkapja az összes erőforrást és nem akad be memóriahiány miatt.
    Max timeout: 10 perc. Ha tovább tart, a tréning továbblép.

    Mappa struktúra:
        ModellNaplo/
        └── 2max_ppo_v4_4M/
            ├── 2max_ppo_v4_4M.pth   ← modell snapshot
            ├── test_...log           ← részletes log
            └── test_...json          ← géppel olvasható eredmény
    """
    milestone_m = milestone_episodes // 1_000_000
    milestone_str = f"{milestone_m}M"
    base_name = os.path.splitext(os.path.basename(filename))[0]

    # Mappa létrehozása
    milestone_dir = os.path.join(MILESTONE_DIR_ROOT, f"{base_name}_{milestone_str}")
    try:
        os.makedirs(milestone_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Mérföldkő mappa nem hozható létre ({milestone_dir}): {e}")
        return

    # Modell snapshot mentése
    milestone_model_path = os.path.join(
        milestone_dir, f"{base_name}_{milestone_str}.pth"
    )
    _save_checkpoint(
        milestone_model_path, model, trainer, reward_norm,
        episodes, time_spent, state_size, action_size,
        num_players=num_players, rlcard_obs_size=rlcard_obs_size,
    )
    logger.info(
        f"🏆 Mérföldkő: {milestone_episodes:,} ep → {milestone_model_path}"
    )

    # test_model_sanity.py elérési útja (projekt gyökérben van)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_script = os.path.join(project_root, "tests", "test_model_sanity.py")

    if not os.path.exists(test_script):
        logger.warning(
            f"Teszt script nem található: {test_script} – teszt kihagyva"
        )
        return

    cmd = [
        sys.executable, test_script,
        milestone_model_path,
        "--num-players", str(num_players),
        "--hands", str(MILESTONE_HANDS),
        "--out-dir", milestone_dir,
        # --winrate NEM fut automatikusan (túl lassú lenne minden 2M-nél).
        # Ha manuálisan akarod: python test_model_sanity.py <pth> --winrate
    ]

    logger.info(
        f"Teszt indítása (tréning szünetel, max 10 perc):\n"
        f"  {' '.join(cmd)}"
    )
    _TAIL = 800  # ennyi karaktert logolunk stdout/stderr végéből hiba esetén
    try:
        result = subprocess.run(
            cmd, check=False, timeout=600,
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info(f"✅ {milestone_str} teszt kész → eredmények: {milestone_dir}")
        else:
            stdout_tail = result.stdout[-_TAIL:] if result.stdout else "(üres)"
            stderr_tail = result.stderr[-_TAIL:] if result.stderr else "(üres)"
            logger.error(
                f"❌ {milestone_str} teszt hibával zárult "
                f"(returncode={result.returncode}) – tréning folytatódik\n"
                f"  stdout (vége): {stdout_tail}\n"
                f"  stderr (vége): {stderr_tail}"
            )
    except subprocess.TimeoutExpired:
        logger.error(
            f"❌ {milestone_str} teszt timeout (>10 perc) – tréning folytatódik"
        )
    except Exception as e:
        logger.error(
            f"❌ {milestone_str} teszt ismeretlen hiba: {e} – tréning folytatódik"
        )


def run_training_session(num_players, filename, episodes_to_run,
                         cfg: TrainingConfig = None):
    if cfg is None:
        cfg = TrainingConfig()
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
        'state_size':      STATE_SIZE,
        'action_size':     ACTION_SIZE,
        'hidden_size':     cfg.hidden_size,
        'num_players':     num_players,
        'rlcard_obs_size': rlcard_obs_size,
        'gru_hidden':      cfg.gru_hidden,
    }
    learner = AdvancedPokerAI(**model_kwargs).to(device)
    buffer = PPOBuffer()
    trainer = PPOTrainer(learner, lr=cfg.learning_rate, device=device)
    pool = OpponentPool(
        AdvancedPokerAI,
        model_kwargs,
        device=device,
        phase=OpponentPool.PHASE_EXPLOITATIVE,    # 30/30/10/10/10/10 eloszlás
        bot_types=['fish', 'nit', 'calling_station', 'lag'],
        bot_weights=[0.8, 1.5, 0.2, 1.5],  # fish:8%, nit:15%, cs:2%, lag:15%
        num_players=num_players,
        state_size=STATE_SIZE,
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
            ck = safe_load_checkpoint(filename, map_location=device)
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
                    # Trainer state külön try: ha az optimizer group mérete
                    # nem egyezik (pl. GRU rétegek hozzáadva), csak WARNING,
                    # nem ERROR – a model weights és start_episode megmaradnak.
                    try:
                        trainer.load_state_dict(ck['trainer'])
                    except Exception as trainer_exc:
                        logger.warning(
                            f'Trainer state nem töltve (architektúra változás, '
                            f'friss optimizerrel folytat): {trainer_exc}'
                        )
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
        num_envs=cfg.num_envs,
        model=learner,
        device=device,
        num_players=num_players,
        action_mapper=action_mapper,
        model_kwargs=model_kwargs,
        pool=pool,
        rlcard_obs_size=rlcard_obs_size,  # [RF-4 FIX] dinamikusan kiszámított fent
    )

    target_episodes   = start_episode + episodes_to_run
    total_collected = start_episode

    # ── Mérföldkő inicializálás ───────────────────────────────────────────
    # Ha folytatott tréning (pl. betöltöttük a 4M-es checkpointot),
    # a last_milestone = 4M lesz, így nem teszteli újra az already-done
    # mérföldköveket – csak a következőtől (6M) indul el.
    last_milestone = (start_episode // cfg.milestone_interval) * cfg.milestone_interval
    logger.info(
        f"Mérföldkő rendszer: interval={cfg.milestone_interval:,} | "
        f"következő: {last_milestone + cfg.milestone_interval:,} ep | "
        f"mentési hely: {cfg.milestone_dir_root}/"
    )

    session_start = time.time()
    metrics = {}

    logger.info("=" * 70)
    logger.info(f"Tréning indul | {num_players}p | PPO + Self-Play v4 OPTIMIZED")
    logger.info(f"Cél: {target_episodes:,} | State: {STATE_SIZE} | "
                f"Device: {device} | Envs: {NUM_ENVS} | "
                f"Buffer: {BUFFER_COLLECT_SIZE} | PPO epochs: 8")
    logger.info("=" * 70)

    while total_collected < target_episodes:
        to_collect = min(cfg.buffer_collect_size, target_episodes - total_collected)

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

        # [RF-3 FIX] Kiszámítjuk V(s_{T+1})-t a buffer utolsó állapotához.
        # Ha az utolsó epizód terminálisan zárult, last_value=0.0 (helyes).
        # Ha nem (buffer tele lett, de az epizód még fut), a learner értéke
        # pontosabb bootstrap-ot ad, mint a korábbi hardcode 0.0.
        last_value = 0.0
        if buffer.episode_ends and not buffer.episode_ends[-1]:
            try:
                last_state = buffer.states[-1].unsqueeze(0).to(device)
                last_legal = buffer.legal_actions[-1]
                with torch.inference_mode():
                    _, lv, _ = learner.forward(last_state, last_legal)
                last_value = float(lv.item())
            except Exception as exc:
                logger.debug(f"last_value bootstrap hiba (fallback 0.0): {exc}")

        try:
            metrics = trainer.update(buffer, last_value=last_value)
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

            # Normál rolling checkpoint
            _save_checkpoint(
                filename, learner, trainer, reward_norm,
                total_collected, total_time_spent + elapsed,
                STATE_SIZE, ACTION_SIZE,
                num_players=num_players, rlcard_obs_size=rlcard_obs_size,
            )

            # ── Mérföldkő ellenőrzés ──────────────────────────────────────
            # Pl: total_collected=2_001_500 → current_milestone=2_000_000
            # Ha last_milestone=0, akkor 2M > 0 → trigger!
            # Ha last_milestone=2M, akkor 2M > 2M → False, nem triggerel újra.
            current_milestone = (
                (total_collected // MILESTONE_INTERVAL) * MILESTONE_INTERVAL
            )
            if current_milestone > last_milestone and current_milestone > 0:
                last_milestone = current_milestone
                _run_milestone(
                    filename, learner, trainer, reward_norm,
                    total_collected, total_time_spent + elapsed,
                    STATE_SIZE, ACTION_SIZE,
                    num_players, current_milestone,
                    rlcard_obs_size=rlcard_obs_size,
                )

    if writer:
        writer.close()
    elapsed = time.time() - session_start
    _save_checkpoint(
        filename, learner, trainer, reward_norm,
        target_episodes, total_time_spent + elapsed,
        STATE_SIZE, ACTION_SIZE,
        num_players=num_players, rlcard_obs_size=rlcard_obs_size,
    )
    logger.info(f"KÉSZ  →  {filename}  ({target_episodes:,} epizód)")
