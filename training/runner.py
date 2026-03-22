"""
training/runner.py  --  Trening session és menü rendszer (v4.2.2-OPT)

Változások v4.2.2 (eredeti):
    [ARCH-FIX] global MILESTONE_DIR_ROOT mutáció eltávolítva.

Változások v4.2.2-OPT (ez a verzió):
    [OPT-1/2/3] BatchedSyncCollector példányosításakor átadja a cfg-ből
                az equity paramétereket (equity_n_sim, equity_min_sims,
                equity_cache_size), hogy a collector a konfigurált értékeket
                használja a hardcode-olt 200/50/20_000 helyett.

    [OPT-CLEANUP] collector.close() hívás trening végén – ThreadPoolExecutor
                  leállítása, erőforrás szivárgás elkerülése.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIMALIZÁCIÓK (v4.2.1-ből megőrizve):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  OPT-5: NUM_ENVS 256->512 – jobb GPU batch kihasználtság
  OPT-6: BUFFER_COLLECT_SIZE 1024->2048 – hatékonyabb PPO update
  OPT-7: torch.compile(mode="reduce-overhead") ha elerheto

MÉRFÖLDKŐ RENDSZER:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  cfg.milestone_interval epizódonként a runner:
    1. Elmenti a modellt: {milestone_dir_root}/{name}_{N}M/...
    2. Elindítja a test_model_sanity.py-t subprocessként
    3. A teszt eredménye (.log + .json) ugyanabba a mappába kerül

  [COLAB MOD v1] milestone_str fix: sub-million intervallumoknál
  a mappa neve helyesen {N}k formátumú, nem 0M.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from typing import Optional

import rlcard
import torch

from config import TrainingConfig
from core.action_mapper import PokerActionMapper
from core.features import compute_state_size
from core.model import AdvancedPokerAI
from core.opponent_tracker import NUM_HUD_STATS
from training.buffer import PPOBuffer
from training.collector import BatchedSyncCollector
from training.normalizer import RunningMeanStd
from training.opponent_pool import OpponentPool
from training.trainer import PPOTrainer
from utils.checkpoint_utils import safe_load_checkpoint

logger = logging.getLogger("PokerAI")

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False

# ── Konfiguráció – modul-szintű defaults ─────────────────────────────────────
_DEFAULT_CFG        = TrainingConfig()
NUM_ENVS            = _DEFAULT_CFG.num_envs
BUFFER_COLLECT_SIZE = _DEFAULT_CFG.buffer_collect_size
HIDDEN_SIZE         = _DEFAULT_CFG.hidden_size
LEARNING_RATE       = _DEFAULT_CFG.learning_rate
MILESTONE_INTERVAL  = _DEFAULT_CFG.milestone_interval
MILESTONE_HANDS     = _DEFAULT_CFG.milestone_hands
MILESTONE_DIR_ROOT  = _DEFAULT_CFG.milestone_dir_root  # ← READ-ONLY, ne mutáld!


# ─────────────────────────────────────────────────────────────────────────────
# Segédfüggvények
# ─────────────────────────────────────────────────────────────────────────────

def _get_model_info(filename: str) -> str:
    if not os.path.exists(filename):
        return "[ Ures / Új modell ]"
    try:
        ck = safe_load_checkpoint(filename, map_location="cpu")
        if isinstance(ck, dict) and "episodes_trained" in ck:
            eps = ck["episodes_trained"]
            t   = ck.get("time_spent", 0.0)
            alg = ck.get("algorithm", "PPO")
            ss  = ck.get("state_size", "?")
            spd = (
                f"~{(t / eps) * 100_000 / 60:.1f} perc/100k"
                if eps > 0 and t > 0
                else "n/a"
            )
            return f"[ {eps:,} ep | {alg} | state={ss} | {spd} ]"
        return "[ Régi formátum ]"
    except Exception:
        return "[ Hiba a fájl olvasásakor ]"


def menu_system():
    """Interaktív CLI menürendszer a trening indításához."""
    print("\n" + "=" * 70)
    print("  POKER AI v4  --  PPO + Self-Play (OPTIMIZED)")
    print("=" * 70)
    print("\n  Jatekosszam:")
    print("    2 = Heads-Up  |  6 = 6-max  |  9 = Full ring  |  A = osszes")
    print(
        f"\n  Env-ek: {NUM_ENVS} | "
        f"Buffer: {BUFFER_COLLECT_SIZE} | PPO epochs: 8"
    )
    print(
        f"  Merfoldko: minden {MILESTONE_INTERVAL:,} ep -> "
        f"mentés + auto-teszt"
    )
    print("=" * 70)
    choice = input("\n  Választás [2/6/9/A]: ").strip().upper()
    if choice == "A":
        return "ALL", None
    try:
        num_players = int(choice)
        if not (2 <= num_players <= 9):
            raise ValueError
    except ValueError:
        print("  Hibas bemenet -> 6 max.")
        num_players = 6
    default_name = f"{num_players}max_ppo_v4.pth"
    info = _get_model_info(default_name)
    print(f"\n  Fajl: {default_name}  {info}")
    custom = input("  Más név? (ENTER = alapértelmezett): ").strip()
    filename = custom if custom else default_name
    if not filename.endswith(".pth"):
        filename += ".pth"
    print(f"  -> {filename}")
    print("=" * 70 + "\n")
    return num_players, filename


def _try_compile(model: AdvancedPokerAI, device: torch.device) -> AdvancedPokerAI:
    """OPT-7: torch.compile ha elerheto (PyTorch 2.0+)."""
    if not hasattr(torch, "compile"):
        logger.info("torch.compile nem elerheto (PyTorch <2.0)")
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        logger.info(
            "torch.compile(mode='reduce-overhead') aktív – warmup..."
        )
        dummy_state_size = model.input_proj.in_features
        dummy = torch.randn(1, dummy_state_size, device=device)
        with torch.inference_mode():
            compiled._encode(dummy)
        logger.info("torch.compile warmup kesz")
        return compiled
    except Exception as exc:
        logger.warning(
            f"torch.compile sikertelen, fallback eredeti modellre: {exc}"
        )
        return model


def _save_checkpoint(
    filename:        str,
    model:           AdvancedPokerAI,
    trainer:         PPOTrainer,
    reward_norm:     RunningMeanStd,
    episodes:        int,
    time_spent:      float,
    state_size:      int,
    action_size:     int,
    num_players:     int   = 2,
    rlcard_obs_size: int   = 54,
) -> None:
    """Checkpoint mentése. Kezeli a torch.compile _orig_mod prefixet."""
    try:
        if hasattr(model, "_orig_mod"):
            sd = model._orig_mod.state_dict()
        else:
            sd = model.state_dict()

        torch.save(
            {
                "state_dict":       sd,
                "trainer":          trainer.state_dict(),
                "reward_norm":      reward_norm.state_dict(),
                "episodes_trained": episodes,
                "time_spent":       time_spent,
                "algorithm":        "PPO_SelfPlay_v4",
                "state_size":       state_size,
                "action_size":      action_size,
                "num_players":      num_players,
                "rlcard_obs_size":  rlcard_obs_size,
            },
            filename,
        )
        logger.debug(f"Checkpoint mentve: {filename!r}")
    except Exception as exc:
        logger.error(f"Checkpoint mentesi hiba: {exc}", exc_info=True)


def _milestone_str(milestone_episodes: int) -> str:
    """Merfoldko string: sub-million -> '{N}k', million+ -> '{N}M'."""
    if milestone_episodes < 1_000_000:
        return f"{milestone_episodes // 1_000}k"
    return f"{milestone_episodes // 1_000_000}M"


def _run_milestone(
    filename:           str,
    model:              AdvancedPokerAI,
    trainer:            PPOTrainer,
    reward_norm:        RunningMeanStd,
    episodes:           int,
    time_spent:         float,
    state_size:         int,
    action_size:        int,
    num_players:        int,
    milestone_episodes: int,
    milestone_dir_root: str,
    rlcard_obs_size:    int            = 54,
    milestone_hands:    Optional[int]  = None,
) -> None:
    """Merfoldko: snapshot mentés + automatikus sanity teszt."""
    ms_str    = _milestone_str(milestone_episodes)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    _hands    = milestone_hands if milestone_hands is not None else MILESTONE_HANDS

    milestone_dir = os.path.join(
        milestone_dir_root, f"{base_name}_{ms_str}"
    )
    try:
        os.makedirs(milestone_dir, exist_ok=True)
    except OSError as exc:
        logger.error(
            f"Merfoldko mappa nem hozhato letre "
            f"({milestone_dir!r}): {exc}"
        )
        return

    milestone_model_path = os.path.join(
        milestone_dir, f"{base_name}_{ms_str}.pth"
    )
    _save_checkpoint(
        milestone_model_path, model, trainer, reward_norm,
        episodes, time_spent, state_size, action_size,
        num_players=num_players, rlcard_obs_size=rlcard_obs_size,
    )
    logger.info(
        f"[MILESTONE] {milestone_episodes:,} ep -> {milestone_model_path!r}"
    )

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    test_script = os.path.join(
        project_root, "tests", "test_model_sanity.py"
    )

    if not os.path.exists(test_script):
        logger.warning(
            f"Teszt script nem található: {test_script!r} – teszt kihagyva"
        )
        return

    cmd = [
        sys.executable, test_script,
        milestone_model_path,
        "--num-players", str(num_players),
        "--hands",       str(_hands),
        "--out-dir",     milestone_dir,
    ]

    logger.info(
        f"Teszt inditasa (trening szunetel, max 10 perc):\n"
        f"  {' '.join(cmd)}"
    )
    _TAIL = 800
    try:
        result = subprocess.run(
            cmd,
            check=False,
            timeout=600,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info(
                f"[OK] {ms_str} teszt kesz -> eredmények: {milestone_dir!r}"
            )
        else:
            stdout_tail = result.stdout[-_TAIL:] if result.stdout else "(ures)"
            stderr_tail = result.stderr[-_TAIL:] if result.stderr else "(ures)"
            logger.error(
                f"[FAIL] {ms_str} teszt hibával zarult "
                f"(returncode={result.returncode}) – trening folytatodik\n"
                f"  stdout (vége): {stdout_tail}\n"
                f"  stderr (vége): {stderr_tail}"
            )
    except subprocess.TimeoutExpired:
        logger.error(
            f"[FAIL] {ms_str} teszt timeout (>10 perc) – trening folytatodik"
        )
    except Exception as exc:
        logger.error(
            f"[FAIL] {ms_str} teszt ismeretlen hiba: {exc} – trening folytatodik"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fő trening session
# ─────────────────────────────────────────────────────────────────────────────

def run_training_session(
    num_players:     int,
    filename:        str,
    episodes_to_run: int,
    cfg:             Optional[TrainingConfig] = None,
) -> None:
    """Egyetlen trening session futtatása."""
    if cfg is None:
        cfg = TrainingConfig()

    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    from utils.logging_setup import setup_logging
    setup_logging(log_file="training.log", num_players=num_players)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── State méret meghatározása ─────────────────────────────────────────
    temp_env = rlcard.make(
        "no-limit-holdem",
        config={"game_num_players": num_players},
    )
    temp_state, _ = temp_env.reset()
    rlcard_obs_size = len(temp_state["obs"])
    del temp_env

    STATE_SIZE  = compute_state_size(rlcard_obs_size, num_players)
    ACTION_SIZE = PokerActionMapper.NUM_CUSTOM_ACTIONS

    logger.info(
        f"STATE_SIZE ({num_players}p): "
        f"rlcard_obs={rlcard_obs_size} TOTAL={STATE_SIZE}"
    )

    # ── Modell és komponensek inicializálása ──────────────────────────────
    model_kwargs = {
        "state_size":      STATE_SIZE,
        "action_size":     ACTION_SIZE,
        "hidden_size":     cfg.hidden_size,
        "num_players":     num_players,
        "rlcard_obs_size": rlcard_obs_size,
        "gru_hidden":      cfg.gru_hidden,
    }
    learner      = AdvancedPokerAI(**model_kwargs).to(device)
    buffer       = PPOBuffer()
    trainer      = PPOTrainer(learner, lr=cfg.learning_rate, device=device)
    pool         = OpponentPool(
        AdvancedPokerAI,
        model_kwargs,
        device=device,
        phase=OpponentPool.PHASE_EXPLOITATIVE,
        bot_types=cfg.opponent_bot_types,
        bot_weights=cfg.opponent_bot_weights,
        num_players=num_players,
        state_size=STATE_SIZE,
    )
    action_mapper = PokerActionMapper()
    reward_norm   = RunningMeanStd()

    # ── TensorBoard ───────────────────────────────────────────────────────
    writer = None
    if _TB:
        log_dir = f"runs/{num_players}max_v4_{int(time.time())}"
        writer  = torch.utils.tensorboard.SummaryWriter(log_dir)
        logger.info(f"TensorBoard: {log_dir}")

    # ── Checkpoint betöltés ───────────────────────────────────────────────
    start_episode    = 0
    total_time_spent = 0.0

    if os.path.exists(filename):
        logger.info(f"Checkpoint betoltese: {filename!r}")
        try:
            ck = safe_load_checkpoint(filename, map_location=device)
            if isinstance(ck, dict) and "state_dict" in ck:
                model_dict = learner.state_dict()
                pretrained = {
                    k: v
                    for k, v in ck["state_dict"].items()
                    if k in model_dict and v.size() == model_dict[k].size()
                }
                n_loaded = len(pretrained)
                n_total  = len(model_dict)
                if n_loaded < n_total:
                    logger.warning(
                        f"Reszleges betöltés: {n_loaded}/{n_total} layer"
                    )
                model_dict.update(pretrained)
                learner.load_state_dict(model_dict)
                start_episode    = ck.get("episodes_trained", 0)
                total_time_spent = ck.get("time_spent", 0.0)
                if "trainer" in ck:
                    try:
                        trainer.load_state_dict(ck["trainer"])
                    except Exception as trainer_exc:
                        logger.warning(
                            f"Trainer state nem töltve "
                            f"(architektúra változás): {trainer_exc}"
                        )
                if "reward_norm" in ck:
                    reward_norm.load_state_dict(ck["reward_norm"])
            else:
                learner.load_state_dict(ck, strict=False)
            logger.info(f"Folytatas: {start_episode:,}. epizodtol.")
        except Exception as exc:
            logger.error(
                f"Checkpoint betoltesi hiba: {exc}", exc_info=True
            )

    # ── torch.compile ─────────────────────────────────────────────────────
    learner = _try_compile(learner, device)
    pool.snapshot(learner)

    # ── Collector ─────────────────────────────────────────────────────────
    # [OPT-1/2/3] equity paraméterek cfg-ből – nem hardcode-olva a collectorban
    logger.info(
        f"BatchedSyncCollector inicializálása "
        f"({cfg.num_envs} env | equity: n_sim={cfg.equity_n_sim}, "
        f"min_sims={cfg.equity_min_sims}, cache={cfg.equity_cache_size:,})..."
    )
    collector = BatchedSyncCollector(
        num_envs         = cfg.num_envs,
        model            = learner,
        device           = device,
        num_players      = num_players,
        action_mapper    = action_mapper,
        model_kwargs     = model_kwargs,
        pool             = pool,
        rlcard_obs_size  = rlcard_obs_size,
        equity_n_sim     = cfg.equity_n_sim,       # [OPT-1]
        equity_min_sims  = cfg.equity_min_sims,    # [OPT-1]
        equity_cache_size= cfg.equity_cache_size,  # [OPT-1]
    )

    target_episodes = start_episode + episodes_to_run
    total_collected = start_episode

    # ── Merfoldko inicializálás ───────────────────────────────────────────
    milestone_dir_root = cfg.milestone_dir_root
    last_milestone     = (
        (start_episode // cfg.milestone_interval) * cfg.milestone_interval
    )

    logger.info(
        f"Merfoldko rendszer: interval={cfg.milestone_interval:,} | "
        f"kovetkezo: {last_milestone + cfg.milestone_interval:,} ep | "
        f"mentesi hely: {milestone_dir_root!r}"
    )

    session_start = time.time()
    metrics       = {}

    logger.info("=" * 70)
    logger.info(
        f"Trening indul | {num_players}p | PPO + Self-Play v4 OPTIMIZED"
    )
    logger.info(
        f"Cél: {target_episodes:,} | State: {STATE_SIZE} | "
        f"Device: {device} | "
        f"Envs: {cfg.num_envs} | "
        f"Buffer: {cfg.buffer_collect_size} | "
        f"PPO epochs: {PPOTrainer.PPO_EPOCHS}"
    )
    logger.info("=" * 70)

    # ── Fő trening loop ───────────────────────────────────────────────────
    try:
        while total_collected < target_episodes:
            to_collect = min(
                cfg.buffer_collect_size,
                target_episodes - total_collected,
            )

            # Snapshot az opponent pool-ba
            prev_snap = total_collected // OpponentPool.SNAPSHOT_INTERVAL
            next_snap = (
                (total_collected + to_collect) // OpponentPool.SNAPSHOT_INTERVAL
            )
            if next_snap > prev_snap and total_collected > start_episode:
                pool.snapshot(learner)
                collector.update_pool()

            # Gyűjtés
            learner.eval()
            try:
                all_episodes = collector.collect(to_collect)
            except Exception as exc:
                logger.error(f"Gyujtesi hiba: {exc}", exc_info=True)
                continue

            # PPO update
            learner.train()
            if not all_episodes:
                logger.warning("Ures gyujtesi batch")
                continue

            for steps, bb_reward in all_episodes:
                reward_norm.update(bb_reward)
                norm_r  = reward_norm.normalize(bb_reward)
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
            total_collected     += collected_this_batch

            # GAE bootstrap
            last_value = 0.0
            if buffer.episode_ends and not buffer.episode_ends[-1]:
                last_value = collector.get_bootstrap_value(learner, device)
            try:
                metrics = trainer.update(buffer, last_value=last_value)
            except Exception as exc:
                logger.error(f"PPO update hiba: {exc}", exc_info=True)
                buffer.reset()
                metrics = {}

            # TensorBoard logging
            if writer and metrics:
                for k, mv in metrics.items():
                    writer.add_scalar(f"Loss/{k}", mv, total_collected)
                writer.add_scalar(
                    "Reward/mean_bb", reward_norm.mean, total_collected
                )
                writer.add_scalar(
                    "Reward/std_bb",
                    reward_norm.var ** 0.5,
                    total_collected,
                )

            # ── 1000 epizódonkénti log + checkpoint ──────────────────────
            prev_1k = (total_collected - collected_this_batch) // 1_000
            curr_1k = total_collected // 1_000

            if curr_1k > prev_1k:
                elapsed = time.time() - session_start
                done_so_far = total_collected - start_episode
                remaining   = target_episodes - total_collected
                eps_sec     = done_so_far / max(elapsed, 1e-6)
                eta_str = time.strftime(
                    "%H:%M:%S",
                    time.gmtime(remaining / max(eps_sec, 1e-6)),
                )
                ela_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                lr = trainer.optimizer.param_groups[0]["lr"]

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
                    num_players=num_players,
                    rlcard_obs_size=rlcard_obs_size,
                )

                # ── Merfoldko ellenőrzés ──────────────────────────────
                current_milestone = (
                    (total_collected // cfg.milestone_interval)
                    * cfg.milestone_interval
                )
                if current_milestone > last_milestone and current_milestone > 0:
                    last_milestone = current_milestone
                    _run_milestone(
                        filename=filename,
                        model=learner,
                        trainer=trainer,
                        reward_norm=reward_norm,
                        episodes=total_collected,
                        time_spent=total_time_spent + elapsed,
                        state_size=STATE_SIZE,
                        action_size=ACTION_SIZE,
                        num_players=num_players,
                        milestone_episodes=current_milestone,
                        milestone_dir_root=milestone_dir_root,
                        rlcard_obs_size=rlcard_obs_size,
                        milestone_hands=cfg.milestone_hands,
                    )

    finally:
        # [OPT-CLEANUP] ThreadPoolExecutor leállítása – erőforrás szivárgás elkerülése.
        # A finally blokk garantálja, hogy Ctrl+C vagy kivétel esetén is lefut.
        collector.close()

    # ── Trening vége ──────────────────────────────────────────────────────
    if writer:
        writer.close()

    elapsed = time.time() - session_start
    _save_checkpoint(
        filename, learner, trainer, reward_norm,
        target_episodes, total_time_spent + elapsed,
        STATE_SIZE, ACTION_SIZE,
        num_players=num_players,
        rlcard_obs_size=rlcard_obs_size,
    )
    logger.info(
        f"KÉSZ  ->  {filename!r}  ({target_episodes:,} epizód)"
    )
