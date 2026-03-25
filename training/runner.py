"""
training/runner.py  --  Tréning session (v4.3 CONFIG-FULL)

[CONFIG-FULL v4.3] Változások:
  - PPOTrainer-nek átadja a teljes cfg-et (cfg=cfg)
    → trainer saját maga olvassa a clip_eps, ppo_epochs, entropy_decay, stb.
  - BatchedSyncCollector-nek átadja:
    → reward_cfg dict (cfg mezőkből összeszedi)
    → max_steps_per_hand
    → equity_n_sim / equity_min_sims / equity_cache_size
  Így a kód soha nem tartalmaz hardkódolt hiperparamétert.
"""

from __future__ import annotations

import glob
import json
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
from training.buffer import PPOBuffer
from training.collector import BatchedSyncCollector
from training.normalizer import RunningMeanStd
from training.opponent_pool import OpponentPool
from training.trainer import PPOTrainer
from utils.checkpoint_utils import safe_load_checkpoint
from training.model_manager import LifecycleLogger

logger = logging.getLogger("PokerAI")

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False

# ── Visszafelé kompatibilis modul-szintű default értékek ─────────────────────
_DEFAULT_CFG        = TrainingConfig()
NUM_ENVS            = _DEFAULT_CFG.num_envs
BUFFER_COLLECT_SIZE = _DEFAULT_CFG.buffer_collect_size
HIDDEN_SIZE         = _DEFAULT_CFG.hidden_size
LEARNING_RATE       = _DEFAULT_CFG.learning_rate
MILESTONE_INTERVAL  = _DEFAULT_CFG.milestone_interval
MILESTONE_HANDS     = _DEFAULT_CFG.milestone_hands
MILESTONE_DIR_ROOT  = _DEFAULT_CFG.milestone_dir_root


# ─────────────────────────────────────────────────────────────────────────────
# Segédfüggvények
# ─────────────────────────────────────────────────────────────────────────────

def _get_model_info(filename: str) -> str:
    if not os.path.exists(filename):
        return "[ Üres / Új modell ]"
    try:
        ck = safe_load_checkpoint(filename, map_location="cpu")
        if isinstance(ck, dict) and "episodes_trained" in ck:
            eps = ck["episodes_trained"]
            t   = ck.get("time_spent", 0.0)
            alg = ck.get("algorithm", "PPO")
            ss  = ck.get("state_size", "?")
            spd = (
                f"~{(t / eps) * 100_000 / 60:.1f} perc/100k"
                if eps > 0 and t > 0 else "n/a"
            )
            return f"[ {eps:,} ep | {alg} | state={ss} | {spd} ]"
        return "[ Régi formátum ]"
    except Exception:
        return "[ Hiba a fájl olvasásakor ]"


def menu_system():
    print("\n" + "=" * 70)
    print("  POKER AI v4  --  PPO + Self-Play (CONFIG-FULL v4.3)")
    print("=" * 70)
    print("\n  Játékosszám:")
    print("    2 = Heads-Up  |  6 = 6-max  |  9 = Full ring  |  A = összes")
    print(f"\n  Env-ek: {NUM_ENVS} | Buffer: {BUFFER_COLLECT_SIZE} | PPO epochs: 8")
    print(f"  Mérföldkő: minden {MILESTONE_INTERVAL:,} ep → mentés + auto-teszt")
    print("=" * 70)
    choice = input("\n  Választás [2/6/9/A]: ").strip().upper()
    if choice == "A":
        return "ALL", None
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
    if not filename.endswith(".pth"):
        filename += ".pth"
    print(f"  → {filename}")
    print("=" * 70 + "\n")
    return num_players, filename


def _try_compile(model, device):
    if not hasattr(torch, "compile"):
        return model
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        dummy_state_size = model.input_proj.in_features
        dummy = torch.randn(1, dummy_state_size, device=device)
        with torch.inference_mode():
            compiled._encode(dummy)
        logger.info("torch.compile warmup kész")
        return compiled
    except Exception as exc:
        logger.warning(f"torch.compile sikertelen, fallback: {exc}")
        return model


def _save_checkpoint(
    filename, model, trainer, reward_norm,
    episodes, time_spent, state_size, action_size,
    num_players=2, rlcard_obs_size=54,
):
    try:
        sd = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
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
        logger.error(f"Checkpoint mentési hiba: {exc}", exc_info=True)


def _milestone_str(milestone_episodes: int) -> str:
    if milestone_episodes < 1_000_000:
        return f"{milestone_episodes // 1_000}k"
    return f"{milestone_episodes // 1_000_000}M"


def _run_milestone(
    filename, model, trainer, reward_norm, episodes, time_spent,
    state_size, action_size, num_players, milestone_episodes,
    milestone_dir_root, rlcard_obs_size=54, milestone_hands=None,
    lifecycle=None,
):
    ms_str        = _milestone_str(milestone_episodes)
    base_name     = os.path.splitext(os.path.basename(filename))[0]
    _hands        = milestone_hands if milestone_hands is not None else MILESTONE_HANDS
    milestone_dir = os.path.join(milestone_dir_root, f"{base_name}_{ms_str}")

    try:
        os.makedirs(milestone_dir, exist_ok=True)
    except OSError as exc:
        logger.error(f"Mérföldkő mappa nem hozható létre ({milestone_dir!r}): {exc}")
        return

    milestone_model_path = os.path.join(milestone_dir, f"{base_name}_{ms_str}.pth")
    _save_checkpoint(
        milestone_model_path, model, trainer, reward_norm,
        episodes, time_spent, state_size, action_size,
        num_players=num_players, rlcard_obs_size=rlcard_obs_size,
    )
    logger.info(f"🏆 Mérföldkő: {milestone_episodes:,} ep → {milestone_model_path!r}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_script  = os.path.join(project_root, "tests", "test_model_sanity.py")

    if not os.path.exists(test_script):
        logger.warning(f"Teszt script nem található: {test_script!r}")
        return

    cmd = [
        sys.executable, test_script,
        milestone_model_path,
        "--num-players", str(num_players),
        "--hands",       str(_hands),
        "--out-dir",     milestone_dir,
    ]
    try:
        result = subprocess.run(cmd, check=False, timeout=600, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ {ms_str} teszt kész → {milestone_dir!r}")
            # Lifecycle naplózás: teszteredmény JSON visszaolvasása
            if lifecycle is not None:
                try:
                    test_jsons = sorted(glob.glob(os.path.join(milestone_dir, "*.json")))
                    test_data = {}
                    for tj in test_jsons:
                        if os.path.basename(tj) != os.path.basename(milestone_model_path):
                            with open(tj, "r", encoding="utf-8") as f:
                                test_data = json.load(f)
                            break
                    lifecycle.log_milestone(
                        episode=milestone_episodes,
                        test_name=f"sanity_check_{ms_str}",
                        results=test_data if test_data else {"milestone_dir": milestone_dir},
                    )
                except Exception as lc_exc:
                    logger.warning(f"[Lifecycle] Mérföldkő naplózási hiba: {lc_exc}")
        else:
            _TAIL = 800
            logger.error(
                f"❌ {ms_str} teszt hiba (rc={result.returncode})\n"
                f"  stdout: {result.stdout[-_TAIL:]}\n"
                f"  stderr: {result.stderr[-_TAIL:]}"
            )
    except subprocess.TimeoutExpired:
        logger.error(f"❌ {ms_str} teszt timeout (>10 perc)")
    except Exception as exc:
        logger.error(f"❌ {ms_str} teszt hiba: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Fő tréning session
# ─────────────────────────────────────────────────────────────────────────────

def run_training_session(
    num_players:     int,
    filename:        str,
    episodes_to_run: int,
    cfg: Optional[TrainingConfig] = None,
) -> None:
    """
    Egyetlen tréning session futtatása.

    [CONFIG-FULL v4.3] A cfg objektum MINDEN paramétert tartalmaz –
    a kód nem hardkódol semmit.
    """
    if cfg is None:
        cfg = TrainingConfig()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_setup import setup_logging
    setup_logging(log_file="training.log", num_players=num_players)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── State méret ───────────────────────────────────────────────────────
    temp_env = rlcard.make("no-limit-holdem", config={"game_num_players": num_players})
    temp_state, _ = temp_env.reset()
    rlcard_obs_size = len(temp_state["obs"])
    del temp_env

    STATE_SIZE  = compute_state_size(rlcard_obs_size, num_players)
    ACTION_SIZE = PokerActionMapper.NUM_CUSTOM_ACTIONS

    logger.info(f"STATE_SIZE ({num_players}p): rlcard_obs={rlcard_obs_size} TOTAL={STATE_SIZE}")

    # ── Modell és komponensek ─────────────────────────────────────────────
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

    # [CONFIG-FULL v4.3] Trainer megkapja a teljes cfg-et
    trainer      = PPOTrainer(
        learner,
        lr=cfg.learning_rate,
        device=device,
        cfg=cfg,           # ← ÖSSZES hiperparaméter ebből jön
    )

    pool = OpponentPool(
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

    # ── Lifecycle napló inicializálása ────────────────────────────────────
    model_name = os.path.splitext(os.path.basename(filename))[0]
    lifecycle_path = os.path.join(
        os.path.dirname(filename) if os.path.dirname(filename) else ".",
        "lifecycle.json",
    )
    lifecycle = LifecycleLogger(lifecycle_path, model_id=model_name)

    if os.path.exists(filename):
        logger.info(f"Checkpoint betöltése: {filename!r}")
        try:
            ck = safe_load_checkpoint(filename, map_location=device)
            if isinstance(ck, dict) and "state_dict" in ck:
                model_dict = learner.state_dict()
                pretrained = {
                    k: v for k, v in ck["state_dict"].items()
                    if k in model_dict and v.size() == model_dict[k].size()
                }
                n_loaded = len(pretrained)
                n_total  = len(model_dict)
                if n_loaded < n_total:
                    logger.warning(f"Részleges betöltés: {n_loaded}/{n_total} layer")
                model_dict.update(pretrained)
                learner.load_state_dict(model_dict)
                start_episode    = ck.get("episodes_trained", 0)
                total_time_spent = ck.get("time_spent", 0.0)
                if "trainer" in ck:
                    try:
                        # [CONFIG-FULL v4.3] trainer.load_state_dict() figyelembe
                        # veszi a reset_optimizer_on_load flaget (cfg-ből jön)
                        trainer.load_state_dict(ck["trainer"])
                    except Exception as trainer_exc:
                        logger.warning(f"Trainer state nem töltve: {trainer_exc}")
                if "reward_norm" in ck:
                    reward_norm.load_state_dict(ck["reward_norm"])
            else:
                learner.load_state_dict(ck, strict=False)
            logger.info(f"Folytatás: {start_episode:,}. epizódtól.")
        except Exception as exc:
            logger.error(f"Checkpoint betöltési hiba: {exc}", exc_info=True)

    learner = _try_compile(learner, device)
    pool.snapshot(learner)

    # ── [CONFIG-FULL v4.3] Reward cfg összerakása a cfg-ből ──────────────
    reward_cfg = {
        'draw_fold_penalty':               cfg.draw_fold_penalty,
        'draw_equity_threshold':           cfg.draw_equity_threshold,
        'street_reward_scale':             cfg.street_reward_scale,
        'allin_penalty_enabled':           cfg.allin_penalty_enabled,
        'allin_penalty_equity_threshold':  cfg.allin_penalty_equity_threshold,
        'allin_penalty_amount':            cfg.allin_penalty_amount,
        'fold_bonus_enabled':              cfg.fold_bonus_enabled,
        'fold_bonus_equity_threshold':     cfg.fold_bonus_equity_threshold,
        'fold_bonus_amount':               cfg.fold_bonus_amount,
        'stack_blindness_penalty_enabled': cfg.stack_blindness_penalty_enabled,
        'stack_blindness_bb_threshold':    cfg.stack_blindness_bb_threshold,
        'stack_blindness_penalty_amount':  cfg.stack_blindness_penalty_amount,
    }

    logger.info(
        f"BatchedSyncCollector inicializálása ({cfg.num_envs} env)..."
    )
    collector = BatchedSyncCollector(
        num_envs            = cfg.num_envs,
        model               = learner,
        device              = device,
        num_players         = num_players,
        action_mapper       = action_mapper,
        model_kwargs        = model_kwargs,
        pool                = pool,
        rlcard_obs_size     = rlcard_obs_size,
        max_steps_per_hand  = cfg.max_steps_per_hand,   # ← cfg-ből
        reward_cfg          = reward_cfg,               # ← cfg-ből
        equity_n_sim        = cfg.equity_n_sim,         # ← cfg-ből
        equity_min_sims     = cfg.equity_min_sims,      # ← cfg-ből
        equity_cache_size   = cfg.equity_cache_size,    # ← cfg-ből
    )

    target_episodes    = start_episode + episodes_to_run
    total_collected    = start_episode
    milestone_dir_root = cfg.milestone_dir_root
    last_milestone     = (
        (start_episode // cfg.milestone_interval) * cfg.milestone_interval
    )

    logger.info(
        f"Mérföldkő: interval={cfg.milestone_interval:,} | "
        f"következő: {last_milestone + cfg.milestone_interval:,} ep | "
        f"mentési hely: {milestone_dir_root!r}"
    )

    session_start = time.time()
    metrics       = {}

    logger.info("=" * 70)
    logger.info(
        f"Tréning indul | {num_players}p | PPO + Self-Play v4.3 CONFIG-FULL"
    )
    logger.info(
        f"Cél: {target_episodes:,} | State: {STATE_SIZE} | Device: {device} | "
        f"Envs: {cfg.num_envs} | Buffer: {cfg.buffer_collect_size} | "
        f"LR scheduler: {cfg.lr_scheduler} | entropy_decay: {cfg.entropy_decay:,}"
    )
    if cfg.allin_penalty_enabled or cfg.fold_bonus_enabled or cfg.stack_blindness_penalty_enabled:
        logger.info(
            f"Reward shaping (extra): "
            f"allin_penalty={cfg.allin_penalty_enabled} "
            f"fold_bonus={cfg.fold_bonus_enabled} "
            f"stack_blindness={cfg.stack_blindness_penalty_enabled}"
        )
    if cfg.reset_optimizer_on_load:
        logger.info("reset_optimizer_on_load=True: friss optimizer-rel indult")
    logger.info("=" * 70)

    # Lifecycle: session megnyitása az aktuális config snapshotjával
    lifecycle.start_session(cfg.to_dict(), start_episode)

    # ── Fő tréning loop ───────────────────────────────────────────────────
    try:
        while total_collected < target_episodes:
            to_collect = min(cfg.buffer_collect_size, target_episodes - total_collected)

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

            last_value = 0.0
            if buffer.episode_ends and not buffer.episode_ends[-1]:
                last_value = collector.get_bootstrap_value(learner, device)

            try:
                metrics = trainer.update(buffer, last_value=last_value)
            except Exception as exc:
                logger.error(f"PPO update hiba: {exc}", exc_info=True)
                buffer.reset()
                metrics = {}

            if writer and metrics:
                for k, mv in metrics.items():
                    writer.add_scalar(f"Loss/{k}", mv, total_collected)
                writer.add_scalar("Reward/mean_bb", reward_norm.mean, total_collected)
                writer.add_scalar("Reward/std_bb", reward_norm.var ** 0.5, total_collected)

            prev_1k = (total_collected - collected_this_batch) // 1_000
            curr_1k = total_collected // 1_000

            if curr_1k > prev_1k:
                elapsed      = time.time() - session_start
                done_so_far  = total_collected - start_episode
                remaining    = target_episodes - total_collected
                eps_sec      = done_so_far / max(elapsed, 1e-6)
                eta_str      = time.strftime("%H:%M:%S", time.gmtime(remaining / max(eps_sec, 1e-6)))
                ela_str      = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                lr           = trainer.optimizer.param_groups[0]["lr"]

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
                    num_players=num_players, rlcard_obs_size=rlcard_obs_size,
                )

                current_milestone = (
                    (total_collected // cfg.milestone_interval) * cfg.milestone_interval
                )
                if current_milestone > last_milestone and current_milestone > 0:
                    last_milestone = current_milestone
                    _run_milestone(
                        filename=filename, model=learner, trainer=trainer,
                        reward_norm=reward_norm, episodes=total_collected,
                        time_spent=total_time_spent + elapsed,
                        state_size=STATE_SIZE, action_size=ACTION_SIZE,
                        num_players=num_players,
                        milestone_episodes=current_milestone,
                        milestone_dir_root=milestone_dir_root,
                        rlcard_obs_size=rlcard_obs_size,
                        milestone_hands=cfg.milestone_hands,
                        lifecycle=lifecycle,
                    )

    except KeyboardInterrupt:
        logger.info("Tréning kézzel megszakítva (Ctrl+C). Session mentése...")
    finally:
        if writer:
            writer.close()

        elapsed = time.time() - session_start
        _save_checkpoint(
            filename, learner, trainer, reward_norm,
            total_collected, total_time_spent + elapsed,
            STATE_SIZE, ACTION_SIZE,
            num_players=num_players, rlcard_obs_size=rlcard_obs_size,
        )

        # Lifecycle: session lezárása aggregált metrikákkal
        lifecycle.close_session(
            end_episode=total_collected,
            metrics={
                "mean_actor_loss":  metrics.get("actor",   0.0) if metrics else 0.0,
                "mean_critic_loss": metrics.get("critic",  0.0) if metrics else 0.0,
                "mean_entropy":     metrics.get("entropy", 0.0) if metrics else 0.0,
                "mean_bb_reward":   float(reward_norm.mean) if hasattr(reward_norm, "mean") else 0.0,
                "execution_time_hours": elapsed / 3600,
            },
        )
        logger.info(f"KÉSZ  →  {filename!r}  ({total_collected:,} epizód)")
