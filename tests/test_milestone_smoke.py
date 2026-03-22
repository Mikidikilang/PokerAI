#!/usr/bin/env python3
"""
tests/test_milestone_smoke.py  –  Gyors füstteszt a mérföldkő rendszerhez

Változások v4.2.2:
    [ARCH-FIX] A global MILESTONE_DIR_ROOT direkt mutációja eltávolítva.
    Az eredeti kód:
        runner_module.MILESTONE_DIR_ROOT = "ModellNaplo_SMOKE_TEST"
        ...
        runner_module.MILESTONE_DIR_ROOT = original_root  # visszaállítás
    Ez fragilis volt: ha a _run_milestone() belsejében kivétel keletkezett
    a finally-blokk előtt, a globális módosítva maradt – és a következő
    tréning session rossz mappába mentett volna.

    A javított verzió:
        - A _run_milestone() explicit milestone_dir_root paramétert kap
        - Nincs globális mutáció → nincs szükség visszaállításra sem
        - A teszt izolált: nem hat a modul állapotára

Mit tesztel:
    1. Egyedi tesztkömyvtár létrejön-e
    2. A snapshot .pth fájl létrejön-e benne
    3. A test_model_sanity.py subprocess elindul-e és lefut-e
    4. A .log és .json fájlok létrejönnek-e

Használat:
    python tests/test_milestone_smoke.py 2max_ppo_v4.pth

Ha minden OK, ezt látod:
    ✅ Mappa létrejött
    ✅ Modell snapshot mentve
    ✅ Teszt lefutott
    ✅ Log fájl létrejött
    ✅ JSON fájl létrejött
"""

from __future__ import annotations

import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    if len(sys.argv) < 2:
        print("Használat: python tests/test_milestone_smoke.py <modell.pth>")
        print("Példa:     python tests/test_milestone_smoke.py 2max_ppo_v4.pth")
        sys.exit(1)

    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"❌ Nem találom: {model_path!r}")
        sys.exit(1)

    print(f"Modell: {model_path}")
    print("Teszt indul...\n")

    # ── Importok ──────────────────────────────────────────────────────────
    import torch
    from core.model import AdvancedPokerAI
    from core.features import compute_state_size
    from training.normalizer import RunningMeanStd
    from training.trainer import PPOTrainer
    from training.runner import _run_milestone, _save_checkpoint
    from utils.checkpoint_utils import safe_load_checkpoint

    # ── Checkpoint betöltés ───────────────────────────────────────────────
    ck          = safe_load_checkpoint(model_path, map_location="cpu")
    state_size  = ck.get("state_size",  475)
    action_size = ck.get("action_size", 7)
    episodes    = ck.get("episodes_trained", 0)

    # num_players inferálás state_size-ból
    num_players = None
    for n in range(2, 10):
        if compute_state_size(54, n) == state_size:
            num_players = n
            break

    if num_players is None:
        print(
            f"❌ Nem sikerült kitalálni a num_players-t "
            f"(state_size={state_size})"
        )
        sys.exit(1)

    print(
        f"  state_size={state_size}, "
        f"num_players={num_players}, "
        f"ep={episodes:,}"
    )

    # ── Modell és komponensek ─────────────────────────────────────────────
    model = AdvancedPokerAI(
        state_size=state_size,
        action_size=action_size,
    )
    model.load_state_dict(ck["state_dict"], strict=False)
    trainer     = PPOTrainer(model, device=torch.device("cpu"))
    reward_norm = RunningMeanStd()

    # ── Teszt könyvtár meghatározása ──────────────────────────────────────
    # [ARCH-FIX] A dedikált smoke-test könyvtárat explicit adjuk át
    # a _run_milestone() milestone_dir_root paraméterének.
    # Nincs szükség globális mutációra – nincs mit visszaállítani.
    smoke_test_root = "ModellNaplo_SMOKE_TEST"
    fake_milestone  = 2_000_000  # szimulált 2M mérföldkő

    print(f"  Mérföldkő könyvtár: {smoke_test_root!r}")
    print(f"  Szimulált epizód:   {fake_milestone:,}\n")

    # ── _run_milestone() hívás ────────────────────────────────────────────
    t0 = time.time()
    try:
        _run_milestone(
            filename=model_path,
            model=model,
            trainer=trainer,
            reward_norm=reward_norm,
            episodes=episodes,
            time_spent=0.0,
            state_size=state_size,
            action_size=action_size,
            num_players=num_players,
            milestone_episodes=fake_milestone,
            milestone_dir_root=smoke_test_root,  # explicit, nem globális!
            milestone_hands=500,
        )
    except Exception as exc:
        print(f"\n❌ _run_milestone() kivételt dobott: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0

    # ── Ellenőrzések ──────────────────────────────────────────────────────
    base_name    = os.path.splitext(os.path.basename(model_path))[0]
    expected_dir = os.path.join(smoke_test_root, f"{base_name}_2M")
    expected_pth = os.path.join(expected_dir, f"{base_name}_2M.pth")

    print("\n── Eredmény ──────────────────────────────────────────────────────")

    checks = []

    ok = os.path.isdir(expected_dir)
    print(f"  {'✅' if ok else '❌'} Mappa létrejött:      {expected_dir!r}")
    checks.append(ok)

    ok = os.path.isfile(expected_pth)
    print(f"  {'✅' if ok else '❌'} Modell snapshot:      {expected_pth!r}")
    checks.append(ok)

    logs  = glob.glob(os.path.join(expected_dir, "*.log"))
    ok    = len(logs) > 0
    print(
        f"  {'✅' if ok else '❌'} Log fájl létrejött:   "
        f"{logs[0] if logs else '(nincs)'}"
    )
    checks.append(ok)

    jsons = glob.glob(os.path.join(expected_dir, "*.json"))
    ok    = len(jsons) > 0
    print(
        f"  {'✅' if ok else '❌'} JSON fájl létrejött:  "
        f"{jsons[0] if jsons else '(nincs)'}"
    )
    checks.append(ok)

    print(f"\n  Teljes idő: {elapsed:.1f}s")
    print("──────────────────────────────────────────────────────────────────")

    if all(checks):
        print(
            f"\n✅ Minden OK – a mérföldkő rendszer működik.\n"
            f"\n  A teszt kimenet itt van: {expected_dir!r}"
            f"\n  (Töröld kézzel ha nem kell: "
            f"rmdir /s /q {smoke_test_root}  [Win] "
            f"/ rm -rf {smoke_test_root}  [Unix])\n"
        )
    else:
        n_fail = checks.count(False)
        print(f"\n❌ {n_fail} ellenőrzés sikertelen – nézd meg a fenti hibákat.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
