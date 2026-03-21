#!/usr/bin/env python3
"""
test_milestone_smoke.py  –  Gyors füstteszt a mérföldkő rendszerhez

Mit tesztel:
  1. ModellNaplo/ mappa létrejön-e
  2. A snapshot .pth fájl létrejön-e benne
  3. A test_model_sanity.py subprocess elindul-e és lefut-e
  4. A .log és .json fájlok létrejönnek-e a mappában

Használat:
  python test_milestone_smoke.py 2max_ppo_v4.pth

Ha minden OK, ezt látod a végén:
  ✅ Mappa létrejött
  ✅ Modell snapshot mentve
  ✅ Teszt lefutott
  ✅ Log fájl létrejött
  ✅ JSON fájl létrejött
"""

import sys, os, glob, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    if len(sys.argv) < 2:
        print("Használat: python test_milestone_smoke.py <modell.pth>")
        print("Példa:     python test_milestone_smoke.py 2max_ppo_v4.pth")
        sys.exit(1)

    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"❌ Nem találom: {model_path}")
        sys.exit(1)

    print(f"Modell: {model_path}")
    print(f"Teszt indul...\n")

    # ── Betöltjük a runner-ből a szükséges dolgokat ───────────────────────────
    import torch
    from core.model import AdvancedPokerAI
    from core.action_mapper import PokerActionMapper
    from training.normalizer import RunningMeanStd
    from training.trainer import PPOTrainer
    from training.runner import _run_milestone, _save_checkpoint, MILESTONE_DIR_ROOT
    from utils.checkpoint_utils import safe_load_checkpoint

    ck = safe_load_checkpoint(model_path, map_location='cpu')
    state_size  = ck.get('state_size', 475)
    action_size = ck.get('action_size', 7)
    episodes    = ck.get('episodes_trained', 0)

    # Kitaláljuk a num_players-t state_size-ból
    from core.features import compute_state_size
    num_players = None
    for n in range(2, 10):
        if compute_state_size(54, n) == state_size:
            num_players = n
            break
    if num_players is None:
        print(f"❌ Nem sikerült kitalálni a num_players-t (state_size={state_size})")
        sys.exit(1)

    print(f"  state_size={state_size}, num_players={num_players}, ep={episodes:,}")

    # Modell betöltés
    model = AdvancedPokerAI(state_size=state_size, action_size=action_size)
    model.load_state_dict(ck['state_dict'])
    trainer     = PPOTrainer(model, device=torch.device('cpu'))
    reward_norm = RunningMeanStd()

    # ── _run_milestone() hívás szimulált 2M-es mérföldkővel ──────────────────
    # Egy kicsit trükkös milestone_episodes értéket adunk meg hogy ne írjon
    # a valódi epizódszám nevű mappába, hanem egy "SMOKE_TEST" nevűbe.
    # Ehhez ideiglenesen felülírjuk a MILESTONE_DIR_ROOT-ot.

    import training.runner as runner_module
    original_root = runner_module.MILESTONE_DIR_ROOT
    runner_module.MILESTONE_DIR_ROOT = "ModellNaplo_SMOKE_TEST"

    fake_milestone = 2_000_000  # szimulált 2M mérföldkő

    print(f"  Mérföldkő könyvtár: {runner_module.MILESTONE_DIR_ROOT}/")
    print(f"  Szimulált epizód:   {fake_milestone:,}\n")

    t0 = time.time()
    try:
        _run_milestone(
            filename       = model_path,
            model          = model,
            trainer        = trainer,
            reward_norm    = reward_norm,
            episodes       = episodes,
            time_spent     = 0.0,
            state_size     = state_size,
            action_size    = action_size,
            num_players    = num_players,
            milestone_episodes = fake_milestone,
        )
    except Exception as e:
        print(f"\n❌ _run_milestone() kivételt dobott: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        runner_module.MILESTONE_DIR_ROOT = original_root  # visszaállítás

    elapsed = time.time() - t0

    # ── Ellenőrzések ──────────────────────────────────────────────────────────
    base_name       = os.path.splitext(os.path.basename(model_path))[0]
    expected_dir    = os.path.join("ModellNaplo_SMOKE_TEST", f"{base_name}_2M")
    expected_pth    = os.path.join(expected_dir, f"{base_name}_2M.pth")

    print("\n── Eredmény ──────────────────────────────────────────────────────")

    checks = []

    ok = os.path.isdir(expected_dir)
    print(f"  {'✅' if ok else '❌'} Mappa létrejött:      {expected_dir}")
    checks.append(ok)

    ok = os.path.isfile(expected_pth)
    print(f"  {'✅' if ok else '❌'} Modell snapshot:      {expected_pth}")
    checks.append(ok)

    logs  = glob.glob(os.path.join(expected_dir, "*.log"))
    ok    = len(logs) > 0
    print(f"  {'✅' if ok else '❌'} Log fájl létrejött:   {logs[0] if logs else '(nincs)'}")
    checks.append(ok)

    jsons = glob.glob(os.path.join(expected_dir, "*.json"))
    ok    = len(jsons) > 0
    print(f"  {'✅' if ok else '❌'} JSON fájl létrejött:  {jsons[0] if jsons else '(nincs)'}")
    checks.append(ok)

    print(f"\n  Teljes idő: {elapsed:.1f}s")
    print("──────────────────────────────────────────────────────────────────")

    if all(checks):
        print("\n✅ Minden OK – a mérföldkő rendszer működik.\n")
        print(f"  A teszt kimenet itt van: {expected_dir}/")
        print(f"  (Töröld kézzel ha nem kell: rmdir /s /q ModellNaplo_SMOKE_TEST)")
    else:
        n_fail = checks.count(False)
        print(f"\n❌ {n_fail} ellenőrzés sikertelen – nézd meg a fenti hibákat.\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
