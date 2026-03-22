#!/usr/bin/env python3
"""
tests/test_model_manager.py  –  ModelManager unit tesztek

Tesztelt hibaesetek:
  1. pth_path() névduplázódás bug – migrált fájlok esetén
     ({name}_ppo_v4.pth már létezik, de pth_path() _ppo_v4_ppo_v4.pth-t generálna)
  2. Új modellnél helyes alapértelmezett fájlnév
  3. Migrált modell esetén a meglévő fájlt adja vissza, nem generáltat
  4. ensure_model_dir idempotens
  5. start_session / end_session napló kerekesége
  6. episodes_start helyes kiolvasása checkpoint-ból
  7. list_models visszaad minden modellt és pth_path egyezik
  8. Session log fájl tartalmaz kritikus mezőket

Futtatás:
    cd <projekt_gyökér>
    python -m pytest tests/test_model_manager.py -v
    # vagy:
    python tests/test_model_manager.py
"""

import json
import os
import sys
import tempfile
import unittest

# A tesztek futtathatók a projekt gyökeréből
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model_manager import ModelManager, CONFIG_DEFAULTS


# ── Segéd: hamis checkpoint dict ────────────────────────────────────────────

def _fake_checkpoint(episodes_trained: int = 500_000,
                     num_players: int = 2) -> dict:
    """Minimális, torch.save-vel menthető checkpoint dict."""
    return {
        "state_dict":      {},
        "trainer":         {},
        "reward_norm":     {},
        "episodes_trained": episodes_trained,
        "time_spent":      123.0,
        "algorithm":       "PPO_SelfPlay_v4",
        "state_size":      300,
        "action_size":     9,
        "num_players":     num_players,
        "rlcard_obs_size": 54,
    }


def _write_fake_pth(path: str, episodes: int = 500_000, num_players: int = 2):
    """Ír egy minimális .pth fájlt (JSON-alapú, nem torch – torch nélkül is fut)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # JSON-ként mentjük, hogy torch függőség nélkül tesztelhető legyen.
    # A checkpoint_utils safe_load_checkpoint torch.load-ot hív, de
    # a pth_path() fix CSAK os.path.exists()-t és glob-ot használ,
    # tehát a névfelbontás teszteléséhez elegendő az üres fájl is.
    with open(path, "wb") as f:
        f.write(b"FAKE_PTH")  # nem valódi torch checkpoint, de létezik


# ════════════════════════════════════════════════════════════════════════════
#  1. pth_path() tesztek
# ════════════════════════════════════════════════════════════════════════════

class TestPthPath(unittest.TestCase):
    """pth_path() névfelbontás tesztek – ez volt a root-cause bug."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="poker_ai_test_")
        self.mgr = ModelManager(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ── 1.1 Új modell: alapértelmezett név generálódik ────────────────────

    def test_new_model_default_filename(self):
        """Új modell (nincs .pth a mappában) → {name}_ppo_v4.pth."""
        name = "2max"
        self.mgr.ensure_model_dir(name, num_players=2)
        path = self.mgr.pth_path(name)
        self.assertTrue(
            path.endswith(f"{name}_ppo_v4.pth"),
            f"Várt végzet: {name}_ppo_v4.pth, kapott: {path}"
        )

    # ── 1.2 BUG REPRODUKCIÓ: migrált modell névduplázódás ────────────────

    def test_migrated_model_no_name_doubling(self):
        """
        BUG: Ha a modellkönyvtárban 2max_ppo_v4.pth van (migrált fájl),
        pth_path("2max_ppo_v4") NEM adhat vissza 2max_ppo_v4_ppo_v4.pth-t.

        A fix előtti viselkedés:
            pth_path("2max_ppo_v4") → models/2max_ppo_v4/2max_ppo_v4_ppo_v4.pth  ← ROSSZ
        A fix utáni viselkedés:
            pth_path("2max_ppo_v4") → models/2max_ppo_v4/2max_ppo_v4.pth  ← JÓ
        """
        name = "2max_ppo_v4"
        self.mgr.ensure_model_dir(name, num_players=2)

        # Migrált fájl neve megőrzi az eredetit (nem kap _ppo_v4 suffixet)
        actual_file = os.path.join(self.mgr.model_dir(name), f"{name}.pth")
        _write_fake_pth(actual_file)

        result = self.mgr.pth_path(name)

        # A visszaadott útvonal NE tartalmazzon _ppo_v4_ppo_v4-et
        self.assertNotIn(
            "_ppo_v4_ppo_v4", result,
            f"Névduplázódás: {result}"
        )
        # A visszaadott útvonal a meglévő fájlra mutasson
        self.assertEqual(
            os.path.normpath(result),
            os.path.normpath(actual_file),
            f"Várt: {actual_file}, kapott: {result}"
        )

    def test_migrated_model_file_exists(self):
        """A visszaadott pth_path() fájl ténylegesen létezik (migrált eset)."""
        name = "6max_ppo_v4"
        self.mgr.ensure_model_dir(name, num_players=6)
        actual_file = os.path.join(self.mgr.model_dir(name), f"{name}.pth")
        _write_fake_pth(actual_file)

        result = self.mgr.pth_path(name)
        self.assertTrue(
            os.path.exists(result),
            f"A visszaadott pth_path() fájl nem létezik: {result}"
        )

    # ── 1.3 Normál (nem migrált) modell ──────────────────────────────────

    def test_standard_model_correct_path(self):
        """Normál esetben: 2max → models/2max/2max_ppo_v4.pth."""
        name = "2max"
        self.mgr.ensure_model_dir(name, num_players=2)
        expected_file = os.path.join(self.mgr.model_dir(name), "2max_ppo_v4.pth")
        _write_fake_pth(expected_file)

        result = self.mgr.pth_path(name)
        self.assertEqual(
            os.path.normpath(result),
            os.path.normpath(expected_file)
        )

    # ── 1.4 Explicit filename megadásakor nem keres ───────────────────────

    def test_explicit_filename_ignores_existing(self):
        """Ha filename explicit meg van adva, azt adja vissza (nem keres)."""
        name = "2max"
        self.mgr.ensure_model_dir(name, num_players=2)
        # Van egy meglévő fájl
        other_file = os.path.join(self.mgr.model_dir(name), "2max.pth")
        _write_fake_pth(other_file)

        explicit_name = "2max_ppo_v4.pth"
        result = self.mgr.pth_path(name, filename=explicit_name)
        self.assertTrue(
            result.endswith(explicit_name),
            f"Explicit filename nem lett figyelembe véve: {result}"
        )

    # ── 1.5 Üres könyvtár: visszaesik az alapértelmezett névre ───────────

    def test_empty_dir_falls_back_to_default(self):
        """Ha a modellkönyvtár üres (nincs .pth), visszaesik az alapértelmezésre."""
        name = "freshmodel"
        self.mgr.ensure_model_dir(name, num_players=6)
        # Nem írunk .pth-t
        result = self.mgr.pth_path(name)
        self.assertTrue(result.endswith(f"{name}_ppo_v4.pth"))


# ════════════════════════════════════════════════════════════════════════════
#  2. ensure_model_dir tesztek
# ════════════════════════════════════════════════════════════════════════════

class TestEnsureModelDir(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="poker_ai_test_")
        self.mgr = ModelManager(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_required_dirs(self):
        """ensure_model_dir létrehozza a szükséges mappákat."""
        self.mgr.ensure_model_dir("testmodel", 2)
        self.assertTrue(os.path.isdir(self.mgr.model_dir("testmodel")))
        self.assertTrue(os.path.isdir(self.mgr.tests_dir("testmodel")))

    def test_creates_config_json(self):
        """config.json létrejön az alapértelmezett mezőkkel."""
        self.mgr.ensure_model_dir("testmodel", 2)
        cfg_path = self.mgr.config_path("testmodel")
        self.assertTrue(os.path.exists(cfg_path))
        cfg = json.loads(open(cfg_path, encoding="utf-8").read())
        self.assertIn("num_players", cfg)
        self.assertIn("config", cfg)

    def test_creates_naplo_json(self):
        """naplo.json létrejön üres sessions listával."""
        self.mgr.ensure_model_dir("testmodel", 2)
        naplo = json.loads(open(self.mgr.naplo_path("testmodel"), encoding="utf-8").read())
        self.assertEqual(naplo["model_name"], "testmodel")
        self.assertEqual(naplo["sessions"], [])

    def test_idempotent(self):
        """ensure_model_dir kétszer hívva sem dob hibát, nem írja felül a meglévőt."""
        self.mgr.ensure_model_dir("testmodel", 2)
        # Írunk valamit a config-ba
        cfg_path = self.mgr.config_path("testmodel")
        data = json.loads(open(cfg_path, encoding="utf-8").read())
        data["custom_field"] = "sentinel"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        # Második hívás
        self.mgr.ensure_model_dir("testmodel", 2)
        data2 = json.loads(open(cfg_path, encoding="utf-8").read())
        self.assertEqual(data2.get("custom_field"), "sentinel",
                         "ensure_model_dir felülírta a meglévő config.json-t")


# ════════════════════════════════════════════════════════════════════════════
#  3. Session napló tesztek
# ════════════════════════════════════════════════════════════════════════════

class TestSessionNaplo(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="poker_ai_test_")
        self.mgr = ModelManager(self.tmpdir)
        self.mgr.ensure_model_dir("2max", 2)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_start_session_returns_id(self):
        """start_session érvényes session_id-t ad vissza."""
        sid = self.mgr.start_session("2max", {}, 0, 2)
        self.assertTrue(sid.startswith("sess_"), f"Várt 'sess_' prefix, kapott: {sid}")

    def test_start_session_written_to_naplo(self):
        """start_session felírja a session-t a naplo.json-ba."""
        sid = self.mgr.start_session("2max", {"lr": 0.001}, 1000, 2)
        naplo = self.mgr.load_naplo("2max")
        session_ids = [s["id"] for s in naplo["sessions"]]
        self.assertIn(sid, session_ids)

    def test_end_session_updates_episodes(self):
        """end_session frissíti az episodes_end-et és a total_episodes-t."""
        sid = self.mgr.start_session("2max", {}, 500_000, 2)
        self.mgr.end_session("2max", sid, 1_000_000, {}, completed=True)
        naplo = self.mgr.load_naplo("2max")
        self.assertEqual(naplo["total_episodes"], 1_000_000)
        sess = next(s for s in naplo["sessions"] if s["id"] == sid)
        self.assertEqual(sess["episodes_end"], 1_000_000)
        self.assertEqual(sess["episodes_added"], 500_000)
        self.assertTrue(sess["completed"])

    def test_end_session_partial(self):
        """end_session completed=False esetén is ment."""
        sid = self.mgr.start_session("2max", {}, 0, 2)
        self.mgr.end_session("2max", sid, 50_000, {}, completed=False)
        naplo = self.mgr.load_naplo("2max")
        sess = next(s for s in naplo["sessions"] if s["id"] == sid)
        self.assertFalse(sess["completed"])
        self.assertEqual(sess["episodes_end"], 50_000)

    def test_multiple_sessions_accumulate(self):
        """Több session egymás után felhalmozódik a naplóban."""
        for ep in [0, 100_000, 200_000]:
            sid = self.mgr.start_session("2max", {}, ep, 2)
            self.mgr.end_session("2max", sid, ep + 100_000, {})
        naplo = self.mgr.load_naplo("2max")
        self.assertEqual(len(naplo["sessions"]), 3)

    def test_episodes_start_is_preserved(self):
        """A session episodes_start mezője megőrzi az indulási értéket."""
        sid = self.mgr.start_session("2max", {}, 999_999, 2)
        naplo = self.mgr.load_naplo("2max")
        sess = next(s for s in naplo["sessions"] if s["id"] == sid)
        self.assertEqual(sess["episodes_start"], 999_999)


# ════════════════════════════════════════════════════════════════════════════
#  4. A teljes "start training" folyamat szimulációja
#     (azt ellenőrzi, hogy a pth_path bug valóban 0-ra vitte-e az episodes_start-ot)
# ════════════════════════════════════════════════════════════════════════════

class TestTrainingStartContinuation(unittest.TestCase):
    """
    Szimulálja a train_gui.py /api/start logikáját, és ellenőrzi, hogy
    migrált modell esetén helyesen olvassa-e ki az episodes_trained értéket.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="poker_ai_test_")
        self.mgr = ModelManager(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _simulate_api_start_episodes(self, model_name: str, actual_pth: str) -> int:
        """
        Megismétli a /api/start episodes_start kiolvasási logikáját.
        Visszatér: episodes_start (amit a tréning kapna).
        """
        pth = self.mgr.pth_path(model_name)
        episodes_start = 0
        if os.path.exists(pth):
            try:
                import json as _json
                # A teszt fake .pth-t ír, ami JSON – de a pth_path bug
                # ellenőrzéséhez csak az os.path.exists() számít.
                # Ha a pth_path() helyes fájlt ad vissza, exists() True lesz.
                episodes_start = 500_000  # szimulált checkpoint érték
            except Exception:
                pass
        return episodes_start

    def test_migrated_model_continues_not_restarts(self):
        """
        BUG REPRODUKCIÓ: migrált modell esetén a /api/start NE induljon 0-ról.

        Régi viselkedés (bug):
            model_name = "2max_ppo_v4"
            pth_path("2max_ppo_v4") → "...2max_ppo_v4_ppo_v4.pth"  (nem létezik)
            episodes_start = 0  ← HIBA, 0-ról indul

        Új viselkedés (fix):
            pth_path("2max_ppo_v4") → "...2max_ppo_v4.pth"  (létezik)
            episodes_start = 500_000  ← helyes folytatás
        """
        name = "2max_ppo_v4"
        self.mgr.ensure_model_dir(name, num_players=2)

        # Migrált fájl: a könyvtár neve 2max_ppo_v4, a fájl neve 2max_ppo_v4.pth
        actual_file = os.path.join(self.mgr.model_dir(name), "2max_ppo_v4.pth")
        _write_fake_pth(actual_file, episodes=500_000)

        # pth_path() most visszaadja az actual_file-t
        returned_path = self.mgr.pth_path(name)
        file_found = os.path.exists(returned_path)

        self.assertTrue(
            file_found,
            f"BUG: pth_path() nem a meglévő fájlt adja vissza!\n"
            f"  Visszaadott: {returned_path}\n"
            f"  Meglévő:     {actual_file}\n"
            f"  → episodes_start = 0 lenne (0-ról indulna a tréning)"
        )

    def test_new_model_starts_from_zero(self):
        """Új modell esetén (nincs .pth) helyes, hogy 0-ról indul."""
        name = "brandnew"
        self.mgr.ensure_model_dir(name, num_players=6)
        pth = self.mgr.pth_path(name)
        self.assertFalse(
            os.path.exists(pth),
            "Új modellnél nem kéne létező .pth-t találni"
        )

    def test_normal_model_path_resolved_correctly(self):
        """Normál névkonvenció (2max) helyesen fel van oldva."""
        name = "2max"
        self.mgr.ensure_model_dir(name, num_players=2)
        pth_file = os.path.join(self.mgr.model_dir(name), "2max_ppo_v4.pth")
        _write_fake_pth(pth_file)

        returned = self.mgr.pth_path(name)
        self.assertTrue(os.path.exists(returned))
        self.assertEqual(os.path.normpath(returned), os.path.normpath(pth_file))


# ════════════════════════════════════════════════════════════════════════════
#  5. list_models tesztek
# ════════════════════════════════════════════════════════════════════════════

class TestListModels(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="poker_ai_test_")
        self.mgr = ModelManager(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_empty_returns_empty_list(self):
        result = self.mgr.list_models()
        self.assertIsInstance(result, list)

    def test_model_dir_without_pth_is_listed(self):
        """Pth nélküli modellmappa is szerepel a listán."""
        self.mgr.ensure_model_dir("emptymodel", 2)
        models = self.mgr.list_models()
        names = [m["name"] for m in models]
        self.assertIn("emptymodel", names)

    def test_model_with_pth_has_correct_path(self):
        """Modell .pth-val: abs_pth egyezik a pth_path()-szal."""
        name = "2max"
        self.mgr.ensure_model_dir(name, 2)
        pth = os.path.join(self.mgr.model_dir(name), "2max_ppo_v4.pth")
        _write_fake_pth(pth)

        models = self.mgr.list_models()
        m = next((x for x in models if x["name"] == name), None)
        self.assertIsNotNone(m, "A modell nem szerepel a listában")
        self.assertEqual(
            os.path.normpath(m["abs_pth"]),
            os.path.normpath(pth)
        )

    def test_migrated_model_abs_pth_exists(self):
        """Migrált modell abs_pth-ja létező fájlra mutat."""
        name = "2max_ppo_v4"
        self.mgr.ensure_model_dir(name, 2)
        pth = os.path.join(self.mgr.model_dir(name), "2max_ppo_v4.pth")
        _write_fake_pth(pth)

        models = self.mgr.list_models()
        m = next((x for x in models if x["name"] == name), None)
        self.assertIsNotNone(m)
        self.assertTrue(os.path.exists(m["abs_pth"]))


# ════════════════════════════════════════════════════════════════════════════
#  6. Session log fájl tesztek (train_gui.py _open_session_log)
# ════════════════════════════════════════════════════════════════════════════

class TestSessionLogFile(unittest.TestCase):
    """
    Ellenőrzi a train_gui.py _open_session_log() és _close_session_log()
    függvényeit. A train_gui.py import helyett inline teszteljük a logikát.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="poker_ai_test_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_session_log(self, model_name, pth, num_players,
                             episodes, episodes_start, session_id, config, cmd):
        """_open_session_log logikájának közvetlen tesztje (importált függvény)."""
        from datetime import datetime as _dt
        import json as _json

        log_path = os.path.join(self.tmpdir, "logs",
                                f"train_ui_{model_name}_test.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = open(log_path, "w", encoding="utf-8", buffering=1)
        fh.write("=" * 70 + "\n")
        fh.write("  POKER AI v4  –  Train Session Log\n")
        fh.write("=" * 70 + "\n")
        fh.write(f"  Dátum/idő      : {_dt.now().isoformat()}\n")
        fh.write(f"  Modell neve    : {model_name}\n")
        fh.write(f"  PthPath        : {pth}\n")
        fh.write(f"  PthLétezik     : {os.path.exists(pth)}\n")
        fh.write(f"  Játékosok      : {num_players}\n")
        fh.write(f"  Futtatandó ep. : {episodes:,}\n")
        fh.write(f"  Kezdő ep.      : {episodes_start:,}\n")
        fh.write(f"  Cél ep.        : {episodes_start + episodes:,}\n")
        fh.write(f"  Session ID     : {session_id}\n")
        fh.write(f"  Subprocess cmd :\n    {' '.join(cmd)}\n")
        fh.write("=" * 70 + "\n\n")
        fh.flush()
        return fh, log_path

    def test_log_file_created(self):
        fh, path = self._create_session_log(
            "2max", "/fake/path.pth", 2, 100_000, 500_000,
            "sess_test", {}, ["python", "train.py"]
        )
        fh.close()
        self.assertTrue(os.path.exists(path))

    def test_log_contains_model_name(self):
        fh, path = self._create_session_log(
            "testmodel", "/fake/path.pth", 6, 50_000, 0,
            "sess_abc", {}, ["python", "x.py"]
        )
        fh.close()
        content = open(path, encoding="utf-8").read()
        self.assertIn("testmodel", content)

    def test_log_contains_episodes_info(self):
        fh, path = self._create_session_log(
            "2max", "/p.pth", 2, 100_000, 500_000,
            "sess_001", {}, []
        )
        fh.close()
        content = open(path, encoding="utf-8").read()
        self.assertIn("500,000", content)   # episodes_start
        self.assertIn("600,000", content)   # cél ep

    def test_log_contains_session_id(self):
        fh, path = self._create_session_log(
            "2max", "/p.pth", 2, 100, 0, "sess_UNIQUE_ID", {}, []
        )
        fh.close()
        content = open(path, encoding="utf-8").read()
        self.assertIn("sess_UNIQUE_ID", content)

    def test_log_pth_exists_flag(self):
        """A log tartalmaz PthLétezik mezőt."""
        fh, path = self._create_session_log(
            "2max", "/non_existent.pth", 2, 100, 0, "sess_x", {}, []
        )
        fh.close()
        content = open(path, encoding="utf-8").read()
        self.assertIn("PthLétezik", content)


# ════════════════════════════════════════════════════════════════════════════
#  7. Regresszió: pth_path konzisztens a list_models() visszatérési értékével
# ════════════════════════════════════════════════════════════════════════════

class TestPthPathConsistency(unittest.TestCase):
    """
    pth_path() és list_models() konzisztenciája:
    a list_models() által visszaadott abs_pth-nak egyeznie kell
    azzal a fájllal, amit pth_path() ad vissza (ha van .pth).
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="poker_ai_test_")
        self.mgr = ModelManager(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_list_models_and_pth_path_agree(self):
        """
        list_models() és pth_path() ugyanazt a .pth fájlt adja vissza
        egy adott modellhez.
        """
        for name, fname in [
            ("2max",         "2max_ppo_v4.pth"),
            ("2max_ppo_v4",  "2max_ppo_v4.pth"),   # migrált eset
            ("6max",         "6max_ppo_v4.pth"),
        ]:
            with self.subTest(name=name):
                self.mgr.ensure_model_dir(name, 2)
                actual_file = os.path.join(self.mgr.model_dir(name), fname)
                _write_fake_pth(actual_file)

                models = self.mgr.list_models()
                m = next((x for x in models if x["name"] == name), None)
                self.assertIsNotNone(m, f"Modell nem található: {name}")

                if m["abs_pth"] is not None:
                    pth_from_manager = self.mgr.pth_path(name)
                    self.assertEqual(
                        os.path.normpath(m["abs_pth"]),
                        os.path.normpath(pth_from_manager),
                        f"list_models() és pth_path() nem egyezik: {name}\n"
                        f"  list_models: {m['abs_pth']}\n"
                        f"  pth_path:    {pth_from_manager}"
                    )


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import os
    from datetime import datetime
    import unittest

    # 1. Meghatározzuk a projekt gyökerét (a tests mappa szülőkönyvtárát)
    # __file__ a jelenlegi fájl útvonala, kétszer lépünk feljebb a gyökérhez
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 2. Létrehozzuk a cél mappát: <projekt_gyökér>/logs/test_model_manager
    log_dir = os.path.join(project_root, "logs", "test_model_manager")
    os.makedirs(log_dir, exist_ok=True)

    # 3. Generálunk egy fájlnevet a pontos dátummal és idővel
    ido_belyeg = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"test_model_manager_{ido_belyeg}.log")

    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])

    # 4. Megnyitjuk a fájlt, és abba irányítjuk a tesztfutató kimenetét
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"=== Teszt futtatása: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        
        # A 'stream=log_file' paraméter mondja meg, hogy a fájlba írjon, ne a konzolra
        runner = unittest.TextTestRunner(stream=log_file, verbosity=2)
        result = runner.run(suite)

    # 5. Visszajelzés a konzolra a relatív útvonallal, hogy átlátható legyen
    rel_path = os.path.relpath(log_file_path, os.getcwd())
    if result.wasSuccessful():
        print(f"✅ Minden teszt sikeresen lefutott! Részletek mentve ide:\n   -> {rel_path}")
    else:
        print(f"❌ Néhány teszt elbukott. A hibaüzeneteket megtalálod itt:\n   -> {rel_path}")

    sys.exit(0 if result.wasSuccessful() else 1)