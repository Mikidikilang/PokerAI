#!/usr/bin/env python3
"""
train_gui.py  --  Poker AI v4 Training Manager GUI (v2)

Elindítja a böngészős tréning kezelőt.

Használat:
    python train_gui.py [--port 8081]
    → http://localhost:8081
"""

import argparse
import http.server
import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.model_manager import ModelManager, CONFIG_DEFAULTS, STYLE_PRESETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainGUI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mgr      = ModelManager(BASE_DIR)

# ── Futó tréning állapot ─────────────────────────────────────────────────────
_training_proc       = None
_training_model      = None
_training_session_id = None
_training_lock       = threading.Lock()
_last_metrics        = {}
_metrics_lock        = threading.Lock()
_training_err_file   = None   # stderr fájl handle a subprocess-hez
_training_err_path   = None   # stderr log fájl elérési útja
_session_log_file    = None   # per-session részletes log handle
_session_log_path    = None   # per-session log fájl elérési útja


def _open_session_log(model_name: str, pth: str, num_players: int,
                      episodes: int, episodes_start: int,
                      session_id: str, config: dict, cmd: list) -> tuple:
    """
    Létrehoz egy részletes per-session log fájlt a logs/ mappában.

    Formátum: logs/train_ui_{model_name}_{timestamp}.log
    Tartalmazza a session metaadatokat fejlécként, utána a subprocess
    stdout+stderr outputját, végül a session záró adatokat.

    Visszatér: (file_handle, log_path)
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    log_path = os.path.join(BASE_DIR, "logs", f"train_ui_{safe_name}_{ts}.log")
    try:
        fh = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered
        fh.write("=" * 70 + "\n")
        fh.write("  POKER AI v4  –  Train Session Log\n")
        fh.write("=" * 70 + "\n")
        fh.write(f"  Dátum/idő      : {datetime.now().isoformat()}\n")
        fh.write(f"  Modell neve    : {model_name}\n")
        fh.write(f"  PthPath        : {pth}\n")
        fh.write(f"  PthLétezik     : {os.path.exists(pth)}\n")
        fh.write(f"  Játékosok      : {num_players}\n")
        fh.write(f"  Futtatandó ep. : {episodes:,}\n")
        fh.write(f"  Kezdő ep.      : {episodes_start:,}\n")
        fh.write(f"  Cél ep.        : {episodes_start + episodes:,}\n")
        fh.write(f"  Session ID     : {session_id}\n")
        fh.write(f"  Config         :\n")
        for line in json.dumps(config, indent=4, ensure_ascii=False,
                               default=str).splitlines():
            fh.write(f"    {line}\n")
        fh.write(f"  Subprocess cmd :\n    {' '.join(cmd)}\n")
        fh.write("=" * 70 + "\n")
        fh.write("  SUBPROCESS KIMENET (stdout + stderr):\n")
        fh.write("=" * 70 + "\n\n")
        fh.flush()
        return fh, log_path
    except Exception as e:
        logger.warning(f"Session log fájl nem hozható létre ({log_path}): {e}")
        return None, None


def _close_session_log(fh, log_path: str, exit_code=None,
                       episodes_end: int = None, completed: bool = False):
    """Lezárja a session log fájlt összefoglaló sorral."""
    if fh is None:
        return
    try:
        fh.write("\n" + "=" * 70 + "\n")
        fh.write("  SESSION VÉGE\n")
        fh.write(f"  Időpont     : {datetime.now().isoformat()}\n")
        if exit_code is not None:
            fh.write(f"  Kilépési kód: {exit_code}\n")
        if episodes_end is not None:
            fh.write(f"  Ep. vége    : {episodes_end:,}\n")
        fh.write(f"  Completed   : {completed}\n")
        fh.write("=" * 70 + "\n")
        fh.flush()
        fh.close()
        logger.info(f"Session log lezárva: {log_path}")
    except Exception as e:
        logger.debug(f"Session log lezárási hiba: {e}")


def _read_training_error() -> str:
    """Visszaadja a tréning subprocess stderr log utolsó 20 sorát (ha van)."""
    try:
        if _training_err_path and os.path.exists(_training_err_path):
            with open(_training_err_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-20:]).strip()
    except Exception:
        pass
    return ""


def _parse_latest_log() -> dict:
    log_dir = os.path.join(BASE_DIR, "logs")
    if not os.path.isdir(log_dir):
        return {}
    import glob as _g
    logs = sorted(_g.glob(os.path.join(log_dir, "session_*.log")))
    if not logs:
        return {}
    try:
        with open(logs[-1], "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        for line in reversed(lines[-80:]):
            if "Ep " in line and "Actor" in line:
                m = {}
                # [FIX P0-1] A logging formátum: "TIMESTAMP [LEVEL] MESSAGE"
                # A "] " token után következik a tényleges log üzenet.
                # Nélküle a split("|") első eleme a timestamp-et is tartalmazza,
                # pl. "2026-03-21 16:26:00 [INFO] Ep    1,000/100,000"
                # ami NEM kezdődik "Ep "-vel → current_ep soha nem töltődött be.
                raw_msg = line.split("] ", 1)[-1] if "] " in line else line
                for part in raw_msg.split("|"):
                    part = part.strip()
                    if part.startswith("Ep "):
                        try:
                            ep_part = part.replace("Ep ", "").split("/")
                            m["current_ep"] = int(ep_part[0].replace(",", "").strip())
                            m["target_ep"]  = int(ep_part[1].replace(",", "").strip())
                        except Exception:
                            pass
                    elif part.startswith("Eltelt "):
                        m["elapsed"] = part.replace("Eltelt ", "").strip()
                    elif part.startswith("ETA "):
                        m["eta"] = part.replace("ETA ", "").strip()
                    elif part.startswith("LR "):
                        m["lr"] = part.replace("LR ", "").strip()
                    elif part.startswith("Actor "):
                        try: m["actor"] = float(part.replace("Actor ", "").strip())
                        except: pass
                    elif part.startswith("Critic "):
                        try: m["critic"] = float(part.replace("Critic ", "").strip())
                        except: pass
                    elif part.startswith("Ent "):
                        try: m["entropy"] = float(part.replace("Ent ", "").strip())
                        except: pass
                    elif part.startswith("Pool "):
                        try: m["pool"] = int(part.replace("Pool ", "").strip())
                        except: pass
                if m:
                    return m
    except Exception:
        pass
    return {}


def _metrics_poller():
    while True:
        with _metrics_lock:
            _last_metrics.update(_parse_latest_log())
        time.sleep(2)


def _ensure_cli_script():
    """
    Biztosítja, hogy _train_session_cli.py létezik.
    Ha a fájl hiányzik (pl. véletlen törlés), újragenerálja a launcher-alapú verziót.
    """
    cli_path = os.path.join(BASE_DIR, "_train_session_cli.py")
    if os.path.exists(cli_path):
        return

    # A fájl tartalma azonos a projektben lévő _train_session_cli.py-val,
    # de beégetett fallback – ha a fájl megvan, ez soha nem fut le.
    content = '''\
#!/usr/bin/env python3
"""
_train_session_cli.py  --  GUI subprocess wrapper (ne futtasd közvetlenül!)
Auto-generált fallback – a tényleges verzió a projektben van.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logging_setup import setup_logging
from training.model_manager import ModelManager
from training.launcher import build_training_config, launch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",         required=True)
    parser.add_argument("--pth-path",           required=True)
    parser.add_argument("--players",            type=int, default=6)
    parser.add_argument("--episodes",           type=int, default=100_000)
    parser.add_argument("--config-json",        default="{}")
    parser.add_argument("--session-id",         default=None)
    parser.add_argument("--milestone-interval", type=int, default=None)
    parser.add_argument("--drive-output-dir",   default=None)
    args = parser.parse_args()

    setup_logging("training.log", num_players=args.players)

    output_base = args.drive_output_dir or BASE_DIR
    if args.drive_output_dir:
        os.makedirs(args.drive_output_dir, exist_ok=True)

    mgr = ModelManager(output_base)
    config_dict = json.loads(args.config_json)

    cfg = build_training_config(
        config_dict=config_dict,
        model_name=args.model_name,
        mgr=mgr,
        milestone_interval=args.milestone_interval,
    )

    launch(
        model_name=args.model_name,
        pth_path=args.pth_path,
        num_players=args.players,
        episodes=args.episodes,
        cfg=cfg,
        mgr=mgr,
        session_id=args.session_id,
        output_base=output_base,
    )


if __name__ == "__main__":
    main()
'''
    with open(cli_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"CLI script (fallback) létrehozva: {cli_path}")


class GUIHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.debug("HTTP %s", args[0] if args else "")

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            try:
                return json.loads(self.rfile.read(length))
            except Exception:
                return {}
        return {}

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_html(os.path.join(BASE_DIR, "train_gui.html"))
        elif path == "/api/models":
            self._send_json({"models": mgr.list_models()})
        elif path.startswith("/api/config/"):
            self._send_json(mgr.load_config(path.replace("/api/config/", "")))
        elif path.startswith("/api/naplo/"):
            self._send_json(mgr.load_naplo(path.replace("/api/naplo/", "")))
        elif path.startswith("/api/tests/"):
            self._send_json({"tests": mgr.list_tests(path.replace("/api/tests/", ""))})
        elif path == "/api/defaults":
            self._send_json({"defaults": CONFIG_DEFAULTS})
        elif path == "/api/presets":
            self._send_json(STYLE_PRESETS)
        elif path == "/api/status":
            # [FIX P1-1] Az összes _training_* globálist egyszerre olvassuk a lock
            # alatt – race condition volt, mikor a lock felszabadult és a /api/stop
            # kinullázta a globálisokat, mielőtt a válasz összeállt volna.
            with _training_lock:
                if _training_proc is not None:
                    running = _training_proc.poll() is None
                else:
                    running = False
                snap_model      = _training_model
                snap_session_id = _training_session_id
            with _metrics_lock:
                metrics = dict(_last_metrics)
            err_msg = "" if running else _read_training_error()
            self._send_json({"running": running, "model": snap_model,
                             "session_id": snap_session_id, "metrics": metrics,
                             "last_error": err_msg})
        else:
            self.send_error(404)

    def do_POST(self):
        global _training_proc, _training_model, _training_session_id
        path = urlparse(self.path).path
        try:
            body = self._read_body()

            if path.startswith("/api/config/"):
                mgr.save_config(path.replace("/api/config/", ""), body)
                self._send_json({"ok": True})

            elif path == "/api/create_model":
                name        = body.get("name", "").strip()
                num_players = int(body.get("num_players", 6))
                if not name:
                    self._send_json({"error": "Hiányzó modellnév"}, 400); return
                mgr.ensure_model_dir(name, num_players)
                pth = mgr.pth_path(name)
                self._send_json({"name": name, "model_dir": mgr.model_dir(name),
                                 "pth_path": pth, "pth_exists": os.path.exists(pth)})

            elif path == "/api/migrate":
                rel_pth     = body.get("rel_pth", "")
                name        = body.get("name", "")
                num_players = int(body.get("num_players", 6))
                abs_pth     = os.path.join(BASE_DIR, rel_pth)
                if not os.path.exists(abs_pth):
                    self._send_json({"error": "Fájl nem található"}, 404); return
                new_pth = mgr.migrate_to_models_dir(abs_pth, name, num_players)
                self._send_json({"ok": True, "new_path": new_pth, "name": name})

            elif path == "/api/start":
                global _training_err_file, _training_err_path
                global _session_log_file, _session_log_path
                model_name  = body.get("model_name", "")
                num_players = int(body.get("num_players", 6))
                episodes    = int(body.get("episodes", 100_000))
                config      = body.get("config", {})

                with _training_lock:
                    if _training_proc is not None and _training_proc.poll() is None:
                        self._send_json({"error": "Már fut egy tréning!"}); return

                _ensure_cli_script()
                mgr.ensure_model_dir(model_name, num_players)
                pth = mgr.pth_path(model_name)

                episodes_start = 0
                pth_exists = os.path.exists(pth)
                if pth_exists:
                    try:
                        from utils.checkpoint_utils import safe_load_checkpoint
                        ck = safe_load_checkpoint(pth, map_location="cpu")
                        if isinstance(ck, dict):
                            episodes_start = ck.get("episodes_trained", 0)
                    except Exception as ck_exc:
                        logger.warning(f"Checkpoint olvasási hiba ({pth}): {ck_exc}")
                else:
                    logger.info(
                        f"Checkpoint nem található ({pth}) → új modell, 0-ról indul"
                    )

                logger.info(
                    f"/api/start | modell={model_name!r} | pth={pth} | "
                    f"pth_létezik={pth_exists} | episodes_start={episodes_start:,} | "
                    f"futtatandó={episodes:,} | cél={episodes_start + episodes:,}"
                )

                session_id = mgr.start_session(model_name, config, episodes_start, num_players)

                cmd = [sys.executable, os.path.join(BASE_DIR, "_train_session_cli.py"),
                       "--model-name", model_name, "--pth-path", pth,
                       "--players", str(num_players), "--episodes", str(episodes),
                       "--config-json", json.dumps(config), "--session-id", session_id]

                os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

                # Per-session részletes log (stdout + stderr a subprocesstől)
                if _session_log_file is not None:
                    try: _session_log_file.close()
                    except Exception: pass
                _session_log_file, _session_log_path = _open_session_log(
                    model_name, pth, num_players, episodes,
                    episodes_start, session_id, config, cmd
                )

                # training_err.log megtartjuk visszafelé kompatibilitásból
                # (a session log fájl tartalmazza az stderr-t is)
                err_path = os.path.join(BASE_DIR, "logs", "training_err.log")
                if _training_err_file is not None:
                    try: _training_err_file.close()
                    except Exception: pass
                _training_err_file = open(err_path, "w", encoding="utf-8")
                _training_err_path = err_path

                with _training_lock:
                    _training_proc = subprocess.Popen(
                        cmd, cwd=BASE_DIR,
                        stdout=_session_log_file if _session_log_file else subprocess.DEVNULL,
                        stderr=_session_log_file if _session_log_file else _training_err_file,
                        start_new_session=True,
                    )
                    _training_model      = model_name
                    _training_session_id = session_id

                logger.info(
                    f"Subprocess elindítva | PID={_training_proc.pid} | "
                    f"session_id={session_id} | log={_session_log_path}"
                )

                cfg_data = mgr.load_config(model_name)
                cfg_data["num_players"] = num_players
                cfg_data["config"]      = config
                mgr.save_config(model_name, cfg_data)

                self._send_json({
                    "ok": True,
                    "pid": _training_proc.pid,
                    "session_id": session_id,
                    "pth_path": pth,
                    "episodes_start": episodes_start,
                    "session_log": _session_log_path,
                })

            elif path == "/api/stop":
                with _training_lock:
                    if _training_proc is not None and _training_proc.poll() is None:
                        # [FIX P1-3] Process group kill: a milestone teszt
                        # subprocess-t is leöli, nem csak a CLI wrapper-t.
                        try:
                            import os as _os, signal as _sig
                            _os.killpg(_os.getpgid(_training_proc.pid), _sig.SIGTERM)
                        except (AttributeError, ProcessLookupError, OSError):
                            _training_proc.terminate()
                        try:
                            _training_proc.wait(timeout=8)
                        except subprocess.TimeoutExpired:
                            try:
                                import os as _os, signal as _sig
                                _os.killpg(_os.getpgid(_training_proc.pid), _sig.SIGKILL)
                            except (AttributeError, ProcessLookupError, OSError):
                                _training_proc.kill()
                            _training_proc.wait()
                        exit_code = _training_proc.returncode
                        ep_end = 0
                        if _training_model and _training_session_id:
                            try:
                                from utils.checkpoint_utils import safe_load_checkpoint
                                pth = mgr.pth_path(_training_model)
                                if os.path.exists(pth):
                                    ck = safe_load_checkpoint(pth, map_location="cpu")
                                    if isinstance(ck, dict):
                                        ep_end = ck.get("episodes_trained", 0)
                                mgr.end_session(_training_model, _training_session_id,
                                               ep_end, {}, completed=False)
                            except Exception as e:
                                logger.warning(f"Session lezárás hiba: {e}")
                        _close_session_log(
                            _session_log_file, _session_log_path,
                            exit_code=exit_code, episodes_end=ep_end, completed=False
                        )
                        _training_proc = _training_model = _training_session_id = None
                self._send_json({"ok": True})

            elif path == "/api/apply_preset":
                result = mgr.apply_style_preset(body.get("config", {}), body.get("style", "exploitative"))
                self._send_json({"config": result})

            elif path.startswith("/api/naplo_note/"):
                name = path.replace("/api/naplo_note/", "")
                mgr.add_naplo_note(name, body.get("session_id", ""), body.get("note", ""))
                self._send_json({"ok": True})

            elif path.startswith("/api/test_log/"):
                parts = path.replace("/api/test_log/", "").split("/", 1)
                name = parts[0]; filename = parts[1] if len(parts) > 1 else ""
                log = mgr.get_test_log(name, filename.replace(".json", ".log"))
                self._send_json({"log": log})

            else:
                self.send_error(404)

        except Exception as e:
            logger.error(f"API hiba ({path}): {e}\n{traceback.format_exc()}")
            self._send_json({"error": str(e)}, 500)


def main():
    parser = argparse.ArgumentParser(description="Poker AI v4 Training Manager v2")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    html_path = os.path.join(BASE_DIR, "train_gui.html")
    if not os.path.exists(html_path):
        print(f"⚠ Hiányzik: train_gui.html")
        return

    _ensure_cli_script()
    t = threading.Thread(target=_metrics_poller, daemon=True)
    t.start()

    server = http.server.HTTPServer(("0.0.0.0", args.port), GUIHandler)
    url    = f"http://localhost:{args.port}"

    print("\n" + "=" * 60)
    print("  🃏  POKER AI v4  –  Training Manager  v2")
    print("=" * 60)
    print(f"  URL:          {url}")
    print(f"  Modell mappa: {os.path.join(BASE_DIR, 'models')}/")
    print(f"  Ctrl+C leállításhoz")
    print("=" * 60 + "\n")

    if not args.no_browser:
        try: webbrowser.open(url)
        except: pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Leállítva.")
        with _training_lock:
            if _training_proc and _training_proc.poll() is None:
                # [FIX P1-4] terminate() + wait() párban – nélküle Unix-on
                # zombie process marad (SIGTERM elküldve, de szülő soha nem
                # hívja wait()-et, a kernel nem tudja felszabadítani az entry-t).
                try:
                    import os as _os, signal as _sig
                    _os.killpg(_os.getpgid(_training_proc.pid), _sig.SIGTERM)
                except (AttributeError, ProcessLookupError, OSError):
                    _training_proc.terminate()
                try:
                    _training_proc.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    try:
                        import os as _os, signal as _sig
                        _os.killpg(_os.getpgid(_training_proc.pid), _sig.SIGKILL)
                    except (AttributeError, ProcessLookupError, OSError):
                        _training_proc.kill()
                    _training_proc.wait()
        server.server_close()


if __name__ == "__main__":
    main()
