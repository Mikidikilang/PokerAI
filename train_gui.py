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
                for part in line.split("|"):
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
    cli_path = os.path.join(BASE_DIR, "_train_session_cli.py")
    if os.path.exists(cli_path):
        return
    content = r'''#!/usr/bin/env python3
"""_train_session_cli.py -- CLI wrapper. Ne futtasd közvetlenül."""
import sys, os, json, argparse, atexit, glob as _g
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.runner import run_training_session
from training.model_manager import ModelManager, CONFIG_DEFAULTS
from config import TrainingConfig
from utils.checkpoint_utils import safe_load_checkpoint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mgr = ModelManager(BASE_DIR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",  required=True)
    parser.add_argument("--pth-path",    required=True)
    parser.add_argument("--players",     type=int, default=6)
    parser.add_argument("--episodes",    type=int, default=100_000)
    parser.add_argument("--config-json", default="{}")
    parser.add_argument("--session-id",  default=None)
    args = parser.parse_args()

    raw      = json.loads(args.config_json)
    bot_pool = raw.pop("bot_pool", CONFIG_DEFAULTS["bot_pool"])
    raw.pop("training_style", None); raw.pop("training_phase", None)

    valid = set(TrainingConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in raw.items() if k in valid}
    filtered["opponent_bot_types"]   = [k for k,v in bot_pool.items() if v.get("enabled",True)]
    filtered["opponent_bot_weights"] = [v.get("weight",1.0) for k,v in bot_pool.items() if v.get("enabled",True)]
    filtered["milestone_dir_root"]   = mgr.tests_dir(args.model_name)

    try:    cfg = TrainingConfig(**filtered)
    except: cfg = TrainingConfig()

    episodes_start = 0
    if os.path.exists(args.pth_path):
        try:
            ck = safe_load_checkpoint(args.pth_path, map_location="cpu")
            if isinstance(ck, dict):
                episodes_start = ck.get("episodes_trained", 0)
        except: pass

    session_id = args.session_id or mgr.start_session(
        args.model_name, json.loads(args.config_json), episodes_start, args.players)

    def _cleanup():
        try:
            ep_end = episodes_start + args.episodes
            if os.path.exists(args.pth_path):
                ck = safe_load_checkpoint(args.pth_path, map_location="cpu")
                if isinstance(ck, dict): ep_end = ck.get("episodes_trained", ep_end)
            metrics = {}
            logs = sorted(_g.glob(os.path.join(BASE_DIR, "logs", "session_*.log")))
            if logs:
                with open(logs[-1], "r", errors="replace") as f:
                    lines = f.readlines()
                for line in reversed(lines[-50:]):
                    if "Actor" in line and "Ep" in line:
                        for part in line.split("|"):
                            p = part.strip()
                            if p.startswith("Actor "):
                                try: metrics["actor_loss"] = float(p.split()[-1])
                                except: pass
                            elif p.startswith("Critic "):
                                try: metrics["critic_loss"] = float(p.split()[-1])
                                except: pass
                            elif p.startswith("Ent "):
                                try: metrics["entropy"] = float(p.split()[-1])
                                except: pass
                        break
            mgr.end_session(args.model_name, session_id, ep_end, metrics, completed=True)
        except Exception as ex:
            print(f"Cleanup hiba: {ex}")
    atexit.register(_cleanup)

    run_training_session(
        num_players=args.players,
        filename=args.pth_path,
        episodes_to_run=args.episodes,
        cfg=cfg,
    )

if __name__ == "__main__":
    main()
'''
    with open(cli_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"CLI script létrehozva: {cli_path}")


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
            running = False
            with _training_lock:
                if _training_proc is not None:
                    running = _training_proc.poll() is None
            with _metrics_lock:
                metrics = dict(_last_metrics)
            self._send_json({"running": running, "model": _training_model,
                             "session_id": _training_session_id, "metrics": metrics})
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
                if os.path.exists(pth):
                    try:
                        from utils.checkpoint_utils import safe_load_checkpoint
                        ck = safe_load_checkpoint(pth, map_location="cpu")
                        if isinstance(ck, dict):
                            episodes_start = ck.get("episodes_trained", 0)
                    except Exception:
                        pass

                session_id = mgr.start_session(model_name, config, episodes_start, num_players)

                cmd = [sys.executable, os.path.join(BASE_DIR, "_train_session_cli.py"),
                       "--model-name", model_name, "--pth-path", pth,
                       "--players", str(num_players), "--episodes", str(episodes),
                       "--config-json", json.dumps(config), "--session-id", session_id]

                with _training_lock:
                    _training_proc       = subprocess.Popen(cmd, cwd=BASE_DIR,
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    _training_model      = model_name
                    _training_session_id = session_id

                cfg_data = mgr.load_config(model_name)
                cfg_data["num_players"] = num_players
                cfg_data["config"]      = config
                mgr.save_config(model_name, cfg_data)

                self._send_json({"ok": True, "pid": _training_proc.pid, "session_id": session_id})

            elif path == "/api/stop":
                with _training_lock:
                    if _training_proc is not None and _training_proc.poll() is None:
                        _training_proc.terminate()
                        try: _training_proc.wait(timeout=5)
                        except subprocess.TimeoutExpired: _training_proc.kill()
                        if _training_model and _training_session_id:
                            try:
                                from utils.checkpoint_utils import safe_load_checkpoint
                                pth = mgr.pth_path(_training_model)
                                ep_end = 0
                                if os.path.exists(pth):
                                    ck = safe_load_checkpoint(pth, map_location="cpu")
                                    if isinstance(ck, dict):
                                        ep_end = ck.get("episodes_trained", 0)
                                mgr.end_session(_training_model, _training_session_id,
                                               ep_end, {}, completed=False)
                            except Exception as e:
                                logger.warning(f"Session lezárás hiba: {e}")
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
                _training_proc.terminate()
        server.server_close()


if __name__ == "__main__":
    main()
