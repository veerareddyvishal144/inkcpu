"""Shared test helpers (plain scripts, engine venv python, no pytest needed, no network, mock backend only)."""
import json
import os
import socket
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(os.path.dirname(HERE))
if ENGINE not in sys.path:
    sys.path.insert(0, ENGINE)
FIX = os.path.join(HERE, "fixtures")


def fixture(name):
    with open(os.path.join(FIX, name), "r", encoding="utf-8") as f:
        return json.load(f)


_tok = None


def tok():
    global _tok
    if _tok is None:
        from server.tokenizer import InkTokenizer
        _tok = InkTokenizer()
    return _tok


class LiveServer:
    """Run a FastAPI app with uvicorn in a thread on an ephemeral port."""

    def __init__(self, app):
        import uvicorn
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        t0 = time.time()
        while not self.server.started:
            if time.time() - t0 > 20:
                raise RuntimeError("server did not start")
            time.sleep(0.02)
        # wait until the engine thread reports ready
        import urllib.request
        while time.time() - t0 < 20:
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2):
                    break
            except Exception:
                time.sleep(0.05)

    @property
    def url(self):
        return "http://127.0.0.1:%d" % self.port

    def stop(self):
        self.server.should_exit = True
        self.thread.join(10)


class Skip(Exception):
    """Raise inside a test to skip it (e.g. the real-engine tests without INK_GGUF / INK_LIB)."""


def run_module(mod_globals):
    """Run every test_* function of a module; return number of failures."""
    fails = 0
    for name in sorted(k for k in mod_globals if k.startswith("test_") and callable(mod_globals[k])):
        t0 = time.time()
        try:
            mod_globals[name]()
            print("PASS %-48s %.2fs" % (name, time.time() - t0), flush=True)
        except Skip as e:
            print("SKIP %-48s %s" % (name, e), flush=True)
        except Exception:
            fails += 1
            print("FAIL %-48s %.2fs" % (name, time.time() - t0), flush=True)
            traceback.print_exc()
    return fails
