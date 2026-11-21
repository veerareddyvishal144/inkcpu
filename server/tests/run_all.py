#!/usr/bin/env python
"""Run all server tests with the engine venv python:  /data/engine/.venv/bin/python server/tests/run_all.py
Mock backend only, no network, < 1 minute."""
import glob
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers  # noqa: E402

t0 = time.time()
fails = 0
n = 0
for path in sorted(glob.glob(os.path.join(helpers.HERE, "test_*.py"))):
    print("== %s" % os.path.basename(path), flush=True)
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    n += sum(1 for k in vars(mod) if k.startswith("test_") and callable(vars(mod)[k]))
    fails += helpers.run_module(vars(mod))
print("\n%d tests, %d failures, %.1fs" % (n, fails, time.time() - t0))
sys.exit(1 if fails else 0)
