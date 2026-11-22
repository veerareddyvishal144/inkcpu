#!/usr/bin/env python3
# ctypes smoke test of libink.so (include/ink/ink_abi.h): load, slots, prefill via ink_forward n>1, decode, slot semantics.
#   /data/engine/.venv/bin/python tests/engine/test_capi.py [path/to/libink.so]   (env: INK_GGUF model path, INK_THREADS, INK_CPUS)
import ctypes, sys, os, time
import numpy as np
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "build-c1b/capi/libink.so"))
lib.ink_load.restype = ctypes.c_void_p
lib.ink_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.ink_last_error.restype = ctypes.c_char_p
lib.ink_version.restype = ctypes.c_char_p
for f in ("ink_n_vocab", "ink_slot_new"):
    getattr(lib, f).argtypes = [ctypes.c_void_p]
lib.ink_slot_pos.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.ink_slot_free.argtypes = lib.ink_slot_reset.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.ink_forward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
lib.ink_last_timing.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
lib.ink_free.argtypes = [ctypes.c_void_p]
print("version:", lib.ink_version().decode())
t0 = time.time()
gguf = os.environ.get("INK_GGUF", "/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf")   # S4: either GGUF layout
h = lib.ink_load(gguf.encode(), int(os.environ.get("INK_THREADS", "12")), 2, 2048)
assert h, lib.ink_last_error()
print("loaded in %.1f s" % (time.time() - t0))
nv = lib.ink_n_vocab(h)
assert nv == 200058, nv
toks = np.fromfile(os.path.join(root, "tools/oracle/s1_en_capital/tokens.i32"), "<i4")
orc = np.fromfile(os.path.join(root, "tools/oracle/s1_en_capital/logits_last.f32"), "<f4")
s = lib.ink_slot_new(h); s2 = lib.ink_slot_new(h)
assert s == 0 and s2 == 1 and lib.ink_slot_new(h) == -1
out = np.zeros(nv, np.float32)
fp = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
rc = lib.ink_forward(h, s, toks.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), len(toks), fp)
assert rc == 0, rc
assert lib.ink_slot_pos(h, s) == len(toks)
cos = float(out @ orc / np.linalg.norm(out) / np.linalg.norm(orc))
print("prefill 13 tokens: argmax", int(out.argmax()), "oracle", int(orc.argmax()), "cosine %.6f" % cos)
assert out.argmax() == orc.argmax() and cos > 0.999
a = ctypes.c_double(); b = ctypes.c_double()
lib.ink_last_timing(h, ctypes.byref(a), ctypes.byref(b)); print("last timing total_us %.0f kernel_us %.0f" % (a.value, b.value))
# decode one more token (greedy) and check pos, then reset
nxt = np.array([int(out.argmax())], np.int32)
rc = lib.ink_forward(h, s, nxt.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1, fp)
assert rc == 0 and lib.ink_slot_pos(h, s) == len(toks) + 1
lib.ink_slot_reset(h, s); assert lib.ink_slot_pos(h, s) == 0
assert lib.ink_forward(h, 5, nxt.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1, fp) != 0   # bad slot
lib.ink_slot_free(h, s); lib.ink_slot_free(h, s2)
assert lib.ink_slot_new(h) == 0
lib.ink_free(h)
print("C ABI smoke test PASS")
