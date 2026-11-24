#!/usr/bin/env python3
# K1 gate (b): dump the DECODE-path logits of the 6 oracle prompts (token by token through ink_forward n=1) plus 32 greedy continuation
# steps of s1_en_capital, so two engine configurations can be compared bit for bit with cmp (e.g. INK_NVFP4R=0 vs INK_NVFP4R=1).
#   INK_NVFP4R=1 INK_PREFILL_MIN=0 /data/engine/.venv/bin/python tests/kernels_gemv/dump_logits_decode.py build-k1/src/capi/libink.so OUTDIR
import ctypes, os, sys, time
import numpy as np
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
lib = ctypes.CDLL(sys.argv[1])
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
lib.ink_load.restype = ctypes.c_void_p
lib.ink_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.ink_last_error.restype = ctypes.c_char_p
lib.ink_n_vocab.argtypes = lib.ink_slot_new.argtypes = [ctypes.c_void_p]
lib.ink_slot_reset.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.ink_forward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
lib.ink_free.argtypes = [ctypes.c_void_p]
gguf = os.environ.get("INK_GGUF", "/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf")
t0 = time.time()
h = lib.ink_load(gguf.encode(), int(os.environ.get("INK_THREADS", "16")), 1, 4096)
assert h, lib.ink_last_error()
nv = lib.ink_n_vocab(h)
print("loaded in %.1f s, n_vocab %d, INK_NVFP4R=%s" % (time.time() - t0, nv, os.environ.get("INK_NVFP4R", "<unset>")), flush=True)
s = lib.ink_slot_new(h)
lg = np.zeros(nv, np.float32)
fp = lg.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
def step(tok):
    t = np.array([tok], np.int32)
    rc = lib.ink_forward(h, s, t.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 1, fp)
    assert rc == 0, rc
names = ["s1_en_capital", "s2_code_add", "s3_en_story", "c1_chat_user", "m1_en_llm", "m2_code_c"]
t1 = time.time()
for nm in names:
    toks = np.fromfile(os.path.join(root, "tools/oracle", nm, "tokens.i32"), "<i4")
    lib.ink_slot_reset(h, s)
    for t in toks: step(int(t))
    lg.tofile(os.path.join(out_dir, nm + ".f32"))
    orc = np.fromfile(os.path.join(root, "tools/oracle", nm, "logits_last.f32"), "<f4")
    cos = float(lg @ orc / np.linalg.norm(lg) / np.linalg.norm(orc))
    print("%-14s ntok %3d argmax %6d (oracle %6d) cos %.6f" % (nm, len(toks), int(lg.argmax()), int(orc.argmax()), cos), flush=True)
# greedy continuation of s1 (decode path), all 32 logits rows
toks = np.fromfile(os.path.join(root, "tools/oracle/s1_en_capital/tokens.i32"), "<i4")
lib.ink_slot_reset(h, s)
for t in toks: step(int(t))
rows = []; gen = []
for i in range(32):
    nxt = int(lg.argmax()); gen.append(nxt)
    step(nxt)
    rows.append(lg.copy())
np.concatenate(rows).tofile(os.path.join(out_dir, "s1_greedy32.f32"))
np.array(gen, np.int32).tofile(os.path.join(out_dir, "s1_greedy32_tokens.i32"))
print("greedy32 tokens:", gen)
print("decode of %d tokens in %.1f s" % (sum(len(np.fromfile(os.path.join(root, "tools/oracle", nm, "tokens.i32"), "<i4")) for nm in names) + len(toks) + 32, time.time() - t1))
lib.ink_free(h)
