#!/usr/bin/env python3
# ctypes test of the slot checkpoint / rewind C ABI (agent S3, docs/REWIND.md): ink_slot_checkpoint, ink_slot_checkpoint_deep, ink_slot_rewind,
# ink_slot_checkpoints. One engine process (17 GB); threads = len(INK_CPUS) or 3.
#   INK_CPUS=26,28,30 /data/engine/.venv/bin/python tests/engine/test_rewind_capi.py [path/to/libink.so]
import ctypes, os, sys, time
import numpy as np

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("INK_PF_CHUNK", "512")   # production prefill chunk: checkpoints every 512 tokens
lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "build-s3/src/capi/libink.so"))
c_int, c_void_p, c_int64 = ctypes.c_int, ctypes.c_void_p, ctypes.c_int64
p32, pf = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)
lib.ink_load.restype = c_void_p
lib.ink_load.argtypes = [ctypes.c_char_p, c_int, c_int, c_int]
lib.ink_last_error.restype = ctypes.c_char_p
lib.ink_n_vocab.argtypes = lib.ink_slot_new.argtypes = [c_void_p]
lib.ink_slot_pos.argtypes = lib.ink_slot_reset.argtypes = lib.ink_slot_free.argtypes = [c_void_p, c_int]
lib.ink_forward.argtypes = [c_void_p, c_int, p32, c_int, pf]
lib.ink_free.argtypes = [c_void_p]
lib.ink_slot_checkpoint.argtypes = lib.ink_slot_checkpoint_deep.argtypes = [c_void_p, c_int]
lib.ink_slot_rewind.restype = c_int64
lib.ink_slot_rewind.argtypes = [c_void_p, c_int, c_int64]
lib.ink_slot_checkpoints.argtypes = [c_void_p, c_int, ctypes.POINTER(c_int64), c_int]

n_threads = len([c for c in os.environ.get("INK_CPUS", "").split(",") if c]) or 3
t0 = time.time()
h = lib.ink_load(os.environ.get("INK_GGUF", "/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf").encode(), n_threads, 2, 4096)
assert h, lib.ink_last_error()
print("loaded in %.1f s, %d threads" % (time.time() - t0, n_threads))
nv = lib.ink_n_vocab(h)
W = np.fromfile(os.path.join(root, "build-c1b/data/ring2/tokens.i32"), "<i4")
assert len(W) >= 2100, len(W)


def fwd(slot, toks, logits=True):
    arr = np.ascontiguousarray(np.asarray(toks, np.int32))
    out = np.zeros(nv, np.float32)
    rc = lib.ink_forward(h, slot, arr.ctypes.data_as(p32), len(arr), out.ctypes.data_as(pf) if logits else None)
    assert rc == 0, rc
    return out


def ckpts(slot):
    buf = (c_int64 * 64)()
    n = lib.ink_slot_checkpoints(h, slot, buf, 64)
    return [int(x) for x in buf[:min(n, 64)]]


s0, s1 = lib.ink_slot_new(h), lib.ink_slot_new(h)
t = time.time()
fwd(s0, W[:1300])
print("prefill 1300: %.1f s" % (time.time() - t))
assert ckpts(s0) == [512, 1024, 1300], ckpts(s0)                      # chunk ends + prompt end
for i in range(1300, 1320):                                            # decode steps add no checkpoints
    fwd(s0, [W[i]])
assert lib.ink_slot_pos(h, s0) == 1320 and ckpts(s0) == [512, 1024, 1300], ckpts(s0)
assert lib.ink_slot_rewind(h, s0, 10) == -1 and lib.ink_slot_pos(h, s0) == 1320          # nothing restorable <= 10: refused, unchanged
assert lib.ink_slot_rewind(h, 5, 100) == -1 and lib.ink_slot_checkpoint(h, 5) == -1 and lib.ink_slot_checkpoints(h, 5, (c_int64 * 4)(), 4) == 0
assert lib.ink_slot_rewind(h, s0, 1300) == 1300 and lib.ink_slot_pos(h, s0) == 1300
tail = W[2000:2040]
a = fwd(s0, tail)
fwd(s1, W[:1300])
b = fwd(s1, tail)
assert np.array_equal(a, b), "rewind + tail differs from a fresh slot (max abs %.3e)" % float(np.abs(a - b).max())
print("rewind(1300) + 40-token tail: logits bit-identical to a fresh slot; argmax", int(a.argmax()))
# ten decode steps after the rewind also agree
for i in range(2100, 2110):
    assert np.array_equal(fwd(s0, [W[i]]), fwd(s1, [W[i]])), i
print("10 decode steps after the rewind: bit-identical")
assert ckpts(s0) == [512, 1024, 1300, 1340], ckpts(s0)                 # the tail's prompt end was added, nothing above 1300 survived the rewind
assert lib.ink_slot_checkpoint_deep(h, s0) == 0 and lib.ink_slot_checkpoint(h, s0) == 0 and ckpts(s0) == [512, 1024, 1300, 1340, 1350], ckpts(s0)
assert lib.ink_slot_rewind(h, s0, 1349) == 1340 and lib.ink_slot_pos(h, s0) == 1340 and ckpts(s0) == [512, 1024, 1300, 1340]
assert lib.ink_slot_rewind(h, s0, 1340) == 1340                        # rewind to the live position is a no-op
lib.ink_slot_reset(h, s0)
assert lib.ink_slot_pos(h, s0) == 0 and ckpts(s0) == []               # reset drops every checkpoint
lib.ink_free(h)
print("rewind C ABI test PASS")
