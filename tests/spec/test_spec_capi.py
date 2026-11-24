#!/usr/bin/env python3
# ctypes test of the speculative-decoding C ABI (include/ink/ink_abi.h, agent S2): ink_spec_load, ink_spec_step, ink_sample, ink_spec_synced; the call sequence a server follows.
#   INK_CPUS=10,12,14,16,18,20 /data/engine/.venv/bin/python tests/spec/test_spec_capi.py [path/to/libink.so] [--no-drafter] [--n 40]
# Checks: ink_slot_pos is exact after every step (pos += n_fed, fed[0] == anchor), greedy spec stream (fixed T = 8 and auto) == plain stream (mode 0) token for token in exact mode,
# stop id mid-window ends the step with stopped = 1 and the slot holding only the tokens before it, max_emit budget, sampled steps reproducible per seed, ink_sample == numpy
# implementation of server/sampling.py's semantics (same kept set, chi-square of 400 draws), timing counters populated.
import ctypes, os, sys, time
import numpy as np

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
no_drafter = "--no-drafter" in sys.argv
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 40
os.environ.setdefault("INK_SPEC_EXACT", "1")   # verify with the decode attention kernels: bit-exact gate
lib = ctypes.CDLL(args[0] if args else os.path.join(root, "build-s2/src/capi/libink.so"))


class SP(ctypes.Structure):
    _fields_ = [("temperature", ctypes.c_float), ("top_k", ctypes.c_int32), ("top_p", ctypes.c_float), ("min_p", ctypes.c_float)]


class Req(ctypes.Structure):
    _fields_ = [("anchor", ctypes.c_int32), ("mode", ctypes.c_int32), ("sp", SP), ("seed", ctypes.c_uint64), ("stop_ids", ctypes.POINTER(ctypes.c_int32)),
                ("n_stop", ctypes.c_int32), ("max_emit", ctypes.c_int32)]


class Res(ctypes.Structure):
    _fields_ = [("tokens", ctypes.c_int32 * 8), ("n_tokens", ctypes.c_int32), ("stopped", ctypes.c_int32), ("fed", ctypes.c_int32 * 8), ("n_fed", ctypes.c_int32),
                ("T", ctypes.c_int32), ("drafted", ctypes.c_int32), ("accepted", ctypes.c_int32), ("rc", ctypes.c_int32), ("draft_us", ctypes.c_double),
                ("verify_us", ctypes.c_double), ("accept_us", ctypes.c_double), ("total_us", ctypes.c_double), ("next_seed", ctypes.c_uint64)]


P = ctypes.POINTER
lib.ink_load.restype = ctypes.c_void_p
lib.ink_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.ink_last_error.restype = ctypes.c_char_p
lib.ink_version.restype = ctypes.c_char_p
lib.ink_n_vocab.argtypes = lib.ink_slot_new.argtypes = [ctypes.c_void_p]
lib.ink_slot_pos.argtypes = lib.ink_slot_reset.argtypes = lib.ink_slot_free.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.ink_forward.argtypes = [ctypes.c_void_p, ctypes.c_int, P(ctypes.c_int32), ctypes.c_int, P(ctypes.c_float)]
lib.ink_last_timing.argtypes = [ctypes.c_void_p, P(ctypes.c_double), P(ctypes.c_double)]
lib.ink_free.argtypes = [ctypes.c_void_p]
lib.ink_spec_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.ink_spec_free.argtypes = [ctypes.c_void_p]
lib.ink_spec_enable.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.ink_spec_mode.argtypes = [ctypes.c_void_p]
lib.ink_spec_synced.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.ink_spec_step.argtypes = [ctypes.c_void_p, ctypes.c_int, P(Req), P(Res)]
lib.ink_sample.restype = ctypes.c_int32
lib.ink_sample.argtypes = [ctypes.c_void_p, P(ctypes.c_float), P(SP), ctypes.c_uint64, P(ctypes.c_uint64)]

print("version:", lib.ink_version().decode())
t0 = time.time()
h = lib.ink_load(os.environ.get("INK_GGUF", "/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf").encode(), int(os.environ.get("INK_THREADS", "6")), 2, 2048)
assert h, lib.ink_last_error()
print("engine loaded in %.1f s" % (time.time() - t0))
nv = lib.ink_n_vocab(h)
assert nv == 200058, nv
have_drafter = False
if not no_drafter:
    t0 = time.time()
    rc = lib.ink_spec_load(h, os.environ.get("INK_DRAFTER", "/data/inkling-mini/models/dspark-Inkling-2-Mini-Q8_0.gguf").encode())
    print("ink_spec_load rc", rc, "" if rc == 0 else lib.ink_last_error().decode(), "(%.1f s)" % (time.time() - t0))
    have_drafter = rc == 0
assert lib.ink_spec_enable(h, 1) == 0 and lib.ink_spec_mode(h) in (0, 1, 4, 8)
toks = np.fromfile(os.path.join(root, "tests/spec/prompts/chat_code.i32"), "<i4")
p = len(toks)
s = lib.ink_slot_new(h)
logits = np.zeros(nv, np.float32)
fp = logits.ctypes.data_as(P(ctypes.c_float))
greedy = SP(0.0, 0, 1.0, 0.0)


def prefill(slot):
    lib.ink_slot_reset(h, slot)
    rc = lib.ink_forward(h, slot, toks.ctypes.data_as(P(ctypes.c_int32)), p, fp)
    assert rc == 0 and lib.ink_slot_pos(h, slot) == p
    return int(logits.argmax())


def step(slot, anchor, mode, sp=greedy, seed=0, stops=(), max_emit=8):
    rq = Req()
    rq.anchor = anchor
    rq.mode = mode
    rq.sp = sp
    rq.seed = seed
    arr = (ctypes.c_int32 * max(1, len(stops)))(*stops)
    rq.stop_ids = arr
    rq.n_stop = len(stops)
    rq.max_emit = max_emit
    res = Res()
    pos0 = lib.ink_slot_pos(h, slot)
    rc = lib.ink_spec_step(h, slot, ctypes.byref(rq), ctypes.byref(res))
    assert rc == 0 and res.rc == 0, (rc, res.rc, lib.ink_last_error())
    assert 1 <= res.n_tokens <= max_emit and res.n_fed == res.n_tokens
    assert lib.ink_slot_pos(h, slot) == pos0 + res.n_fed, (lib.ink_slot_pos(h, slot), pos0, res.n_fed)   # V's requirement: exact position after every step
    assert res.fed[0] == anchor and list(res.fed[1:res.n_fed]) == list(res.tokens[:res.n_tokens - 1])
    assert res.total_us > 0 and (res.T == 1 or res.verify_us > 0)
    return res


def run(slot, mode, n, **kw):
    anchor = prefill(slot)
    out, steps, acc, dr, hist = [], 0, 0, 0, {}
    while len(out) < n:
        r = step(slot, anchor, mode, **kw)
        out += list(r.tokens[:r.n_tokens])
        anchor = r.tokens[r.n_tokens - 1]
        steps += 1
        acc += r.accepted
        dr += r.drafted
        hist[r.T] = hist.get(r.T, 0) + 1
        if r.stopped:
            break
    return out, steps, acc, dr, hist


# 1. plain stream (mode 0) vs fixed T=8 vs auto, greedy, exact mode: token for token
a0 = prefill(s)
sampled = lib.ink_sample(h, fp, ctypes.byref(greedy), 0, None)
assert sampled == a0, (sampled, a0)
print("synced after prefill:", lib.ink_spec_synced(h, s), "(drafter %s)" % ("loaded" if have_drafter else "absent"))
assert lib.ink_spec_synced(h, s) == (1 if have_drafter else 0)
t0 = time.time()
plain, st_p, _, _, _ = run(s, 0, N)
t_plain = time.time() - t0
print("plain: %d tokens in %d steps, %.2f tok/s" % (len(plain), st_p, len(plain) / t_plain))
for mode, name in ((8, "fixed T=8"), (1, "auto")):
    t0 = time.time()
    out, steps, acc, dr, hist = run(s, mode, N)
    dt = time.time() - t0
    k = min(len(out), len(plain))
    same = out[:k] == plain[:k]
    print("%s: %d tokens in %d steps, acc %d/%d, T hist %s, %.2f tok/s, %s" % (name, len(out), steps, acc, dr, hist, len(out) / dt, "== plain" if same else "DIFFERS from plain"))
    assert same, (out[:k], plain[:k])
    if have_drafter and mode == 8:
        assert dr > 0, "no drafts although the drafter is loaded and synced"

# 2. stop id in the middle of a verify window; the slot holds exactly the tokens before the stop
anchor = prefill(s)
r = step(s, anchor, 8 if have_drafter else 0, stops=(plain[4],))
if have_drafter and r.accepted >= 4:
    assert r.n_tokens == 5 and r.stopped == 1 and list(r.tokens[:5]) == plain[:5], (list(r.tokens[:r.n_tokens]), r.stopped)
    assert lib.ink_slot_pos(h, s) == p + 5
    print("stop mid-window: n_tokens 5, stopped 1, pos p+5 ok (accepted %d/%d)" % (r.accepted, r.drafted))
else:
    print("stop mid-window: step emitted %d tokens (accepted %d/%d); continuing to the stop" % (r.n_tokens, r.accepted, r.drafted))
    out = list(r.tokens[:r.n_tokens])
    while not r.stopped:
        r = step(s, r.tokens[r.n_tokens - 1], 8 if have_drafter else 0, stops=(plain[4],))
        out += list(r.tokens[:r.n_tokens])
    assert out == plain[:5] and lib.ink_slot_pos(h, s) == p + 5, (out, plain[:5])
# the slot continues like plain after the stop (the stop token was never fed): feeding it now must reproduce plain[5:]
r = step(s, plain[4], 0)
assert r.tokens[0] == plain[5]
print("state after stop == plain state (next token matches)")

# 3. max_emit budget
anchor = prefill(s)
r = step(s, anchor, 8 if have_drafter else 0, max_emit=3)
assert r.n_tokens <= 3 and list(r.tokens[:r.n_tokens]) == plain[:r.n_tokens] and lib.ink_slot_pos(h, s) == p + r.n_tokens
r2 = step(s, r.tokens[r.n_tokens - 1], 4 if have_drafter else 0)
assert list(r2.tokens[:r2.n_tokens]) == plain[r.n_tokens:r.n_tokens + r2.n_tokens]
print("max_emit 3: emitted %d then continued correctly" % r.n_tokens)

# 4. sampled mode: reproducible per seed, next_seed chains
sp = SP(0.8, 8, 1.0, 0.0)
res = []
for rep in range(2):
    anchor = prefill(s)
    a = lib.ink_sample(h, fp, ctypes.byref(sp), 123, None)
    r = step(s, a, 4 if have_drafter else 0, sp=sp, seed=777)
    r2 = step(s, r.tokens[r.n_tokens - 1], 4 if have_drafter else 0, sp=sp, seed=r.next_seed)
    res.append((a, list(r.tokens[:r.n_tokens]), r.accepted, list(r2.tokens[:r2.n_tokens])))
assert res[0] == res[1], res
print("sampled steps reproducible per seed:", res[0][1], res[0][3])

# 5. ink_sample vs a numpy implementation of server/sampling.py (temperature -> top_k -> top_p -> min_p)
prefill(s)
x = logits.astype(np.float32)
e = np.exp(np.maximum((x - x.max()) * np.float32(1.0 / 0.8), -80.0).astype(np.float32))
idx = np.argsort(-e, kind="stable")[:8]
probs = e[idx] / e[idx].sum(dtype=np.float64)
cnt = {int(t): 0 for t in idx}
for i in range(400):
    t = lib.ink_sample(h, fp, ctypes.byref(sp), 5000 + i, None)
    assert t in cnt, "ink_sample drew %d outside the top-8 set" % t
    cnt[t] += 1
chi = sum((cnt[int(t)] - 400 * q) ** 2 / (400 * q) for t, q in zip(idx, probs) if 400 * q >= 5)
print("ink_sample 400 draws: all inside the top-8 set, chi-square %.2f" % chi)
assert chi < 30
a = ctypes.c_double()
b = ctypes.c_double()
lib.ink_last_timing(h, ctypes.byref(a), ctypes.byref(b))
print("last timing total_us %.0f kernel_us %.0f" % (a.value, b.value))
lib.ink_spec_free(h)
assert lib.ink_spec_synced(h, s) == 0
anchor = prefill(s)
r = step(s, anchor, 1)
assert r.T == 1 and r.tokens[0] == plain[0]
print("after ink_spec_free: plain steps still work")
lib.ink_free(h)
print("spec C ABI test PASS")
