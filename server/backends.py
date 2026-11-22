"""Backends: the engine behind one narrow interface, always driven from ONE dedicated thread.

    Backend.start()                  load the model (called in the engine thread)
    slot_new() / slot_free(s) / slot_reset(s) / slot_pos(s)
    prefill(slot, tokens, want_logits=True) -> logits   append >= 1 tokens, return float32[n_vocab] logits of the LAST token
                                     (None when want_logits is False: the lm head is skipped). The returned array may be a VIEW
                                     of a per-slot buffer: it stays valid until the next forward call on the SAME slot.
    step(slot, token)     -> logits  append one token (decode)
    spec_step(slot, anchor, SpecRequest) -> StepResult   speculative decoding (only when supports_spec): the backend samples
                                     itself and returns >= 1 EMITTED tokens per call (see StepResult)
    close()

MockBackend       pure python, deterministic pseudo-logits (optionally scripted output), records every call.
LlamaServerBackend  HTTP to a llama.cpp server (testing/parity only; the full distribution comes from n_probs).
InkBackend        ctypes over libink.so as declared in include/ink/ink_abi.h (needs the engine library to be built).
"""
from __future__ import annotations

import ctypes
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

N_VOCAB = 200058


class BackendError(RuntimeError):
    pass


class ContextFull(BackendError):
    """The slot cannot take the tokens (engine code -2)."""


@dataclass
class SpecRequest:
    """What a speculative backend needs to sample on its own (agent S2's ink_spec_step request)."""
    temperature: float = 0.0               # 0 = greedy
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    seed: Optional[int] = None
    stop_ids: Sequence[int] = ()           # token ids that end the generation (never fed to the engine)
    max_emit: int = 1                      # emit at most this many tokens in this call
    mode: int = -1                         # ink_spec_request.mode: -1 engine default, 0 plain, 1 auto, 2..8 fixed verify width


@dataclass
class StepResult:
    """Result of one speculative step.
    tokens   the tokens EMITTED by this call (>= 1), in order, new text for the client. The LAST one is the next anchor
             (sampled, not yet processed by the engine) unless it is a stop token.
    fed      the tokens the engine APPENDED to the slot history in this call, in order: the anchor, then the emitted tokens that
             were processed (accepted drafts). Without a stop token that is [anchor] + tokens[:-1]; if a stop token is among
             the emitted tokens the stop token and everything after it are NOT fed (as plain decode never feeds it).
             The server trusts `fed` (and checks it against slot_pos) for the slot history, never the emitted list.
    logits   logits after fed[-1] if the backend has them (plain decode), else None
    drafted / accepted   counters of this step (statistics only)
    """
    tokens: List[int]
    fed: Optional[List[int]] = None        # None: derived with derive_fed() from the anchor and the emitted tokens (ABI without a fed list)
    logits: Optional[np.ndarray] = None
    drafted: int = 0
    accepted: int = 0
    t_used: int = 0                        # verify width used (statistics)
    verify_us: float = 0.0
    draft_us: float = 0.0
    stopped: bool = False                  # the engine saw a stop id among the emitted tokens (tokens[-1])
    next_seed: int = 0                     # sampled mode: seed of the NEXT step (the engine chains it; the scheduler passes it back)


def derive_fed(anchor: int, emitted: Sequence[int], stop_ids: Sequence[int]) -> List[int]:
    """Tokens the engine appended to the slot in one speculative step, from the SPEC_CONTRACT rules: the anchor is always processed;
    every emitted token is processed except the LAST one (the next anchor, not yet processed); if a stop token is among the emitted
    tokens, the engine state ends as if only the tokens BEFORE it had been processed (the stop token and anything after it are
    never fed)."""
    stops = set(int(t) for t in stop_ids)
    for i, t in enumerate(emitted):
        if int(t) in stops:
            return [int(anchor)] + [int(x) for x in emitted[:i]]
    return [int(anchor)] + [int(x) for x in emitted[:-1]]


class Backend:
    name = "base"
    supports_spec = False               # True: spec_step() is implemented (the scheduler may use it for greedy / simple sampling)

    def __init__(self, n_slots: int, max_ctx: int, n_vocab: int = N_VOCAB):
        self.n_slots, self.max_ctx, self.n_vocab = n_slots, max_ctx, n_vocab

    def start(self) -> None: ...
    def close(self) -> None: ...
    def slot_new(self) -> int: raise NotImplementedError
    def slot_free(self, slot: int) -> None: raise NotImplementedError
    def slot_reset(self, slot: int) -> None: raise NotImplementedError
    def slot_pos(self, slot: int) -> int: raise NotImplementedError
    def prefill(self, slot: int, tokens: Sequence[int], want_logits: bool = True) -> Optional[np.ndarray]: raise NotImplementedError
    def step(self, slot: int, token: int) -> np.ndarray: return self.prefill(slot, [token])
    def spec_step(self, slot: int, anchor: int, req: "SpecRequest") -> "StepResult":
        raise NotImplementedError("%s has no speculative decoding" % self.name)
    def sample(self, logits: np.ndarray, req: "SpecRequest", seed: int) -> "tuple[int, int]":
        """First anchor of a speculative request: the backend's own sampler on a logits row -> (token, next_seed), so that the whole
        request is one deterministic chain of (seed -> step -> next_seed) inside the engine."""
        raise NotImplementedError("%s has no sampler" % self.name)
    spec_default_mode = 1               # mode applied to requests that do not name one (ink_spec_mode of the engine); 0 = plain

    def info(self) -> Dict[str, object]:
        return {"backend": self.name, "n_slots": self.n_slots, "max_ctx": self.max_ctx, "n_vocab": self.n_vocab}


class MockBackend(Backend):
    """Deterministic fake model. logits = f(hash of the whole slot history) so identical histories give identical logits.
    `responder(prompt_ids) -> list[int]` scripts the output (the target token gets +50, so greedy and sampled output equal
    the script; a missing tail means the stop token 200006). `delay` simulates compute time per call."""
    name = "mock"
    STOP = 200006

    def __init__(self, n_slots: int = 2, max_ctx: int = 8192, responder: Optional[Callable[[List[int]], List[int]]] = None,
                 delay: float = 0.0, noise_std: float = 3.0, n_vocab: int = N_VOCAB):
        super().__init__(n_slots, max_ctx, n_vocab)
        self.responder, self.delay, self.noise_std = responder, delay, noise_std
        self.hist: Dict[int, List[int]] = {}
        self.prompt_len: Dict[int, int] = {}
        self.hash: Dict[int, int] = {}
        self.free_ids = list(range(n_slots))
        self.calls: List[tuple] = []        # (op, slot, n_tokens)
        self._script_cache: Dict[int, List[int]] = {}

    def slot_new(self) -> int:
        if not self.free_ids:
            return -1
        s = self.free_ids.pop(0)
        self.slot_reset(s)
        self.calls.append(("slot_new", s, 0))
        return s

    def slot_free(self, slot: int) -> None:
        self.calls.append(("slot_free", slot, 0))
        self.hist.pop(slot, None)
        self.free_ids.append(slot)

    def slot_reset(self, slot: int) -> None:
        self.calls.append(("slot_reset", slot, 0))
        self.hist[slot], self.prompt_len[slot], self.hash[slot] = [], 0, 1469598103934665603

    def slot_pos(self, slot: int) -> int:
        return len(self.hist.get(slot, ()))

    def _feed(self, slot: int, tokens: Sequence[int]) -> None:
        h = self.hash[slot]
        for t in tokens:
            h = ((h ^ (int(t) + 0x9E3779B9)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        self.hash[slot] = h
        self.hist[slot].extend(int(t) for t in tokens)

    def _logits(self, slot: int) -> np.ndarray:
        rng = np.random.default_rng(self.hash[slot])
        x = (rng.standard_normal(self.n_vocab) * self.noise_std).astype(np.float32)
        x[self.STOP] -= 6.0  # random stops are rare
        if self.responder is not None:
            hist, pl = self.hist[slot], self.prompt_len[slot]
            out = self._script_cache.get(slot)
            if out is None:
                out = self._script_cache[slot] = list(self.responder(hist[:pl]))
            pos = len(hist) - pl  # generated tokens fed so far
            x[out[pos] if pos < len(out) else self.STOP] += 50.0
        return x

    def _forward(self, slot: int, tokens: Sequence[int], is_prefill: bool) -> np.ndarray:
        if self.delay:
            time.sleep(self.delay)
        if self.slot_pos(slot) + len(tokens) > self.max_ctx:
            raise BackendError("context overflow")
        self.calls.append(("prefill" if is_prefill else "step", slot, len(tokens)))
        self._feed(slot, tokens)
        if is_prefill:  # everything fed by prefill() belongs to the prompt (origin of the scripted response)
            self.prompt_len[slot] = len(self.hist[slot])
            self._script_cache.pop(slot, None)
        return self._logits(slot)

    def prefill(self, slot: int, tokens: Sequence[int], want_logits: bool = True) -> np.ndarray:
        return self._forward(slot, tokens, True)

    def step(self, slot: int, token: int) -> np.ndarray:
        return self._forward(slot, [token], False)


class MockSpecBackend(MockBackend):
    """MockBackend with speculative steps: spec_step() feeds the anchor and emits 1-4 tokens per call (the counts cycle through
    `emit_pattern`, capped by req.max_emit), feeding the emitted tokens it 'accepted' exactly like a verify step would: the last
    emitted token is NOT fed (it is the next anchor); a stop token ends the step and is never fed. `overshoot` > 0 makes the
    mock append that many extra tokens AFTER a stop token to the emitted list (a misbehaving engine: the server must drop them).
    `spec_log` records (slot, anchor, emitted, fed) of every step."""
    name = "mockspec"
    supports_spec = True

    def __init__(self, *a, emit_pattern: Sequence[int] = (1, 3, 2, 4), overshoot: int = 0, **kw):
        super().__init__(*a, **kw)
        self.emit_pattern, self.overshoot = tuple(emit_pattern), int(overshoot)
        self.spec_log: List[tuple] = []
        self._n_steps = 0

    def _pick(self, slot: int, req: SpecRequest) -> int:
        x = self._logits(slot)
        if req.temperature <= 0.0:
            return int(np.argmax(x))
        from .sampling import Sampler, SamplingParams
        seed = (self.hash[slot] ^ (int(req.seed or 0) * 0x9E3779B97F4A7C15)) & 0x7FFFFFFFFFFFFFFF
        return Sampler(SamplingParams(temperature=req.temperature, top_k=req.top_k, top_p=req.top_p, min_p=req.min_p, seed=seed)).sample(x)

    def sample(self, logits: np.ndarray, req: SpecRequest, seed: int) -> "tuple[int, int]":
        if req.temperature <= 0.0:
            return int(np.argmax(logits)), seed
        from .sampling import Sampler, SamplingParams
        tok = Sampler(SamplingParams(temperature=req.temperature, top_k=req.top_k, top_p=req.top_p, min_p=req.min_p,
                                     seed=int(seed) & 0x7FFFFFFFFFFFFFFF)).sample(logits)
        return tok, (int(seed) * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF

    def spec_step(self, slot: int, anchor: int, req: SpecRequest) -> StepResult:
        if self.delay:
            time.sleep(self.delay)
        k = max(1, min(self.emit_pattern[self._n_steps % len(self.emit_pattern)], int(req.max_emit)))
        self._n_steps += 1
        if self.slot_pos(slot) + 1 > self.max_ctx:
            raise ContextFull("context overflow")
        self._feed(slot, [anchor])
        fed, emitted = [int(anchor)], []
        for j in range(k):
            tok = self._pick(slot, req)
            emitted.append(tok)
            if tok in req.stop_ids:
                break
            if j < k - 1:
                if self.slot_pos(slot) + 1 > self.max_ctx:
                    raise ContextFull("context overflow")
                self._feed(slot, [tok])
                fed.append(tok)
        if self.overshoot and emitted[-1] in req.stop_ids:
            emitted += [7] * self.overshoot
        self.calls.append(("spec", slot, len(fed)))
        self.spec_log.append((slot, int(anchor), list(emitted), list(fed)))
        stopped = emitted[-1] in req.stop_ids or (self.overshoot > 0 and any(t in req.stop_ids for t in emitted))
        return StepResult(emitted, fed, None, drafted=k - 1, accepted=len(fed) - 1, t_used=k, stopped=stopped,
                          next_seed=(int(req.seed or 0) * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF)


class LlamaServerBackend(Backend):
    """Test backend over a llama.cpp server (e.g. port 8001): logits are recovered as log-probabilities (n_probs =
    whole vocabulary). Slow: one HTTP call and a full-vocabulary sort per token. The history is resent every call; the
    server's prompt cache keeps it cheap."""
    name = "llama"

    def __init__(self, url: str = "http://127.0.0.1:8001", n_slots: int = 1, max_ctx: int = 4096):
        super().__init__(n_slots, max_ctx)
        self.url = url.rstrip("/")
        self.hist: Dict[int, List[int]] = {}
        self.free_ids = list(range(n_slots))

    def start(self) -> None:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=10) as r:
                if r.status != 200:
                    raise BackendError("llama-server not ready")
        except Exception as e:
            raise BackendError("cannot reach llama-server at %s: %s" % (self.url, e))

    def slot_new(self) -> int:
        if not self.free_ids:
            return -1
        s = self.free_ids.pop(0)
        self.hist[s] = []
        return s

    def slot_free(self, slot: int) -> None:
        self.hist.pop(slot, None)
        self.free_ids.append(slot)

    def slot_reset(self, slot: int) -> None:
        self.hist[slot] = []

    def slot_pos(self, slot: int) -> int:
        return len(self.hist[slot])

    def prefill(self, slot: int, tokens: Sequence[int], want_logits: bool = True) -> Optional[np.ndarray]:
        self.hist[slot].extend(int(t) for t in tokens)
        if not want_logits:
            return None
        body = {"prompt": self.hist[slot], "n_predict": 1, "n_probs": self.n_vocab, "temperature": 0.0, "top_k": 1,
                "cache_prompt": True, "stream": False, "post_sampling_probs": False}
        req = urllib.request.Request(self.url + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read())
        probs = d["completion_probabilities"][0]
        top = probs.get("top_logprobs") or probs.get("top_probs") or []
        x = np.full(self.n_vocab, -1e30, dtype=np.float32)
        for e in top:
            i = e["id"]
            if i < self.n_vocab:
                x[i] = e["logprob"] if "logprob" in e else float(np.log(max(e["prob"], 1e-30)))
        return x


class InkBackend(Backend):
    """ctypes binding of include/ink/ink_abi.h. All calls (including ink_load) must come from the single engine thread.

    CPU partition: the engine pins its pool to the CPUs of env INK_CPUS (comma list, one per thread); pass n_threads = len(list).
    Logits: one float32[n_vocab] buffer per slot; prefill()/step() return a VIEW of the slot's buffer (valid until the next
    forward on that slot), so a decode step copies nothing. Engine return codes: -1 bad slot, -2 context full, -3 bad token."""
    name = "ink"
    _RC = {-1: "bad slot", -2: "context full", -3: "bad token id", -4: "no tokens", -10: "bad arguments", -11: "engine exception"}

    def __init__(self, gguf_path: str, lib_path: Optional[str] = None, n_threads: int = 16, n_slots: int = 2,
                 max_ctx: int = 8192, timer_slack_us: int = 5000, drafter_path: Optional[str] = None, spec_mode: Optional[int] = None):
        super().__init__(n_slots, max_ctx)
        self.gguf_path, self.n_threads = gguf_path, n_threads
        self.lib_path = lib_path
        self.drafter_path = drafter_path            # DSpark drafter GGUF: speculative steps need it AND the ink_spec_* symbols
        self.spec_mode = spec_mode                  # --spec: engine default for requests without `speculation` (None = leave the engine's: env INK_SPEC or auto)
        self.spec_default_mode = 0
        self.supports_spec = False                  # set by start() when the library has the spec ABI and the drafter loaded
        self.spec_info: Dict[str, object] = {}
        self.timer_slack_us = timer_slack_us
        self._lib = None
        self._eng = None
        self._bufs = None
        self._tok1 = np.zeros(1, dtype=np.int32)
        self._pf = ctypes.POINTER(ctypes.c_float)
        self._pi = ctypes.POINTER(ctypes.c_int32)
        self.load_seconds = 0.0

    @staticmethod
    def candidate_paths() -> List[str]:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        c = [os.environ.get("INK_LIB", "")]
        for d in ("build", "build/src/capi", "build/src/engine", "build/lib", "build-release", "build-rel", "build-v/src/capi",
                  "build-orch/src/capi"):
            c.append(os.path.join(root, d, "libink.so"))
        return [p for p in c if p]

    def start(self) -> None:
        paths = [self.lib_path] if self.lib_path else self.candidate_paths()
        lib = None
        for p in paths:
            if p and os.path.exists(p):
                try:
                    lib = ctypes.CDLL(p)
                    self.lib_path = p
                    break
                except OSError as e:
                    raise BackendError("cannot load %s: %s" % (p, e))
        if lib is None:
            raise BackendError(
                "libink.so not found (searched: %s). Build the engine first (cmake -S . -B build && cmake --build build, "
                "target ink / libink.so from src/capi), or set INK_LIB=/path/to/libink.so or pass --lib. "
                "For development use --backend mock." % ", ".join(paths))
        if not os.path.exists(self.gguf_path):
            raise BackendError("GGUF file not found: %s" % self.gguf_path)
        cpus = [c for c in os.environ.get("INK_CPUS", "").split(",") if c.strip()]
        if cpus and len(cpus) != self.n_threads:
            print("WARNING: INK_CPUS lists %d CPUs but --threads is %d: pool threads would share CPUs and the spin barriers "
                  "convoy (4-5 tok/s instead of 13)" % (len(cpus), self.n_threads), flush=True)
        c_int, c_char_p, c_void_p = ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p
        lib.ink_load.restype = c_void_p
        lib.ink_load.argtypes = [c_char_p, c_int, c_int, c_int]
        lib.ink_free.argtypes = [c_void_p]
        lib.ink_last_error.restype = c_char_p
        lib.ink_version.restype = c_char_p
        lib.ink_n_vocab.restype = c_int
        lib.ink_n_vocab.argtypes = [c_void_p]
        lib.ink_slot_new.restype = c_int
        lib.ink_slot_new.argtypes = [c_void_p]
        lib.ink_slot_free.argtypes = [c_void_p, c_int]
        lib.ink_slot_reset.argtypes = [c_void_p, c_int]
        lib.ink_slot_pos.restype = c_int
        lib.ink_slot_pos.argtypes = [c_void_p, c_int]
        lib.ink_forward.restype = c_int
        lib.ink_forward.argtypes = [c_void_p, c_int, self._pi, c_int, self._pf]
        lib.ink_last_timing.argtypes = [c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self._lib = lib
        if self.timer_slack_us > 0:
            # The pool's idle workers poll with 20 us sleeps (ThreadPool, include/ink/threadpool.h): with the kernel's default 50 us
            # timer slack that is ~30% of one core for an IDLE server (measured). The slack of the creating thread is inherited by
            # the pool threads ink_load creates, so setting it here makes the idle poll ~1 wake-up per slack interval; the cost is
            # at most that much extra latency for the first job after the pool went idle (> 2 ms without work).
            try:
                libc = ctypes.CDLL(None, use_errno=True)
                libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
                if libc.prctl(29, self.timer_slack_us * 1000, 0, 0, 0) != 0:   # PR_SET_TIMERSLACK
                    print("WARNING: prctl(PR_SET_TIMERSLACK) failed: errno %d" % ctypes.get_errno(), flush=True)
            except Exception as e:  # noqa: BLE001
                print("WARNING: cannot set the timer slack: %s" % e, flush=True)
        t0 = time.monotonic()
        eng = lib.ink_load(self.gguf_path.encode(), self.n_threads, self.n_slots, self.max_ctx)
        if not eng:
            raise BackendError("ink_load failed: %s" % (lib.ink_last_error() or b"").decode(errors="replace"))
        self.load_seconds = time.monotonic() - t0
        self._eng = ctypes.c_void_p(eng)
        self.n_vocab = int(lib.ink_n_vocab(self._eng))
        self._bufs = np.zeros((self.n_slots, self.n_vocab), dtype=np.float32)
        self._setup_spec(lib)

    def info(self) -> Dict[str, object]:
        d = super().info()
        if self._lib is not None:
            d["lib"] = self.lib_path
            d["version"] = (self._lib.ink_version() or b"").decode(errors="replace")
            d["load_seconds"] = round(self.load_seconds, 2)
        d["spec"] = dict(self.spec_info, supported=self.supports_spec, default_mode=self.spec_default_mode)
        return d

    def close(self) -> None:
        if self._lib is not None and self._eng is not None:
            self._lib.ink_free(self._eng)
            self._eng = None

    def slot_new(self) -> int:
        s = int(self._lib.ink_slot_new(self._eng))
        if s >= self.n_slots:
            raise BackendError("engine returned slot %d but only %d slots were requested" % (s, self.n_slots))
        return s

    def slot_free(self, slot: int) -> None:
        self._lib.ink_slot_free(self._eng, slot)

    def slot_reset(self, slot: int) -> None:
        self._lib.ink_slot_reset(self._eng, slot)

    def slot_pos(self, slot: int) -> int:
        return int(self._lib.ink_slot_pos(self._eng, slot))

    def _forward(self, slot: int, arr: np.ndarray, want_logits: bool) -> Optional[np.ndarray]:
        out = self._bufs[slot]
        rc = self._lib.ink_forward(self._eng, slot, arr.ctypes.data_as(self._pi), int(arr.size),
                                   out.ctypes.data_as(self._pf) if want_logits else None)
        if rc != 0:
            msg = "ink_forward failed (rc=%d %s): %s" % (rc, self._RC.get(rc, "?"), (self._lib.ink_last_error() or b"").decode(errors="replace"))
            raise (ContextFull(msg) if rc == -2 else BackendError(msg))
        return out if want_logits else None

    def prefill(self, slot: int, tokens: Sequence[int], want_logits: bool = True) -> Optional[np.ndarray]:
        arr = np.ascontiguousarray(np.asarray(tokens, dtype=np.int32))
        return self._forward(slot, arr, want_logits)

    def step(self, slot: int, token: int) -> np.ndarray:
        self._tok1[0] = token
        return self._forward(slot, self._tok1, True)

    def timing(self):
        tot, ker = ctypes.c_double(0), ctypes.c_double(0)
        self._lib.ink_last_timing(self._eng, ctypes.byref(tot), ctypes.byref(ker))
        return tot.value, ker.value

    # ------------------------------------------------------------------ speculative decoding (agent S2's C ABI, include/ink/ink_abi.h)
    # Structs copied field for field from ink_abi.h (ink_sampling_params nested in ink_spec_request). The server probes the symbols at
    # start-up (dlsym); a library without them keeps supports_spec False and nothing else changes. Call sequence (ink_abi.h): ink_load ->
    # ink_spec_load BEFORE any slot is filled (installs the tap hook) -> per request: prefill via ink_forward, first anchor = ink_sample(logits),
    # then ink_spec_step(anchor) until the stop / the budget; the engine applies stop ids and max_emit; slot_pos advances by n_fed per step.
    SPEC_SYMBOLS = ("ink_spec_load", "ink_spec_free", "ink_spec_enable", "ink_spec_mode", "ink_spec_synced", "ink_spec_step", "ink_sample")
    SPEC_MODES = {"off": 0, "plain": 0, "none": 0, "auto": 1}

    class SamplingParamsC(ctypes.Structure):          # ink_sampling_params
        _fields_ = [("temperature", ctypes.c_float), ("top_k", ctypes.c_int32), ("top_p", ctypes.c_float), ("min_p", ctypes.c_float)]

    class SpecRequestC(ctypes.Structure):             # ink_spec_request
        pass

    class SpecResultC(ctypes.Structure):              # ink_spec_result
        _fields_ = [("tokens", ctypes.c_int32 * 8), ("n_tokens", ctypes.c_int32), ("stopped", ctypes.c_int32), ("fed", ctypes.c_int32 * 8),
                    ("n_fed", ctypes.c_int32), ("T", ctypes.c_int32), ("drafted", ctypes.c_int32), ("accepted", ctypes.c_int32),
                    ("rc", ctypes.c_int32), ("draft_us", ctypes.c_double), ("verify_us", ctypes.c_double), ("accept_us", ctypes.c_double),
                    ("total_us", ctypes.c_double), ("next_seed", ctypes.c_uint64)]

    @staticmethod
    def parse_spec_mode(v) -> Optional[int]:
        """`speculation` request field / --spec flag -> ink mode: None (absent: engine default), 0 off, 1 auto, 2..8 fixed verify width.
        Raises ValueError for anything else."""
        if v is None:
            return None
        if isinstance(v, bool):
            return 1 if v else 0
        if isinstance(v, str):
            k = v.strip().lower()
            if k in InkBackend.SPEC_MODES:
                return InkBackend.SPEC_MODES[k]
            if k.isdigit():
                v = int(k)
            else:
                raise ValueError("speculation must be auto, off or an integer 2..8 (verify width)")
        if isinstance(v, int) and (v == 0 or v == 1 or 2 <= v <= 8):
            return int(v)
        raise ValueError("speculation must be auto, off or an integer 2..8 (verify width)")

    @classmethod
    def _bind_spec(cls, lib) -> Optional[str]:
        """Bind the ink_spec_* / ink_sample symbols if the library exports them all. Returns None when usable, else the reason
        (the server then runs without speculation)."""
        for name in cls.SPEC_SYMBOLS:
            if not hasattr(lib, name):
                return "library has no %s" % name
        c_int, c_void_p, c_char_p = ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p
        lib.ink_spec_load.restype = c_int
        lib.ink_spec_load.argtypes = [c_void_p, c_char_p]
        lib.ink_spec_free.restype = None
        lib.ink_spec_free.argtypes = [c_void_p]
        lib.ink_spec_enable.restype = c_int
        lib.ink_spec_enable.argtypes = [c_void_p, c_int]
        lib.ink_spec_mode.restype = c_int
        lib.ink_spec_mode.argtypes = [c_void_p]
        lib.ink_spec_synced.restype = c_int
        lib.ink_spec_synced.argtypes = [c_void_p, c_int]
        lib.ink_spec_step.restype = c_int
        lib.ink_spec_step.argtypes = [c_void_p, c_int, ctypes.POINTER(cls.SpecRequestC), ctypes.POINTER(cls.SpecResultC)]
        lib.ink_sample.restype = ctypes.c_int32
        lib.ink_sample.argtypes = [c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(cls.SamplingParamsC), ctypes.c_uint64,
                                   ctypes.POINTER(ctypes.c_uint64)]
        return None

    def _setup_spec(self, lib) -> None:
        why = self._bind_spec(lib)
        if why is not None:
            self.spec_info = {"reason": why}
            if self.drafter_path:
                print("WARNING: --drafter given but %s: running without speculative decoding" % why, flush=True)
            return
        if not self.drafter_path:
            self.spec_info = {"reason": "no --drafter given"}
            return
        t0 = time.monotonic()
        rc = int(lib.ink_spec_load(self._eng, self.drafter_path.encode()))
        if rc != 0:
            self.spec_info = {"reason": "ink_spec_load failed (rc=%d): %s" % (rc, (lib.ink_last_error() or b"").decode(errors="replace"))}
            print("WARNING: %s; running without speculative decoding" % self.spec_info["reason"], flush=True)
            return
        if self.spec_mode is not None and int(lib.ink_spec_enable(self._eng, int(self.spec_mode))) != 0:
            print("WARNING: ink_spec_enable(%d) failed: %s" % (self.spec_mode, (lib.ink_last_error() or b"").decode(errors="replace")), flush=True)
        self.spec_default_mode = int(lib.ink_spec_mode(self._eng))
        self.spec_info = {"drafter": self.drafter_path, "load_seconds": round(time.monotonic() - t0, 2),
                          "exact_verify": os.environ.get("INK_SPEC_EXACT", "0")}
        self.supports_spec = True

    def spec_synced(self, slot: int) -> bool:
        """ink_spec_synced: the drafter context equals the slot (speculation possible on it). Engine thread only."""
        return bool(self.supports_spec and self._lib.ink_spec_synced(self._eng, slot))

    @staticmethod
    def _sp_of(req: SpecRequest) -> "InkBackend.SamplingParamsC":
        return InkBackend.SamplingParamsC(float(req.temperature or 0.0), int(req.top_k or 0),
                                          float(req.top_p if req.top_p is not None else 1.0), float(req.min_p or 0.0))

    def sample(self, logits: np.ndarray, req: SpecRequest, seed: int) -> "tuple[int, int]":
        sp = self._sp_of(req)
        nxt = ctypes.c_uint64(0)
        tok = int(self._lib.ink_sample(self._eng, logits.ctypes.data_as(self._pf), ctypes.byref(sp), int(seed) & 0xFFFFFFFFFFFFFFFF, ctypes.byref(nxt)))
        if tok < 0:
            raise BackendError("ink_sample failed (%d): %s" % (tok, (self._lib.ink_last_error() or b"").decode(errors="replace")))
        return tok, int(nxt.value)

    def spec_step(self, slot: int, anchor: int, req: SpecRequest) -> StepResult:
        stops = np.ascontiguousarray(np.asarray(list(req.stop_ids) or [0], dtype=np.int32))
        rq = self.SpecRequestC()
        rq.anchor, rq.mode = int(anchor), int(req.mode)
        rq.sp = self._sp_of(req)
        rq.seed = int(req.seed or 0) & 0xFFFFFFFFFFFFFFFF
        rq.stop_ids, rq.n_stop = stops.ctypes.data_as(self._pi), len(req.stop_ids)
        rq.max_emit = max(1, min(8, int(req.max_emit)))
        res = self.SpecResultC()
        rc = int(self._lib.ink_spec_step(self._eng, slot, ctypes.byref(rq), ctypes.byref(res)))
        if rc != 0:
            msg = "ink_spec_step failed (rc=%d %s): %s" % (rc, self._RC.get(rc, "?"), (self._lib.ink_last_error() or b"").decode(errors="replace"))
            raise (ContextFull(msg) if rc == -2 else BackendError(msg))
        return self._result_to_step(res, int(anchor))

    @staticmethod
    def _result_to_step(res, anchor: int) -> StepResult:
        """ink_spec_result -> StepResult: the emitted tokens, `fed` straight from the engine (fed[0] must be the anchor)."""
        n, nf = int(res.n_tokens), int(res.n_fed)
        if n < 1 or n > 8:
            raise BackendError("ink_spec_step returned %d tokens" % n)
        if nf < 1 or nf > 8 or nf > n:
            raise BackendError("ink_spec_step: n_fed %d inconsistent with %d emitted tokens" % (nf, n))
        toks = [int(res.tokens[i]) for i in range(n)]
        fed = [int(res.fed[i]) for i in range(nf)]
        if fed[0] != anchor or fed[1:] != toks[:nf - 1]:
            raise BackendError("ink_spec_step: fed %s is not anchor %d + emitted %s" % (fed, anchor, toks))
        return StepResult(toks, fed, None, drafted=int(res.drafted), accepted=int(res.accepted), t_used=int(res.T),
                          verify_us=float(res.verify_us), draft_us=float(res.draft_us), stopped=bool(res.stopped), next_seed=int(res.next_seed))


InkBackend.SpecRequestC._fields_ = [("anchor", ctypes.c_int32), ("mode", ctypes.c_int32), ("sp", InkBackend.SamplingParamsC), ("seed", ctypes.c_uint64),
                                    ("stop_ids", ctypes.POINTER(ctypes.c_int32)), ("n_stop", ctypes.c_int32), ("max_emit", ctypes.c_int32)]
