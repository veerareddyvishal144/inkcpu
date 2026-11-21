"""Backends: the engine behind one narrow interface, always driven from ONE dedicated thread.

    Backend.start()                  load the model (called in the engine thread)
    slot_new() / slot_free(s) / slot_reset(s) / slot_pos(s)
    prefill(slot, tokens) -> logits  append >= 1 tokens, return float32[n_vocab] logits of the LAST token
    step(slot, token)     -> logits  append one token (decode)
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
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

N_VOCAB = 200058


class BackendError(RuntimeError):
    pass


class Backend:
    name = "base"

    def __init__(self, n_slots: int, max_ctx: int, n_vocab: int = N_VOCAB):
        self.n_slots, self.max_ctx, self.n_vocab = n_slots, max_ctx, n_vocab

    def start(self) -> None: ...
    def close(self) -> None: ...
    def slot_new(self) -> int: raise NotImplementedError
    def slot_free(self, slot: int) -> None: raise NotImplementedError
    def slot_reset(self, slot: int) -> None: raise NotImplementedError
    def slot_pos(self, slot: int) -> int: raise NotImplementedError
    def prefill(self, slot: int, tokens: Sequence[int]) -> np.ndarray: raise NotImplementedError
    def step(self, slot: int, token: int) -> np.ndarray: return self.prefill(slot, [token])

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

    def prefill(self, slot: int, tokens: Sequence[int]) -> np.ndarray:
        return self._forward(slot, tokens, True)

    def step(self, slot: int, token: int) -> np.ndarray:
        return self._forward(slot, [token], False)


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

    def prefill(self, slot: int, tokens: Sequence[int]) -> np.ndarray:
        self.hist[slot].extend(int(t) for t in tokens)
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
    """ctypes binding of include/ink/ink_abi.h. All calls (including ink_load) must come from the single engine thread."""
    name = "ink"

    def __init__(self, gguf_path: str, lib_path: Optional[str] = None, n_threads: int = 16, n_slots: int = 2,
                 max_ctx: int = 8192):
        super().__init__(n_slots, max_ctx)
        self.gguf_path, self.n_threads = gguf_path, n_threads
        self.lib_path = lib_path
        self._lib = None
        self._eng = None
        self._buf = None
        self._slots: set = set()

    @staticmethod
    def candidate_paths() -> List[str]:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        c = [os.environ.get("INK_LIB", "")]
        for d in ("build", "build/src/capi", "build/src/engine", "build/lib", "build-release", "build-rel"):
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
        lib.ink_forward.argtypes = [c_void_p, c_int, ctypes.POINTER(ctypes.c_int32), c_int, ctypes.POINTER(ctypes.c_float)]
        lib.ink_last_timing.argtypes = [c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self._lib = lib
        eng = lib.ink_load(self.gguf_path.encode(), self.n_threads, self.n_slots, self.max_ctx)
        if not eng:
            raise BackendError("ink_load failed: %s" % (lib.ink_last_error() or b"").decode(errors="replace"))
        self._eng = ctypes.c_void_p(eng)
        self.n_vocab = int(lib.ink_n_vocab(self._eng))
        self._buf = np.empty(self.n_vocab, dtype=np.float32)

    def info(self) -> Dict[str, object]:
        d = super().info()
        if self._lib is not None:
            d["lib"] = self.lib_path
            d["version"] = (self._lib.ink_version() or b"").decode(errors="replace")
        return d

    def close(self) -> None:
        if self._lib is not None and self._eng is not None:
            self._lib.ink_free(self._eng)
            self._eng = None

    def slot_new(self) -> int:
        return int(self._lib.ink_slot_new(self._eng))

    def slot_free(self, slot: int) -> None:
        self._lib.ink_slot_free(self._eng, slot)

    def slot_reset(self, slot: int) -> None:
        self._lib.ink_slot_reset(self._eng, slot)

    def slot_pos(self, slot: int) -> int:
        return int(self._lib.ink_slot_pos(self._eng, slot))

    def prefill(self, slot: int, tokens: Sequence[int]) -> np.ndarray:
        n = len(tokens)
        arr = (ctypes.c_int32 * n)(*tokens)
        rc = self._lib.ink_forward(self._eng, slot, arr, n, self._buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if rc != 0:
            raise BackendError("ink_forward failed (rc=%d): %s" % (rc, (self._lib.ink_last_error() or b"").decode(errors="replace")))
        return self._buf.copy()

    def timing(self):
        tot, ker = ctypes.c_double(0), ctypes.c_double(0)
        self._lib.ink_last_timing(self._eng, ctypes.byref(tot), ctypes.byref(ker))
        return tot.value, ker.value
