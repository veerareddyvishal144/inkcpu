"""Engine thread, slot policy (append-only prefix reuse) and request scheduling.

SLOT POLICY (append-only prefix reuse)
  The engine can only append tokens to a slot (ink_forward) or reset it completely (ink_slot_reset); it cannot rewind.
  So for every request we look at the idle slots and use the one whose FULL token history is a prefix of the new prompt
  (longest history wins, then most recently used): only the tokens after that prefix are prefilled. If no slot qualifies
  the request takes an empty slot, else the least recently used idle slot, which is reset (its cache is lost).
  A prompt identical to a slot's history reuses the logits cached from the slot's last forward call (nothing is prefilled).
  The slot history is exactly the list of tokens fed to the engine: the prompt, then every generated token except the last
  sampled one (the stop token / the token that hit max_tokens is never fed). A chat turn that sends the previous assistant
  message back therefore continues the slot, provided the re-rendered prompt starts with the old history token for token.
  Caveat: the model's thinking is part of the history; a client that drops `reasoning_content` from the assistant message
  changes the rendered prompt and gets a reset (prefill from scratch). Clients that send `reasoning_content` back hit the cache.

SPECULATIVE STEPS (backends with supports_spec, e.g. MockSpecBackend today, agent S2's ink_spec_* later)
  The first token after the prefill is sampled here from the logits (it becomes the ANCHOR: emitted to the client, not yet
  processed by the engine). Every following scheduler step hands the anchor to backend.spec_step(), which samples inside the
  engine and returns >= 1 emitted tokens plus `fed`, the tokens the engine really appended to the slot. The last emitted token is
  the next anchor. Tokens after a stop token / a stop string / the max_tokens limit are dropped; the slot history is `fed`
  (engine truth, cross-checked with slot_pos), never the emitted list. Requests with penalties / logit_bias use the plain path, and so
  does any request whose effective mode is 0 (`speculation: "off"`, or the engine default is off and the request names none).
  Sampled requests: the first anchor comes from backend.sample(logits, seed) and every step returns `next_seed` for the following
  one, so a request is one deterministic chain inside the engine (reproducible per seed).

THREADING
  ONE engine thread owns the backend (load, prefill, step, slot calls; the C ABI is single-thread). It round-robins between
  active sequences, one work unit at a time (a prefill chunk of <= prefill_chunk tokens, or one decode step), samples with
  numpy, runs the incremental parser and the stop-string filter, and hands results to each request's `sink` callable
  (the HTTP layer wraps loop.call_soon_threadsafe). Requests wait in a bounded FIFO; when it is full submit() raises
  QueueFull(retry_after) which the HTTP layer turns into 429 + Retry-After.
"""
from __future__ import annotations

import collections
import math
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from .backends import Backend, BackendError, SpecRequest, derive_fed
from .parser import Event, ParsedMessage, StopMatcher, StreamParser
from .sampling import Sampler, SamplingParams
from .tokenizer import InkTokenizer


class QueueFull(Exception):
    def __init__(self, retry_after: int):
        super().__init__("request queue is full")
        self.retry_after = retry_after


@dataclass
class SlotInfo:
    id: int
    tokens: List[int] = field(default_factory=list)       # exactly what was fed to the engine
    last_logits: Optional[np.ndarray] = None               # logits after tokens[-1]
    busy: bool = False
    last_used: float = 0.0


class SlotTable:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.slots: Dict[int, SlotInfo] = {}

    def init(self) -> None:
        for _ in range(self.backend.n_slots):
            sid = self.backend.slot_new()
            if sid < 0:
                break
            self.slots[sid] = SlotInfo(sid)

    def acquire(self, prompt: List[int]) -> Optional[Tuple[SlotInfo, int]]:
        """Pick a slot for `prompt`; returns (slot, n_reused_tokens) or None when every slot is busy."""
        best: Optional[SlotInfo] = None
        n_prompt = len(prompt)
        for s in self.slots.values():
            if s.busy or not s.tokens:
                continue
            n = len(s.tokens)
            if n > n_prompt or (n == n_prompt and s.last_logits is None):
                continue
            if prompt[:n] != s.tokens:
                continue
            if best is None or n > len(best.tokens) or (n == len(best.tokens) and s.last_used > best.last_used):
                best = s
        if best is not None:
            best.busy = True
            return best, len(best.tokens)
        idle = [s for s in self.slots.values() if not s.busy]
        if not idle:
            return None
        s = min(idle, key=lambda x: (len(x.tokens) > 0, x.last_used))  # empty slots first, then least recently used
        self.reset(s)
        s.busy = True
        return s, 0

    def reset(self, s: SlotInfo) -> None:
        if s.tokens or s.last_logits is not None:
            self.backend.slot_reset(s.id)
        s.tokens, s.last_logits = [], None

    def release(self, s: SlotInfo, healthy: bool = True) -> None:
        if not healthy:
            self.backend.slot_reset(s.id)
            s.tokens, s.last_logits = [], None
        s.busy = False
        s.last_used = time.monotonic()


@dataclass
class GenRequest:
    id: str
    prompt_ids: List[int]
    sampling: SamplingParams
    max_tokens: int
    sink: Callable[[tuple], None]
    mode: str = "chat"                                  # chat (parse structure) | raw (plain text)
    stop: List[str] = field(default_factory=list)       # stop strings (content channel in chat mode)
    parser_prefix_ids: List[int] = field(default_factory=list)  # forced tokens already inside prompt_ids, fed to the parser
    skip_special: bool = True                            # raw mode: hide special tokens in the text
    stop_after_tool: bool = False                        # forced tool_choice: the turn ends when the forced call is closed
    spec_mode: Optional[int] = None                      # `speculation` request field: None = server default, 0 off, 1 auto, 2..8 fixed width
    cancel: threading.Event = field(default_factory=threading.Event)
    created: float = field(default_factory=time.monotonic)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Finish:
    finish_reason: str
    message: ParsedMessage
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    queue_s: float
    prefill_s: float
    decode_s: float
    total_s: float
    ttft_s: float
    errors: List[str] = field(default_factory=list)
    decode_engine_s: float = 0.0        # time inside backend calls during the decode phase
    sample_s: float = 0.0               # time in the numpy sampler
    token_ids: List[int] = field(default_factory=list)   # generated tokens that were emitted (without the stop token)
    spec_steps: int = 0                 # speculative steps (0 = plain decode)
    spec_drafted: int = 0
    spec_accepted: int = 0
    spec_emitted: int = 0               # tokens emitted by speculative steps (the first anchor is sampled outside a step)
    spec_mode: Optional[int] = None     # mode passed to the engine (-1 = engine default) when the speculative path was used
    spec_synced: Optional[bool] = None  # ink_spec_synced of the slot at the first speculative step (None: plain path)
    spec_t_hist: Dict[int, int] = field(default_factory=dict)   # verify width -> number of steps


class Sequence:
    """One running generation (engine thread only)."""

    def __init__(self, sched: "Scheduler", req: GenRequest, slot: SlotInfo, n_reused: int):
        self.sched, self.req, self.slot = sched, req, slot
        self.n_reused = n_reused
        self.pos = n_reused
        self.logits: Optional[np.ndarray] = slot.last_logits if n_reused == len(req.prompt_ids) else None
        self.phase = "decode" if self.logits is not None else "prefill"
        tok = sched.tokenizer
        self.sampler = Sampler(req.sampling, req.prompt_ids)
        self.parser = StreamParser(tok) if req.mode == "chat" else None
        self.raw_dec = tok.stream_decoder(req.skip_special) if req.mode == "raw" else None
        self.stopm = StopMatcher(req.stop)
        self.n_completion = 0
        self.reasoning_tokens = 0
        self.finish_reason: Optional[str] = None
        self.error: Optional[str] = None
        self.t_start = time.monotonic()
        self.t_prefill = 0.0
        self.t_decode = 0.0
        self.t_first: Optional[float] = None
        self.t_sample = 0.0
        self.t_engine_dec = 0.0
        self.token_ids: List[int] = []
        self.anchor: Optional[int] = None                 # emitted but not yet processed by the engine (speculative path)
        be = sched.backend
        eff = req.spec_mode if req.spec_mode is not None else getattr(be, "spec_default_mode", 1)
        self.spec_mode = req.spec_mode if req.spec_mode is not None else -1       # -1: let the engine apply its default
        self.use_spec = bool(sched.spec_enabled and be.supports_spec and not req.sampling.has_penalties and eff != 0)
        self.spec_steps = self.spec_drafted = self.spec_accepted = self.spec_emitted = 0
        self.spec_synced: Optional[bool] = None
        self.spec_t_hist: Dict[int, int] = {}
        seed = req.sampling.seed
        self.spec_seed = (int(seed) if seed is not None else int.from_bytes(os.urandom(8), "little")) & 0xFFFFFFFFFFFFFFFF
        self._content: List[str] = []
        self._reasoning: List[str] = []
        if self.parser is not None and req.parser_prefix_ids:  # forced tokens (e.g. tool_choice) are part of the output
            for t in req.parser_prefix_ids:
                self._emit(self.parser.feed_token(t))

    # ---------------------------------------------------------------- one work unit; True when finished
    def step(self) -> bool:
        if self.req.cancel.is_set():
            self.finish_reason = "cancelled"
            return True
        be = self.sched.backend
        if self.phase == "prefill":
            t0 = time.monotonic()
            chunk = self.req.prompt_ids[self.pos:self.pos + self.sched.prefill_chunk]
            last = self.pos + len(chunk) >= len(self.req.prompt_ids)
            logits = be.prefill(self.slot.id, chunk, want_logits=last)   # only the last chunk needs the lm head
            self.slot.tokens.extend(chunk)
            self.slot.last_logits = logits
            self.pos += len(chunk)
            self.t_prefill += time.monotonic() - t0
            if last:
                self.phase, self.logits = "decode", logits
            return False
        t0 = time.monotonic()
        try:
            if self.anchor is None:                       # a token is sampled from the logits (plain decode / first token)
                ts = time.perf_counter()
                if self.use_spec:                         # the engine's sampler (ink_sample): first link of the request's seed chain
                    tok, self.spec_seed = be.sample(self.logits, self._spec_request(), self.spec_seed)
                else:
                    tok = self.sampler.sample(self.logits)
                self.sampler.accept(tok)
                self.t_sample += time.perf_counter() - ts
                if self.t_first is None:
                    self.t_first = time.monotonic()
                if self._on_token(tok):
                    return True
                if self.use_spec:
                    self.anchor = tok                     # the next step processes it together with the drafts
                else:
                    te = time.perf_counter()
                    self.logits = be.step(self.slot.id, tok)
                    self.t_engine_dec += time.perf_counter() - te
                    self.slot.tokens.append(tok)
                    self.slot.last_logits = self.logits
                return False
            return self._spec_step(be)
        finally:
            self.t_decode += time.monotonic() - t0

    def _spec_request(self) -> SpecRequest:
        sp = self.req.sampling
        remaining = self.req.max_tokens - self.n_completion          # >= 1: _on_token ends the request at max_tokens
        return SpecRequest(temperature=sp.temperature, top_k=sp.top_k, top_p=sp.top_p, min_p=sp.min_p, seed=self.spec_seed,
                           stop_ids=tuple(self.sched.tokenizer.stop_ids), max_emit=max(1, min(8, remaining)), mode=self.spec_mode)

    def _spec_step(self, be: Backend) -> bool:
        slot = self.slot
        sreq = self._spec_request()
        if self.spec_synced is None and hasattr(be, "spec_synced"):
            self.spec_synced = bool(be.spec_synced(slot.id))         # diagnostics: an unsynced slot runs plain steps until it is reset
        te = time.perf_counter()
        res = be.spec_step(slot.id, self.anchor, sreq)
        self.t_engine_dec += time.perf_counter() - te
        self.spec_seed = int(res.next_seed) & 0xFFFFFFFFFFFFFFFF
        fed = res.fed if res.fed is not None else derive_fed(self.anchor, res.tokens, sreq.stop_ids)
        slot.tokens.extend(fed)                                       # the slot history is what the engine really processed
        slot.last_logits = res.logits
        if be.slot_pos(slot.id) != len(slot.tokens):
            raise BackendError("slot %d history out of sync: engine holds %d tokens, server expects %d"
                               % (slot.id, be.slot_pos(slot.id), len(slot.tokens)))
        if not res.tokens:
            raise BackendError("speculative step returned no token")
        self.spec_steps += 1
        self.spec_drafted += res.drafted
        self.spec_accepted += res.accepted
        self.spec_emitted += len(res.tokens)
        self.spec_t_hist[res.t_used] = self.spec_t_hist.get(res.t_used, 0) + 1
        for tok in res.tokens:
            self.sampler.accept(tok)
            if self._on_token(tok):
                return True                                           # tokens after the stop / the limit are dropped
        self.anchor = res.tokens[-1]
        return False

    def _emit(self, events: List[Event]) -> None:
        if not events:
            return
        out = []
        for e in events:
            if e.kind == "content":
                text, matched = self.stopm.push(e.text)
                if text:
                    out.append(Event("content", text))
                    self._content.append(text)
                if matched:
                    self.finish_reason = "stop"
            else:
                if e.kind == "reasoning":
                    self._reasoning.append(e.text)
                elif e.kind == "tool_end" and self.req.stop_after_tool:
                    self.finish_reason = "stop"  # the model would go on (thinking after a call it never decided on): end the turn
                out.append(e)
        if out:
            self.req.sink(("ev", out))

    def _on_token(self, tok: int) -> bool:
        tk = self.sched.tokenizer
        if tok in tk.stop_ids:
            self.finish_reason = "stop"
            if self.parser is not None:
                self.parser.feed_token(tok)
            return True
        self.n_completion += 1
        self.token_ids.append(tok)
        if self.parser is not None:
            self._emit(self.parser.feed_token(tok))
            if self.parser.in_reasoning:
                self.reasoning_tokens += 1
        else:
            t = self.raw_dec.push(tok)
            if t:
                self._emit([Event("content", t)])
        if self.finish_reason is not None:  # stop string matched
            return True
        if self.n_completion >= self.req.max_tokens:
            self.finish_reason = "length"
            return True
        return False

    def conclude(self) -> Finish:
        """Flush parser/stop filter and build the Finish record (also after errors/cancel)."""
        now = time.monotonic()
        if self.error is None and self.finish_reason != "cancelled":
            if self.parser is not None:
                self._emit(self.parser.finish())
            elif self.raw_dec is not None:
                t = self.raw_dec.flush()
                if t:
                    self._emit([Event("content", t)])
            tail = self.stopm.flush()
            if tail and not self.stopm.matched:
                self._content.append(tail)
                self.req.sink(("ev", [Event("content", tail)]))
        pm = self.parser.result() if self.parser is not None else ParsedMessage()
        pm.content = "".join(self._content) or None
        pm.reasoning_content = "".join(self._reasoning) or None
        fr = self.finish_reason or "error"
        if fr == "stop" and pm.tool_calls and self.req.mode == "chat":
            fr = "tool_calls"
        t_first = (self.t_first - self.req.created) if self.t_first else 0.0
        return Finish(fr, pm, len(self.req.prompt_ids), self.n_completion, self.n_reused, self.reasoning_tokens,
                      self.t_start - self.req.created, self.t_prefill, self.t_decode, now - self.req.created, t_first,
                      list(pm.errors) + ([self.error] if self.error else []), self.t_engine_dec, self.t_sample,
                      list(self.token_ids), self.spec_steps, self.spec_drafted, self.spec_accepted, self.spec_emitted,
                      self.spec_mode if self.spec_steps else None, self.spec_synced, dict(self.spec_t_hist))


class Scheduler:
    def __init__(self, backend: Backend, tokenizer: InkTokenizer, max_queue: int = 8, prefill_chunk: int = 256,
                 on_finish: Optional[Callable[[GenRequest, Finish], None]] = None, spec: bool = True):
        self.backend, self.tokenizer = backend, tokenizer
        self.spec_enabled = spec
        self.max_queue, self.prefill_chunk = max_queue, max(1, prefill_chunk)
        self.on_finish = on_finish
        self.table: Optional[SlotTable] = None
        self._pending: Deque[GenRequest] = collections.deque()
        self._active: Deque[Sequence] = collections.deque()
        self._cv = threading.Condition()
        self._stop = False
        self.ready = threading.Event()
        self.load_error: Optional[str] = None
        self._thread = threading.Thread(target=self._run, name="ink-engine", daemon=True)
        self._ema_s = 0.0
        self._n_running = 0   # sequences admitted and not yet finished (they are in no deque while a step runs)

    # ---------------------------------------------------------------- public API (any thread)
    def start(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        self._thread.start()
        if wait:
            self.ready.wait(timeout)
            if self.load_error:
                raise RuntimeError(self.load_error)

    def shutdown(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=10)

    def n_waiting(self) -> int:
        return len(self._pending)

    def slot_snapshot(self) -> List[Dict[str, Any]]:
        """Per-slot view for /slots and /metrics (plain python data only: no engine call, any thread)."""
        if self.table is None:
            return []
        now = time.monotonic()
        return [{"id": s.id, "n_tokens": len(s.tokens), "busy": s.busy, "idle_s": round(now - s.last_used, 1) if s.last_used else None,
                 "has_logits": s.last_logits is not None} for s in list(self.table.slots.values())]

    def n_active(self) -> int:
        return self._n_running

    def retry_after(self) -> int:
        per = self._ema_s if self._ema_s > 0 else 5.0
        return int(min(300, max(1, math.ceil(per * (len(self._pending) + 1) / max(1, self.backend.n_slots)))))

    def submit(self, req: GenRequest) -> None:
        if not self.ready.is_set() or self.load_error:
            raise RuntimeError("engine not ready: %s" % (self.load_error or "loading"))
        with self._cv:
            if len(self._pending) + self._n_running >= self.backend.n_slots + self.max_queue:
                raise QueueFull(self.retry_after())
            self._pending.append(req)
            self._cv.notify_all()

    # ---------------------------------------------------------------- engine thread
    def _run(self) -> None:
        try:
            self.backend.start()
            self.table = SlotTable(self.backend)
            self.table.init()
            if not self.table.slots:
                raise RuntimeError("backend returned no slots")
        except Exception as e:  # noqa: BLE001
            self.load_error = "%s: %s" % (type(e).__name__, e)
            self.ready.set()
            return
        self.ready.set()
        while True:
            with self._cv:
                while not self._stop and not self._pending and not self._active:
                    self._cv.wait(0.5)
                if self._stop:
                    break
            self._admit()
            if not self._active:
                with self._cv:  # pending requests wait for a slot that is not coming (no active work): avoid spinning
                    if self._pending and not self._active:
                        self._cv.wait(0.05)
                continue
            seq = self._active.popleft()
            done = False
            try:
                done = seq.step()
            except Exception as e:  # noqa: BLE001
                seq.error = "%s: %s" % (type(e).__name__, e)
                seq.finish_reason = "error"
                traceback.print_exc()
                done = True
            if done:
                self._finish(seq)
            else:
                self._active.append(seq)
        try:
            self.backend.close()
        except Exception:  # noqa: BLE001
            pass

    def _admit(self) -> None:
        while True:
            with self._cv:
                if not self._pending:
                    return
                req = self._pending[0]
                if req.cancel.is_set():
                    self._pending.popleft()
                    continue
                if len(req.prompt_ids) + 1 > self.backend.max_ctx:
                    self._pending.popleft()
                    req.sink(("error", "prompt of %d tokens exceeds the context of %d" % (len(req.prompt_ids), self.backend.max_ctx), 400))
                    continue
                got = self.table.acquire(req.prompt_ids)
                if got is None:
                    return
                self._pending.popleft()
                self._n_running += 1
            slot, n_reused = got
            self._active.append(Sequence(self, req, slot, n_reused))

    def _finish(self, seq: Sequence) -> None:
        healthy = seq.error is None
        self.table.release(seq.slot, healthy)
        fin = seq.conclude()
        self._ema_s = fin.total_s if self._ema_s == 0 else 0.8 * self._ema_s + 0.2 * fin.total_s
        with self._cv:  # capacity is released BEFORE the client is told it is done (back-to-back requests must not 429)
            self._n_running -= 1
        if seq.error:
            seq.req.sink(("error", seq.error, 500))
        elif fin.finish_reason != "cancelled":
            seq.req.sink(("done", fin))
        if self.on_finish:
            try:
                self.on_finish(seq.req, fin)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
