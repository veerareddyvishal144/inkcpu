"""Tiny Prometheus-style metrics and the JSONL request log."""
from __future__ import annotations

import json
import threading
import time
from typing import Dict, Optional, Tuple

_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300)


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.requests: Dict[Tuple[str, str], int] = {}
        self.rejected = 0
        self.tokens = {"prompt": 0, "completion": 0, "cached": 0, "reasoning": 0}
        self.dur_buckets = [0] * (len(_BUCKETS) + 1)
        self.dur_sum = 0.0
        self.dur_count = 0
        self.ttft_sum = 0.0
        self.ttft_count = 0
        self.prefill_tps = 0.0
        self.decode_tps = 0.0
        self.sec = {"prefill": 0.0, "decode_engine": 0.0, "decode_sampling": 0.0, "decode_other": 0.0, "queue": 0.0}
        self.finished: Dict[str, int] = {}
        self.spec = {"steps": 0, "drafted": 0, "accepted": 0, "emitted": 0}
        self.started = time.time()

    def inc_request(self, endpoint: str, status: int) -> None:
        with self._lock:
            k = (endpoint, str(status))
            self.requests[k] = self.requests.get(k, 0) + 1
            if status == 429:
                self.rejected += 1

    def observe(self, fin, endpoint: str) -> None:
        with self._lock:
            self.tokens["prompt"] += fin.prompt_tokens
            self.tokens["completion"] += fin.completion_tokens
            self.tokens["cached"] += fin.cached_tokens
            self.tokens["reasoning"] += fin.reasoning_tokens
            self.dur_sum += fin.total_s
            self.dur_count += 1
            self.finished[fin.finish_reason] = self.finished.get(fin.finish_reason, 0) + 1
            self.sec["prefill"] += fin.prefill_s
            self.sec["queue"] += fin.queue_s
            self.sec["decode_engine"] += fin.decode_engine_s
            self.sec["decode_sampling"] += fin.sample_s
            self.sec["decode_other"] += max(0.0, fin.decode_s - fin.decode_engine_s - fin.sample_s)
            if fin.spec_steps:
                self.spec["steps"] += fin.spec_steps
                self.spec["drafted"] += fin.spec_drafted
                self.spec["accepted"] += fin.spec_accepted
                self.spec["emitted"] += fin.completion_tokens
            i = 0
            while i < len(_BUCKETS) and fin.total_s > _BUCKETS[i]:
                i += 1
            self.dur_buckets[i] += 1
            if fin.ttft_s:
                self.ttft_sum += fin.ttft_s
                self.ttft_count += 1
            new = fin.prompt_tokens - fin.cached_tokens
            if fin.prefill_s > 0 and new > 1:
                self.prefill_tps = new / fin.prefill_s
            if fin.decode_s > 0 and fin.completion_tokens > 1:
                self.decode_tps = fin.completion_tokens / fin.decode_s

    def render(self, sched, backend_name: str) -> str:
        L = []
        def g(name, help_, typ, val, labels=""):
            L.append("# HELP %s %s\n# TYPE %s %s\n%s%s %s" % (name, help_, name, typ, name, labels, val))
        with self._lock:
            g("ink_up", "server is up", "gauge", 1, '{backend="%s"}' % backend_name)
            g("ink_uptime_seconds", "seconds since start", "gauge", "%.1f" % (time.time() - self.started))
            g("ink_queue_depth", "requests waiting for a slot", "gauge", sched.n_waiting())
            g("ink_active_requests", "requests being generated", "gauge", sched.n_active())
            g("ink_slots_total", "engine slots", "gauge", sched.backend.n_slots)
            L.append("# HELP ink_requests_total requests by endpoint and HTTP status\n# TYPE ink_requests_total counter")
            for (ep, st), n in sorted(self.requests.items()):
                L.append('ink_requests_total{endpoint="%s",status="%s"} %d' % (ep, st, n))
            g("ink_rejected_requests_total", "requests rejected with 429", "counter", self.rejected)
            L.append("# HELP ink_tokens_total tokens processed\n# TYPE ink_tokens_total counter")
            for k, v in self.tokens.items():
                L.append('ink_tokens_total{kind="%s"} %d' % (k, v))
            L.append("# HELP ink_request_duration_seconds request wall time\n# TYPE ink_request_duration_seconds histogram")
            acc = 0
            for b, n in zip(_BUCKETS, self.dur_buckets):
                acc += n
                L.append('ink_request_duration_seconds_bucket{le="%g"} %d' % (b, acc))
            L.append('ink_request_duration_seconds_bucket{le="+Inf"} %d' % self.dur_count)
            L.append("ink_request_duration_seconds_sum %.4f\nink_request_duration_seconds_count %d" % (self.dur_sum, self.dur_count))
            g("ink_ttft_seconds_sum", "sum of time to first token", "counter", "%.4f" % self.ttft_sum)
            g("ink_ttft_seconds_count", "requests with a first token", "counter", self.ttft_count)
            L.append("# HELP ink_finished_total finished generations by finish reason\n# TYPE ink_finished_total counter")
            for k, v in sorted(self.finished.items()):
                L.append('ink_finished_total{reason="%s"} %d' % (k, v))
            L.append("# HELP ink_seconds_total wall time by phase (decode_other = parser, stop strings, queueing the SSE events)\n"
                     "# TYPE ink_seconds_total counter")
            for k, v in self.sec.items():
                L.append('ink_seconds_total{phase="%s"} %.4f' % (k, v))
            L.append("# HELP ink_spec_total speculative decoding counters\n# TYPE ink_spec_total counter")
            for k, v in self.spec.items():
                L.append('ink_spec_total{kind="%s"} %d' % (k, v))
            slots = sched.slot_snapshot()
            if slots:
                L.append("# HELP ink_slot_tokens tokens held in the slot (KV cache fill)\n# TYPE ink_slot_tokens gauge")
                for s_ in slots:
                    L.append('ink_slot_tokens{slot="%d",busy="%d"} %d' % (s_["id"], 1 if s_["busy"] else 0, s_["n_tokens"]))
                g("ink_context_tokens", "context length per slot", "gauge", sched.backend.max_ctx)
            g("ink_prefill_tokens_per_second", "prefill speed of the last request", "gauge", "%.2f" % self.prefill_tps)
            g("ink_decode_tokens_per_second", "decode speed of the last request", "gauge", "%.2f" % self.decode_tps)
        return "\n".join(L) + "\n"


class RequestLog:
    """One JSON object per line; bodies are only logged when log_bodies is set."""

    def __init__(self, path: Optional[str], log_bodies: bool = False):
        self.path, self.log_bodies = path, log_bodies
        self._lock = threading.Lock()
        self._f = open(path, "a", buffering=1, encoding="utf-8") if path else None

    def write(self, rec: dict) -> None:
        if self._f is None:
            return
        rec = dict(rec)
        rec.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        with self._lock:
            self._f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._f:
            self._f.close()
