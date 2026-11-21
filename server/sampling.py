"""Sampling in numpy on the float32 logits of the engine (final softcap already applied).

Order (HF generation order): penalties -> temperature -> top_k -> top_p -> min_p -> draw. Defaults follow the checkpoint's
generation_config.json: temperature 1.0, top_p 1, top_k 0 (off). temperature == 0 (or top_k == 1) is greedy argmax.
A request seed gives a private PCG64 stream: one uniform draw per sampled token, so the same logits + seed + params always
give the same tokens (greedy draws nothing).

Penalties: repetition_penalty (HF/vLLM semantics) over prompt + generated tokens; frequency_penalty / presence_penalty
(OpenAI semantics: logit -= freq*count + pres*[count>0]) over the generated tokens only.

Speed (V, 2026-11-21; vocabulary 200058, one engine decode step takes ~75 ms; numbers in docs/SERVING.md):
  * the logits row is NEVER copied unless a penalty has to modify it (the backend hands out a view of its per-slot buffer);
  * greedy = one np.argmax;
  * pure temperature: e = exp((x - max) / T) into ONE preallocated float32 buffer and a two-level inverse CDF (block sums in
    float64, then one block) instead of a 200k-element float64 cumsum;
  * top_k / top_p / min_p: the candidate set is a THRESHOLD on e (min_p: e >= min_p exactly; top_k / top_p: widened
    1e-3, 1e-6, 1e-12, 0 until the set holds k tokens / covers the nucleus), so the sort only sees a few hundred tokens.
    A threshold set is always a prefix of the fully sorted vocabulary, so the result equals the full-sort reference
    (`Sampler._reference`, used by the tests).
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0

    def validate(self) -> None:
        if not (0.0 <= self.temperature <= 10.0):
            raise ValueError("temperature must be in [0, 10]")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not (0.0 <= self.min_p <= 1.0):
            raise ValueError("min_p must be in [0, 1]")
        if not (-2.0 <= self.frequency_penalty <= 2.0) or not (-2.0 <= self.presence_penalty <= 2.0):
            raise ValueError("frequency_penalty / presence_penalty must be in [-2, 2]")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be > 0")

    @property
    def has_penalties(self) -> bool:
        return self.repetition_penalty != 1.0 or self.frequency_penalty != 0.0 or self.presence_penalty != 0.0


class Sampler:
    _BLOCK = 512                      # block size of the two-level inverse CDF
    _FLOOR = np.float32(-80.0)        # exp(-80) = 1.8e-35 is still a normal float32 (no denormal slow path)

    def __init__(self, params: SamplingParams, prompt_ids: Optional[Iterable[int]] = None):
        self.p = params
        seed = params.seed if params.seed is not None else secrets.randbits(63)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.counts: Dict[int, int] = {}
        self._track = params.has_penalties
        self._prompt_unique = (np.unique(np.fromiter(prompt_ids, dtype=np.int64))
                               if (prompt_ids is not None and params.repetition_penalty != 1.0) else None)
        self._buf: Optional[np.ndarray] = None

    def accept(self, tok: int) -> None:
        if self._track:
            self.counts[tok] = self.counts.get(tok, 0) + 1

    def _penalize(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        if p.repetition_penalty != 1.0:
            idx = self._prompt_unique if self._prompt_unique is not None else np.empty(0, np.int64)
            if self.counts:
                idx = np.union1d(idx, np.fromiter(self.counts, dtype=np.int64, count=len(self.counts)))
            idx = idx[idx < x.size]
            v = x[idx]
            x[idx] = np.where(v > 0, v / p.repetition_penalty, v * p.repetition_penalty)
        if (p.frequency_penalty != 0.0 or p.presence_penalty != 0.0) and self.counts:
            idx = np.fromiter(self.counts, dtype=np.int64, count=len(self.counts))
            cnt = np.fromiter(self.counts.values(), dtype=np.float32, count=len(self.counts))
            ok = idx < x.size
            idx, cnt = idx[ok], cnt[ok]
            x[idx] -= p.frequency_penalty * cnt + p.presence_penalty * (cnt > 0)
        return x

    # ---------------------------------------------------------------- draws
    def _draw(self, probs: np.ndarray) -> int:
        """Inverse CDF over a (small) array of unnormalised probabilities, in array order; one uniform draw."""
        c = np.cumsum(probs, dtype=np.float64)
        u = self.rng.random() * c[-1]
        return int(min(np.searchsorted(c, u, side="right"), probs.size - 1))

    def _draw_full(self, e: np.ndarray) -> int:
        """Inverse CDF in token-id order over the whole vocabulary without a 200k-element float64 cumsum."""
        V, B = e.size, self._BLOCK
        nb = V // B
        if nb < 2:
            return self._draw(e)
        bs = e[:nb * B].reshape(nb, B).sum(axis=1, dtype=np.float64)
        if nb * B < V:
            bs = np.append(bs, e[nb * B:].sum(dtype=np.float64))
        cb = np.cumsum(bs)
        u = self.rng.random() * cb[-1]
        j = min(int(np.searchsorted(cb, u, side="right")), cb.size - 1)
        lo, hi = j * B, min(j * B + B, V)
        inner = np.cumsum(e[lo:hi], dtype=np.float64)
        u2 = u - (cb[j - 1] if j else 0.0)
        return lo + min(int(np.searchsorted(inner, u2, side="right")), hi - lo - 1)

    # ---------------------------------------------------------------- scores
    def _scores(self, x: np.ndarray) -> np.ndarray:
        """e = exp((x - max) / T) in float32 (max == 1), written into the per-sampler buffer."""
        e = self._buf
        if e is None or e.size != x.size:
            e = self._buf = np.empty(x.size, dtype=np.float32)
        np.subtract(x, np.float32(x.max()), out=e)
        if self.p.temperature != 1.0:
            np.multiply(e, np.float32(1.0 / self.p.temperature), out=e)
        np.maximum(e, self._FLOOR, out=e)
        np.exp(e, out=e)
        return e

    @staticmethod
    def _widen(thr: float) -> float:
        return thr * 1e-3 if thr > 1e-12 else 0.0

    @classmethod
    def _topk_candidates(cls, e: np.ndarray, k: int) -> np.ndarray:
        thr = 1e-3
        while True:
            idx = np.flatnonzero(e >= np.float32(thr)) if thr > 0.0 else np.arange(e.size)
            if idx.size >= k or thr <= 0.0:
                break
            thr = cls._widen(thr)
        if idx.size > k:
            idx = idx[np.argpartition(e[idx], idx.size - k)[idx.size - k:]]
        return idx

    def _keep(self, ec: np.ndarray, z: Optional[float]) -> int:
        """Number of leading entries of the descending-sorted scores `ec` that survive top_p / min_p."""
        p = self.p
        keep = ec.size
        if p.top_p < 1.0:
            cum = np.cumsum(ec, dtype=np.float64) / z
            keep = min(keep, int(np.searchsorted(cum, p.top_p, side="left")) + 1)
        if p.min_p > 0.0:
            keep = min(keep, int(np.searchsorted(-ec, -(np.float32(p.min_p) * ec[0]), side="right")))
        return max(keep, 1)

    # ---------------------------------------------------------------- main entry
    def sample(self, logits: np.ndarray) -> int:
        p = self.p
        x = logits
        V = x.size
        if p.repetition_penalty != 1.0 or ((p.frequency_penalty != 0.0 or p.presence_penalty != 0.0) and self.counts):
            x = self._penalize(np.array(x, dtype=np.float32, copy=True))   # the only copy of the logits row
        if p.temperature <= 0.0 or p.top_k == 1:
            return int(np.argmax(x))
        e = self._scores(x)
        use_k = 0 < p.top_k < V
        if not use_k and p.top_p >= 1.0 and p.min_p <= 0.0:
            return self._draw_full(e)
        z = None
        if use_k:                                            # top_k first; top_p renormalises over the top_k set (HF)
            idx = self._topk_candidates(e, p.top_k)
            if p.top_p < 1.0:
                z = float(e[idx].sum(dtype=np.float64))
        elif p.top_p < 1.0:                                  # nucleus: widen the threshold until the set covers top_p
            z = float(e.sum(dtype=np.float64))
            thr = 1e-3
            while True:
                idx = np.flatnonzero(e >= np.float32(thr)) if thr > 0.0 else np.arange(V)
                if (thr <= 0.0 or float(e[idx].sum(dtype=np.float64)) >= p.top_p * z
                        or (p.min_p > 0.0 and thr <= p.min_p)):   # (min_p alone cuts inside a set that holds all e >= min_p)
                    break
                thr = self._widen(thr)
        else:                                                # min_p only: exact threshold, always contains the max token
            idx = np.flatnonzero(e >= np.float32(p.min_p))
        ec = e[idx]
        order = np.argsort(-ec, kind="stable")
        idx, ec = idx[order], ec[order]
        keep = self._keep(ec, z)
        return int(idx[self._draw(ec[:keep])])

    def _reference(self, x: np.ndarray) -> int:
        """Slow full-sort implementation of the same semantics (tests): one uniform draw, whole vocabulary sorted."""
        p = self.p
        if p.temperature <= 0.0 or p.top_k == 1:
            return int(np.argmax(x))
        e = self._scores(np.asarray(x, dtype=np.float32)).copy()
        order = np.argsort(-e, kind="stable")
        es = e[order]
        if 0 < p.top_k < es.size:
            es, order = es[:p.top_k], order[:p.top_k]
        z = float(es.sum(dtype=np.float64))
        keep = self._keep(es, z)
        return int(order[:keep][self._draw(es[:keep])])
