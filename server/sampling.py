"""Sampling in numpy on the float32 logits of the engine (final softcap already applied).

Order (HF generation order): penalties -> temperature -> top_k -> top_p -> min_p -> draw. Defaults follow the checkpoint's
generation_config.json: temperature 1.0, top_p 1, top_k 0 (off). temperature == 0 (or top_k == 1) is greedy argmax.
A request seed gives a private PCG64 stream: one uniform draw per sampled token, so the same logits + seed + params always
give the same tokens (greedy draws nothing).

Penalties: repetition_penalty (HF/vLLM semantics) over prompt + generated tokens; frequency_penalty / presence_penalty
(OpenAI semantics: logit -= freq*count + pres*[count>0]) over the generated tokens only.
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


class Sampler:
    _K0 = 4096  # candidate pre-selection for top_p / min_p without top_k (exactness is checked, else full sort)

    def __init__(self, params: SamplingParams, prompt_ids: Optional[Iterable[int]] = None):
        self.p = params
        seed = params.seed if params.seed is not None else secrets.randbits(63)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.counts: Dict[int, int] = {}
        self._prompt_unique = (np.unique(np.fromiter(prompt_ids, dtype=np.int64))
                               if (prompt_ids is not None and params.repetition_penalty != 1.0) else None)

    def accept(self, tok: int) -> None:
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

    def _draw(self, probs: np.ndarray) -> int:
        c = np.cumsum(probs, dtype=np.float64)
        u = self.rng.random() * c[-1]
        return int(min(np.searchsorted(c, u, side="right"), probs.size - 1))

    def sample(self, logits: np.ndarray) -> int:
        p = self.p
        x = np.array(logits, dtype=np.float32, copy=True)
        V = x.size
        if p.repetition_penalty != 1.0 or self.counts:
            x = self._penalize(x)
        if p.temperature <= 0.0 or p.top_k == 1:
            return int(np.argmax(x))
        if p.temperature != 1.0:
            x /= np.float32(p.temperature)
        cand = None  # None = whole vocabulary, else array of candidate token ids
        if 0 < p.top_k < V:
            cand = np.argpartition(x, V - p.top_k)[V - p.top_k:]
        need_sort = p.top_p < 1.0 or p.min_p > 0.0
        partial = False
        if cand is None and need_sort:
            if V > self._K0:
                cand = np.argpartition(x, V - self._K0)[V - self._K0:]
                partial = True
            else:
                cand = np.arange(V)
        m = float(x.max())
        if cand is None:
            probs = np.exp(x - m, dtype=np.float32).astype(np.float64)
            return self._draw(probs)
        lx = x[cand]
        e = np.exp(lx - m).astype(np.float64)
        if partial:  # normaliser over the FULL vocabulary so that coverage can be checked exactly
            z_full = float(np.exp(x - m, dtype=np.float32).sum(dtype=np.float64))
        else:
            z_full = float(e.sum())
        if need_sort:
            order = np.argsort(-e, kind="stable")
            cand, e = cand[order], e[order]
            pr = e / z_full
            keep = e.size
            if p.top_p < 1.0:
                cum = np.cumsum(pr)
                if partial and cum[-1] < p.top_p:  # candidate set does not cover the nucleus: exact full sort
                    return self._sample_full_sort(x, m)
                keep = min(keep, int(np.searchsorted(cum, p.top_p, side="left")) + 1)
            if p.min_p > 0.0:
                thr = p.min_p * pr[0]
                if partial and pr[-1] >= thr:  # tokens above thr might lie outside the candidate set
                    return self._sample_full_sort(x, m)
                keep = min(keep, int(np.searchsorted(-pr, -thr, side="right")))
            cand, e = cand[:max(keep, 1)], e[:max(keep, 1)]
        return int(cand[self._draw(e)])

    def _sample_full_sort(self, x: np.ndarray, m: float) -> int:
        p = self.p
        e = np.exp(x - m, dtype=np.float32).astype(np.float64)
        order = np.argsort(-e, kind="stable")
        es = e[order]
        pr = es / es.sum()
        keep = es.size
        if p.top_p < 1.0:
            keep = min(keep, int(np.searchsorted(np.cumsum(pr), p.top_p, side="left")) + 1)
        if p.min_p > 0.0:
            keep = min(keep, int(np.searchsorted(-pr, -(p.min_p * pr[0]), side="right")))
        keep = max(keep, 1)
        return int(order[:keep][self._draw(es[:keep])])
