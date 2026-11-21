import queue
import threading

import helpers
from helpers import tok

from server.backends import MockBackend
from server.sampling import SamplingParams
from server.scheduler import GenRequest, QueueFull, Scheduler


def _sched(n_slots=2, chunk=16, **kw):
    be = MockBackend(n_slots=n_slots, max_ctx=4096, **kw)
    s = Scheduler(be, tok(), max_queue=1, prefill_chunk=chunk)
    s.start()
    return be, s


def _run(s, prompt, max_tokens=6, seed=1):
    q = queue.Queue()
    s.submit(GenRequest(id="t", prompt_ids=prompt, sampling=SamplingParams(seed=seed, temperature=0.0), max_tokens=max_tokens,
                        sink=q.put, mode="raw"))
    while True:
        it = q.get(timeout=20)
        if it[0] in ("done", "error"):
            assert it[0] == "done", it
            return it[1]


def _fed(be, since):
    return sum(n for op, _, n in be.calls[since:] if op in ("prefill", "step"))


def _prefilled(be, since):
    return sum(n for op, _, n in be.calls[since:] if op == "prefill")


def test_prefix_reuse_only_prefills_the_new_tokens():
    be, s = _sched(n_slots=1)
    try:
        base = list(range(1000, 1050))
        f1 = _run(s, base)
        assert f1.cached_tokens == 0 and _prefilled(be, 0) == 50
        hist = list(s.table.slots[0].tokens)               # prompt + generated tokens that were fed
        assert hist[:50] == base and len(hist) == 50 + 5   # max_tokens=6: the 6th sampled token is not fed
        mark = len(be.calls)
        f2 = _run(s, hist + [7, 8, 9])
        assert f2.cached_tokens == len(hist)
        assert _prefilled(be, mark) == 3, be.calls[mark:]  # only the 3 new tokens
        assert not [c for c in be.calls[mark:] if c[0] == "slot_reset"]
        assert s.table.slots[0].tokens[:len(hist) + 3] == hist + [7, 8, 9]
    finally:
        s.shutdown()


def test_divergence_resets_and_policy_prefers_empty_then_lru_slot():
    be, s = _sched(n_slots=2)
    try:
        a, b, c = list(range(100, 130)), list(range(200, 230)), list(range(300, 330))
        _run(s, a); _run(s, b)                             # two conversations, one slot each
        ha = [x for x in s.table.slots.values() if x.tokens[:30] == a][0]
        hb = [x for x in s.table.slots.values() if x.tokens[:30] == b][0]
        assert ha.id != hb.id
        mark = len(be.calls)
        f = _run(s, ha.tokens + [5])                       # continues conversation a (not the most recent one)
        assert f.cached_tokens > 0
        assert _prefilled(be, mark) == 1
        mark = len(be.calls)
        f = _run(s, c)                                     # unrelated: resets the least recently used slot (b)
        assert f.cached_tokens == 0 and any(x[0] == "slot_reset" and x[1] == hb.id for x in be.calls[mark:])
        assert hb.tokens[:30] == c and ha.tokens[:30] == a
        mark = len(be.calls)
        _run(s, a[:10] + [999])                            # prefix-but-not-extension of a: no reuse, reset
        assert _prefilled(be, mark) == 11
    finally:
        s.shutdown()


def test_identical_prompt_reuses_cached_logits_without_calling_the_engine():
    be, s = _sched(n_slots=1)
    try:
        p = list(range(500, 540))
        _run(s, p, max_tokens=3)
        hist = list(s.table.slots[0].tokens)
        mark = len(be.calls)
        f = _run(s, hist, max_tokens=2)                    # exactly the history: nothing to prefill
        assert f.cached_tokens == len(hist) and _prefilled(be, mark) == 0
        assert [c[0] for c in be.calls[mark:]] == ["step"]  # one decode step (2 tokens max_tokens=2, last not fed)
    finally:
        s.shutdown()


def test_longest_matching_history_wins():
    be, s = _sched(n_slots=2)
    try:
        p = list(range(700, 740))
        _run(s, p, max_tokens=2)
        h1 = list(s.table.slots[0].tokens)
        # second slot holds a longer continuation of the same conversation
        _run(s, h1 + [1, 2, 3, 4], max_tokens=2)           # extends slot 0 in place (longest match = slot 0 only)
        longer = list(s.table.slots[0].tokens)
        _run(s, p + [9999], max_tokens=2)                  # diverges from slot 0's history -> goes to the empty slot 1
        assert s.table.slots[1].tokens[:41] == p + [9999]
        mark = len(be.calls)
        f = _run(s, longer + [42])
        assert f.cached_tokens == len(longer) and _prefilled(be, mark) == 1
    finally:
        s.shutdown()


def test_queue_full_raises_with_retry_after():
    be, s = _sched(n_slots=1, delay=0.2)
    try:
        qs = []
        def submit(i):
            q = queue.Queue()
            s.submit(GenRequest(id=str(i), prompt_ids=[1000 + i, 2], sampling=SamplingParams(temperature=0.0), max_tokens=3,
                                sink=q.put, mode="raw"))
            qs.append(q)
        submit(0); submit(1)                               # 1 running + 1 queued = capacity (n_slots + max_queue)
        try:
            submit(2)
        except QueueFull as e:
            assert e.retry_after >= 1
        else:
            raise AssertionError("expected QueueFull")
        for q in qs:                                       # accepted requests still finish
            while q.get(timeout=20)[0] != "done":
                pass
    finally:
        s.shutdown()


if __name__ == "__main__":
    raise SystemExit(helpers.run_module(globals()))
