import queue
import threading

import helpers
from helpers import tok

from server.backends import MockBackend
from server.sampling import SamplingParams
from server.scheduler import GenRequest, QueueFull, Scheduler


def _sched(n_slots=2, chunk=16, reuse_min=1024, **kw):
    be = MockBackend(n_slots=n_slots, max_ctx=4096, **kw)
    s = Scheduler(be, tok(), max_queue=1, prefill_chunk=chunk, prefix_reuse_min=reuse_min)
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


# ---------------------------------------------------------------------------------------------- S3: LCP reuse by slot rewind (docs/REWIND.md)
def _ops(be, since, op):
    return [c for c in be.calls[since:] if c[0] == op]


def _invariant(be, s):
    for sl in s.table.slots.values():
        assert be.slot_pos(sl.id) == len(sl.tokens) and be.hist[sl.id] == sl.tokens, (sl.id, be.slot_pos(sl.id), len(sl.tokens))


def test_rewind_lcp_reuses_a_shared_prefix_and_leaves_a_checkpoint_at_the_divergence_point():
    be, s = _sched(n_slots=1, reuse_min=32)
    try:
        base = list(range(1000, 1100))                        # 100 shared tokens; prefill chunks of 16 -> checkpoints 16, 32, .., 96, 103
        f1 = _run(s, base + [1, 2, 3])
        slot = s.table.slots[0]
        assert f1.cached_tokens == 0 and slot.tokens[:103] == base + [1, 2, 3]
        assert be.checkpoints(0)[-2:] == [103, len(slot.tokens)], be.checkpoints(0)   # prompt end (deep) + generation end (shallow)
        assert (0, 103, True) in be.ckpt_calls and (0, len(slot.tokens), False) in be.ckpt_calls
        mark = len(be.calls)
        f2 = _run(s, base + [7, 8, 9, 10])                    # LCP 100: nearest checkpoint <= 100 is 96
        assert f2.cached_tokens == 96, f2.cached_tokens
        assert ("rewind", 0, 96) in be.calls[mark:] and not _ops(be, mark, "slot_reset")
        assert [c[2] for c in _ops(be, mark, "prefill")] == [4, 4], be.calls[mark:]   # 96 -> 100 (the divergence point) -> 104
        assert (0, 100, True) in be.ckpt_calls and (0, 104, True) in be.ckpt_calls      # deep at the divergence point and at the prompt end
        assert slot.tokens[:104] == base + [7, 8, 9, 10]
        _invariant(be, s)
        be2, s2 = _sched(n_slots=1, reuse_min=32)             # the rewound run equals a cold run (the mock's logits are a function of the history)
        try:
            assert _run(s2, base + [7, 8, 9, 10]).token_ids == f2.token_ids
        finally:
            s2.shutdown()
        mark = len(be.calls)
        f3 = _run(s, base + [11, 12])                         # a third conversation with the same prefix hits the divergence checkpoint exactly
        assert f3.cached_tokens == 100 and [c[2] for c in _ops(be, mark, "prefill")] == [2]
        assert s.n_rewinds() == 2
        _invariant(be, s)
    finally:
        s.shutdown()


def test_rewind_below_the_threshold_or_refused_falls_back_to_the_lru_reset():
    be, s = _sched(n_slots=1, reuse_min=64)
    try:
        base = list(range(1000, 1040))                        # 40 shared tokens < 64
        _run(s, base + [1, 2, 3])
        mark = len(be.calls)
        f = _run(s, base + [5, 6, 7])
        assert f.cached_tokens == 0 and _ops(be, mark, "slot_reset") and not _ops(be, mark, "rewind")
        _invariant(be, s)
    finally:
        s.shutdown()
    be, s = _sched(n_slots=1, reuse_min=32, rewind_bound=4)    # every shallow checkpoint <= 100 is dead (12 appended > 4), the deep one is at 103
    try:
        base = list(range(1000, 1100))
        _run(s, base + [1, 2, 3])
        mark = len(be.calls)
        f = _run(s, base + [7, 8, 9, 10])
        assert f.cached_tokens == 0 and _ops(be, mark, "slot_reset") and not _ops(be, mark, "rewind")
        assert [c[2] for c in _ops(be, mark, "prefill")] == [16] * 6 + [8]
        _invariant(be, s)
    finally:
        s.shutdown()


def test_rewind_when_the_prompt_is_a_strict_prefix_of_the_history_processes_at_least_one_token():
    be, s = _sched(n_slots=1, reuse_min=32)
    try:
        base = list(range(1000, 1100))
        _run(s, base + [1, 2, 3])
        mark = len(be.calls)
        f = _run(s, base[:90])                                 # LCP 90 == prompt length: rewind target 89 -> checkpoint 80, prefill 10
        assert f.cached_tokens == 80 and [c[2] for c in _ops(be, mark, "prefill")] == [10]
        assert s.table.slots[0].tokens[:90] == base[:90]
        _invariant(be, s)
    finally:
        s.shutdown()


def test_rewind_picks_the_slot_with_the_longest_common_prefix_over_an_empty_slot():
    be, s = _sched(n_slots=2, reuse_min=32)
    try:
        a, b = list(range(1000, 1100)), list(range(2000, 2060))
        _run(s, a + [1, 2, 3])                                 # slot 0
        mark = len(be.calls)
        f = _run(s, a + [7, 8])                                # LCP 100 with slot 0 beats the empty slot 1
        assert f.cached_tokens == 96 and _ops(be, mark, "rewind")[0][1] == 0 and not s.table.slots[1].tokens
        _run(s, b + [1])                                       # unrelated: empty slot 1
        assert s.table.slots[1].tokens[:61] == b + [1]
        mark = len(be.calls)
        f = _run(s, a[:70] + b[:40])                           # shares 70 with slot 0, nothing with slot 1 -> slot 0 (checkpoint 64)
        assert f.cached_tokens == 64 and _ops(be, mark, "rewind")[0][1] == 0
        mark = len(be.calls)
        f = _run(s, b[:50] + [9])                              # shares 50 with slot 1 (>= 32) -> slot 1 (checkpoint 48)
        assert f.cached_tokens == 48 and _ops(be, mark, "rewind")[0][1] == 1
        _invariant(be, s)
    finally:
        s.shutdown()


def test_rewind_disabled_backend_keeps_the_append_only_policy():
    be, s = _sched(n_slots=1, reuse_min=32, rewind=False)
    try:
        base = list(range(1000, 1100))
        _run(s, base + [1, 2, 3])
        mark = len(be.calls)
        f = _run(s, base + [7, 8, 9, 10])
        assert f.cached_tokens == 0 and _ops(be, mark, "slot_reset") and not be.ckpt_calls
    finally:
        s.shutdown()


if __name__ == "__main__":
    raise SystemExit(helpers.run_module(globals()))


def test_c_sampler_flag_uses_the_backend_sampler_on_the_plain_path_only_without_penalties():
    """--sampler c (agent T): plain decode samples through backend.sample (ink_sample) when the backend has one and the request has no
    penalties; spec_step is never used (spec disabled); penalties fall back to the numpy Sampler."""
    from server.backends import MockSpecBackend
    be = MockSpecBackend(n_slots=1, max_ctx=4096)
    be.supports_sample = True
    calls = []
    orig = be.sample

    def counting_sample(logits, req, seed):
        calls.append(seed)
        return orig(logits, req, seed)
    be.sample = counting_sample
    s = Scheduler(be, tok(), max_queue=1, prefill_chunk=16, spec=False, c_sampler=True)
    s.start()
    try:
        prompt = tok().encode("Once upon a time")
        for sp, expect_c in ((SamplingParams(seed=5, temperature=0.7), True),
                             (SamplingParams(seed=5, temperature=0.7, repetition_penalty=1.2), False),
                             (SamplingParams(seed=5, temperature=0.0), True)):
            calls.clear()
            q = queue.Queue()
            s.submit(GenRequest(id="c", prompt_ids=prompt, sampling=sp, max_tokens=5, sink=q.put, mode="raw"))
            while True:
                it = q.get(timeout=20)
                if it[0] in ("done", "error"):
                    assert it[0] == "done", it
                    fin = it[1]
                    break
            assert fin.completion_tokens == 5, fin
            assert fin.spec_steps == 0 and not be.spec_log, "spec path must stay off"
            if expect_c:
                assert len(calls) == 5, calls                     # one backend sample per generated token
                if sp.temperature > 0:
                    assert len(set(calls)) == 5, "the seed must be chained from draw to draw"
            else:
                assert calls == [], "penalties must use the numpy sampler"
        # the same seeded request twice gives the same tokens (deterministic chain)
        outs = []
        for _ in range(2):
            q = queue.Queue()
            s.submit(GenRequest(id="d", prompt_ids=prompt, sampling=SamplingParams(seed=11, temperature=0.9), max_tokens=6, sink=q.put, mode="raw"))
            while True:
                it = q.get(timeout=20)
                if it[0] in ("done", "error"):
                    outs.append(it[1].token_ids)
                    break
        assert outs[0] == outs[1], outs
    finally:
        s.shutdown()
