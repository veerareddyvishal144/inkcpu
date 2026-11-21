import numpy as np

import helpers

from server.sampling import Sampler, SamplingParams


def _logits(seed=0, n=5000, scale=3.0):
    return (np.random.default_rng(seed).standard_normal(n) * scale).astype(np.float32)


def _draws(params, logits, n=64, prompt=None):
    s = Sampler(params, prompt)
    out = []
    for _ in range(n):
        t = s.sample(logits)
        s.accept(t)
        out.append(t)
    return out


def test_seeded_determinism():
    x = _logits()
    a = _draws(SamplingParams(seed=7), x)
    assert a == _draws(SamplingParams(seed=7), x)
    assert a != _draws(SamplingParams(seed=8), x)
    for kw in (dict(top_k=40), dict(top_p=0.9), dict(min_p=0.05), dict(top_k=50, top_p=0.8, temperature=0.7),
               dict(frequency_penalty=0.5, presence_penalty=0.2, repetition_penalty=1.1)):
        p = SamplingParams(seed=3, **kw)
        assert _draws(p, x) == _draws(SamplingParams(seed=3, **kw), x), kw


def test_greedy_and_top_k_1():
    x = _logits()
    am = int(np.argmax(x))
    assert set(_draws(SamplingParams(temperature=0.0, seed=1), x, 5)) == {am}
    assert set(_draws(SamplingParams(top_k=1, seed=1), x, 5)) == {am}


def test_filters_restrict_the_support():
    x = _logits(1, 3000)
    order = np.argsort(-x)
    top5 = set(order[:5].tolist())
    assert set(_draws(SamplingParams(top_k=5, seed=2), x, 200)) <= top5
    p = np.exp(x - x.max()); p /= p.sum()
    cum = np.cumsum(p[order])
    nucleus = set(order[:int(np.searchsorted(cum, 0.5)) + 1].tolist())
    assert set(_draws(SamplingParams(top_p=0.5, seed=2), x, 200)) <= nucleus
    thr = 0.2 * p.max()
    assert all(p[t] >= thr for t in _draws(SamplingParams(min_p=0.2, seed=2), x, 200))


def test_large_vocab_partial_path_equals_full_sort():
    x = _logits(5, 200058, 2.0)
    for kw in (dict(top_p=0.9), dict(top_p=0.99), dict(min_p=0.01), dict(top_p=0.95, min_p=0.001)):
        for seed in range(6):
            s1, s2 = Sampler(SamplingParams(seed=seed, **kw)), Sampler(SamplingParams(seed=seed, **kw))
            assert s1.sample(x) == s2._sample_full_sort(x, float(x.max())), (kw, seed)


def test_penalties():
    x = np.array([5.0, 4.9, 0.0], dtype=np.float32)
    s = Sampler(SamplingParams(temperature=0.0, presence_penalty=0.5))
    assert s.sample(x) == 0
    s.accept(0)
    assert s.sample(x) == 1                      # presence penalty flips the argmax
    s = Sampler(SamplingParams(temperature=0.0, frequency_penalty=0.06))
    s.accept(0); s.accept(0)
    assert s.sample(x) == 1                      # 5 - 0.12 < 4.9
    s = Sampler(SamplingParams(temperature=0.0, repetition_penalty=2.0), prompt_ids=[0])
    assert s.sample(x) == 1                      # prompt token 0: 5/2 < 4.9
    s = Sampler(SamplingParams(temperature=0.0, repetition_penalty=2.0), prompt_ids=[])
    assert s.sample(np.array([-1.0, -1.5], dtype=np.float32)) == 0
    s.accept(0)
    assert s.sample(np.array([-1.0, -1.5], dtype=np.float32)) == 1  # negative logits are multiplied


def test_distribution_follows_temperature():
    x = np.log(np.array([0.7, 0.2, 0.1], dtype=np.float32))
    d = np.bincount(_draws(SamplingParams(seed=11), x, 4000), minlength=3) / 4000.0
    assert abs(d[0] - 0.7) < 0.03 and abs(d[1] - 0.2) < 0.03
    cold = np.bincount(_draws(SamplingParams(seed=11, temperature=0.3), x, 2000), minlength=3) / 2000.0
    assert cold[0] > 0.9


if __name__ == "__main__":
    raise SystemExit(helpers.run_module(globals()))
