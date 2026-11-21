#!/data/engine/.venv/bin/python
"""Routing statistics from GGML_ROUTE_TRACE files (agent E).  Interpreter: /data/engine/.venv/bin/python

usage: route_stats.py name=trace.rtrace [name=trace.rtrace ...] [--bytes-json tools/bytes_per_token.json] [--min-tokens 64]
                      [--drafter-mb 635.795744] [--out report.md]

Trace format (written by the local patch in llama.cpp ggml/src/ggml-cpu/ggml-cpu.c): records of
  int32 header[8] = {0x52545243, fwd, visit, layer, n_tokens, n_expert_used, 0, 0}  followed by  int32 ids[n_tokens][n_expert_used]
one record per gate_up_exps MUL_MAT_ID node; a complete forward has 48 records (visit 0..47 = layers 0-5, 6-17 x3, 18-23).
Computes (a) distinct experts of T consecutive tokens vs the independent-routing expectation, (b) overlap of the expert sets of the
same token between the 3 passes of the looped layers, (c) popularity skew (share of the 16 most used experts per layer),
(d) expert bytes of a T-token verify step vs a one-token step, and the bandwidth break-even L*(T) of speculative decoding.
"""
import argparse
import json
import sys

import numpy as np

MAGIC = 0x52545243
N_EXP, N_USED = 128, 8
CORE0, CORE1, PASSES = 6, 18, 3
VISIT_LAYER = list(range(CORE0)) + list(range(CORE0, CORE1)) * PASSES + list(range(CORE1, 24))
TS = list(range(1, 9))
GROUPS = [('prelude L0-5', range(0, 6)), ('core pass 1', range(6, 18)), ('core pass 2', range(18, 30)),
          ('core pass 3', range(30, 42)), ('postlude L18-19 (NVFP4)', range(42, 44)), ('postlude L20-23 (Q8_0)', range(44, 48))]


def indep(T):
    return N_EXP * (1 - (1 - N_USED / N_EXP) ** T)


def load_trace(path, min_tokens):
    a = np.fromfile(path, dtype=np.int32)
    pos, recs = 0, {}
    while pos + 8 <= a.size:
        if a[pos] != MAGIC:
            sys.exit(f'{path}: bad magic at int32 {pos}')
        fwd, visit, layer, nt, nu = (int(x) for x in a[pos + 1:pos + 6])
        n = nt * nu
        if pos + 8 + n > a.size:
            break
        assert nu == N_USED
        recs.setdefault(fwd, {})[visit] = (layer, a[pos + 8:pos + 8 + n].reshape(nt, nu))
        pos += 8 + n
    fw, skipped = [], []
    for f in sorted(recs):
        r = recs[f]
        nt = next(iter(r.values()))[1].shape[0]
        if len(r) != 48 or nt < min_tokens:
            skipped.append((f, len(r), nt))
            continue
        for v in range(48):
            assert r[v][0] == VISIT_LAYER[v], (path, f, v, r[v][0])
        fw.append(np.stack([r[v][1] for v in range(48)]).astype(np.int16))  # [48, n_tokens, 8]
    return fw, skipped


def membership(ids):
    m = np.zeros((ids.shape[0], N_EXP), dtype=bool)
    m[np.arange(ids.shape[0])[:, None], ids] = True
    return m


def distinct_stats(fwds):
    """mean number of distinct experts in a window of T consecutive tokens, per visit: [48, len(TS)]"""
    dsum = np.zeros((48, len(TS)))
    dcnt = np.zeros(len(TS))
    for arr in fwds:
        nt = arr.shape[1]
        for v in range(48):
            C = np.concatenate([np.zeros((1, N_EXP), np.int32), np.cumsum(membership(arr[v]).astype(np.int32), axis=0)])
            for i, T in enumerate(TS):
                if T <= nt:
                    dsum[v, i] += ((C[T:] - C[:-T]) > 0).sum()
        for i, T in enumerate(TS):
            if T <= nt:
                dcnt[i] += nt - T + 1
    return dsum / dcnt


def pass_overlap(fwds):
    """per core layer: mean over tokens of |S1&S2|, |S2&S3|, |S1&S3|, |S1&S2&S3|, |S1|S2|S3|  -> [12, 5]"""
    acc = np.zeros((CORE1 - CORE0, 5))
    ntok = 0
    for arr in fwds:
        for k, l in enumerate(range(CORE0, CORE1)):
            M1, M2, M3 = (membership(arr[l + 12 * p]) for p in range(3))
            acc[k] += [(M1 & M2).sum(), (M2 & M3).sum(), (M1 & M3).sum(), (M1 & M2 & M3).sum(), (M1 | M2 | M3).sum()]
        ntok += arr.shape[1]
    return acc / ntok


def skew(fwds, top=16):
    """share of the selections taken by the `top` most used experts: per physical layer, per visit (mean) and pooled over the passes"""
    pooled = np.zeros((24, N_EXP))
    per_visit = np.zeros(24)
    nv = np.zeros(24)
    for arr in fwds:
        for v in range(48):
            l = VISIT_LAYER[v]
            cnt = np.bincount(arr[v].ravel().astype(np.int64), minlength=N_EXP)
            pooled[l] += cnt
            per_visit[l] += np.sort(cnt)[-top:].sum() / cnt.sum()
            nv[l] += 1
    return per_visit / nv, np.sort(pooled, axis=1)[:, -top:].sum(1) / pooled.sum(1)


def null_share(n_tok, rng, top=16, trials=8):
    s = []
    for _ in range(trials):
        sel = np.argpartition(rng.random((n_tok, N_EXP)), N_USED, axis=1)[:, :N_USED]
        cnt = np.bincount(sel.ravel(), minlength=N_EXP)
        s.append(np.sort(cnt)[-top:].sum() / cnt.sum())
    return float(np.mean(s))


def md(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('traces', nargs='+')
    ap.add_argument('--bytes-json', default='/data/inkling-mini/engine/tools/bytes_per_token.json')
    ap.add_argument('--min-tokens', type=int, default=64)
    ap.add_argument('--drafter-mb', type=float, default=635.795744)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    B = json.load(open(args.bytes_json))
    nv_b, q8_b = B['per_expert_bytes']['NVFP4'], B['per_expert_bytes']['Q8_0']
    assert len(nv_b) == 1 and len(q8_b) == 1
    bpe = np.array([nv_b[0] if B['layer_expert_type'][str(VISIT_LAYER[v])] == 'NVFP4' else q8_b[0] for v in range(48)], float)
    is_nv = bpe == nv_b[0]
    rng = np.random.default_rng(0)

    names, data, info = [], {}, []
    for spec in args.traces:
        name, path = spec.split('=', 1)
        fw, skipped = load_trace(path, args.min_tokens)
        if not fw:
            sys.exit(f'{path}: no complete forward with >= {args.min_tokens} tokens (skipped {skipped})')
        names.append(name)
        data[name] = dict(fw=fw, D=distinct_stats(fw), ov=pass_overlap(fw), skew=skew(fw), nt=[a.shape[1] for a in fw])
        info.append([name, len(fw), '+'.join(str(a.shape[1]) for a in fw), sum(a.shape[1] for a in fw), len(skipped)])
    D_all = {n: data[n]['D'] for n in names}
    D_mean = np.mean([D_all[n] for n in names], axis=0)  # equal weight per data set
    out = []
    P = out.append

    P('## Traces\n')
    P(md(['data set', 'forwards used', 'tokens per forward', 'tokens', 'forwards skipped (warm-up etc.)'], info))

    P('\n## (a) distinct experts per visit for T consecutive tokens (mean over all 48 visits and all windows)\n')
    rows = []
    for i, T in enumerate(TS):
        r = [T, f'{indep(T):.2f}']
        r += [f'{D_all[n][:, i].mean():.2f}' for n in names]
        r += [f'{D_mean[:, i].mean():.2f}']
        r += [f'{D_all[n][:, i].mean() / indep(T):.3f}' for n in names]
        r += [f'{D_mean[:, i].mean() / indep(T):.3f}']
        rows.append(r)
    P(md(['T', 'independent 128(1-(1-8/128)^T)'] + [f'{n} distinct' for n in names] + ['mean distinct'] + [f'{n} / indep' for n in names] +
         ['mean / indep'], rows))
    P('\nBy visit group (mean of the data sets), distinct experts and ratio to independent:\n')
    rows = []
    for g, vs in GROUPS:
        r = [g]
        for T in (2, 4, 8):
            d = D_mean[list(vs), TS.index(T)].mean()
            r.append(f'{d:.2f} ({d / indep(T):.3f})')
        rows.append(r)
    P(md(['visit group', 'T=2', 'T=4', 'T=8'], rows))

    P('\n## (b) overlap of the expert sets of the SAME token between the passes of the looped layers (layers 6-17; mean over layers and tokens)\n')
    ind2, ind3, indu = N_USED * N_USED / N_EXP, N_USED ** 3 / N_EXP ** 2, indep(3)
    rows = []
    for n in names + ['mean']:
        ov = np.mean([data[m]['ov'] for m in names], axis=0) if n == 'mean' else data[n]['ov']
        m = ov.mean(axis=0)
        rows.append([n] + [f'{m[j]:.2f} ({m[j] / N_USED * 100:.1f}%)' for j in range(4)] + [f'{m[4]:.2f}'])
    rows.append(['independent', f'{ind2:.2f} ({ind2 / N_USED * 100:.1f}%)', f'{ind2:.2f}', f'{ind2:.2f}', f'{ind3:.3f}', f'{indu:.2f}'])
    P(md(['data set', '|S1 and S2| of 8', '|S2 and S3|', '|S1 and S3|', '|S1 and S2 and S3|', '|S1 or S2 or S3| (distinct over 3 passes, of 24)'], rows))
    P('\nPer core layer (mean of the data sets): |S1&S2|, |S2&S3|, |S1&S3|, union of 3\n')
    ovm = np.mean([data[m]['ov'] for m in names], axis=0)
    P(md(['layer', 'S1&S2', 'S2&S3', 'S1&S3', 'union3'], [[CORE0 + k] + [f'{ovm[k, j]:.2f}' for j in (0, 1, 2, 4)] for k in range(12)]))

    P('\n## (c) popularity skew: share of the selections taken by the 16 most used experts of a layer (uniform: 12.5% + sampling noise)\n')
    rows = []
    lt = B['layer_expert_type']
    for l in range(24):
        visits_l = 3 if CORE0 <= l < CORE1 else 1
        r = [l, lt[str(l)]]
        for n in names:
            r.append(f'{data[n]["skew"][0][l] * 100:.1f}% / {data[n]["skew"][1][l] * 100:.1f}%')
        rows.append(r)
    nulls = []
    for n in names:
        nt_tot = sum(data[n]['nt'])
        nulls.append(f'{null_share(nt_tot, rng) * 100:.1f}% / {null_share(nt_tot * 3, rng) * 100:.1f}%')
    rows.append(['null (random routing, same token count; 1 visit / 3 passes pooled)', ''] + nulls)
    P(md(['layer', 'experts'] + [f'{n}: per visit / pooled over passes' for n in names], rows))
    allsh = np.mean([data[n]['skew'][0] for n in names], axis=0)
    P(f'\nmean per-visit top-16 share over the 24 layers (mean of data sets): {allsh.mean() * 100:.1f}%, min {allsh.min() * 100:.1f}% (L{allsh.argmin()}), max {allsh.max() * 100:.1f}% (L{allsh.argmax()})')

    P('\n## (d) expert bytes of a T-token verify step (MB = 10^6 B; 44 NVFP4 visits + 4 Q8_0 visits; per expert NVFP4 %d B, Q8_0 %d B)\n' % (nv_b[0], q8_b[0]))
    rows = []
    e1 = None
    Etot = {}
    for i, T in enumerate(TS):
        for n in names + ['mean', 'indep']:
            D = D_mean if n == 'mean' else (np.full((48, len(TS)), 0.0) + np.array([indep(t) for t in TS]) if n == 'indep' else D_all[n])
            Etot[(n, T)] = float(D[:, i] @ bpe)
    e1 = Etot[('mean', 1)]
    for i, T in enumerate(TS):
        Dm = D_mean[:, i]
        nvmb, q8mb = float(Dm[is_nv] @ bpe[is_nv]) / 1e6, float(Dm[~is_nv] @ bpe[~is_nv]) / 1e6
        r = [T, f'{nvmb:.1f}', f'{q8mb:.1f}', f'{Etot[("mean", T)] / 1e6:.1f}', f'{Etot[("mean", T)] / e1:.2f}',
             f'{Etot[("mean", T)] / e1 / T:.3f}', f'{Etot[("indep", T)] / 1e6:.1f}', f'{Etot[("indep", T)] / e1:.2f}']
        r += [f'{Etot[(n, T)] / e1:.2f}' for n in names]
        rows.append(r)
    P(md(['T', 'NVFP4 visits MB', 'Q8_0 visits MB', 'experts MB (mean of data sets)', 'x one-token step', 'x one-token step per token',
          'independent MB', 'independent x'] + [f'{n} x' for n in names], rows))

    P('\n## Break-even of speculative decoding on bandwidth grounds\n')
    P('Plain decoding reads B1 = N + E(1) per token. A speculative step reads S(T) = N + E(T) + Dr (N: non-expert bytes incl. lm_head, read once for the whole\n'
      'T-token verify batch; E(T): measured expert bytes of T consecutive tokens; Dr: drafter bytes, %.1f MB + optionally one lm_head pass) and yields L tokens\n'
      '(L in [1, T]: accepted drafts + 1). Speculation wins on bandwidth iff S(T)/L < B1, i.e. L > L*(T) = S(T)/B1. `max x` = T*B1/S(T) is the speed-up if EVERY draft\n'
      'is accepted (L = T): an upper bound, not an estimate.\n' % args.drafter_mb)
    for scen, label in (('C', 'engine plan C (F32->bf16 at load, lm_head 200058 rows)'), ('A', 'GGUF as stored A (F32 router/controllers, lm_head 201024 rows)')):
        N = B['non_expert_read'][scen]
        head = B['lm_head_bytes'][scen]
        B1 = N + e1
        P(f'\n### {label}: N = {N / 1e6:.1f} MB, lm_head = {head / 1e6:.1f} MB, E(1) = {e1 / 1e6:.1f} MB, B1 = {B1 / 1e6:.1f} MB\n')
        for with_head in (True, False):
            Dr = args.drafter_mb * 1e6 + (head if with_head else 0.0)
            rows = []
            for i, T in enumerate(TS):
                if T < 2:
                    continue
                r = [T, f'{(N + Etot[("mean", T)]) / 1e6:.0f}', f'{(N + Etot[("mean", T)] + Dr) / 1e6:.0f}']
                r += [f'{(N + Etot[("indep", T)] + Dr) / B1:.2f}'] + [f'{(N + Etot[(n, T)] + Dr) / B1:.2f}' for n in names]
                r += [f'{(N + Etot[("mean", T)] + Dr) / B1:.2f}', f'{T * B1 / (N + Etot[("mean", T)] + Dr):.2f}']
                rows.append(r)
            P(f'drafter bytes Dr = {Dr / 1e6:.1f} MB ({"drafter + one lm_head pass" if with_head else "drafter only, no lm_head pass"}):\n')
            P(md(['T', 'verify step N+E(T) MB', 'step S(T) MB incl. drafter', 'L* if routing were independent'] + [f'L* {n}' for n in names] +
                 ['L* mean', 'max x at L=T (mean)'], rows))
            P('')
    text = '\n'.join(out)
    print(text)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
