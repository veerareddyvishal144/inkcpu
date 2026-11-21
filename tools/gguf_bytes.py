#!/data/engine/.venv/bin/python
"""Exact bytes-per-token table of the local GGUF (agent E).

usage: gguf_bytes.py [--gguf PATH] [--json OUT.json] [--markdown OUT.md]

Reads only the GGUF tensor table (plus the F32 tensors, to prove they are exactly representable in bf16) and applies the
decode rule of llama.cpp for inkling2_loop at batch 1 (every visit re-reads its weights from DRAM):
  * 48 visits: layers 0-5 once, core layers 6-17 three times (passes 1-3), layers 18-23 once
  * routed experts: 8 of 128 experts per visit (bytes * 8/128; per-expert .scale entries too)
  * loop controllers and loop_msg_query: passes 2 and 3 only (2 visits per core layer)
  * embedding: ONE row per token; lm_head (output.weight): all 201024 padded rows, once
Scenarios: A = tensors as stored in the GGUF; B = A with every exactly-bf16-representable F32 tensor stored as bf16;
           C = B with lm_head trimmed to the 200058 real vocabulary rows.   MB = 10^6 bytes.
"""
import argparse
import json
import re
import sys

sys.path.insert(0, '/data/inkling-mini/llama.cpp/gguf-py')
import numpy as np  # noqa: E402
from gguf import GGUFReader  # noqa: E402

N_LAYERS, CORE0, CORE1, PASSES = 24, 6, 18, 3
N_EXPERT, N_USED = 128, 8
VOCAB_PADDED, VOCAB_REAL = 201024, 200058
EXPERT_W = ('ffn_gate_up_exps.weight', 'ffn_down_exps.weight')
EXPERT_S = ('ffn_gate_up_exps.scale', 'ffn_down_exps.scale')


def visits(layer):
    return PASSES if CORE0 <= layer < CORE1 else 1


def class_visits(cls, layer):
    return PASSES - 1 if cls.startswith('loop_') else visits(layer)


def group_of(cls, ttype):
    if cls in ('attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight', 'attn_gate.weight', 'attn_rel.weight'):
        return 'attention projections (Q8_0)'
    if cls in ('attn_sink_proj.weight', 'attn_k_conv_proj.weight', 'attn_v_conv_proj.weight'):
        return 'attention BF16 (sink_proj, dyn-conv proj)'
    if cls in EXPERT_W:
        return 'routed experts NVFP4 (L0-19)' if ttype == 'NVFP4' else 'routed experts Q8_0 (L20-23)'
    if cls in EXPERT_S:
        return 'expert .scale F32 (selected entries)'
    if cls in ('ffn_gate_inp.weight', 'exp_probs_b.bias'):
        return 'router (ffn_gate_inp F32 + exp_probs_b F32)'
    if cls.startswith('loop_'):
        return 'loop controllers + msg queries (F32, passes 2-3)'
    if cls == 'output.weight':
        return 'lm_head (output.weight, Q8_0)'
    if cls == 'token_embd.weight':
        return 'embedding row (token_embd.weight, 1 row)'
    return 'small F32: convs, norms, clips, rel_bias, queries'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gguf', default='/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf')
    ap.add_argument('--json', default='/data/inkling-mini/engine/tools/bytes_per_token.json')
    ap.add_argument('--markdown', default=None)
    args = ap.parse_args()

    r = GGUFReader(args.gguf)
    rows = []  # one dict per tensor
    for t in r.tensors:
        m = re.match(r'blk\.(\d+)\.(.+)$', t.name)
        layer, cls = (int(m.group(1)), m.group(2)) if m else (-1, t.name)
        ttype = t.tensor_type.name
        nbytes = int(t.n_bytes)
        nel = int(t.n_elements)
        bf16_exact = None
        if ttype == 'F32':
            u = np.ascontiguousarray(t.data).reshape(-1).view(np.uint32)
            bf16_exact = not bool((u & 0xFFFF).any())
        rows.append(dict(name=t.name, layer=layer, cls=cls, type=ttype, shape=[int(x) for x in t.shape], nel=nel, nbytes=nbytes,
                         bf16_exact=bf16_exact))

    def read_bytes(row, scen):
        """bytes read per visit of this tensor under scenario A/B/C"""
        b = row['nbytes']
        if row['type'] == 'F32' and scen in 'BC' and row['bf16_exact']:
            b = row['nel'] * 2
        cls = row['cls']
        if cls in EXPERT_W or cls in EXPERT_S:
            return b * N_USED / N_EXPERT
        if row['name'] == 'token_embd.weight':
            return b / VOCAB_PADDED  # one row
        if row['name'] == 'output.weight' and scen == 'C':
            return b / VOCAB_PADDED * VOCAB_REAL
        return b

    # ---- per class table
    classes = {}
    for row in rows:
        key = (row['cls'], row['type'])  # experts of layers 0-19 (NVFP4) and 20-23 (Q8_0) are separate rows
        cls = row['cls']
        c = classes.setdefault(key, dict(cls=cls, types=set(), n=0, file_bytes=0, visits=0, read={'A': 0.0, 'B': 0.0, 'C': 0.0},
                                         exact=[0, 0]))
        c['types'].add(row['type'])
        c['n'] += 1
        c['file_bytes'] += row['nbytes']
        v = 1 if row['layer'] < 0 else class_visits(cls, row['layer'])
        c['visits'] += v
        for s in 'ABC':
            c['read'][s] += read_bytes(row, s) * v
        if row['bf16_exact'] is not None:
            c['exact'][0] += int(row['bf16_exact'])
            c['exact'][1] += 1
    for c in classes.values():
        c['types'] = sorted(c['types'])

    total = {s: sum(c['read'][s] for c in classes.values()) for s in 'ABC'}
    expert_total = {s: sum(c['read'][s] for c in classes.values() if c['cls'] in EXPERT_W) for s in 'ABC'}
    scale_total = {s: sum(c['read'][s] for c in classes.values() if c['cls'] in EXPERT_S) for s in 'ABC'}

    # ---- per-expert bytes (incl. the 2 x 4 B scale entries) and layer types
    per_expert = {}
    for ty in ('NVFP4', 'Q8_0'):
        sizes = set()
        for l in range(N_LAYERS):
            b = 0
            for row in rows:
                if row['layer'] == l and (row['cls'] in EXPERT_W) and row['type'] == ty:
                    b += row['nbytes'] // N_EXPERT
                if row['layer'] == l and row['cls'] in EXPERT_S and any(x['layer'] == l and x['cls'] in EXPERT_W and x['type'] == ty for x in rows):
                    b += row['nbytes'] // N_EXPERT
            if b:
                sizes.add(b)
        per_expert[ty] = sorted(sizes)
    layer_type = {}
    for row in rows:
        if row['cls'] == 'ffn_gate_up_exps.weight':
            layer_type[row['layer']] = row['type']
    visit_layer = list(range(CORE0)) + list(range(CORE0, CORE1)) * PASSES + list(range(CORE1, N_LAYERS))
    assert len(visit_layer) == 48

    lm = next(x for x in rows if x['name'] == 'output.weight')
    summary = dict(
        mb=1e6, gguf=args.gguf, n_tensors=len(rows), tensor_bytes=sum(x['nbytes'] for x in rows),
        total_read={s: total[s] for s in 'ABC'},
        expert_weight_read={s: expert_total[s] for s in 'ABC'},
        expert_scale_read={s: scale_total[s] for s in 'ABC'},
        non_expert_read={s: total[s] - expert_total[s] - scale_total[s] for s in 'ABC'},
        lm_head_bytes={'A': lm['nbytes'], 'B': lm['nbytes'], 'C': lm['nbytes'] / VOCAB_PADDED * VOCAB_REAL},
        per_expert_bytes={k: v for k, v in per_expert.items()},
        layer_expert_type={str(k): v for k, v in sorted(layer_type.items())},
        visit_layer=visit_layer,
        f32_total_bytes=sum(x['nbytes'] for x in rows if x['type'] == 'F32'),
        f32_not_bf16_exact=[x['name'] for x in rows if x['type'] == 'F32' and not x['bf16_exact']],
        classes=[dict(cls=c['cls'], types=c['types'], n=c['n'], file_bytes=c['file_bytes'], visits=c['visits'], read=c['read'],
                      bf16_exact=c['exact']) for c in classes.values()],
    )
    with open(args.json, 'w') as f:
        json.dump(summary, f, indent=1)

    # ---- markdown
    out = []
    mb = lambda x: f'{x / 1e6:,.2f}'
    out.append('| tensor class (blk.N.<class>) | type | tensors | file MB | visits | read MB/token A (as stored) | B (F32->bf16) | C (B + lm_head 200058) | share of A |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    for c in sorted(classes.values(), key=lambda c: -c['read']['A']):
        t = '/'.join(c['types'])
        if 'F32' in c['types'] and c['exact'][1]:
            t += f" ({c['exact'][0]}/{c['exact'][1]} bf16-exact)"
        out.append(f"| {c['cls']} | {t} | {c['n']} | {mb(c['file_bytes'])} | {c['visits']} | {mb(c['read']['A'])} | {mb(c['read']['B'])} | {mb(c['read']['C'])} | {100 * c['read']['A'] / total['A']:.2f}% |")
    out.append(f"| **total** | | {len(rows)} | {mb(summary['tensor_bytes'])} | | **{mb(total['A'])}** | **{mb(total['B'])}** | **{mb(total['C'])}** | 100% |")
    out.append('')
    groups = {}
    for c in classes.values():
        g = group_of(c['cls'], c['types'][0])
        gg = groups.setdefault(g, {'A': 0.0, 'B': 0.0, 'C': 0.0})
        for s in 'ABC':
            gg[s] += c['read'][s]
    out.append('| group | MB/token A | B | C | share of A |')
    out.append('|---|---|---|---|---|')
    for g, v in sorted(groups.items(), key=lambda kv: -kv[1]['A']):
        out.append(f"| {g} | {mb(v['A'])} | {mb(v['B'])} | {mb(v['C'])} | {100 * v['A'] / total['A']:.2f}% |")
    out.append(f"| **total** | **{mb(total['A'])}** | **{mb(total['B'])}** | **{mb(total['C'])}** | 100% |")
    out.append('')
    out.append(f"per expert (gate_up + down + 2 scale entries): NVFP4 {per_expert['NVFP4']} B, Q8_0 {per_expert['Q8_0']} B; "
               f"expert layers: " + ', '.join(f'{k}:{v}' for k, v in sorted(layer_type.items())))
    not_exact = sorted({re.sub(r'^blk\.\d+\.', '', n) for n in summary['f32_not_bf16_exact']})
    out.append(f"F32 classes with at least one tensor that is NOT exactly bf16-representable (kept F32 in B and C): {not_exact or 'none'}; "
               f"F32 bytes in file {mb(summary['f32_total_bytes'])} MB")
    out.append(f"sum of tensor bytes {summary['tensor_bytes']:,} (file size minus header/alignment gives the rest)")
    text = '\n'.join(out)
    print(text)
    if args.markdown:
        with open(args.markdown, 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
