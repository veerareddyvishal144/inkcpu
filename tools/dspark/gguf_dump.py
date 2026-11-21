#!/usr/bin/env python3
"""Minimal GGUF v3 header dump (KV + tensor table), no dependencies. Usage: gguf_dump.py file.gguf"""
import struct, sys
T = {0:('B',1),1:('b',1),2:('H',2),3:('h',2),4:('I',4),5:('i',4),6:('f',4),7:('?',1),10:('Q',8),11:('q',8),12:('d',8)}
GT = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',30:'BF16',39:'MXFP4',40:'NVFP4'}
f = open(sys.argv[1], 'rb')
def rd(fmt):
    n = struct.calcsize(fmt); return struct.unpack('<' + fmt, f.read(n))[0]
def rs():
    n = rd('Q'); return f.read(n).decode('utf-8', 'replace')
def rv(t):
    if t == 8: return rs()
    if t == 9:
        et = rd('I'); n = rd('Q'); out = [rv(et) for _ in range(n)]; return out
    return rd(T[t][0])
assert f.read(4) == b'GGUF'
ver = rd('I'); nt = rd('Q'); nkv = rd('Q')
print('# version', ver, 'tensors', nt, 'kv', nkv)
for _ in range(nkv):
    k = rs(); t = rd('I'); v = rv(t)
    if isinstance(v, list) and len(v) > 24: v = str(v[:12]) + '...(%d items)' % len(v)
    if isinstance(v, str) and len(v) > 200: v = v[:200] + '...'
    print('KV', k, '=', v)
tot = 0
for _ in range(nt):
    name = rs(); nd = rd('I'); ne = [rd('Q') for _ in range(nd)]; ty = rd('I'); off = rd('Q')
    print('T', name, ne, GT.get(ty, ty))
