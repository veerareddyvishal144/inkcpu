#!/usr/bin/env python3
# Tokenises tests/spec/prompts/lookup_*.txt (agent L: copy-heavy editor / tool-call turns for bench_spec) with the server's tokenizer.json into <name>.i32 (int32 LE),
# like make_prompt_ids.py does for tools/dspark/prompts. No BOS/EOS (docs/MODEL.md 7.13).
import glob, os, numpy as np
from tokenizers import Tokenizer
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tk = Tokenizer.from_file(os.path.join(root, "server/assets/tokenizer.json"))
for fn in sorted(glob.glob(os.path.join(root, "tests/spec/prompts/lookup_*.txt"))):
    text = open(fn, encoding="utf-8").read()
    ids = np.array(tk.encode(text, add_special_tokens=False).ids, dtype="<i4")
    ids.tofile(fn[:-4] + ".i32")
    print("%-22s %5d tokens  first %s last %s" % (os.path.basename(fn)[:-4], len(ids), ids[:4].tolist(), ids[-4:].tolist()))
