#!/usr/bin/env python3
# Tokenises tools/dspark/prompts/*.txt with the server's tokenizer.json into tests/spec/prompts/<name>.i32 (int32 LE), the inputs of the spec tests / bench.
# Special tokens in the text (chat markers) are matched by the tokenizer's added-token list. No BOS/EOS is added (the GGUF does not either, docs/MODEL.md 7.13).
import os, sys, numpy as np
from tokenizers import Tokenizer
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tk = Tokenizer.from_file(os.path.join(root, "server/assets/tokenizer.json"))
src = os.path.join(root, "tools/dspark/prompts")
dst = os.path.join(root, "tests/spec/prompts")
os.makedirs(dst, exist_ok=True)
for fn in sorted(os.listdir(src)):
    if not fn.endswith(".txt"):
        continue
    text = open(os.path.join(src, fn), encoding="utf-8").read()
    enc = tk.encode(text, add_special_tokens=False)
    ids = np.array(enc.ids, dtype="<i4")
    ids.tofile(os.path.join(dst, fn[:-4] + ".i32"))
    print("%-22s %4d tokens  first %s" % (fn[:-4], len(ids), ids[:6].tolist()))
