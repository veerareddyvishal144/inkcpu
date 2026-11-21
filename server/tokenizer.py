"""Inkling 2 Mini tokenizer, built from the ORIGINAL checkpoint's tokenizer.json with HF `tokenizers`.

Facts (checkpoint tokenizer.json + generation_config.json, cross-checked against llama.cpp /tokenize):
  * byte-level BPE (GPT-2 style, `ignore_merges`), 199998 ordinary tokens (ids 0..199997) followed by 60 added
    special tokens (ids 199998..200057); the embedding is padded to 201024 rows, 200058 are real (ink_n_vocab).
  * the ONLY stop token is 200006 `<|content_model_end_sampling|>` (generation_config eos_token_id).
    `<|end_message|>` (200010) closes a message segment and is NOT a stop token; pad is 199999 `<|endoftext|>`.
  * special-token strings inside text are parsed to their ids (HF default and llama.cpp `parse_special=true`);
    `encode(..., parse_special=False)` encodes them as ordinary text instead.

Assets live in server/assets (tokenizer.json, chat_template.jinja, generation_config.json, tokenizer_config.json), downloaded
once from https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/<file> (see assets/README or server/README.md).
"""
from __future__ import annotations

import codecs
import json
import os
from typing import Iterable, List, Optional

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
HF_BASE = "https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/"
ASSET_FILES = ("tokenizer.json", "chat_template.jinja", "generation_config.json", "tokenizer_config.json")

VOCAB_PADDED = 201024
VOCAB_REAL = 200058
PAD_ID = 199999            # <|endoftext|>
STOP_ID = 200006           # <|content_model_end_sampling|>
STOP_IDS = frozenset({STOP_ID})
END_MESSAGE_ID = 200010    # <|end_message|>, NOT a stop token


def ensure_assets(download: bool = True) -> None:
    """Make sure the checkpoint's tokenizer/template files are in server/assets (download once if missing)."""
    missing = [f for f in ASSET_FILES if not os.path.exists(os.path.join(ASSETS_DIR, f))]
    if not missing:
        return
    if not download:
        raise FileNotFoundError("missing assets %s in %s" % (missing, ASSETS_DIR))
    import urllib.request
    os.makedirs(ASSETS_DIR, exist_ok=True)
    for f in missing:
        tmp = os.path.join(ASSETS_DIR, f + ".part")
        with urllib.request.urlopen(HF_BASE + f, timeout=120) as r, open(tmp, "wb") as out:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                out.write(b)
        os.replace(tmp, os.path.join(ASSETS_DIR, f))


def _bytes_to_unicode():
    """GPT-2 byte <-> printable-unicode table (the byte-level BPE alphabet)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class Utf8Stream:
    """Incremental UTF-8 decoder: feed token bytes, get only complete characters (never a split code point)."""

    def __init__(self):
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, data: bytes) -> str:
        return self._dec.decode(data, final=False)

    def flush(self) -> str:
        return self._dec.decode(b"", final=True)


class InkTokenizer:
    def __init__(self, tokenizer_json: Optional[str] = None):
        from tokenizers import Tokenizer
        path = tokenizer_json or os.path.join(ASSETS_DIR, "tokenizer.json")
        if not os.path.exists(path):
            ensure_assets()
        self.path = path
        self._tok = Tokenizer.from_file(path)
        self._plain = None  # lazily built tokenizer without added tokens (for parse_special=False)
        added = self._tok.get_added_tokens_decoder()
        self.special_id_to_str = {i: a.content for i, a in added.items() if a.special}
        self.special_str_to_id = {s: i for i, s in self.special_id_to_str.items()}
        self.n_vocab = VOCAB_REAL
        self.n_vocab_padded = VOCAB_PADDED
        self.stop_ids = STOP_IDS
        self.pad_id = PAD_ID
        # per-id byte strings (ids >= 200058 do not exist in the real vocabulary)
        b2u = _bytes_to_unicode()
        u2b = {u: b for b, u in b2u.items()}
        vocab = self._tok.get_vocab(with_added_tokens=True)
        tb: List[Optional[bytes]] = [None] * VOCAB_REAL
        for s, i in vocab.items():
            if i >= VOCAB_REAL:
                continue
            if i in self.special_id_to_str:
                tb[i] = s.encode("utf-8")
            else:
                tb[i] = bytes(u2b[c] for c in s)
        self._token_bytes = tb
        # named special ids used by the template / parser
        s = self.special_str_to_id
        self.id_message_user = s["<|message_user|>"]
        self.id_message_model = s["<|message_model|>"]
        self.id_message_system = s["<|message_system|>"]
        self.id_message_tool = s["<|message_tool|>"]
        self.id_content_text = s["<|content_text|>"]
        self.id_content_thinking = s["<|content_thinking|>"]
        self.id_content_invoke_tool_json = s["<|content_invoke_tool_json|>"]
        self.id_end_message = s["<|end_message|>"]
        self.id_stop = s["<|content_model_end_sampling|>"]
        assert self.id_stop == STOP_ID and self.id_end_message == END_MESSAGE_ID

    # ------------------------------------------------------------------ encode / decode
    def encode(self, text: str, parse_special: bool = True) -> List[int]:
        if parse_special:
            return self._tok.encode(text, add_special_tokens=False).ids
        if self._plain is None:
            from tokenizers import Tokenizer
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            d["added_tokens"] = []
            self._plain = Tokenizer.from_str(json.dumps(d))
        return self._plain.encode(text, add_special_tokens=False).ids

    def is_special(self, tid: int) -> bool:
        return tid in self.special_id_to_str

    def token_bytes(self, tid: int) -> bytes:
        if tid < 0 or tid >= VOCAB_REAL or self._token_bytes[tid] is None:
            return b""
        return self._token_bytes[tid]

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        sp = self.special_id_to_str
        parts = []
        for t in ids:
            if skip_special and t in sp:
                continue
            parts.append(self.token_bytes(t))
        return b"".join(parts).decode("utf-8", errors="replace")

    def stream_decoder(self, skip_special: bool = False) -> "TokenStreamDecoder":
        return TokenStreamDecoder(self, skip_special)


class TokenStreamDecoder:
    """push(token_id) -> text that is complete (UTF-8 safe); flush() at the end."""

    def __init__(self, tok: InkTokenizer, skip_special: bool = False):
        self.tok = tok
        self.skip_special = skip_special
        self._u = Utf8Stream()

    def push(self, tid: int) -> str:
        if self.skip_special and self.tok.is_special(tid):
            return ""
        return self._u.feed(self.tok.token_bytes(tid))

    def flush(self) -> str:
        return self._u.flush()


_default: Optional[InkTokenizer] = None


def get_tokenizer() -> InkTokenizer:
    global _default
    if _default is None:
        _default = InkTokenizer()
    return _default
