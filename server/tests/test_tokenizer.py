import helpers
from helpers import fixture, tok


def test_parity_with_llama_cpp_tokenize():
    fx = fixture("parity.json")["tokenize"]
    assert len(fx) >= 10
    t = tok()
    for r in fx:
        assert t.encode(r["text"], parse_special=True) == r["special"], r["text"]
        assert t.encode(r["text"], parse_special=False) == r["plain"], r["text"]


def test_special_ids_and_stop():
    t = tok()
    assert t.n_vocab == 200058 and t.n_vocab_padded == 201024
    assert t.stop_ids == frozenset({200006})
    assert t.special_id_to_str[200006] == "<|content_model_end_sampling|>"
    assert t.special_id_to_str[200010] == "<|end_message|>" and 200010 not in t.stop_ids
    assert t.special_id_to_str[199999] == "<|endoftext|>" and t.special_id_to_str[200000] == "<|message_user|>"
    assert t.encode("<|end_message|>") == [200010]
    assert t.encode("<|content_invoke_tool_json|>") == [200049]


def test_decode_roundtrip_and_hf_decode():
    t = tok()
    for r in fixture("parity.json")["tokenize"]:
        ids = t.encode(r["text"], parse_special=False)
        assert t.decode(ids) == r["text"], r["text"]
        ids2 = t.encode(r["text"], parse_special=True)
        assert t.decode(ids2) == r["text"]  # specials decode to their strings
        assert t.decode(ids2) == t._tok.decode(ids2, skip_special_tokens=False) or "�" in r["text"]


def test_utf8_stream_never_splits_a_codepoint():
    from server.tokenizer import Utf8Stream
    data = "a\U0001F600b世界é".encode("utf-8")
    u = Utf8Stream()
    out = [u.feed(bytes([b])) for b in data]
    assert "".join(out) + u.flush() == "a\U0001F600b世界é"
    assert all("�" not in o for o in out)
    d = tok().stream_decoder()
    ids = tok().encode("héllo \U0001F680 世界", parse_special=False)
    assert "".join(d.push(i) for i in ids) + d.flush() == "héllo \U0001F680 世界"


if __name__ == "__main__":
    raise SystemExit(helpers.run_module(globals()))
