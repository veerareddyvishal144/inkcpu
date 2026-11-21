import json

import helpers
from helpers import fixture, tok

from server.parser import StopMatcher, StreamParser, parse_ids, parse_text

SP = "<|message_model|>"
END = "<|end_message|>"
STOP = "<|content_model_end_sampling|>"


def _ids_text(s):
    return tok().encode(s, parse_special=True)


def _real(name):
    c = fixture("real_outputs.json")[name]
    return c, c["forced_prefix_ids"] + c["generated_ids"]


def _collect(parser_feed):
    r = {"reasoning": "", "content": "", "calls": {}}
    order = []
    for e in parser_feed:
        order.append(e.kind)
        if e.kind == "reasoning":
            r["reasoning"] += e.text
        elif e.kind == "content":
            r["content"] += e.text
        elif e.kind == "tool_start":
            r["calls"][e.index] = {"id": e.call_id, "name": e.name, "args": ""}
        elif e.kind == "tool_args":
            r["calls"][e.index]["args"] += e.text
    return r, order


def test_real_output_plain_with_reasoning():
    c, ids = _real("plain_natural")
    m = parse_ids(tok(), ids)
    assert m.reasoning_content.startswith("The user asks to reply with just the word: pong")
    assert m.content == "pong" and not m.tool_calls and m.finished and not m.errors
    assert (m.reasoning_content + m.content) == c["server_content"]  # llama-server's own concatenation
    assert m.to_openai()["reasoning_content"] == m.reasoning_content


def test_real_output_forced_thinking_then_text():
    _, ids = _real("plain_forced")
    m = parse_ids(tok(), ids)
    assert m.reasoning_content == "The user wants me to say pong." and m.content == "pong"
    c, ids = _real("tool_result_forced")
    m = parse_ids(tok(), ids)
    assert m.content == "The weather in Paris is 18°C and sunny." and not m.tool_calls


def test_real_output_tool_call():
    _, ids = _real("tool_forced")
    m = parse_ids(tok(), ids)
    assert m.content is None and len(m.tool_calls) == 1
    tc = m.tool_calls[0]
    assert tc["type"] == "function" and tc["id"].startswith("call_") and tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Paris"}
    o = m.to_openai()
    assert o["content"] is None and o["tool_calls"] == m.tool_calls


def test_real_output_parallel_tool_calls():
    _, ids = _real("parallel_tools_forced")
    m = parse_ids(tok(), ids)
    assert [t["function"]["name"] for t in m.tool_calls] == ["get_weather", "get_time"]
    assert [json.loads(t["function"]["arguments"]) for t in m.tool_calls] == [{"city": "Paris"}, {"city": "Tokyo"}]
    assert len({t["id"] for t in m.tool_calls}) == 2


def test_streaming_events_match_final_and_never_leak_markers():
    for name in ("plain_natural", "tool_forced", "parallel_tools_forced", "tool_result_forced"):
        _, ids = _real(name)
        p = StreamParser(tok())
        evs = []
        for i in ids:
            evs += p.feed_token(i)
        evs += p.finish()
        r, order = _collect(evs)
        m = p.result()
        assert r["reasoning"] == (m.reasoning_content or "") and r["content"] == (m.content or "")
        assert [v["name"] for _, v in sorted(r["calls"].items())] == [t["function"]["name"] for t in m.tool_calls]
        assert [v["args"] for _, v in sorted(r["calls"].items())] == [t["function"]["arguments"] for t in m.tool_calls]
        for e in evs:
            assert "<|" not in e.text and "|>" not in e.text, e
        # a tool call is announced (start) before its arguments, and closed after
        for k, kind in enumerate(order):
            if kind == "tool_args":
                assert "tool_start" in order[:k] and "tool_end" in order[k:]


def test_text_mode_equals_token_mode_for_every_split():
    _, ids = _real("parallel_tools_forced")
    text = tok().decode(ids)
    ref = parse_ids(tok(), ids, id_factory=lambda: "x")
    for cut in range(0, len(text) + 1, 7):  # two chunks, cut anywhere (including inside <|...|> markers)
        p = StreamParser(tok(), id_factory=lambda: "x")
        evs = p.feed_text(text[:cut]) + p.feed_text(text[cut:]) + p.finish()
        for e in evs:
            assert "<|" not in e.text
        m = p.result()
        assert (m.content, m.reasoning_content, m.tool_calls) == (ref.content, ref.reasoning_content, ref.tool_calls), cut
    p = StreamParser(tok())  # 1 char at a time
    evs = []
    for ch in text:
        evs += p.feed_text(ch)
    evs += p.finish()
    assert [t["function"]["name"] for t in p.result().tool_calls] == ["get_weather", "get_time"]
    assert all("<|" not in e.text for e in evs)


def test_synthetic_variants():
    t = tok()
    m = parse_text("<|content_text|>hi<|end_message|>", t)  # header-less start is valid
    assert m.content == "hi" and not m.finished
    m = parse_text(SP + "<|content_text|>a" + END + SP + "<|content_text|>b" + END + STOP, t)
    assert m.content == "ab" and m.finished  # segments concatenated, no stripping
    m = parse_text(SP + "<|content_thinking|>x " + END + SP + "<|content_thinking|>y" + END + SP + "<|content_text|>z" + END + STOP, t)
    assert m.reasoning_content == "x y" and m.content == "z"
    m = parse_text(SP + "<|content_thinking|>still thinking", t)  # truncated by max_tokens
    assert m.reasoning_content == "still thinking" and m.content is None and not m.finished
    m = parse_text(SP + "<|content_tool_error|>boom" + END + STOP, t)
    assert m.content == "boom"
    m = parse_text(SP + "<|content_invoke_tool_json|>" + '{"name":"f","args":{"a":1}}' + END + STOP, t)  # empty header name
    assert m.tool_calls[0]["function"]["name"] == "f" and m.tool_calls[0]["function"]["arguments"] == '{"a":1}'
    m = parse_text(SP + "f<|content_invoke_tool_json|>" + '{"name":"f","args":{"a":' + STOP, t)  # truncated JSON
    assert m.tool_calls[0]["function"]["name"] == "f" and m.errors and m.tool_calls[0]["function"]["arguments"].startswith('{"name"')
    m = parse_text(SP + "<|content_text|>x" + END + END + "<|message_user|>" + STOP, t)  # stray markers are ignored
    assert m.content == "x" and m.errors
    m = parse_text("plain text without any marker", t)  # lenient: free text is content
    assert m.content == "plain text without any marker"
    m = parse_text(SP + "<|content_text|>a<|b|>c" + END, t)  # unknown <|..|> text stays text
    assert m.content == "a<|b|>c"


def test_utf8_split_across_tokens_is_held_back():
    class FakeTok:
        special_id_to_str = {1: SP, 2: "<|content_text|>", 3: END}
        special_str_to_id = {v: k for k, v in special_id_to_str.items()}
        def is_special(self, i): return i in self.special_id_to_str
        def token_bytes(self, i): return {10: b"\xf0\x9f", 11: b"\x98\x80", 12: b"ok"}[i]
    p = StreamParser(FakeTok())
    evs = []
    for i in (1, 2, 10, 11, 12, 3):
        evs += p.feed_token(i)
    texts = [e.text for e in evs if e.kind == "content"]
    assert texts == ["\U0001F600", "ok"], texts


def test_stop_matcher():
    sm = StopMatcher(["STOP"])
    assert sm.push("abST") == ("ab", False)
    assert sm.push("OPxyz") == ("", True) and sm.push("more") == ("", True)
    sm = StopMatcher(["\n\n", "END"])
    out = ""
    for ch in "a\nb\nc EN d":
        t, hit = sm.push(ch)
        out += t
        assert not hit
    assert out + sm.flush() == "a\nb\nc EN d"
    sm = StopMatcher(["xyz"])
    t, hit = sm.push("1xyz2")
    assert (t, hit) == ("1", True) and sm.flush() == ""
    assert StopMatcher([]).push("abc") == ("abc", False)


if __name__ == "__main__":
    raise SystemExit(helpers.run_module(globals()))
