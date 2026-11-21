import helpers
from helpers import fixture, tok
from parity_cases import TEMPLATE_CASES


def _tmpl():
    from server import chat_template
    return chat_template.get_template()


def test_matches_llama_server_jinja_text_and_token_ids():
    fx = fixture("parity.json")["template"]
    ct, t = _tmpl(), tok()
    assert len(TEMPLATE_CASES) >= 15
    for name, c in TEMPLATE_CASES.items():
        ref = fx[name]
        assert ref["llama_prompt"] is not None, (name, ref.get("error"))
        mine = ct.render_openai(c["messages"], c.get("tools"), c.get("reasoning_effort"))
        assert mine == ref["llama_prompt"], name
        assert t.encode(mine, parse_special=True) == ref["llama_ids"], name


def test_shape_of_prompt():
    ct = _tmpl()
    p = ct.render_openai([{"role": "user", "content": "Hi"}])
    assert p == "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hi<|end_message|>"
    assert not p.endswith("<|message_model|>")  # no generation prompt: the model emits <|message_model|> itself
    p = ct.render_openai([{"role": "system", "content": "S"}, {"role": "user", "content": "Hi"}])
    assert p.startswith("<|message_system|><|content_text|>S<|end_message|><|message_system|><|content_text|>Thinking")


def test_effort_values_and_errors():
    from server.chat_template import TemplateError
    ct = _tmpl()
    m = [{"role": "user", "content": "Hi"}]
    assert "effort level: 0.2<|" in ct.render_openai(m, None, "low")
    assert "effort level: 0<|" in ct.render_openai(m, None, 0)
    assert "effort level: 0.99<|" in ct.render_openai(m, None, "0.99")
    for bad in (1.0, -0.1, "bogus"):
        try:
            ct.render_openai(m, None, bad)
        except TemplateError:
            continue
        raise AssertionError("expected TemplateError for %r" % (bad,))


def test_openai_arguments_string_is_canonicalised():
    ct = _tmpl()
    call = lambda a: {"id": "c", "type": "function", "function": {"name": "f", "arguments": a}}
    msgs = lambda a: [{"role": "user", "content": "x"}, {"role": "assistant", "content": None, "tool_calls": [call(a)]},
                      {"role": "tool", "tool_call_id": "c", "content": "r"}]
    a = ct.render_openai(msgs('{"b": 1, "a": "z"}'))
    b = ct.render_openai(msgs({"b": 1, "a": "z"}))
    assert a == b and '"args":{"a":"z","b":1}}' in a
    assert "<|message_tool|>f<|content_text|>r<|end_message|>" in a  # tool name resolved from tool_call_id


def test_unsupported_inputs_rejected():
    from server.chat_template import TemplateError
    ct = _tmpl()
    for bad in ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
                [{"role": "wizard", "content": "x"}], []):
        try:
            ct.render_openai(bad)
        except TemplateError:
            continue
        raise AssertionError("expected TemplateError")


if __name__ == "__main__":
    raise SystemExit(helpers.run_module(globals()))
