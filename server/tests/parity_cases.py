"""Shared case definitions for the tokenizer / chat-template parity checks (llama-server --jinja is the oracle)."""
import json

TOKENIZE_STRINGS = [
    "Hello world",
    "Hello world<|message_user|>",
    "<|message_user|><|content_text|>Hi<|end_message|>",
    "The quick brown fox jumps over the lazy dog.",
    "   leading spaces and\ttabs\n\nnewlines\r\n  trailing  ",
    "héllo wörld, 世界 こんにちは 안녕하세요 \U0001F600\U0001F680",
    "def foo(x):\n    return x**2  # comment\n\tif x: pass\n",
    "12345 1234567890 3.14159 -42 0x1F 1,000,000",
    '{"name":"get_weather","args":{"city":"Paris","units":["c","f"]}}',
    '<|content_invoke_tool_json|>{"a":1}<|end_message|><|content_model_end_sampling|>',
    "don't I'll we're THEY'VE it's",
    "https://example.com/path?q=1&r=2#frag \\\\ \"quoted\" 'single' (paren) [br] {cb}",
    "​zero width nbsp ÀÉÎ ǅ ǆ Ωμέγα مرحبا שלום",
    "<|endoftext|><|message_model|><|content_thinking|>x<|end_message|><|message_tool|>t<|content_xml|>",
    "a",
    "CamelCaseIdentifier snake_case_name HTTPServerError ABCdef",
    "<|not_a_token|> <| |> |> <|",
]

_W = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
_T = {"type": "function", "function": {
    "name": "get_time", "description": "Get the time in a city (é世界).",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}


def u(c): return {"role": "user", "content": c}
def s(c): return {"role": "system", "content": c}
def a(c, **kw): return dict({"role": "assistant", "content": c}, **kw)
def tc(i, name, args): return {"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


TEMPLATE_CASES = {
    "plain": dict(messages=[u("Hello, how are you?")]),
    "system_user": dict(messages=[s("You are a terse assistant."), u("What is 2+2?")]),
    "multi_turn": dict(messages=[u("Hi"), a("Hello! How can I help?"), u("Tell me a joke")]),
    "multi_turn_system": dict(messages=[s("Be brief."), u("Hi"), a("Hello!"), u("Bye"), a("Goodbye!"), u("One more thing")]),
    "reasoning_history": dict(messages=[u("What is 2+2?"), a("4", reasoning_content="2+2 is 4."), u("And 3+3?")]),
    "tools": dict(messages=[u("What is the weather in Paris?")], tools=[_W]),
    "two_tools": dict(messages=[s("Use tools."), u("Weather in Paris and time in Tokyo?")], tools=[_W, _T]),
    "tool_result": dict(messages=[u("What is the weather in Paris?"),
                                  a(None, tool_calls=[tc("call_1", "get_weather", {"city": "Paris"})]),
                                  {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "18C and sunny"}],
                        tools=[_W]),
    "tool_result_by_id": dict(messages=[u("What is the weather in Paris?"),
                                        a(None, tool_calls=[tc("call_1", "get_weather", {"city": "Paris"})]),
                                        {"role": "tool", "tool_call_id": "call_1", "content": "18C and sunny"}],
                              tools=[_W]),
    "tool_roundtrip": dict(messages=[u("Weather in Paris?"),
                                     a(None, tool_calls=[tc("call_1", "get_weather", {"city": "Paris"})]),
                                     {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "18C and sunny"},
                                     a("It is 18C and sunny in Paris."), u("And in Rome?")], tools=[_W]),
    "parallel_calls": dict(messages=[u("Do both."),
                                     a("Calling both.", tool_calls=[
                                         tc("c1", "get_weather", {"units": "celsius", "city": "Paris", "extra": {"z": 1, "y": [1, 2, {"b": 2, "a": 1}]}}),
                                         tc("c2", "get_time", {"city": "Tokyo"})]),
                                     {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "18C"},
                                     {"role": "tool", "tool_call_id": "c2", "name": "get_time", "content": "12:00"}],
                           tools=[_W, _T]),
    "call_with_reasoning": dict(messages=[u("Weather in Paris?"),
                                          a(None, reasoning_content="Need the weather tool.", tool_calls=[tc("c1", "get_weather", {"city": "Paris"})]),
                                          {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "18C"}],
                                tools=[_W]),
    "unicode": dict(messages=[u("héllo 世界 \U0001F600 \"quotes\" and\nnewline <b>html</b> & stuff")], tools=[_T]),
    "parts": dict(messages=[{"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]}]),
    "effort_low": dict(messages=[u("Hi")], reasoning_effort="low"),
    "effort_num": dict(messages=[u("Hi")], reasoning_effort=0.2),
    "effort_zero": dict(messages=[u("Hi")], reasoning_effort=0),
    "effort_max": dict(messages=[s("sys"), u("Hi")], reasoning_effort="max"),
}
