"""Chat-template rendering for Inkling 2 Mini, identical to the model card / HF `apply_chat_template`.

The template (assets/chat_template.jinja, from the original checkpoint) is rendered with the same Jinja environment HF
transformers uses (ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks, loopcontrols, the `generation` tag and an
HF-style `tojson` that does not escape HTML and accepts sort_keys/separators).

Facts about the template (verified against llama-server --jinja, see tests/test_template_parity.py):
  * there is NO generation prompt: the prompt ends after the last message; the model itself emits `<|message_model|>`.
  * a numeric system message `<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>` (default 0.9) is
    inserted before the first non-system message; `reasoning_effort` accepts a number in [0, 0.99] or one of
    none/minimal/low/medium/high/xhigh/max.
  * tool declarations: `<|message_system|>tool_declare<|content_xml|>[sorted-key JSON specs]<|end_message|>`.
  * tool-call arguments must be a parsed object (OpenAI clients send a JSON string): `normalize_messages` canonicalizes.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jinja2
from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .tokenizer import ASSETS_DIR, ensure_assets

DEFAULT_REASONING_EFFORT = 0.9
EFFORT_NAMES = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}


class TemplateError(ValueError):
    """Raised for requests the template cannot render (maps to HTTP 400)."""


class _GenerationTag(Extension):
    """HF's `{% generation %}...{% endgeneration %}` tag (assistant-token masking); here it only renders its body."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("_body"), [], [], body).set_lineno(lineno)

    def _body(self, caller):
        return caller()


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # same override HF transformers installs: no HTML escaping, options exposed
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def _build_env() -> jinja2.Environment:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[_GenerationTag, jinja2.ext.loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    return env


class ChatTemplate:
    def __init__(self, template_path: Optional[str] = None):
        path = template_path or os.path.join(ASSETS_DIR, "chat_template.jinja")
        if not os.path.exists(path):
            ensure_assets()
        with open(path, "r", encoding="utf-8") as f:
            self.source = f.read()
        self._tmpl = _build_env().from_string(self.source)

    def render(self, messages: Sequence[Dict[str, Any]], tools: Optional[Sequence[Dict[str, Any]]] = None,
               reasoning_effort: Any = None) -> str:
        """Render canonicalized messages. Use `normalize_messages` first for OpenAI-style input."""
        kwargs: Dict[str, Any] = {}
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        try:
            return self._tmpl.render(messages=list(messages), tools=list(tools) if tools else None,
                                     add_generation_prompt=True, **kwargs)
        except jinja2.exceptions.TemplateError as e:
            raise TemplateError(str(e)) from e

    def render_openai(self, messages, tools=None, reasoning_effort=None) -> str:
        msgs = normalize_messages(messages)
        return self.render(msgs, normalize_tools(tools), parse_reasoning_effort(reasoning_effort))


def parse_reasoning_effort(v: Any) -> Any:
    """None -> None (template default 0.9); numeric strings -> float; names stay names; validates the range."""
    if v is None:
        return None
    if isinstance(v, bool):
        raise TemplateError("reasoning_effort must be a number or one of %s" % ", ".join(EFFORT_NAMES))
    if isinstance(v, (int, float)):
        f = float(v)
    elif isinstance(v, str):
        key = v.strip()
        if key in EFFORT_NAMES:
            return key
        try:
            f = float(key)
        except ValueError:
            raise TemplateError("Unknown reasoning_effort: %s (use a number in [0, 0.99] or one of %s)"
                                % (v, ", ".join(EFFORT_NAMES)))
    else:
        raise TemplateError("reasoning_effort must be a number or string")
    if f < 0.0 or f > 0.99:
        raise TemplateError("reasoning_effort must be in [0.0, 0.99]")
    return f


def normalize_tools(tools: Optional[Sequence[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict):
            raise TemplateError("each tool must be an object")
        fn = t.get("function", t)
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise TemplateError("each tool needs function.name")
        out.append(t)
    return out


def _text_of_parts(parts: Sequence[Any]) -> List[Any]:
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            ty = p.get("type", "text")
            if ty in ("image", "input_image", "image_url", "audio", "input_audio", "audio_url"):
                raise TemplateError("image/audio content is not supported by this server (text only)")
            if ty not in ("text", "input_text"):
                raise TemplateError("unsupported content part type: %s" % ty)
            out.append({"type": "text", "text": p.get("text", "") if isinstance(p.get("text", ""), str) else ""})
        else:
            raise TemplateError("invalid content part")
    return out


def normalize_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OpenAI chat messages -> what the template accepts (arguments parsed to objects, developer->system, ...)."""
    if not isinstance(messages, (list, tuple)) or not messages:
        raise TemplateError("messages must be a non-empty list")
    out: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict) or "role" not in m:
            raise TemplateError("each message needs a role")
        m = dict(m)
        if m["role"] == "developer":
            m["role"] = "system"
        if m["role"] == "function":  # legacy
            m["role"] = "tool"
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = _text_of_parts(c)
        elif c is None:
            if m["role"] != "assistant":
                m["content"] = ""
        elif not isinstance(c, str):
            m["content"] = json.dumps(c, ensure_ascii=False)
        rc = m.get("reasoning_content")
        if rc is None and isinstance(m.get("reasoning"), str):
            rc = m["reasoning"]  # tolerate the OpenRouter/vLLM alias
        if rc is not None:
            m["reasoning_content"] = rc
        tcs = m.get("tool_calls")
        if tcs:
            new = []
            for tc in tcs:
                tc = copy.deepcopy(tc)
                fn = tc.get("function")
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                    raise TemplateError("tool call function name must be a string")
                a = fn.get("arguments")
                if isinstance(a, str):
                    if a.strip() == "":
                        a = {}
                    else:
                        try:
                            a = json.loads(a)
                        except ValueError:
                            raise TemplateError("tool call arguments must be valid JSON")
                if a is None:
                    a = {}
                if not isinstance(a, dict):
                    raise TemplateError("tool call arguments must be a JSON object")
                fn["arguments"] = a
                new.append(tc)
            m["tool_calls"] = new
        else:
            m.pop("tool_calls", None)
        out.append(m)
    return out


_default: Optional[ChatTemplate] = None


def get_template() -> ChatTemplate:
    global _default
    if _default is None:
        _default = ChatTemplate()
    return _default


def build_prompt(tokenizer, messages, tools=None, reasoning_effort=None,
                 forced_prefix: str = "") -> Tuple[str, List[int]]:
    """Render + tokenize (special-token strings parsed). `forced_prefix` is appended to the prompt (used for a forced
    tool_choice); returns (text, token ids)."""
    text = get_template().render_openai(messages, tools, reasoning_effort) + forced_prefix
    return text, tokenizer.encode(text, parse_special=True)
