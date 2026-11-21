"""Parse Inkling 2 Mini output into OpenAI structures (reasoning_content / content / tool_calls), incrementally.

Output grammar (generation of one assistant turn, see tokenizer_config.json `response_template` of the checkpoint):

    message  := <|message_model|> [NAME] <|content_...|> BODY <|end_message|>
    turn     := message* <|content_model_end_sampling|>          (the stop token, 200006)

    <|content_thinking|>                      -> reasoning_content   (segments are concatenated, no stripping)
    <|content_text|> | <|content_tool_error|> | <|content_invoke_tool_text|>
                                              -> content             (segments are concatenated, no stripping)
    <|content_invoke_tool_json|>              -> tool call: header NAME + body {"name":NAME,"args":{...}}

The `<|message_model|>NAME` header is optional (open_pattern in the checkpoint), so a bare `<|content_text|>` is also a
valid start. The parser is a small state machine fed with token ids (`feed_token`, special tokens are atomic, so a partial
special token can never be emitted) or with text (`feed_text`, which holds back a possible partial `<|...|>` marker).
It yields events; `result()` returns the accumulated ParsedMessage. Streaming and non-streaming share this code path.
"""
from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .tokenizer import InkTokenizer, Utf8Stream

SEG_KINDS = {
    "<|content_thinking|>": "thinking",
    "<|content_text|>": "text",
    "<|content_tool_error|>": "text",
    "<|content_invoke_tool_text|>": "text",
    "<|content_invoke_tool_json|>": "tool",
}
MSG_START = "<|message_model|>"
MSG_END = "<|end_message|>"
TURN_END = "<|content_model_end_sampling|>"
_SPECIAL_RE = re.compile(r"(<\|[A-Za-z0-9_]+\|>)")
_MAX_MARKER = 48  # longest special-token string is < 48 chars; used for the text-mode hold-back


@dataclass
class Event:
    kind: str            # reasoning | content | tool_start | tool_args | tool_end
    text: str = ""
    index: int = 0
    call_id: str = ""
    name: str = ""


@dataclass
class ParsedMessage:
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finished: bool = False                      # saw <|content_model_end_sampling|> (or finish() was called)
    errors: List[str] = field(default_factory=list)

    def to_openai(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": "assistant"}
        msg["content"] = self.content if (self.content or not self.tool_calls) else None
        if msg["content"] is None and not self.tool_calls:
            msg["content"] = ""
        if self.reasoning_content:
            msg["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


def new_call_id() -> str:
    return "call_" + secrets.token_hex(12)


class StreamParser:
    IDLE, HEADER, SEG, DONE = range(4)

    def __init__(self, tokenizer: Optional[InkTokenizer] = None, id_factory: Callable[[], str] = new_call_id):
        self.tok = tokenizer
        self._id_factory = id_factory
        self._state = self.IDLE
        self._header = ""
        self._kind = ""
        self._seg_name = ""
        self._tool_buf: List[str] = []
        self._tool_index = -1
        self._tool_id = ""
        self._tool_started = False
        self._utf8 = Utf8Stream()
        self._pending = ""  # text-mode hold-back of a possible partial marker
        self._reasoning: List[str] = []
        self._content: List[str] = []
        self._calls: List[Dict[str, Any]] = []
        self._errors: List[str] = []
        self.finished = False

    # ---------------------------------------------------------------- public feeds
    def feed_token(self, tid: int) -> List[Event]:
        tok = self.tok
        if tok.is_special(tid):
            ev = self._flush_utf8()
            ev += self._special(tok.special_id_to_str[tid])
            return ev
        return self._text(self._utf8.feed(tok.token_bytes(tid)))

    def feed_text(self, text: str) -> List[Event]:
        """Text-mode feed; special markers may be split across calls (a possible partial marker is held back)."""
        text = self._pending + text
        self._pending = ""
        out: List[Event] = []
        pieces = _SPECIAL_RE.split(text)
        for i, p in enumerate(pieces):
            if not p:
                continue
            if i % 2 == 1 and self._is_known_special(p):
                out += self._special(p)
                continue
            if i == len(pieces) - 1:
                cut = p.rfind("<")
                if cut != -1 and len(p) - cut < _MAX_MARKER and "|>" not in p[cut:]:
                    self._pending = p[cut:]
                    p = p[:cut]
            if p:
                out += self._text(p)
        return out

    def finish(self) -> List[Event]:
        """End of generation: flush held text and close an unterminated segment."""
        out: List[Event] = []
        if self._pending:
            out += self._text(self._pending)
            self._pending = ""
        out += self._flush_utf8()
        if self._state == self.SEG:
            out += self._close_segment(terminated=False)
        self._state = self.DONE
        return out

    @property
    def in_reasoning(self) -> bool:
        return self._state == self.SEG and self._kind == "thinking"

    def result(self) -> ParsedMessage:
        r = "".join(self._reasoning)
        c = "".join(self._content)
        return ParsedMessage(content=c if c else None, reasoning_content=r if r else None,
                             tool_calls=list(self._calls), finished=self.finished, errors=list(self._errors))

    # ---------------------------------------------------------------- internals
    def _is_known_special(self, s: str) -> bool:
        if s in SEG_KINDS or s in (MSG_START, MSG_END, TURN_END):
            return True
        return self.tok is not None and s in self.tok.special_str_to_id

    def _flush_utf8(self) -> List[Event]:
        t = self._utf8.flush()
        return self._text(t) if t else []

    def _text(self, t: str) -> List[Event]:
        if not t or self._state == self.DONE:
            return []
        if self._state == self.HEADER:
            self._header += t
            return []
        if self._state == self.SEG:
            if self._kind == "thinking":
                self._reasoning.append(t)
                return [Event("reasoning", t)]
            if self._kind == "text":
                self._content.append(t)
                return [Event("content", t)]
            self._tool_buf.append(t)
            return []
        # IDLE: free text outside any message header; lenient: treat as content
        self._content.append(t)
        return [Event("content", t)]

    def _special(self, name: str) -> List[Event]:
        out: List[Event] = []
        if self._state == self.DONE:
            return out
        if name == MSG_START:
            if self._state == self.SEG:
                out += self._close_segment(terminated=False)
            self._state = self.HEADER
            self._header = ""
        elif name in SEG_KINDS:
            if self._state == self.SEG:
                out += self._close_segment(terminated=False)
            self._seg_name = self._header.strip() if self._state == self.HEADER else ""
            self._header = ""
            self._kind = SEG_KINDS[name]
            self._state = self.SEG
            if self._kind == "tool":
                self._tool_buf = []
                self._tool_started = False
                self._tool_index += 1
                self._tool_id = self._id_factory()
                if self._seg_name:  # name known from the header: announce the call right away
                    self._tool_started = True
                    out.append(Event("tool_start", index=self._tool_index, call_id=self._tool_id, name=self._seg_name))
        elif name == MSG_END:
            if self._state == self.SEG:
                out += self._close_segment(terminated=True)
            self._state = self.IDLE
            self._header = ""
        elif name == TURN_END:
            if self._state == self.SEG:
                out += self._close_segment(terminated=False)
            self._state = self.DONE
            self.finished = True
        else:
            self._errors.append("unexpected special token %s ignored" % name)
        return out

    def _close_segment(self, terminated: bool) -> List[Event]:
        out: List[Event] = []
        if self._kind == "tool":
            raw = "".join(self._tool_buf)
            name = self._seg_name
            args_text = raw
            try:
                obj = json.loads(raw)
                if not isinstance(obj, dict):
                    raise ValueError("tool call body is not an object")
                if not name:
                    name = obj.get("name") or ""
                args = obj.get("args", obj.get("arguments", obj.get("parameters", {})))
                if isinstance(args, str):  # tolerate a JSON string
                    try:
                        args = json.loads(args)
                    except ValueError:
                        pass
                args_text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, separators=(",", ":"))
            except ValueError as e:
                self._errors.append("invalid tool call JSON (%s)%s" % (e, "" if terminated else ", truncated"))
            if not self._tool_started:
                out.append(Event("tool_start", index=self._tool_index, call_id=self._tool_id, name=name))
            out.append(Event("tool_args", text=args_text, index=self._tool_index))
            out.append(Event("tool_end", index=self._tool_index))
            self._calls.append({"id": self._tool_id, "type": "function",
                                "function": {"name": name, "arguments": args_text}})
        self._kind = ""
        self._state = self.IDLE
        return out


# ---------------------------------------------------------------------------- convenience wrappers
def parse_ids(tokenizer: InkTokenizer, ids, id_factory: Callable[[], str] = new_call_id) -> ParsedMessage:
    p = StreamParser(tokenizer, id_factory)
    for t in ids:
        p.feed_token(t)
    p.finish()
    return p.result()


def parse_text(text: str, tokenizer: Optional[InkTokenizer] = None,
               id_factory: Callable[[], str] = new_call_id) -> ParsedMessage:
    p = StreamParser(tokenizer, id_factory)
    p.feed_text(text)
    p.finish()
    return p.result()


class StopMatcher:
    """Stop-string filter for the content channel. push(text) -> (text safe to emit, matched). Holds back the
    longest suffix that could still grow into a stop string; nothing after a match is ever emitted."""

    def __init__(self, stops: List[str]):
        self.stops = [s for s in stops if s]
        self._hold = ""
        self.matched = False
        self._maxlen = max((len(s) for s in self.stops), default=0)

    def push(self, text: str):
        if not self.stops:
            return text, False
        if self.matched:
            return "", True
        buf = self._hold + text
        best = -1
        for s in self.stops:
            i = buf.find(s)
            if i != -1 and (best == -1 or i < best):
                best = i
        if best != -1:
            self.matched = True
            self._hold = ""
            return buf[:best], True
        keep = 0
        for s in self.stops:  # longest suffix of buf that is a proper prefix of some stop string
            for k in range(min(len(s) - 1, len(buf)), 0, -1):
                if buf.endswith(s[:k]):
                    keep = max(keep, k)
                    break
        emit, self._hold = buf[:len(buf) - keep], buf[len(buf) - keep:]
        return emit, False

    def flush(self) -> str:
        h, self._hold = self._hold, ""
        return h
