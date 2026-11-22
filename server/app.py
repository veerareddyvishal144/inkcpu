"""OpenAI-compatible FastAPI server for Inkling 2 Mini.

Endpoints: POST /v1/chat/completions, POST /v1/completions, GET /v1/models, GET /health, GET /metrics.
The engine runs in ONE dedicated thread (scheduler.Scheduler); this module only validates requests, renders the chat
template, tokenizes, submits a GenRequest and turns the engine's events into JSON / SSE.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from . import chat_template as ct
from .backends import Backend, InkBackend
from .metrics import Metrics, RequestLog
from .sampling import SamplingParams
from .scheduler import Finish, GenRequest, QueueFull, Scheduler
from .tokenizer import InkTokenizer


@dataclass
class ServerConfig:
    served_name: str = "inkling-2-mini"
    api_key: Optional[str] = None          # None -> env INK_API_KEY; empty/None = open server
    max_queue: int = 8                      # waiting requests beyond the busy slots before 429
    default_max_tokens: int = 8192          # chat default when max_tokens is absent (clamped to the context)
    request_log: Optional[str] = None       # JSONL path; None -> env INK_REQUEST_LOG; "" disables
    log_bodies: bool = False                # also log prompts/outputs (privacy: off by default)
    prefill_chunk: int = 256
    default_reasoning_effort: Optional[str] = None   # used when a request carries no reasoning_effort (None = template default 0.9)
    spec: bool = True                       # use the backend's speculative steps when it has them (penalties always use plain decode)


class ApiError(Exception):
    def __init__(self, status: int, message: str, type_: str = "invalid_request_error", code: Optional[str] = None,
                 param: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.status, self.message, self.type, self.code, self.param, self.headers = status, message, type_, code, param, headers or {}

    def body(self) -> Dict[str, Any]:
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


def _sse(obj: Any) -> bytes:
    return ("data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n\n").encode("utf-8")


def _num(body: Dict[str, Any], key: str, default, kind=float, lo=None, hi=None):
    v = body.get(key)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ApiError(400, "%s must be a number" % key, param=key)
    v = kind(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ApiError(400, "%s out of range" % key, param=key)
    return v


def sampling_from_body(body: Dict[str, Any]) -> SamplingParams:
    rp = body.get("repetition_penalty", body.get("repeat_penalty"))
    sp = SamplingParams(
        temperature=_num(body, "temperature", 1.0, float, 0.0, 10.0),
        top_p=_num(body, "top_p", 1.0, float, 1e-6, 1.0),
        top_k=_num(body, "top_k", 0, int, 0),
        min_p=_num(body, "min_p", 0.0, float, 0.0, 1.0),
        seed=_num(body, "seed", None, int, 0),
        frequency_penalty=_num(body, "frequency_penalty", 0.0, float, -2.0, 2.0),
        presence_penalty=_num(body, "presence_penalty", 0.0, float, -2.0, 2.0),
        repetition_penalty=float(rp) if isinstance(rp, (int, float)) and not isinstance(rp, bool) and rp > 0 else 1.0)
    try:
        sp.validate()
    except ValueError as e:
        raise ApiError(400, str(e))
    return sp


def stops_from_body(body: Dict[str, Any]) -> List[str]:
    s = body.get("stop")
    if s is None:
        return []
    if isinstance(s, str):
        s = [s]
    if not isinstance(s, list) or not all(isinstance(x, str) for x in s) or len(s) > 16:
        raise ApiError(400, "stop must be a string or a list of up to 16 strings", param="stop")
    return [x for x in s if x]



def _no_think(effort) -> bool:
    """True for reasoning_effort "none" / 0 / "0" / 0.0: answer directly, no thinking segment."""
    if effort is None:
        return False
    if isinstance(effort, str):
        if effort.strip().lower() == "none":
            return True
        try:
            return float(effort) == 0.0
        except ValueError:
            return False
    return isinstance(effort, (int, float)) and not isinstance(effort, bool) and float(effort) == 0.0


def create_app(backend: Backend, cfg: Optional[ServerConfig] = None, tokenizer: Optional[InkTokenizer] = None) -> FastAPI:
    cfg = cfg or ServerConfig()
    api_key = cfg.api_key if cfg.api_key is not None else os.environ.get("INK_API_KEY", "")
    log_path = cfg.request_log if cfg.request_log is not None else os.environ.get("INK_REQUEST_LOG", "")
    tok = tokenizer or InkTokenizer()
    metrics = Metrics()
    reqlog = RequestLog(log_path or None, cfg.log_bodies)

    def on_finish(req: GenRequest, fin: Finish) -> None:
        status = 200 if fin.finish_reason not in ("cancelled", "error") else (499 if fin.finish_reason == "cancelled" else 500)
        metrics.observe(fin, req.meta.get("endpoint", "chat"))
        rec = {"id": req.id, "endpoint": req.meta.get("endpoint"), "model": req.meta.get("model"), "stream": req.meta.get("stream"),
               "client": req.meta.get("client"), "status": status, "finish_reason": fin.finish_reason,
               "prompt_tokens": fin.prompt_tokens, "cached_tokens": fin.cached_tokens,
               "completion_tokens": fin.completion_tokens, "reasoning_tokens": fin.reasoning_tokens,
               "queue_ms": round(fin.queue_s * 1e3, 1), "prefill_ms": round(fin.prefill_s * 1e3, 1),
               "decode_ms": round(fin.decode_s * 1e3, 1), "engine_decode_ms": round(fin.decode_engine_s * 1e3, 1),
               "sample_ms": round(fin.sample_s * 1e3, 1), "spec_steps": fin.spec_steps or None,
               "spec_drafted": fin.spec_drafted or None, "spec_accepted": fin.spec_accepted or None, "spec_mode": fin.spec_mode,
               "ttft_ms": round(fin.ttft_s * 1e3, 1),
               "total_ms": round(fin.total_s * 1e3, 1),
               "decode_tps": round(fin.completion_tokens / fin.decode_s, 2) if fin.decode_s > 0 and fin.completion_tokens > 1 else None,
               "temperature": req.sampling.temperature, "top_p": req.sampling.top_p, "top_k": req.sampling.top_k,
               "seed": req.sampling.seed, "n_tools": req.meta.get("n_tools", 0), "errors": fin.errors or None}
        if cfg.log_bodies:
            rec["request"] = req.meta.get("body")
            m = fin.message
            rec["response"] = {"content": m.content, "reasoning_content": m.reasoning_content, "tool_calls": m.tool_calls}
        reqlog.write(rec)

    sched = Scheduler(backend, tok, max_queue=cfg.max_queue, prefill_chunk=cfg.prefill_chunk, on_finish=on_finish, spec=cfg.spec)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sched.start(wait=False)
        yield
        sched.shutdown()
        reqlog.close()

    app = FastAPI(title="inkcpu OpenAI-compatible server", lifespan=lifespan)
    app.state.sched, app.state.metrics, app.state.tokenizer, app.state.cfg = sched, metrics, tok, cfg

    # ------------------------------------------------------------------ errors / auth helpers
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        metrics.inc_request(request.url.path, exc.status)
        reqlog.write({"endpoint": request.url.path, "status": exc.status, "error": exc.message,
                      "client": request.client.host if request.client else None})
        return JSONResponse(exc.body(), status_code=exc.status, headers=exc.headers)

    def check_auth(request: Request) -> None:
        if not api_key:
            return
        h = request.headers.get("authorization", "")
        got = h[7:].strip() if h.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
        if not got or not hmac.compare_digest(got.encode(), api_key.encode()):
            raise ApiError(401, "Incorrect API key provided.", "invalid_request_error", "invalid_api_key")

    async def read_json(request: Request) -> Dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            raise ApiError(400, "request body is not valid JSON")
        if not isinstance(body, dict):
            raise ApiError(400, "request body must be a JSON object")
        return body

    def submit(gen: GenRequest) -> None:
        try:
            sched.submit(gen)
        except QueueFull as e:
            raise ApiError(429, "server busy: all slots are in use and the request queue is full, retry later",
                           "rate_limit_error", "queue_full", headers={"Retry-After": str(e.retry_after)})
        except RuntimeError as e:
            raise ApiError(503, str(e), "server_error", "engine_unavailable", headers={"Retry-After": "5"})

    def usage_of(fin: Finish) -> Dict[str, Any]:
        return {"prompt_tokens": fin.prompt_tokens, "completion_tokens": fin.completion_tokens,
                "total_tokens": fin.prompt_tokens + fin.completion_tokens,
                "prompt_tokens_details": {"cached_tokens": fin.cached_tokens},
                "completion_tokens_details": {"reasoning_tokens": fin.reasoning_tokens}}

    def timings_of(fin: Finish) -> Dict[str, Any]:
        """Non-standard `timings` object; field names follow llama.cpp's (prompt_n = NEW prompt tokens, cache_n = reused)."""
        new = fin.prompt_tokens - fin.cached_tokens
        out = {"queue_ms": round(fin.queue_s * 1e3, 1), "cache_n": fin.cached_tokens, "prompt_n": new,
               "prompt_ms": round(fin.prefill_s * 1e3, 1),
               "prompt_per_second": round(new / fin.prefill_s, 2) if fin.prefill_s > 0 and new > 0 else None,
               "predicted_n": fin.completion_tokens, "predicted_ms": round(fin.decode_s * 1e3, 1),
               "predicted_per_second": round(fin.completion_tokens / fin.decode_s, 2) if fin.decode_s > 0 else None,
               "engine_ms": round(fin.decode_engine_s * 1e3, 1), "sample_ms": round(fin.sample_s * 1e3, 1),
               "ttft_ms": round(fin.ttft_s * 1e3, 1), "total_ms": round(fin.total_s * 1e3, 1)}
        if fin.spec_steps:
            out.update(spec_steps=fin.spec_steps, spec_drafted=fin.spec_drafted, spec_accepted=fin.spec_accepted,
                       spec_tokens_per_step=round(fin.spec_emitted / fin.spec_steps, 3),
                       spec_accept_rate=round(fin.spec_accepted / fin.spec_drafted, 3) if fin.spec_drafted else None,
                       spec_mode=fin.spec_mode, spec_synced=fin.spec_synced,
                       spec_t_hist={str(k): v for k, v in sorted(fin.spec_t_hist.items())})
        return out

    def make_gen(request: Request, body: Dict[str, Any], prompt_ids: List[int], max_tokens: int, mode: str, stream: bool,
                 endpoint: str, loop, q: "asyncio.Queue", **kw) -> GenRequest:
        def sink(item):
            if item[0] == "ev" and not stream:
                return  # non-streaming responses are built from the final record only
            try:
                loop.call_soon_threadsafe(q.put_nowait, item)
            except RuntimeError:
                pass
        sp = sampling_from_body(body)
        try:
            spec_mode = InkBackend.parse_spec_mode(body.get("speculation"))     # non-standard: "auto" | "off" | 2..8 (verify width)
        except ValueError as e:
            raise ApiError(400, str(e), param="speculation")
        if body.get("logit_bias"):                                             # not applied by any path today; never speculate on it
            spec_mode = 0
        meta = {"endpoint": endpoint, "model": body.get("model") or cfg.served_name, "stream": stream,
                "client": request.client.host if request.client else None, "n_tools": len(body.get("tools") or [])}
        if cfg.log_bodies:
            meta["body"] = body
        return GenRequest(id=("chatcmpl-" if mode == "chat" else "cmpl-") + uuid.uuid4().hex[:24], prompt_ids=prompt_ids,
                          sampling=sp, max_tokens=max_tokens, sink=sink, mode=mode, stop=stops_from_body(body),
                          meta=meta, spec_mode=spec_mode, **kw)

    def clamp_max_tokens(body: Dict[str, Any], n_prompt: int, default: int) -> int:
        mt = body.get("max_completion_tokens", body.get("max_tokens"))
        if mt is None:
            mt = default
        if isinstance(mt, bool) or not isinstance(mt, int) or mt < 1:
            raise ApiError(400, "max_tokens must be a positive integer", param="max_tokens")
        room = backend.max_ctx - n_prompt
        if room < 1:
            raise ApiError(400, "This model's maximum context length is %d tokens, but the prompt has %d tokens."
                           % (backend.max_ctx, n_prompt), code="context_length_exceeded", param="messages")
        return min(mt, room)

    # ------------------------------------------------------------------ simple endpoints
    @app.get("/health")
    async def health():
        if sched.load_error:
            return JSONResponse({"status": "error", "error": sched.load_error}, status_code=503)
        if not sched.ready.is_set():
            return JSONResponse({"status": "loading"}, status_code=503)
        return {"status": "ok", "backend": backend.name, "slots": backend.n_slots, "ctx": backend.max_ctx,
                "active": sched.n_active(), "queued": sched.n_waiting()}

    @app.get("/slots")
    async def slots_ep():
        return {"slots": sched.slot_snapshot(), "ctx": backend.max_ctx, "queued": sched.n_waiting(), "active": sched.n_active()}

    @app.get("/metrics")
    async def metrics_ep():
        return PlainTextResponse(metrics.render(sched, backend.name), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models(request: Request):
        check_auth(request)
        metrics.inc_request("/v1/models", 200)
        return {"object": "list", "data": [{"id": cfg.served_name, "object": "model", "created": int(metrics.started),
                                            "owned_by": "inkcpu", "context_length": backend.max_ctx}]}

    # ------------------------------------------------------------------ chat completions
    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        check_auth(request)
        body = await read_json(request)
        if body.get("n") not in (None, 1):
            raise ApiError(400, "n > 1 is not supported", param="n")
        if body.get("logprobs") or body.get("top_logprobs"):
            raise ApiError(400, "logprobs are not supported", param="logprobs")
        rf = body.get("response_format")
        if isinstance(rf, dict) and rf.get("type") not in (None, "text"):
            raise ApiError(400, "response_format %s is not supported (no constrained decoding)" % rf.get("type"), param="response_format")
        stream = bool(body.get("stream"))
        try:
            tools = ct.normalize_tools(body.get("tools"))
            tool_choice = body.get("tool_choice")
            forced = None
            if tools:
                if tool_choice == "none":
                    tools = None
                elif isinstance(tool_choice, dict):
                    forced = (tool_choice.get("function") or {}).get("name")
                    names = [(t.get("function", t)).get("name") for t in tools]
                    if forced not in names:
                        raise ApiError(400, "tool_choice names an unknown function", param="tool_choice")
                elif tool_choice not in (None, "auto", "required"):
                    raise ApiError(400, "unsupported tool_choice", param="tool_choice")
            effort = body.get("reasoning_effort")
            if effort is None and isinstance(body.get("reasoning"), dict):
                effort = body["reasoning"].get("effort")
            if effort is None and isinstance(body.get("chat_template_kwargs"), dict):
                effort = body["chat_template_kwargs"].get("reasoning_effort")
            if effort is None and cfg.default_reasoning_effort is not None:
                effort = cfg.default_reasoning_effort   # server default (--default-reasoning-effort), e.g. "low" for editor clients that omit it
            text = ct.get_template().render_openai(body.get("messages"), tools, effort)
        except ct.TemplateError as e:
            raise ApiError(400, str(e), param="messages")
        prompt_ids = tok.encode(text, parse_special=True)
        prefix_ids: List[int] = []
        if forced:  # forced function call: prefill the call header, the model continues with the arguments
            prefix = '<|message_model|>%s<|content_invoke_tool_json|>{"name":%s,"args":' % (forced, json.dumps(forced))
            prefix_ids = tok.encode(prefix, parse_special=True)
            prompt_ids = prompt_ids + prefix_ids
        elif _no_think(effort) and not tools:
            # reasoning_effort "none" (or 0): skip the thinking message entirely by prefilling the answer header. The model's own
            # effort 0 still thinks for ~150 tokens, which eats small budgets (editor titles, Tab). Only without tools: a forced text
            # header would stop the model from opening with a tool call.
            prefix_ids = tok.encode("<|message_model|><|content_text|>", parse_special=True)
            prompt_ids = prompt_ids + prefix_ids
        max_tokens = clamp_max_tokens(body, len(prompt_ids), cfg.default_max_tokens)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        gen = make_gen(request, body, prompt_ids, max_tokens, "chat", stream, "/v1/chat/completions", loop, q,
                       parser_prefix_ids=prefix_ids, stop_after_tool=bool(forced))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        model = body.get("model") or cfg.served_name
        created = int(time.time())
        base = {"id": gen.id, "created": created, "model": model, "system_fingerprint": "inkcpu-" + backend.name}
        submit(gen)
        metrics.inc_request("/v1/chat/completions", 200)
        if not stream:
            fin = await _wait_done(request, q, gen)
            msg = fin.message.to_openai()
            out = dict(base, object="chat.completion",
                       choices=[{"index": 0, "message": msg, "finish_reason": fin.finish_reason, "logprobs": None}],
                       usage=usage_of(fin), timings=timings_of(fin))
            if body.get("return_token_ids"):   # non-standard (vLLM-style): exact token-level comparisons in tests
                out["prompt_token_ids"] = prompt_ids
                out["choices"][0]["token_ids"] = fin.token_ids
            return JSONResponse(out)

        async def gen_sse():
            def chunk(delta, finish=None):
                return _sse(dict(base, object="chat.completion.chunk",
                                 choices=[{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}]))
            try:
                yield chunk({"role": "assistant", "content": ""})
                while True:
                    item = await q.get()
                    if item[0] == "ev":
                        for e in item[1]:
                            if e.kind == "reasoning":
                                yield chunk({"reasoning_content": e.text})
                            elif e.kind == "content":
                                yield chunk({"content": e.text})
                            elif e.kind == "tool_start":
                                yield chunk({"tool_calls": [{"index": e.index, "id": e.call_id, "type": "function",
                                                             "function": {"name": e.name, "arguments": ""}}]})
                            elif e.kind == "tool_args":
                                yield chunk({"tool_calls": [{"index": e.index, "function": {"arguments": e.text}}]})
                    elif item[0] == "done":
                        fin = item[1]
                        yield chunk({}, fin.finish_reason)
                        if include_usage:
                            yield _sse(dict(base, object="chat.completion.chunk", choices=[], usage=usage_of(fin)))
                        yield b"data: [DONE]\n\n"
                        return
                    else:  # error
                        yield _sse({"error": {"message": item[1], "type": "server_error", "code": item[2]}})
                        yield b"data: [DONE]\n\n"
                        return
            finally:
                gen.cancel.set()  # client went away (or finished): free the slot

        return StreamingResponse(gen_sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def _wait_done(request: Request, q: "asyncio.Queue", gen: GenRequest) -> Finish:
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        gen.cancel.set()
                        raise ApiError(499, "client closed request")
                    continue
                if item[0] == "done":
                    return item[1]
                if item[0] == "error":
                    raise ApiError(item[2], item[1], "server_error" if item[2] >= 500 else "invalid_request_error")
        except asyncio.CancelledError:
            gen.cancel.set()
            raise

    # ------------------------------------------------------------------ raw completions
    @app.post("/v1/completions")
    async def completions(request: Request):
        check_auth(request)
        body = await read_json(request)
        if body.get("n") not in (None, 1) or body.get("best_of") not in (None, 1):
            raise ApiError(400, "n / best_of > 1 is not supported")
        if body.get("logprobs") or body.get("echo") or body.get("suffix"):
            raise ApiError(400, "logprobs / echo / suffix are not supported")
        prompt = body.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], (str, list)):
            prompt = prompt[0]
        if isinstance(prompt, str):
            prompt_ids = tok.encode(prompt, parse_special=True)
        elif isinstance(prompt, list) and prompt and all(isinstance(t, int) and not isinstance(t, bool) for t in prompt):
            if any(t < 0 or t >= backend.n_vocab for t in prompt):
                raise ApiError(400, "token id out of range", param="prompt")
            prompt_ids = list(prompt)
        else:
            raise ApiError(400, "prompt must be a string or a list of token ids", param="prompt")
        if not prompt_ids:
            raise ApiError(400, "prompt is empty", param="prompt")
        stream = bool(body.get("stream"))
        max_tokens = clamp_max_tokens(body, len(prompt_ids), 16)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        gen = make_gen(request, body, prompt_ids, max_tokens, "raw", stream, "/v1/completions", loop, q,
                       skip_special=bool(body.get("skip_special_tokens", True)))
        model = body.get("model") or cfg.served_name
        base = {"id": gen.id, "created": int(time.time()), "model": model, "system_fingerprint": "inkcpu-" + backend.name}
        submit(gen)
        metrics.inc_request("/v1/completions", 200)
        if not stream:
            fin = await _wait_done(request, q, gen)
            reason = "stop" if fin.finish_reason == "tool_calls" else fin.finish_reason
            out = dict(base, object="text_completion",
                       choices=[{"text": fin.message.content or "", "index": 0, "logprobs": None, "finish_reason": reason}],
                       usage=usage_of(fin), timings=timings_of(fin))
            if body.get("return_token_ids"):
                out["prompt_token_ids"] = prompt_ids
                out["choices"][0]["token_ids"] = fin.token_ids
            return JSONResponse(out)
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        async def gen_sse():
            def chunk(text, finish=None):
                return _sse(dict(base, object="text_completion",
                                 choices=[{"text": text, "index": 0, "logprobs": None, "finish_reason": finish}]))
            try:
                while True:
                    item = await q.get()
                    if item[0] == "ev":
                        for e in item[1]:
                            if e.kind == "content":
                                yield chunk(e.text)
                    elif item[0] == "done":
                        fin = item[1]
                        yield chunk("", fin.finish_reason)
                        if include_usage:
                            yield _sse(dict(base, object="text_completion", choices=[], usage=usage_of(fin)))
                        yield b"data: [DONE]\n\n"
                        return
                    else:
                        yield _sse({"error": {"message": item[1], "type": "server_error", "code": item[2]}})
                        yield b"data: [DONE]\n\n"
                        return
            finally:
                gen.cancel.set()

        return StreamingResponse(gen_sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
