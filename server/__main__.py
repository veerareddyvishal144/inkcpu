"""python -m server --backend mock|llama|ink ...   (run from the engine/ directory)"""
import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m server", description="OpenAI-compatible server for Inkling 2 Mini (inkcpu)")
    ap.add_argument("--backend", choices=["mock", "llama", "ink"], default="mock")
    ap.add_argument("--model", default="/data/inkling-mini/models/Inkling-2-Mini-UD-Q4_K_XL.gguf", help="GGUF path (ink backend)")
    ap.add_argument("--lib", default=None, help="path of libink.so (ink backend; default: INK_LIB or build dirs)")
    ap.add_argument("--llama-url", default="http://127.0.0.1:8001", help="llama.cpp server URL (llama backend)")
    ap.add_argument("--threads", type=int, default=None,
                    help="engine threads = number of CPUs in env INK_CPUS when that is set, else 16")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=8192, help="max tokens per slot")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-queue", type=int, default=8, help="waiting requests beyond busy slots before 429")
    ap.add_argument("--default-max-tokens", type=int, default=8192)
    ap.add_argument("--prefill-chunk", type=int, default=256)
    ap.add_argument("--served-name", default="inkling-2-mini")
    ap.add_argument("--timer-slack-us", type=int, default=5000,
                    help="timer slack of the engine threads (ink backend): the idle pool polls every 20 us, which costs ~30%% of a core "
                         "with the kernel default of 50 us; 0 = leave the default")
    ap.add_argument("--shutdown-grace", type=int, default=10, help="seconds uvicorn waits for open streams after SIGTERM")
    ap.add_argument("--no-spec", action="store_true", help="never use the backend's speculative decoding path")
    ap.add_argument("--drafter", default=os.environ.get("INK_DRAFTER") or None,
                    help="DSpark drafter GGUF (ink backend): enables speculative steps when libink exports ink_spec_* (else a warning)")
    ap.add_argument("--spec", default=None, metavar="auto|off|2..8",
                    help="engine default for requests without a `speculation` field: auto (policy picks the width), off (plain decode), "
                         "or a fixed verify width 2..8; default: the engine's (env INK_SPEC, else auto)")
    ap.add_argument("--no-keep-off-engine-cpus", action="store_true",
                    help="do not move the HTTP/Python threads off the CPUs listed in INK_CPUS")
    ap.add_argument("--request-log", default=None, help="JSONL request log path (default env INK_REQUEST_LOG, '' disables)")
    ap.add_argument("--log-bodies", action="store_true", help="also log prompts and outputs (privacy!)")
    a = ap.parse_args(argv)

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    from server.app import ServerConfig, create_app
    from server import backends

    cpus = [int(c) for c in os.environ.get("INK_CPUS", "").split(",") if c.strip().isdigit()]
    if a.threads is None:
        a.threads = len(cpus) if cpus else 16
    if a.backend == "ink" and cpus and not a.no_keep_off_engine_cpus and hasattr(os, "sched_setaffinity"):
        # the spin-barrier pool convoys when a Python thread lands on one of its CPUs: keep this process's own threads away
        # (threads created later inherit the mask; the engine re-pins its own pool threads explicitly)
        allowed = set(os.sched_getaffinity(0)) - set(cpus)
        if allowed:
            os.sched_setaffinity(0, allowed)
    if a.backend == "mock":
        be = backends.MockBackend(n_slots=a.slots, max_ctx=a.ctx)
    elif a.backend == "llama":
        be = backends.LlamaServerBackend(a.llama_url, n_slots=1, max_ctx=min(a.ctx, 4096))
    else:
        try:
            spec_mode = backends.InkBackend.parse_spec_mode(a.spec)
        except ValueError as e:
            ap.error("--spec: %s" % e)
        be = backends.InkBackend(a.model, a.lib, a.threads, a.slots, a.ctx, timer_slack_us=a.timer_slack_us, drafter_path=a.drafter,
                                 spec_mode=spec_mode)
    cfg = ServerConfig(served_name=a.served_name, max_queue=a.max_queue, default_max_tokens=a.default_max_tokens,
                       request_log=a.request_log, log_bodies=a.log_bodies, prefill_chunk=a.prefill_chunk, spec=not a.no_spec)
    app = create_app(be, cfg)
    import threading
    import time
    import uvicorn

    def fail_fast():  # a failed model load must kill the process (a supervisor restarts it) instead of serving 503 forever
        sched = app.state.sched
        while not sched.ready.wait(1.0):
            pass
        if sched.load_error:
            print("FATAL: model load failed: %s" % sched.load_error, file=sys.stderr, flush=True)
            time.sleep(1.0)
            os._exit(3)
    threading.Thread(target=fail_fast, daemon=True).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="info", timeout_graceful_shutdown=a.shutdown_grace)


if __name__ == "__main__":
    main()
