#!/bin/bash
# inkcpu server control: tools/serve.sh start|stop|restart|status|logs [n]
# Binds to 127.0.0.1 only; reach it from another machine through an SSH tunnel (ssh -L 8000:127.0.0.1:8000 <vm>).
# env overrides: INK_PORT (8000; other ports get their own pid/log files so test instances can coexist), INK_THREADS (16),
#   INK_CPUS (16 physical cores), INK_SLOTS (2), INK_CTX (65536), INK_PREFILL_CHUNK (512), INK_DEFAULT_EFFORT (none = answer
#   without thinking when the client sends no reasoning_effort), INK_SPEC (auto|off|4|8), INK_MODEL_DIR (/data/inkling-mini/models),
#   INK_LIB (build-orch/src/capi/libink.so), INK_NO_DOWNLOAD=1 (fail instead of fetching missing model files), INK_API_KEY (optional bearer key)
cd /data/inkling-mini/engine || exit 1
PY=/data/engine/.venv/bin/python
RUN=/data/inkling-mini/engine/run
PORT=${INK_PORT:-8000}
SFX=""; [ "$PORT" != "8000" ] && SFX="-$PORT"
PIDF=$RUN/server$SFX.pid; LOG=$RUN/server$SFX.log
THREADS=${INK_THREADS:-16}
export INK_CPUS=${INK_CPUS:-2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,0}
MDIR=${INK_MODEL_DIR:-/data/inkling-mini/models}
MODEL=$MDIR/Inkling-2-Mini-UD-Q4_K_XL.gguf
DRAFTER=$MDIR/dspark-Inkling-2-Mini-Q8_0.gguf
LIB=${INK_LIB:-/data/inkling-mini/engine/build-orch/src/capi/libink.so}
SPEC=${INK_SPEC:-auto}
mkdir -p "$RUN"
alive() { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
case "${1:-status}" in
  start)
    if alive; then echo "already running (pid $(cat "$PIDF"))"; exit 0; fi
    if [ ! -s "$MODEL" ] || { [ "$SPEC" != "off" ] && [ ! -s "$DRAFTER" ]; }; then
      if [ "${INK_NO_DOWNLOAD:-0}" = "1" ]; then echo "model files missing in $MDIR and INK_NO_DOWNLOAD=1 (see tools/get_model.sh)"; exit 1; fi
      echo "model files missing in $MDIR: downloading with tools/get_model.sh (17.6 GB + 0.4 GB, resumable)"; INK_MODEL_DIR="$MDIR" tools/get_model.sh || exit 1
    fi
    DR=(); [ "$SPEC" != "off" ] && DR=(--drafter "$DRAFTER")
    echo "starting on 127.0.0.1:$PORT, $THREADS threads on CPUs $INK_CPUS, model dir $MDIR, log $LOG"
    nohup setsid "$PY" -m server --backend ink --model "$MODEL" --lib "$LIB" "${DR[@]}" --spec "$SPEC" \
        --threads "$THREADS" --slots "${INK_SLOTS:-2}" --ctx "${INK_CTX:-65536}" --prefill-chunk "${INK_PREFILL_CHUNK:-512}" \
        --default-reasoning-effort "${INK_DEFAULT_EFFORT:-none}" --host 127.0.0.1 --port "$PORT" \
        --request-log "$RUN/requests$SFX.jsonl" >> "$LOG" 2>&1 &
    echo $! > "$PIDF"; echo "pid $!" ;;
  stop)
    if alive; then kill "$(cat "$PIDF")"; sleep 2; alive && kill -9 "$(cat "$PIDF")"; echo stopped; else echo "not running"; fi
    rm -f "$PIDF" ;;
  restart) "$0" stop; "$0" start ;;
  status)
    if alive; then echo "running (pid $(cat "$PIDF")), rss $(awk '/VmRSS/{printf "%.1f GB", $2/1048576}' /proc/$(cat "$PIDF")/status)"; curl -s "127.0.0.1:$PORT/health"; echo; else echo "not running"; fi ;;
  logs) tail -n "${2:-40}" "$LOG" ;;
  *) echo "usage: $0 start|stop|restart|status|logs [n]"; exit 2 ;;
esac
