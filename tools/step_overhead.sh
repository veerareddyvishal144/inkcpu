#!/bin/bash
# Per-token server overhead on a TEST server (agent T, wave 1). Holds the bench lock for the whole life of the test server
# (one full model resident besides production). Engine speed is irrelevant here: the overhead is measured in absolute ms per token.
#   tools/step_overhead.sh [lib] [port] [cpus]      defaults: build-t/src/capi/libink.so, 8010, 29,31
# Output: the step-trace lines of the server log (INK_STEP_TRACE=1) for greedy / sampled / top-k+top-p / penalty requests, stream and non-stream.
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT" || exit 1
LIB=${1:-$ROOT/build-t/src/capi/libink.so}; PORT=${2:-8010}; CPUS=${3:-29,31}
NT=$(echo "$CPUS" | tr ',' '\n' | wc -l)
export INK_PORT=$PORT INK_CPUS=$CPUS INK_THREADS=$NT INK_LIB=$LIB INK_SPEC=${INK_SPEC:-off} INK_STEP_TRACE=1 INK_CTX=${INK_CTX:-4096} INK_SLOTS=1 INK_NO_DOWNLOAD=1
PY=/data/engine/.venv/bin/python
inner() {
  local avail; avail=$(free -g | awk '/Mem:/{print $7}'); [ "$avail" -ge 40 ] || { echo "only $avail GB available"; exit 3; }
  tools/serve.sh stop >/dev/null 2>&1
  tools/serve.sh start || exit 1
  for i in $(seq 1 120); do curl -sf "127.0.0.1:$PORT/health" | grep -q '"ok"' && break; sleep 2; done
  curl -s "127.0.0.1:$PORT/health"; echo
  echo "server threads (comm psr allowed):"; for t in /proc/$(cat run/server-$PORT.pid)/task/*; do printf "%s %s %s\n" "$(cat $t/comm)" "$(awk '{print $39}' $t/stat)" "$(grep Cpus_allowed_list $t/status | cut -f2)"; done | sort | uniq -c | sort -rn | head -8
  P='Write a short story about a lighthouse keeper who discovers a message in a bottle. Begin:'
  req() { # label, extra json, stream
    local label=$1 extra=$2 stream=$3
    echo "--- $label (stream=$stream) ---"
    if [ "$stream" = true ]; then
      curl -s -N "127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' -d "{\"prompt\":\"$P\",\"max_tokens\":${NTOK:-48},\"stream\":true,$extra}" | grep -c '^data:' | sed 's/^/sse events: /'
    else
      curl -s "127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' -d "{\"prompt\":\"$P\",\"max_tokens\":${NTOK:-48},$extra}" | $PY -c 'import json,sys; d=json.load(sys.stdin); t=d["timings"]; print("timings:", {k:t[k] for k in ("predicted_n","predicted_ms","engine_ms","sample_ms","predicted_per_second")})'
    fi
    sleep 0.5; grep 'step-trace' run/server-$PORT.log | tail -1
  }
  req greedy '"temperature":0' false
  req greedy '"temperature":0' true
  req "T=0.6" '"temperature":0.6,"seed":7' true
  req "T=0.6 top_k40 top_p0.9" '"temperature":0.6,"top_k":40,"top_p":0.9,"seed":7' true
  req "T=0.6 rep-penalty 1.1" '"temperature":0.6,"repetition_penalty":1.1,"seed":7' true
  req "chat greedy stream" '"temperature":0' true
  echo "--- chat endpoint, greedy, stream ---"
  curl -s -N "127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":${NTOK:-48},\"temperature\":0,\"stream\":true}" | grep -c '^data:' | sed 's/^/sse events: /'
  sleep 0.5; grep 'step-trace' run/server-$PORT.log | tail -1
  echo "--- idle CPU of the test server over 10 s (/proc/pid/stat utime+stime), pool mode: ${INK_POOL_PARK:-park (default)} ---"
  SP=$(cat run/server-$PORT.pid); t0=$(awk '{print $14+$15}' /proc/$SP/stat); sleep 10; t1=$(awk '{print $14+$15}' /proc/$SP/stat)
  echo "idle server: $(awk -v d=$((t1-t0)) 'BEGIN{printf "%.1f%% of one core", d/10}') ($NT engine threads)"
  tools/serve.sh stop
}
export -f inner; export PORT NTOK PY
tools/with_bench_lock.sh bash -c inner
