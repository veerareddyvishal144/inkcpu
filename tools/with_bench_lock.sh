#!/bin/bash
# Serialise timing-sensitive runs between agents:  tools/with_bench_lock.sh <command...>
# Waits up to 30 minutes for the lock. Correctness tests and builds do not need it (run them with `nice -n 10`).
exec flock -w 1800 /data/inkling-mini/engine/.bench.lock "$@"
