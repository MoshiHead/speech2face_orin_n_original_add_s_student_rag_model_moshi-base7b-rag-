#!/usr/bin/env bash
# run_watchdog_x86.sh -- keep the x86 server up: restart it if it exits or hits a CUDA
# device-side assert.  RUN_SCRIPT picks the configuration: ./run_x86.sh (legacy full
# PersonaPlex, the default) or ./run_x86_rag.sh (base7b RAG + avatar).
#   RUN_SCRIPT=run_x86_rag.sh ./run_watchdog_x86.sh &
# Log: run.log (current), crashlogs/ (saved logs of crashed runs).
# Stop: pkill -f run_watchdog_x86.sh; pkill -f speech2face/src
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"; mkdir -p crashlogs
RUN_SCRIPT="${RUN_SCRIPT:-run_x86.sh}"
exec 9>"$HERE/.watchdog_x86.lock"; flock -n 9 || { echo "watchdog already running"; exit 1; }
while true; do
  ./"$RUN_SCRIPT" > run.log 2>&1 &
  PID=$!; echo "[$(date +%F_%T)] started pid $PID ($RUN_SCRIPT)" >> crashlogs/watchdog.log
  sleep 60
  while kill -0 $PID 2>/dev/null; do
    if grep -q "device-side assert" run.log; then
      cp run.log crashlogs/crash_$(date +%m%d_%H%M%S).log
      echo "[$(date +%F_%T)] device-side assert -> restarting" >> crashlogs/watchdog.log
      kill -9 $PID; break
    fi
    sleep 5
  done
  wait $PID 2>/dev/null; sleep 5
done
