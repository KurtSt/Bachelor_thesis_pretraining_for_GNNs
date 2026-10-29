#!/usr/bin/env bash
# Runs a command and stops it cleanly when the free memory of the machine drops below a
# limit, before the machine starts swapping.
#
# Usage: scripts/memory_guard.sh <min free MiB> <command...>
set -uo pipefail

MIN_FREE_MB="$1"; shift
"$@" &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
  FREE_MB=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
  if [ "$FREE_MB" -lt "$MIN_FREE_MB" ]; then
    echo "[memory_guard] only ${FREE_MB} MiB available, stopping $*" >&2
    kill -TERM "$PID"
    sleep 10
    kill -KILL "$PID" 2>/dev/null
    wait "$PID"
    exit 137
  fi
  sleep 2
done
wait "$PID"
