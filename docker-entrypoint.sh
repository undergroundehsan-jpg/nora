#!/bin/sh
# Seed the call history on first boot.
#
# /app/logs is where the app reads and writes call history. On Railway it is a
# mounted volume, which starts empty and shadows anything baked into the image,
# so the seed copy lives in /app/seed-logs and is copied across only when the
# volume has no history yet. After that the volume is left alone and new calls
# accumulate on top.
set -e

if [ -d /app/seed-logs ]; then
  mkdir -p /app/logs
  seeded=0
  for src in /app/seed-logs/*.jsonl; do
    [ -f "$src" ] || continue
    dest="/app/logs/$(basename "$src")"
    if [ ! -s "$dest" ]; then
      cp "$src" "$dest"
      seeded=$((seeded + 1))
      echo "[seed] $(basename "$src") -> $(wc -l < "$dest") lines"
    fi
  done
  if [ "$seeded" -eq 0 ]; then
    echo "[seed] existing call history found; nothing seeded"
  fi
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 1 --ws websockets
