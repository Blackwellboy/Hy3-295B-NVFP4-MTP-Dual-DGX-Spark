#!/usr/bin/env bash
# Post-boot sanity check. Run from the head once the container is up.
#   ./scripts/smoke_test.sh [host] [port]
# SUCCESS = a chat reply AND (Phase 2) finish_reason: tool_calls.
set -euo pipefail
HOST="${1:-localhost}"
PORT="${2:-8000}"
BASE="http://${HOST}:${PORT}"

echo "== health =="
curl -fsS "${BASE}/health" && echo "  OK" || { echo "  FAIL"; exit 1; }

echo "== models =="
curl -fsS "${BASE}/v1/models" | python3 -m json.tool

echo "== chat (no_think) =="
curl -fsS -X POST "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "hy3",
    "temperature": 0.9,
    "top_p": 1.0,
    "messages": [{"role":"user","content":"Reply with exactly: hy3 online"}],
    "chat_template_kwargs": {"reasoning_effort": "no_think"}
  }' | python3 -m json.tool

echo "== tool call (Phase 2 — verifies the :opensource parser patch) =="
curl -fsS -X POST "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "hy3",
    "temperature": 0.9,
    "messages": [{"role":"user","content":"What is the weather in Sydney? Use the tool."}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
      }
    }],
    "tool_choice": "auto"
  }' | python3 -c 'import sys,json; d=json.load(sys.stdin); fr=d["choices"][0]["finish_reason"]; print("finish_reason:", fr); print("PASS: structured tool_calls" if fr=="tool_calls" else "WARN: no structured tool_calls (parser patch / ENABLE_TOOLS=1?)")'
