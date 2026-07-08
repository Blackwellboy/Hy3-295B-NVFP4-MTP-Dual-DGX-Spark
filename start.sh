#!/usr/bin/env bash
# Start Hy3 across both Sparks from the HEAD node only. mp multi-node:
# the worker (headless) comes up first and blocks on the master port, then the head.
#
#   ./start.sh
#
# Prereqs: .env present+configured on BOTH nodes (head: NODE_RANK=0; worker: NODE_RANK=1);
# passwordless SSH head->worker (test: ssh $WORKER_HOST hostname); run ON the head.
set -euo pipefail
cd "$(dirname "$0")"

[[ -f .env ]] || { echo "start.sh: no .env — cp .env.example .env (and edit)" >&2; exit 1; }
# shellcheck disable=SC1091
set -a; source .env; set +a

: "${WORKER_HOST:?set WORKER_HOST in .env (worker SSH host/alias)}"
: "${MASTER_ADDR:?set MASTER_ADDR in .env (head fabric IP)}"
REPO_DIR_REMOTE="${REPO_DIR_REMOTE:-$(pwd)}"
PORT="${PORT:-8000}"
API_TIMEOUT="${API_TIMEOUT:-2400}"

echo "==> [worker] ${WORKER_HOST}: docker compose up -d (rank/headless from its OWN .env)"
ssh "${WORKER_HOST}" "cd '${REPO_DIR_REMOTE}' && docker compose up -d" \
  || { echo "start.sh: failed to start worker over SSH" >&2; exit 1; }

echo "==> [head] $(hostname): docker compose up -d"
docker compose up -d

echo "==> waiting for API on http://127.0.0.1:${PORT}/v1/models (timeout ${API_TIMEOUT}s)"
start_ts=$(date +%s)
while true; do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "==> READY: Hy3 serving on http://127.0.0.1:${PORT}"
    curl -fsS "http://127.0.0.1:${PORT}/v1/models" | python3 -m json.tool 2>/dev/null || true
    break
  fi
  now=$(date +%s)
  if (( now - start_ts > API_TIMEOUT )); then
    echo "start.sh: timed out. Check logs:" >&2
    echo "  head:   docker logs -f hy3-vllm" >&2
    echo "  worker: ssh ${WORKER_HOST} docker logs -f hy3-vllm" >&2
    exit 1
  fi
  sleep 10; printf '.'
done
echo
echo "Next: ./scripts/smoke_test.sh"
