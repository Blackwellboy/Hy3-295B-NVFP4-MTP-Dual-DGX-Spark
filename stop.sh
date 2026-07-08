#!/usr/bin/env bash
# Stop Hy3 on both Sparks from the HEAD only. Targets only the "hy3" compose
# project (name: hy3) — leaves any other stacks alone.
#   ./stop.sh
set -euo pipefail
cd "$(dirname "$0")"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  set -a; source .env; set +a
fi
REPO_DIR_REMOTE="${REPO_DIR_REMOTE:-$(pwd)}"

echo "==> [head] $(hostname): docker compose down (project hy3)"
docker compose down || true

if [[ -n "${WORKER_HOST:-}" ]]; then
  echo "==> [worker] ${WORKER_HOST}: docker compose down"
  ssh "${WORKER_HOST}" "cd '${REPO_DIR_REMOTE}' && docker compose down" \
    || echo "stop.sh: worker teardown failed (may already be down)"
fi
echo "==> stopped."
