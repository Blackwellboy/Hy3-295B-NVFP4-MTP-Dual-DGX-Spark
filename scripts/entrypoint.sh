#!/usr/bin/env bash
# Hy3-295B NVFP4 on 2x DGX Spark (GB10, SM_121) — mp multi-node, TP=2.
# Runs INSIDE the container on BOTH nodes; NODE_RANK selects head(0)/worker(1).
# The verified GB10 recipe is encoded in the defaults; see README + FINDINGS.md.
set -euo pipefail

# --------------------------- model / serve config ----------------------------
MODEL="${MODEL:-/models}"                 # local NVFP4 weights, bind-mounted :ro
SERVED_NAME="${SERVED_NAME:-hy3}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"  # 128K
GMU="${GMU:-0.90}"                        # 0.92 OOMs on ~122GiB GB10
MOE_BACKEND="${MOE_BACKEND:-marlin}"      # only correct NVFP4-MoE path for W4A16 on SM121
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-1}"   # pos-2 MTP accept too low -> spec-1 wins
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
PORT="${PORT:-8000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ENABLE_TOOLS="${ENABLE_TOOLS:-0}"         # 1 => patch parsers + add hy_v3 tool/reasoning

# ------------------------------ multi-node (mp) ------------------------------
NODE_RANK="${NODE_RANK:-0}"
NNODES="${NNODES:-2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"   # HEAD fabric IP (rendezvous) — set in .env
MASTER_PORT="${MASTER_PORT:-29500}"

# ------------- fabric: force the head<->worker 200G link ---------------------
# Some images bake interface defaults that point at a DOWN/other NIC. Force the
# real fabric iface/HCA here so NCCL uses the 200G link. Verify with:
#   ip -br link   /   ibdev2netdev -v
FABRIC_IF="${NCCL_SOCKET_IFNAME:-enp1s0f0np0}"
FABRIC_HCA="${NCCL_IB_HCA:-rocep1s0f0}"
export NCCL_SOCKET_IFNAME="$FABRIC_IF"
export GLOO_SOCKET_IFNAME="$FABRIC_IF"
export TP_SOCKET_IFNAME="$FABRIC_IF"
export MN_IF_NAME="$FABRIC_IF"
export OMPI_MCA_btl_tcp_if_include="$FABRIC_IF"
export UCX_NET_DEVICES="${FABRIC_HCA}:1"
export NCCL_IB_HCA="$FABRIC_HCA"
export NCCL_IB_DISABLE=0
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
# advertise THIS node's own fabric IP so vLLM doesn't pick a LAN/WiFi/tailscale addr
export VLLM_HOST_IP="${VLLM_HOST_IP:-$MASTER_ADDR}"
# GB10 NVFP4 safety: keep FlashInfer FP4 MoE OFF; do NOT force trtllm allreduce.
export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-0}"
unset VLLM_FLASHINFER_ALLREDUCE_BACKEND 2>/dev/null || true

HEADLESS_FLAG=""
[ "$NODE_RANK" = "1" ] && HEADLESS_FLAG="--headless"

# ----------------- Phase 2: :opensource parser patch + flags -----------------
TOOL_ARGS=""
if [ "$ENABLE_TOOLS" = "1" ]; then
  bash /opt/hy3/patch_hy3_parsers.sh || echo "entrypoint: parser patch reported an issue (continuing)."
  TOOL_ARGS="--tool-call-parser hy_v3 --reasoning-parser hy_v3 --enable-auto-tool-choice"
fi

echo "entrypoint: NODE_RANK=${NODE_RANK}/${NNODES} MASTER=${MASTER_ADDR}:${MASTER_PORT} TP=${TP_SIZE} IF=${FABRIC_IF} HCA=${FABRIC_HCA} HOST_IP=${VLLM_HOST_IP} tools=${ENABLE_TOOLS} headless='${HEADLESS_FLAG}'"

set -x
exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  ${HEADLESS_FLAG} \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GMU}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --moe-backend "${MOE_BACKEND}" \
  --enforce-eager \
  --trust-remote-code \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}" \
  ${TOOL_ARGS} \
  --host 0.0.0.0 \
  --port "${PORT}" \
  ${EXTRA_ARGS}
