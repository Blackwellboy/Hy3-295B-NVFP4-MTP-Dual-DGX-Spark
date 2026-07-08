# Hy3-295B · NVFP4 · MTP · Dual DGX Spark

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Hardware](https://img.shields.io/badge/hardware-2%C3%97%20DGX%20Spark%20(GB10%2FSM121)-76B900)
![vLLM](https://img.shields.io/badge/vLLM-0.23.1-orange)
![Model](https://img.shields.io/badge/model-Hy3--295B%20NVFP4%20W4A16-8A2BE2)
![Parallelism](https://img.shields.io/badge/TP-2%20(mp%20multi--node)-informational)

**Serve Tencent [Hy3 / Hunyuan-3](https://huggingface.co/tencent/Hy3) (295B MoE / 21B active) as an NVFP4 W4A16 build across two NVIDIA DGX Spark (GB10) nodes** — TP=2 over the 200G ConnectX-7 fabric, native MTP speculative decoding, FP8 KV-cache, structured `hy_v3` tool-calling, and an OpenAI-compatible API. You run one script on the head node.

> **The single most important lesson of this repo:** on GB10 / SM121 with a CUDA-13.0 driver, the marlin NVFP4-MoE path fails with `cudaErrorUnsupportedPtxVersion` (error 222) **if the image's default `ptxas` is newer than the driver (e.g. CUDA 13.2)**. Use a vLLM image whose default toolchain is **CUDA 13.0** — [`eugr/spark-vllm`](https://github.com/eugr/spark-vllm-docker) — and it just works. Full story in [Troubleshooting](#troubleshooting) and [`FINDINGS.md`](FINDINGS.md).

---

## Results (measured, this cluster)

| Metric | Value | Notes |
| --- | --- | --- |
| Single-stream | **13.1 tok/s** | no_think, 131K ctx, marlin weight-only FP4 |
| Single-stream (thinking) | 9.4 tok/s | `reasoning_effort: high` |
| 6-way aggregate | **36.9 tok/s** | 6.14 tok/s/stream |
| MTP acceptance (pos-1) | **63.5 %** | spec-1 (pos-2 doesn't exist at n=1) |
| GPU KV cache | 284,688 tokens | 2.17× concurrency @ 131,072 ctx |
| Peak memory / node | ~120 / 121 GiB | 84.5 GiB weights + 22 GiB KV + overhead (tight) |
| Cold-start | ~14 min | two-phase load (target + MTP drafter) |
| Structured tool calls | ✅ `finish_reason: tool_calls` | via patched `hy_v3` `:opensource` parsers |

> These numbers are on the **prebuilt** `eugr/spark-vllm` image, which runs **weight-only FP4** on `sm_120` cubins (Blackwell forward-compat) — correct, but ~40% below the native-FP4 target (~21.8 tok/s). A from-source `sm_121a` build is the lever to chase that speed; see [`FINDINGS.md`](FINDINGS.md).

---

## Requirements

**Hardware**
- 2 × DGX Spark / GB10 (SM_121, cc 12.1, arm64), ~122 GiB unified memory each
- 200G ConnectX-7 fabric between the nodes (MTU 9000 recommended)
- ~200 GB free local NVMe per node for the 181 GB checkpoint

**Software**
- Docker + `docker compose` v2.24+, NVIDIA Container Toolkit (default on DGX Spark)
- **Passwordless SSH from head → worker** (`ssh $WORKER_HOST hostname` must work)
- GPU driver **580.x / CUDA 13.0** and a vLLM image whose default `ptxas` is **CUDA 13.0** (see below)

---

## Quick start

### 1. Get the weights onto **both** nodes' local NVMe
Recommended: pull once to the head, then copy over the 200G fabric (not parallel WAN pulls).
```bash
hf download kodelow/Hy3-NVFP4-W4A16 --local-dir /path/to/Hy3-NVFP4-W4A16 --max-workers 4
# then, from the head, fabric-copy to the worker over the 200G link:
rsync -a /path/to/Hy3-NVFP4-W4A16/ <worker-fabric-host>:/path/to/Hy3-NVFP4-W4A16/
```

### 2. Get the image on both nodes
```bash
docker pull eugr/spark-vllm:latest        # CUDA-13.0 default toolchain — avoids the 222 error
```

### 3. Configure `.env` on **both** nodes
```bash
cp .env.example .env
```
Set per node (identical except `NODE_RANK` / `VLLM_HOST_IP`):

| Variable | Head | Worker |
| --- | --- | --- |
| `NODE_RANK` | `0` | `1` |
| `VLLM_HOST_IP` | head fabric IP | worker fabric IP |
| `MASTER_ADDR` | head fabric IP | *(same head IP)* |
| `WORKER_HOST` | worker SSH host | — |
| `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` | your 200G iface / RoCE HCA (`ip -br link`, `ibdev2netdev -v`) | *(same)* |
| `MODEL_DIR` | local checkpoint dir | *(same path)* |

### 4. Launch (from the head only)
```bash
./start.sh          # brings up the worker (headless) over SSH, then the head, then polls the API
```

### 5. Verify
```bash
./scripts/smoke_test.sh          # health + models + chat + structured tool call
```

### 6. Enable tools (optional, phase 2)
Set `ENABLE_TOOLS=1` in `.env` on both nodes and relaunch — the entrypoint applies the `:opensource` parser patch in-container and adds `--tool-call-parser hy_v3 --reasoning-parser hy_v3 --enable-auto-tool-choice`.

---

## Architecture

```
┌──────────────────────────┐      200G fabric        ┌──────────────────────────┐
│   spark1 · HEAD           │◄──── NCCL (mp) ────────►│  spark2 · WORKER          │
│   NODE_RANK=0             │   MASTER_ADDR:29500      │  NODE_RANK=1 --headless   │
│   ┌────────────────────┐  │   RoCE / PyNCCL          │  ┌────────────────────┐   │
│   │ vLLM  TP rank 0    │  │                          │  │ vLLM  TP rank 1    │   │
│   │ OpenAI API :8000   │  │                          │  │ (GPU only)         │   │
│   └────────────────────┘  │                          │  └────────────────────┘   │
└──────────────────────────┘                          └──────────────────────────┘
        ▲  start.sh / stop.sh run here, drive the worker over SSH
```

### Verified vLLM flags (source-confirmed on the image's vLLM)

| Flag | Value | Why |
| --- | --- | --- |
| `--tensor-parallel-size` | `2` | 2-node model. TP=3 impossible (8 KV heads). |
| `--distributed-executor-backend` | `mp` | multi-node via `--nnodes/--node-rank/--master-addr/--master-port` |
| `--headless` | worker only | **the worker MUST be headless** — otherwise two rank-0 servers → rendezvous hang |
| `--moe-backend` | `marlin` | the only correct NVFP4-MoE path for W4A16 on SM121 (CUTLASS is guarded off / FlashInfer freezes) |
| `--enforce-eager` | on | CUDA graphs off (GB10 marlin path) |
| `--kv-cache-dtype` | `fp8_e4m3` | ~128K ctx |
| `--gpu-memory-utilization` | `0.90` | 0.92 OOMs (node runs ~120/121 GiB — tight) |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":1}` | pos-2 acceptance too low → spec-1 wins |
| `--tool-call-parser` / `--reasoning-parser` | `hy_v3` | + the `:opensource` token patch (see `scripts/patch_hy3_parsers.sh`) |

---

## Troubleshooting

- **`cudaErrorUnsupportedPtxVersion` (error 222) in `marlin_utils_fp4.py: repack_weight`.** The image's default `ptxas`/`nvcc` is newer (e.g. CUDA 13.2/13.3) than your CUDA-13.0 driver's PTX-JIT accepts, so a runtime-JIT'd kernel is rejected. **Not fixable by env** (`TRITON_PTXAS_PATH` only affects Triton; the culprit is elsewhere). **Fix:** use an image whose *default* toolchain is CUDA 13.0 (`eugr/spark-vllm`), whose compiled kernels ship no PTX to JIT. `docker run --rm <img> bash -c 'which ptxas; ptxas --version'` should show 13.0.
- **`NvFp4 MoE backend 'EMULATION' does not support … QuantKey(u8, …)`.** Emulation is gated off for the W4A16 quant scheme — marlin is the only path; don't switch backends.
- **Worker won't join / two rank-0 servers.** The worker's `.env` must have `NODE_RANK=1`; the entrypoint adds `--headless` for rank 1.
- **NCCL hang / SymmMem "not supported".** SymmMem is unavailable on cc 12.1 → vLLM auto-falls-back to PyNCCL over RoCE; that's normal. Verify `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` match `ip -br link` / `ibdev2netdev -v` on both nodes; MTU 9000.
- **CUDA OOM at boot.** Lower `GMU` to `0.85` or reduce `MAX_MODEL_LEN`.
- **Slow / weight-only FP4 warning.** The prebuilt image runs `sm_120` weight-only FP4 (forward-compat) — correct but ~40% slower than native FP4. A from-source `sm_121a` build is the speed lever.

---

## Credits

Packaging and bring-up notes — the hard discovery work is theirs:
- [**kodelow**](https://huggingface.co/kodelow/Hy3-NVFP4-W4A16) — NVFP4 W4A16 quant (experts-only, MTP preserved).
- [**tonyd2wild**](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark) — methodology and dual-Spark packaging approach this repo's structure is modeled on (his DeepSeek-V4-Flash 2-Spark NVFP4 recipe).
- [**Mia / MiaAI-Lab**](https://github.com/MiaAI-Lab) — the dual-Spark `start.sh`/`stop.sh` + `mp` multi-node pattern this is modelled on.
- [**eugr / spark-vllm-docker**](https://github.com/eugr/spark-vllm-docker) — the CUDA-13.0 GB10 vLLM image that avoids the marlin FP4 PTX error.
- [**jasl**](https://github.com/jasl) — GB10 / SM121 vLLM enablement.
- [**Tencent Hunyuan Team**](https://huggingface.co/tencent/Hy3) — Hy3 (Apache 2.0).

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
