# FINDINGS — Hy3-295B NVFP4 on 2× DGX Spark (GB10)

Real, measured results from bring-up. Hardware: 2× DGX Spark (GB10, SM_121 / cc 12.1,
arm64), 200G ConnectX-7 fabric, GPU driver 580.x / CUDA 13.0. Model:
`kodelow/Hy3-NVFP4-W4A16` (Tencent Hunyuan3 295B MoE / 21B active, W4A16 NVFP4, MTP layer),
181 GB, TP=2, mp multi-node (`--headless` worker), `--enforce-eager`, fp8_e4m3 KV, MTP spec-1.

## Working image (the one thing that mattered most)
- **`eugr/spark-vllm:latest`** — vLLM `0.23.1rc1.dev899+g378a4be5d`, digest
  `sha256:739f8c2719ec42e28a08ecb798424e83bcc4858541244e1bccb013abad58a795`.
- Serves Hy3 correctly with `--moe-backend marlin`. Coherent output + structured tool calls verified.

### Why a mismatched-toolchain image FAILS (the 222 story)
A prebuilt NVFP4 image can have HYV3 + hy_v3 parsers + marlin and still fail: if its **default
`ptxas` is a newer CUDA (e.g. 13.2/13.3)** than the 580.x / CUDA-13.0 driver's PTX-JIT accepts,
the NVFP4-MoE marlin repack (`marlin_utils_fp4.py: repack_weight`) triggers a runtime JIT that the
driver rejects:
```
torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain
(cudaErrorUnsupportedPtxVersion, error 222)
```
It is **node-independent** (both GB10s, identical driver). **Not fixable by env**: `TRITON_PTXAS_PATH`
does not help (the culprit is not Triton), and `--moe-backend emulation` is rejected for this W4A16
quant scheme (`EMULATION does not support QuantKey(u8, scale(f8e4m3fn,...))`). CUTLASS/FlashInfer FP4
are guarded off / freeze on SM121 — so **marlin is the only path, and it needs a driver-compatible
CUDA-13.0 toolchain.** `eugr/spark-vllm` ships default `ptxas`/`nvcc` = **CUDA 13.0** and its compiled
kernels carry **no PTX to JIT**, so error 222 cannot fire.

## Benchmarks (measured on the 2-node cluster, 131,072 ctx, marlin, spec-1)
| Metric | Value | Recipe claim | Note |
| --- | --- | --- | --- |
| **Single-stream (no_think)** | **13.1 tok/s** | ~21.8 | best of 2 (13.1 / 11.7) |
| Single-stream (thinking / `high`) | 9.4 tok/s | — | reasoning mode |
| **6-way aggregate (no_think)** | **36.9 tok/s** | ~59.7 | 6.14 tok/s/stream, 6/6 ok |
| **MTP acceptance, position-1** | **63.5 %** | 62–76 % | 922 accepted / 1451 drafts; spec-1 → no pos-2 |
| Cold-start (container → API ready) | ~14 min | — | weights + MTP drafter (two-phase) |
| Model weights / node | 84.5 GiB | — | TP=2 shard |
| Peak memory / node | ~120 / 121 GiB | — | 84.5 weights + 22 KV + overhead; some swap (tight) |
| GPU KV cache | 284,688 tokens | — | 22 GiB/node; 2.17× concurrency @ 131,072 ctx |
| Max context (inferred) | ~256K (config 262,144) | 128K default | KV pool > 262,144 ⇒ 256K likely fits; **not OOM-probed** |

### Performance verdict
The eugr **prebuilt** marlin path is **weight-only FP4** on `sm_120` cubins (Blackwell
forward-compat to SM121): correct output, but **~40% below the native-FP4 target** (13.1 vs ~21.8
single, 36.9 vs ~59.7 aggregate), and it logs *"GPU does not have native support for FP4
computation … may degrade performance for compute-heavy workloads."* A **from-source build with
native `sm_121a` FP4 tensor-core kernels** is the lever to chase that speed — not yet attempted.

## Phase 2 — structured tool calls: PASS
`ENABLE_TOOLS=1` applies the 11-token `:opensource` patch in-container (writable layer only) and
adds `--tool-call-parser hy_v3 --reasoning-parser hy_v3 --enable-auto-tool-choice`. Verified: a
coherent chat reply **and** `finish_reason: tool_calls` with a correct structured call
(`get_weather {"city":"Sydney"}`).

## Key GB10 / SM121 recipe confirmations
- `--moe-backend marlin` is the **only** correct MoE path for this W4A16 checkpoint on SM121.
- `--enforce-eager` required (CUDA graphs off).
- `--kv-cache-dtype fp8_e4m3`, `--gpu-memory-utilization 0.90` (0.92 OOMs; node runs ~120/121 GiB — tight).
- MTP `num_speculative_tokens=1` (pos-1 accept ~63%; pos-2 doesn't exist at n=1).
- Multi-node = vLLM **mp** backend with `--nnodes 2 --node-rank {0,1} --master-addr <head> --master-port <p>` and **`--headless` on the worker** (missing `--headless` = a two-rank-0 rendezvous hang). NCCL over RoCE; PyNCCL all-reduce (SymmMem unsupported on cc 12.1 → auto-fallback).

## Model staging tip
Pull the checkpoint once to the head node's NVMe, then copy to the other node over the 200G
fabric — not parallel Hugging Face pulls (WAN duplication + rate limits). Both nodes need identical
local weights at the same path for TP=2.

## Open / not yet done
- tok/s of a **from-source `sm_121a`** build (the speed lever — deferred).
- Definitive max-context OOM probe (needs a relaunch at higher `--max-model-len`).
