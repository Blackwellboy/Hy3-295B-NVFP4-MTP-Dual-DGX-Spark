# FINDINGS — Hy3-295B NVFP4 on 2× DGX Spark (GB10)

Real, measured results from bring-up. Hardware: 2× DGX Spark (GB10, SM_121 / cc 12.1,
arm64), 200G ConnectX-7 fabric, GPU driver 580.159.03 / CUDA 13.0. Model:
`kodelow/Hy3-NVFP4-W4A16` (Tencent Hunyuan3 295B MoE / 21B active, W4A16 NVFP4, MTP layer),
181 GB, TP=2, mp multi-node (`--headless` worker), `--enforce-eager`, fp8_e4m3 KV, MTP spec-1.

## Working image (the one thing that mattered most)
- **`eugr/spark-vllm:latest`** — vLLM `0.23.1rc1.dev899+g378a4be5d`, digest
  `sha256:739f8c2719ec42e28a08ecb798424e83bcc4858541244e1bccb013abad58a795`.
- Serves Hy3 correctly with `--moe-backend marlin`. Coherent output + structured tool calls verified.

### Why a mismatched-toolchain image FAILS (the 222 story)
A prebuilt NVFP4 image can have HYV3 + hy_v3 parsers + marlin and still fail: if its **default
`ptxas` is a newer CUDA (e.g. 13.2/13.3)** than the 580.159.03 / CUDA-13.0 driver's PTX-JIT accepts,
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

## Benchmarks (scored slice) — accuracy vs. published Hy3
Scored slices to check the weight-only-FP4 quant preserves *quality*, not just the 7 vibe-check
puzzles. **N=100 each · greedy (temp 0) · seed 42 · 4 concurrent workers**, on the eugr weight-only-FP4
(`sm_120`) stack. Harness: [`bench/eval_hy3.py`](bench/eval_hy3.py) (samples with the fixed seed and
captures `finish_reason` + full `content`/`reasoning_content` per item, so truncation and parse-fails
are separable). A sanitized 5-item sample of that captured per-item JSONL is in
[`bench/sample_gsm8k.jsonl`](bench/sample_gsm8k.jsonl) (4 completed + 1 truncated, to show the
`finish_reason` capture); the full per-item data regenerates by re-running the harness.

| Suite | Slice | This quant | Tencent official Hy3 |
| --- | --- | --- | --- |
| GSM8K (`openai/gsm8k`, main/test) | N=100 | **75/76 = 98.7% on completed items** | 95.37 |
| MMLU (`cais/mmlu`, all/test) | N=100 | **90/100** | 87.42 (5-shot, base) |

- **GSM8K:** of the 100 sampled problems, **76 completed and 75 were correct (98.7%)**. The other
  **24 truncated** (`finish_reason: length`) — high-effort reasoning ran past the harness's 2048-token
  output budget *before* emitting a final answer; all 24 were verified from the saved responses as
  **cut-off reasoning chains, not model errors**. (A first pass with a smaller budget reported a
  misleading ~62% purely for this reason.) The single non-truncated miss was a *suspect GSM8K gold
  label* — the model's arithmetic was correct. High-effort reasoning here is genuinely verbose: even
  re-running truncated items at an **8192-token** budget, some still did not finish.
- **MMLU:** 90/100 — zero parse-fails, zero truncation.
- **Concordance with published scores:** these land **at or slightly above** Tencent's official Hy3
  numbers (GSM8K **95.37**, MMLU **87.42** 5-shot base) — within the noise of a 100-item slice. So the
  **weight-only-FP4 path reproduces the model's published accuracy: it costs ~40% speed, not
  correctness.** This is *concordance within noise* — **not** a like-for-like comparison and **not**
  "beating" the numbers: Tencent's MMLU is 5-shot with their harness; ours is a zero/few-shot N=100
  slice with our own answer extraction.

**Measurement meta-lesson (worth more than the number):** the first harness reported a misleadingly
low GSM8K for two reasons — (1) a 1500–2048-token budget that **truncated** verbose high-effort
reasoning before the final answer, and (2) it **discarded per-item responses**, making the low score
un-diagnosable. Fix: capture `finish_reason` + full response text per item, size the token budget for
the reasoning path, and **always inspect truncation and parse-fails separately before trusting — or
blaming the quant for — a local-eval number.**

**Caveats:** N=100 slices (not the full suites). Only the eugr weight-only-FP4 path is measured;
**native `sm_121a` build accuracy is untested.**

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
