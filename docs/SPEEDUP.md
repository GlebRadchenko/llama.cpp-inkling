# Inkling-2-Mini (UD-Q4_K_XL) speedup on M1 Max

Measured 2026-11-24 on an M1 Max (64 GB). Same GGUF, same quant, same reasoning effort, no loop reduction.
Base: Unsloth llama.cpp `b12651-mix-c7059d6` (source commit d426a37); ported to `b12675-mix-d14fe45` (see the end).

## Result

| setup | stock Unsloth llama.cpp | patched | gain |
|---|---|---|---|
| llama-bench decode (tg64) | 21.7 tok/s | 37–38 tok/s | **+72–75%** |
| llama-server, Unsloth Studio's flags, 256-token decode | 21.1 | 33.8 | **+60%** |
| Studio-flags chat, effort high (long reasoning outputs) | 21.1 | 31.1–31.4 | +48% |
| Studio-like, 4 concurrent requests | — | ~55 tok/s aggregate | — |
| quiz (10 math questions), effort low / high / 4 parallel | 10/10 | 10/10 / 10/10 / 10/10 | no loss |

Decode GPU time per token: 41.9 ms → ~26 ms; Metal dispatches per token 5682 → 2371, barriers 4378 → 1703.

## Why it was slow

Decode was not bandwidth-bound: ~3.6 GB/token is ~10 ms at M1 Max bandwidth, but the GPU spent ~42 ms per token.
Inkling's looped architecture (48 layer visits of many tiny ops) produced ~5.7k serial kernel dispatches per token,
each costing ~5–8 µs of launch/barrier latency. Every patch below removes dispatches or serial barrier levels.

## Patches

| patch | change | A/B decode |
|---|---|---|
| 0001 | NVFP4 mul_mv: word loads, LUT-free E2M1 decode, NR0=4 | +6.7% |
| 0002 | clip = t·l2_norm(x, eps=t): 23 → 8 dispatches per layer | +8% |
| 0003 | load-time derived weights; conv residual+scale folded into taps; q/k/v loop controllers batched | +8.5% |
| 0004 | new `GGML_GLU_OP_SITU` (softcapped SwiGLU) CPU+Metal: 9 → 1 dispatch | +3.4% |
| 0005 | 1-token relative bias: gather→project, sink handed to flash-attn; padded table at load | +10.6% |
| 0006 | clip factors folded into attention scale / bias table | +0.5% |
| 0007 | new `GGML_OP_DEPTH_MIX` (attention over depth) CPU+Metal, kernel specialized per shape | +10% |
| 0008 | Metal fusion CONCAT+CPY+SSM_CONV (short conv + state store); legacy message queries stacked | +4.8% |
| 0009 | MoE per-expert scales folded (SITU row scale; down scale into router weights) | +1.8% |
| 0010 | new `GGML_OP_LOOP_GATE` (loop controller update) elementwise | +2.6% |
| 0011 | removed experiment counters; test cases (`INK_EVAL=1 test-backend-ops test`, `INK_PERF=1 ... perf`) | — |

## How quality was verified

- **KLD against a CPU-backend reference** (stock binary, all ops on CPU, 4×512 tokens), on three code paths:
  prefill (ubatch 512), small batch (ub 2), single-token decode (ub 1).
- The looped model amplifies float-rounding differences: **stock Metal vs CPU is already KLD 0.00158 (prefill) / 0.00127 (decode)**.
  Rule used: a change must stay at or below that floor.
- Final build: **KLD 0.00129 (prefill) / 0.00117 (decode)**, same-top-token 99.6% / 99.5% — at or below stock Metal.
- Every new op/kernel checked against the CPU implementation with `test-backend-ops` (39/39 pass).
- Quiz 10/10 at effort low and high, and with 4 concurrent requests.

## Failures and corrections

- **Conv fusion first version was wrong** (KLD 0.10–0.20): it skipped writing the concat output, which the
  full-attention layers' dynamic taps also read. Caught by KLD before shipping; fixed (bit-exact afterwards).
- **Loop-gate with the RMS norm fused in was 1.5% slower** (one threadgroup per row = 1–3 GPU cores). Shipped the
  elementwise form instead; the norm-fused form stays behind `INK_GATE_NORM=1`.
- **Splitting DEPTH_MIX over more threadgroups: no gain** — reverted.
- **MLX path (mlx-vlm 0.8.1, original NVFP4 weights): slower** — 18.6 tok/s plain, 18.7–19.2 with its DSpark drafter.
- **DSpark speculative decoding is now a net loss**: 24.8 vs 35.0 tok/s (Studio-like). The 2-token verify costs
  43.5 ms vs 28.1 ms for one token (MoE reads experts per token), and prose drafts get rejected often.
- A Q8_0-experts requant did not help.

## Port to b12675-mix-d14fe45 (2026-11-24)

- Rebased with real 3-way merges (old release tree → new release tree). Conflicts: 0001, 0006, 0009.
- **0001:** upstream adopted its own LUT-free NVFP4 decode (one nibble at a time). The patch keeps its own kernel
  (two nibbles per half2) and NR0=4: upstream's kernel measured 35.0 tok/s, this one 36.5 (tg64, A/B).
- **0006:** upstream dropped the reserve-only dense block (taken) and added a shift for padded banded spans of 2–32
  tokens; the shift now reads the clip-folded padded table.
- **0009:** upstream's MoE amax rescale (other archs) kept; the down scale still goes into the router weights.
- Speed: stock 23.2 → patched 36.5 tok/s (tg64). Studio flags, effort high: 22.7–22.8 → 31.0–33.6 tok/s.
- Quality: 39/39 op tests; 32-chunk prefill KLD stock 0.00295 vs patched 0.00277 (max 1.82 vs 0.57); 8-chunk
  decode (ub 1) stock 0.00279 vs patched 0.00274; quiz 10/10 at effort high.
- 4-chunk KLD is too noisy to compare builds here: stock b12651 → b12675 moved 0.00158 → 0.00128 and single patches
  moved it ±0.0002 although they are exact rewrites (the conv fold alone: 0.00123 → 0.00141). 32 chunks settle it.
- DSpark: Studio b12675 enables it by itself (`speculativeType: auto`). Still a loss with the patches: 22.8–28.7
  vs 31.0–33.6 tok/s; on stock it is roughly neutral (20.0–25.4 vs 22.7–22.8).

## Caveats

- Tuned and verified for Metal (M1 Max) and this model. The new ops have CPU implementations; other backends
  (CUDA/Vulkan) would fall back to CPU for them.
- The conv fusion does not apply when llama.cpp keeps recurrent rollback slots (speculative decoding on), so the
  spec path is slower than it could be.
- Not tested: image/video input, very long contexts (> 8k tokens of KV) for speed.

## What is left (estimated)

- Make the 2-token verify cheaper (n=2 paths, expert de-dup) so DSpark pays off again: maybe +10–20% on code.
- Fuse the bias f16 copy + mask add before flash-attn; merge same-input Q/rel/gate matmuls: ~2–4%.
