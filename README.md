# llama.cpp-inkling

> **Moved:** this patch set now lives as a proper fork with an update and install pipeline: https://github.com/GlebRadchenko/llama.cpp (branch `inkling`). This repo is archived.

Faster **Inkling-2-Mini** on Apple Silicon: Unsloth's llama.cpp plus 11 patches (Metal kernels and fused ggml ops).
Same GGUF, same quality (compared against a CPU reference, as close as stock Metal is), **~1.7× decode on an M1 Max**.

| M1 Max, Inkling-2-Mini UD-Q4_K_XL | stock Unsloth b12675 | patched |
|---|---|---|
| `llama-bench` decode (tg64) | 23.2 tok/s | **36.5 tok/s** |
| `llama-server`, Unsloth Studio's flags, chat at effort high | 22.7–22.8 tok/s | **31.0–33.6 tok/s** |
| same, as Studio ships it (DSpark on) | 20.0–25.4 tok/s | — |
| quiz, 10 math questions, effort high | 10/10 | 10/10 |

Why it helps: decode was not memory-bound. Inkling runs its middle layers in a loop, so each token launched ~5.7k tiny
GPU kernels one after another. The patches fuse them (~2.4k per token). Details: [docs/SPEEDUP.md](docs/SPEEDUP.md).

## Install

Needs macOS on Apple Silicon, Xcode command-line tools and `cmake` (`brew install cmake`).

```bash
git clone https://github.com/GlebRadchenko/llama.cpp-inkling
cd llama.cpp-inkling
./install.sh
```

It downloads Unsloth's llama.cpp source (13 MB), applies `patches/`, builds static Metal binaries and checks the new ops
against the CPU. Result: `bin/llama-server`, `bin/llama-bench`, `bin/llama-perplexity`. It never touches `~/.unsloth`.

**Unsloth Studio:** Settings → llama.cpp folder → `<this repo>/bin`, set speculative decoding to **Off**, reload the model.
Undo: clear that setting and Studio goes back to its own binary.

**Plain llama.cpp:** use `bin/llama-server` like any llama-server.

## Update to a newer Unsloth llama.cpp

```bash
gh release list -R unslothai/llama.cpp -L 5     # pick a tag
./install.sh b12700-mix-xxxxxxx                 # example tag
```

The default tag (`b12675-mix-d14fe45`) is the one the patches were last ported to and verified on.
If a patch no longer applies, the script stops before touching `bin/` and prints the four commands to port it.
After porting, check speed and quality (below) before relying on it.

## Check

```bash
M=Inkling-2-Mini-UD-Q4_K_XL.gguf
bin/llama-bench -m $M -p 512 -n 64                                    # speed

# quality: KL divergence vs a CPU-only reference (any ~10 KB text file)
bin/llama-perplexity -m $M -f text.txt -c 512 --chunks 4 -fa on -dev none -t 8 --kl-divergence-base cpu.kld
bin/llama-perplexity -m $M -f text.txt -c 512 --chunks 4 -fa on --kl-divergence-base cpu.kld --kl-divergence        # prefill
bin/llama-perplexity -m $M -f text.txt -c 512 --chunks 4 -fa on -ub 1 --kl-divergence-base cpu.kld --kl-divergence  # decode
```

Compare mean KLD with a stock build on the same text: it should not be higher. The looped model amplifies float rounding,
so with 4 chunks any build can move ±0.0003; use `--chunks 32` for a decision.
b12675, 32 chunks: stock 0.0030, patched 0.0028 (prefill); 8 chunks at `-ub 1`: stock 0.0028, patched 0.0027.

## Limits

- Tuned for Metal. The new ops also have CPU code, but CUDA/Vulkan would run them on the CPU: not for NVIDIA boxes.
- Turn speculative decoding (DSpark) off. Studio b12675 turns it on by itself (`auto`); with these patches it is
  slower than plain decode (23–29 vs 31–34 tok/s).
- Not measured: image/video input, speed beyond ~8k tokens of context.
