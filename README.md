# llama.cpp-inkling

Faster **Inkling-2-Mini** on Apple Silicon: Unsloth's llama.cpp plus 11 patches (Metal kernels and fused ggml ops).
Same GGUF, same quality (compared against a CPU reference, as close as stock Metal is), **~1.7× decode on an M1 Max**.

| M1 Max, Inkling-2-Mini UD-Q4_K_XL | stock Unsloth b12651 | patched |
|---|---|---|
| `llama-bench` decode (tg64) | 21.7 tok/s | **37.5 tok/s** |
| `llama-server`, Unsloth Studio's flags | 21.1 tok/s | **33.8 tok/s** |
| quiz, 10 math questions | 10/10 | 10/10 |

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

**Unsloth Studio:** Settings → llama.cpp folder → `<this repo>/bin` → reload the model.
Undo: clear that setting and Studio goes back to its own binary.

**Plain llama.cpp:** use `bin/llama-server` like any llama-server.

## Update to a newer Unsloth llama.cpp

```bash
gh release list -R unslothai/llama.cpp -L 5     # pick a tag
./install.sh b12700-mix-xxxxxxx                 # example tag
```

The default tag (`b12651-mix-c7059d6`) is the one the patches were made and verified on.
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

Mean KLD should stay ≤ ~0.0016. That is how far stock Metal already is from the CPU; this build measured 0.0013 / 0.0012.

## Limits

- Tuned for Metal. The new ops also have CPU code, but CUDA/Vulkan would run them on the CPU: not for NVIDIA boxes.
- Leave speculative decoding (DSpark) off. With these patches it is slower than plain decode (24.8 vs 35.0 tok/s).
- Not measured: image/video input, speed beyond ~8k tokens of context.
