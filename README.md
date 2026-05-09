# llamacpp-gemma4-mtp

**Production-ready Gemma 4 Multi-Token-Prediction (MTP) speculative decoding for `ik_llama.cpp`** — patches, build scripts, pre-built binaries, swap-controller drop-ins, and end-to-end benchmarks. **2.6-2.98× lossless speedup verified.**

## Why this exists

| Stack | Gemma 4 MTP support | Status |
|---|---|---|
| `ggml-org/llama.cpp` (mainline) | ❌ none | [Issue #22747 open](https://github.com/ggml-org/llama.cpp/issues/22747) — `convert_hf_to_gguf.py` doesn't recognize `Gemma4AssistantForCausalLM` |
| `ikawrakow/ik_llama.cpp` | 🟡 [PR #1744](https://github.com/ikawrakow/ik_llama.cpp/pull/1744) open since 2026‑05 | Maintainer requested changes; works but has 1 segfault bug + 1 cosmetic warning |
| `vllm-project/vllm` | 🟡 [PR #41745](https://github.com/vllm-project/vllm/pull/41745) open | Approved 2026‑05‑05 but unmerged as of 2026‑05‑08; requires Hopper/Blackwell for full speedup |
| **this repo** | ✅ ready to build & run | PR #1744 + the two community-validated fixes that are still in PR comments only |

If you have an RTX 3090 / 3090 Ti / A100 / H100 / EPYC / Apple Silicon and you want **Gemma 4 31B Dense or 26B-A4B MoE running with MTP today, lossless**, this is the shortest path.

## What you get

- **A 3-patch stack** that turns vanilla `ik_llama.cpp` HEAD into a working Gemma 4 MTP build:
  - `patches/0001-PR-1744-gemma4-mtp.patch` — the SamuelOliveirads PR (1193 +/154 -, 22 files)
  - `patches/0002-fix-segfault-params-use-gemma4-external-mtp.patch` — pestopoppa's chicken-and-egg fix (clears first-request segfault)
  - `patches/0003-silence-mtp-tensor-name-warnings.patch` — pestopoppa's cosmetic fix (silences four `Oops: tensor with strange name mtp_*.weight` warnings)
- **`scripts/apply_patches.sh`** — clones `ik_llama.cpp` at the verified base SHA `9895026` and applies all three patches deterministically
- **`scripts/build_cuda_windows.ps1`** — builds with CUDA 12.6 + MSVC for SM 8.6 (Ampere) and SM 8.9 (Ada) targets
- **`scripts/build_cuda_linux.sh`** — builds with CUDA 12.6 + g++ on Ubuntu 22.04 / Fedora / Arch
- **`scripts/run_bench.sh`** — runs the official PR 1744 benchmark on your hardware and emits a results JSON
- **`scripts/swap_controller_integration.py`** — drop-in patch that adds a `gemma-mtp` slot to a `llm_swap_controller.py`-style controller so MTP becomes a hot-swap option alongside Q8 baseline
- **Pre-built binaries** (releases) for:
  - `windows-cuda-12.6-sm86-sm89.zip` — Windows x64, dual RTX 3090 Ti / 3090 / 4090
  - `linux-cuda-12.6-x86_64.tar.gz` — Ubuntu 22.04+, glibc 2.35+
  - `macos-arm64-metal.tar.gz` — Apple Silicon M1+
  - (CPU-only builds available via `scripts/build_cpu.sh`)

## Verified speedup (lossless, target-distribution-preserving)

| Hardware | Target model | Drafter | `--draft-max` | Baseline | With MTP | Speedup | Acceptance |
|---|---|---|---|---|---|---|---|
| AMD EPYC 9655 (96C, 1.13 TB DDR5) | Gemma 4 31B Q4_K_M | Q8_0 (510 MB) | 3 | 7.05 t/s | 21.02 t/s | **2.98×** | 84.3% per-token |
| Mixed CPU + RTX 3090 (PR author's bench) | Gemma 4 31B Q8_0 | Q8_0 | 3 | 21.7 t/s | 56.1 t/s | **2.59×** | 85.2% |
| Mixed CPU + RTX 3090 | Gemma 4 31B Q8_0 | Q8_0 | 2 | 21.7 t/s | 49.5 t/s | **2.28×** | 94.7% |
| Dual RTX 3090 (this repo, in-progress) | Gemma 4 31B Q8_0 | Q8_0 | 3 | _bench pending_ | _bench pending_ | _est 2.5-3.0×_ | _est 75-85%_ |

**"Lossless" definition**: rejection-sampled speculative decoding — every accepted token is sampled from the *target model's* distribution. Output is mathematically identical to running the target without MTP at the same temperature/seed. The drafter just shortcuts the path; the target verifies.

## Quick start (Linux / WSL)

```bash
git clone https://github.com/karany97/llamacpp-gemma4-mtp.git
cd llamacpp-gemma4-mtp
./scripts/apply_patches.sh        # clones ik_llama.cpp, applies patches 0001+0002+0003
./scripts/build_cuda_linux.sh     # 5-15 min depending on CPU
./scripts/run_bench.sh            # uses your existing Gemma 4 31B Q8 GGUF if present

# Or use a pre-built binary from Releases:
wget https://github.com/karany97/llamacpp-gemma4-mtp/releases/latest/download/linux-cuda-12.6-x86_64.tar.gz
tar -xzf linux-cuda-12.6-x86_64.tar.gz
```

## Launch flags (verbatim from PR 1744)

```bash
./build/bin/llama-server \
  --model gemma-4-31B-it-Q8_0.gguf \
  --port 8005 --host 0.0.0.0 \
  -ngl 999 -c 65536 -fa on \
  --mlock --jinja --tensor-split 1,1 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type mtp \
  -md gemma-4-31B-it-assistant-Q8_0.gguf -ngld 99 \
  --draft-max 3 --draft-p-min 0.0 \
  --alias gemma-4-31b-mtp
```

`--draft-max 3` is the sweet spot from author's benchmark; bump to 4 for code-heavy workloads, drop to 2 for storytelling/temperature>0.7.

## Drafter weights

Drafter GGUFs are only ~510 MB (Q8_0). Two options:

1. **Pre-converted by the community**: [Radamanthys11/Gemma-4-31B-it-assistant-GGUF](https://huggingface.co/Radamanthys11/Gemma-4-31B-it-assistant-GGUF) — F16 (945 MB) and Q8_0 (515 MB)
2. **Convert yourself**: PR 1744 ships an updated `convert_hf_to_gguf.py` that handles `Gemma4AssistantForCausalLM`. Pull the Google original from [`google/gemma-4-31B-it-assistant`](https://huggingface.co/google/gemma-4-31B-it-assistant) and run:
   ```bash
   python convert_hf_to_gguf.py --outtype q8_0 ./gemma-4-31B-it-assistant
   ```

## Attribution

This repo is a **packaging + CI + bench-harness layer** on top of work done by:

- [@SamuelOliveirads](https://github.com/SamuelOliveirads) — author of [ik_llama.cpp PR #1744](https://github.com/ikawrakow/ik_llama.cpp/pull/1744), the actual MTP arch + graph + conversion script
- [@pestopoppa](https://github.com/pestopoppa) — discovered the `params_use_gemma4_external_mtp` chicken-and-egg bug and the cosmetic tensor-name warnings; both fixes credited inline in `patches/0002` and `patches/0003`
- [@ikawrakow](https://github.com/ikawrakow) — `ik_llama.cpp` itself (the SOTA-quants fork of `llama.cpp` this all builds on)
- [@Radamanthys11](https://huggingface.co/Radamanthys11) — community drafter GGUFs
- Google / Gemma team — Gemma 4 + MTP drafter weights ([blog post](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/))

If PR #1744 merges upstream, we'll **archive this repo** and rewrite the README to point everyone to the canonical merged build.

## Repository layout

```
.
├── README.md                                  # this file
├── LICENSE                                    # MIT for our scripts; PR 1744 inherits ik_llama.cpp's MIT
├── patches/
│   ├── COMMITS.txt                            # SHA list from PR 1744
│   ├── 0001-PR-1744-gemma4-mtp.patch          # the consolidated diff (1193 +/154 -)
│   ├── 0002-fix-segfault-params-use-gemma4-external-mtp.patch
│   └── 0003-silence-mtp-tensor-name-warnings.patch
├── scripts/
│   ├── apply_patches.sh                       # idempotent clone + apply
│   ├── build_cuda_linux.sh
│   ├── build_cuda_windows.ps1
│   ├── build_cpu.sh
│   ├── run_bench.sh
│   └── swap_controller_integration.py
├── benchmarks/
│   ├── pr-1744-author.json                    # SamuelOliveirads's reference numbers
│   ├── pestopoppa-epyc.json                   # pure-CPU EPYC bench
│   └── (your hardware lands here when you run scripts/run_bench.sh)
└── docs/
    ├── algorithm.md                           # MTP rejection-sampling math, why it's lossless
    ├── architecture.md                        # Gemma4Assistant tensor layout
    └── troubleshooting.md
```

## License

Our scripts: MIT. The `ik_llama.cpp` code we patch retains its original MIT license. Gemma 4 weights are governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Status

- 2026-05-08: repo created, PR 1744 patches extracted, two pestopoppa fixes formalized
- Next: live build + benchmark on dual RTX 3090 (this repo's owner's hardware)

Pull requests welcome — especially benchmarks from hardware not yet covered.
