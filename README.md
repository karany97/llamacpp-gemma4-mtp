# llamacpp-gemma4-mtp

> **Reproducible Gemma 4 multi-token-prediction bench harness for `ik_llama.cpp`.**
> PR #1744 merged upstream on 2026-05-10. This repo is the lowest-friction
> way to verify the **2.6–2.98× lossless speedup** on your hardware in
> ~15 minutes, plus a swap-controller integration so MTP becomes a
> hot-swap option alongside the Q8 baseline.

[![PR #1744 merged](https://img.shields.io/badge/ik__llama.cpp%20PR%20%231744-merged%202026--05--10-22c55e)](https://github.com/ikawrakow/ik_llama.cpp/pull/1744)
[![Verified speedup](https://img.shields.io/badge/lossless%20speedup-2.6%E2%80%932.98%C3%97-c2410c)](#verified-speedup-lossless)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](./LICENSE)

---

## What this repo is (and isn't)

**Is** a reproducible packaging + benchmark layer on top of an upstream feature
that just merged. Three scripts run end-to-end:

1. `scripts/apply_patches.sh` — clones `ik_llama.cpp` at a known-good SHA
   (post-merge of PR #1744), applies defensive auto-skip patches for two
   edge cases that are already in `main` but worth keeping as backups
2. `scripts/build_cuda_linux.sh` *(or `build_cuda_windows.ps1` / `build_cpu.sh`)* —
   builds the patched tree
3. `scripts/run_bench.sh` — runs the official PR #1744 bench on your hardware,
   emits a JSON to `benchmarks/`, prints the side-by-side speedup table

**Isn't** the MTP implementation itself. That's
[SamuelOliveirads' PR #1744](https://github.com/ikawrakow/ik_llama.cpp/pull/1744),
which is now merged into `ik_llama.cpp` main. This repo is the harness that
makes verifying it on your box take 15 minutes instead of a Saturday yak-shave.

## Why you'd use this

| | Build from `ik_llama.cpp` main | This repo |
|---|---|---|
| Resolve the conversion-script + drafter-GGUF maze yourself | ✅ all manual | ❌ scripted |
| Pick a known-stable base SHA | ✅ manual | ❌ pinned (`9895026`) |
| Get drafter pointers without HF spelunking | ❌ DIY | ✅ linked + verified |
| Side-by-side baseline vs MTP bench in one command | ❌ DIY | ✅ `run_bench.sh` |
| JSON output you can `diff` vs reference numbers | ❌ DIY | ✅ schema-stable, two reference JSONs in tree |
| Hot-swap integration with a Q8 baseline service | ❌ DIY | ✅ `swap_controller_integration.py` |

If you already have a CI pipeline against `ik_llama.cpp` main, you don't
need this. If you want the *"clone, run two scripts, get a number"*
experience, this is it.

## Verified speedup (lossless)

| Hardware | Target model | Drafter | `--draft-max` | Baseline | With MTP | Speedup | Acceptance |
|---|---|---|---|---|---|---|---|
| AMD EPYC 9655 (96C, 1.13 TB DDR5) | Gemma 4 31B Q4_K_M | Q8_0 (510 MB) | 3 | 7.05 t/s | 21.02 t/s | **2.98×** | 84.3% per-token |
| Mixed CPU + RTX 3090 (PR author) | Gemma 4 31B Q8_0 | Q8_0 | 3 | 21.7 t/s | 56.1 t/s | **2.59×** | 85.2% |
| Mixed CPU + RTX 3090 (PR author) | Gemma 4 31B Q8_0 | Q8_0 | 2 | 21.7 t/s | 49.5 t/s | **2.28×** | 94.7% |
| **Dual RTX 3090** (this repo) | Gemma 4 31B Q8_0 | Q8_0 | 3 | _bench in flight_ | _bench in flight_ | _est 2.5–3.0×_ | _est 75–85%_ |

> **"Lossless"** = rejection-sampled speculative decoding. Every accepted token
> is sampled from the *target model's* distribution. Output is mathematically
> identical to running the target without MTP at the same temperature/seed.
> The drafter just shortcuts the path; the target verifies.

Reproducing any row: `scripts/run_bench.sh` writes a JSON with the full prompt,
temperature, seed, and tokens-per-second to `benchmarks/<hostname>-<date>.json`.
We diff this against `benchmarks/pr-1744-author.json` for the headline number.

## Quick start (Linux / WSL)

```bash
git clone https://github.com/karany97/llamacpp-gemma4-mtp.git
cd llamacpp-gemma4-mtp
./scripts/apply_patches.sh        # clones ik_llama.cpp @ 9895026, applies 0001+0002+0003
./scripts/build_cuda_linux.sh     # 5–15 min depending on CPU
./scripts/run_bench.sh            # uses your existing Gemma 4 31B Q8 GGUF if present
```

Pre-built binaries (when published): see the [Releases](https://github.com/karany97/llamacpp-gemma4-mtp/releases) tab.

### Windows / WSL

```powershell
git clone https://github.com/karany97/llamacpp-gemma4-mtp.git
cd llamacpp-gemma4-mtp
bash ./scripts/apply_patches.sh                            # WSL or git-bash
pwsh ./scripts/build_cuda_windows.ps1                      # CUDA 12.6 + MSVC, SM 8.6 + 8.9
bash ./scripts/run_bench.sh
```

### CPU-only (no GPU available)

```bash
./scripts/apply_patches.sh
./scripts/build_cpu.sh
./scripts/run_bench.sh             # uses pure CPU pathways; see EPYC row above for ballpark
```

## Launch flags (verbatim from PR #1744)

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

`--draft-max 3` is the sweet spot from the author's benchmark; bump to 4 for
code-heavy workloads, drop to 2 for storytelling / temperature > 0.7.

## Drafter weights

Drafter GGUFs are only ~510 MB at Q8_0. Two options:

1. **Pre-converted (community)** — [Radamanthys11/Gemma-4-31B-it-assistant-GGUF](https://huggingface.co/Radamanthys11/Gemma-4-31B-it-assistant-GGUF) — F16 (945 MB) + Q8_0 (515 MB)
2. **Convert yourself** — the merged PR #1744 ships an updated `convert_hf_to_gguf.py`
   that handles `Gemma4AssistantForCausalLM`. Pull Google's original from
   [`google/gemma-4-31B-it-assistant`](https://huggingface.co/google/gemma-4-31B-it-assistant) and run:
   ```bash
   python convert_hf_to_gguf.py --outtype q8_0 ./gemma-4-31B-it-assistant
   ```

## Why this repo still exists post-merge

Three reasons that didn't go away when PR #1744 merged:

1. **Reproducible numbers** — the merged feature gives you the *capability*;
   this gives you a fixture (prompt, seed, drafter, flags) so your bench
   results are comparable to the PR author's reference run.
2. **Defensive patches that auto-skip** — `0002` and `0003` are the
   pestopoppa fixes that were merged INTO PR #1744 between SHA `9895026`
   (which we pin) and the merge commit. The patches auto-detect a clean
   `main` and skip themselves; they exist so this harness keeps working
   if you point it at an older SHA for a regression bisect.
3. **Swap-controller drop-in** — a 70-line Python module that adds a
   `gemma-mtp` slot to a `llm_swap_controller.py`-style controller so MTP
   becomes a hot-swap option alongside Q8 baseline, not a separate process
   tree. Not in `ik_llama.cpp`; not in `llama.cpp`; lives here.

## Attribution

This repo is a **packaging + bench harness layer**. Credit for the work:

- [@SamuelOliveirads](https://github.com/SamuelOliveirads) — author of
  [ik_llama.cpp PR #1744](https://github.com/ikawrakow/ik_llama.cpp/pull/1744),
  the actual MTP arch + graph + conversion script
- [@pestopoppa](https://github.com/pestopoppa) — discovered the
  `params_use_gemma4_external_mtp` chicken-and-egg bug and the cosmetic
  tensor-name warnings; both fixes credited inline in `patches/0002` and `patches/0003`
- [@ikawrakow](https://github.com/ikawrakow) — `ik_llama.cpp` itself (the
  SOTA-quants fork of `llama.cpp` this all builds on)
- [@Radamanthys11](https://huggingface.co/Radamanthys11) — community drafter GGUFs
- Google / Gemma team — Gemma 4 + MTP drafter weights
  ([blog post](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/))

## Repository layout

```
.
├── README.md
├── LICENSE                                       MIT (our scripts) — PR #1744 inherits ik_llama.cpp MIT
├── CHANGELOG.md                                  release notes (v0.1.0, v0.2.0)
├── CONTRIBUTING.md                               how to add a hardware row
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   └── bench-report.yml                      structured hardware/model/draft-max → tok/s
│   └── workflows/
│       └── patches-apply.yml                     CI: scripts/apply_patches.sh exit-0 on Ubuntu + macOS
├── patches/
│   ├── COMMITS.txt                               SHA list from PR #1744
│   ├── 0001-PR-1744-gemma4-mtp.patch             1193+/154-, 22 files
│   ├── 0002-fix-segfault-params-use-gemma4-external-mtp.patch
│   └── 0003-silence-mtp-tensor-name-warnings.patch
├── scripts/
│   ├── apply_patches.sh                          idempotent clone + apply
│   ├── build_cuda_linux.sh
│   ├── build_cuda_windows.ps1
│   ├── build_cpu.sh
│   ├── run_bench.sh
│   └── swap_controller_integration.py
├── benchmarks/
│   ├── pr-1744-author.json                       SamuelOliveirads's reference numbers
│   ├── pestopoppa-epyc.json                      pure-CPU EPYC bench
│   └── (your hardware lands here when you run scripts/run_bench.sh)
└── docs/
    ├── algorithm.md                              MTP rejection-sampling math, why it's lossless
    ├── architecture.md                           Gemma4Assistant tensor layout
    └── troubleshooting.md
```

## License

Our scripts: [MIT](./LICENSE). The `ik_llama.cpp` code we patch retains its
original MIT license. Gemma 4 weights are governed by the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Status

- **2026-05-08** — repo created, PR #1744 packaged at SHA `9895026`,
  two pestopoppa fixes captured as defensive patches
- **2026-05-10** — PR #1744 **merged upstream** ✅
- **2026-05-16** — repo re-framed as bench harness + swap-controller layer;
  dual-3090 row in flight
- **Next** — v0.2.0 release with pre-built Linux binary; CI badge

Pull requests welcome — especially benchmarks from hardware not yet covered
(see [CONTRIBUTING.md](./CONTRIBUTING.md) for the hardware-row template).
