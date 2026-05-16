# Contributing to llamacpp-gemma4-mtp

The most useful thing you can contribute is **a bench row from your
hardware**. The MTP speedup is workload-dependent (CPU/GPU ratio,
memory bandwidth, draft acceptance rate), so every row from a different
configuration sharpens the table.

## Adding a hardware bench row

### 1. Run the bench

```bash
git clone https://github.com/karany97/llamacpp-gemma4-mtp.git
cd llamacpp-gemma4-mtp
./scripts/apply_patches.sh
./scripts/build_cuda_linux.sh    # or build_cuda_windows.ps1 / build_cpu.sh
./scripts/run_bench.sh           # auto-emits benchmarks/<hostname>-<YYYYMMDD>.json
```

`run_bench.sh` runs both **baseline** (no `--spec-type mtp`) and **with MTP**
(the launch flags from the README) on the same prompt, same seed, same
drafter, then writes the JSON.

### 2. Verify the JSON

The schema (excerpt — see `benchmarks/pr-1744-author.json` for the full
example):

```json
{
  "hardware": {
    "cpu": "AMD EPYC 9655 (96C, 1.13 TB DDR5)",
    "gpu": null,
    "ram_gb": 1152,
    "os": "Ubuntu 24.04 LTS"
  },
  "model": {
    "target": "gemma-4-31B-it-Q4_K_M.gguf",
    "drafter": "gemma-4-31B-it-assistant-Q8_0.gguf",
    "context_len": 65536
  },
  "flags": {
    "draft_max": 3,
    "draft_p_min": 0.0,
    "ngl": 999,
    "ngld": 99
  },
  "baseline": {
    "tokens_per_second": 7.05,
    "prompt_eval_tps": 168.4
  },
  "with_mtp": {
    "tokens_per_second": 21.02,
    "prompt_eval_tps": 168.7,
    "acceptance_rate": 0.843
  },
  "speedup": 2.98,
  "ik_llama_cpp_sha": "9895026",
  "patches_applied": ["0001", "0002", "0003"],
  "bench_date": "2026-05-16",
  "submitter": "your-github-handle"
}
```

### 3. Open a PR

Branch name: `bench/<hostname>-<date>` (e.g. `bench/titan-dual-rtx3090-20260516`).

PR title: `bench: add <hardware> row`

PR body must include:

- One paragraph describing the workload (any prompt class — code, prose,
  multi-turn — that's different from the existing rows)
- The verbatim `run_bench.sh` output (terminal capture)
- The new JSON file under `benchmarks/<hostname>-<YYYYMMDD>.json`
- An updated row in the README's "Verified speedup" table

The CI will (when wired) verify the JSON shape against the schema before
the PR is reviewable.

## Adding a fixture to `run_bench.sh`

If you want to add a different prompt class to the bench (current default
is the PR-1744 author's mixed prose+code prompt), open an issue first to
discuss the rationale. Adding fixtures changes the comparability of every
existing row, so it's a multi-day discussion not a one-PR change.

## Filing bugs

Use the `bench-report` issue template if your bench failed or produced
suspicious numbers (e.g. acceptance < 50%, speedup < 1.5×). Include:

- The verbatim error output from `run_bench.sh`
- Your hardware JSON (even partial)
- Your `ik_llama_cpp_sha` (from `./build/ik_llama.cpp/.git/HEAD`)
- The `nvidia-smi` / `lscpu` output if relevant

## Code changes (rare)

This repo is intentionally thin — patches + scripts. If you find a bug in
a script, send a PR with:

- One commit, focused on the bug
- A line in CHANGELOG.md under `[Unreleased]`
- An `Issue:` line in the commit body if it relates to an open issue

If you find a bug in `ik_llama.cpp` proper, please report it
[upstream](https://github.com/ikawrakow/ik_llama.cpp/issues), not here.

## Code of conduct

Be kind to the people who built the things this repo packages — they did
the actual work. Specifically: PR #1744 was authored by SamuelOliveirads;
the defensive patches credit pestopoppa; `ik_llama.cpp` itself is
ikawrakow. Issues that complain about upstream behavior get redirected
politely.
