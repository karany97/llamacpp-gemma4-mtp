# Changelog

All notable changes to this project will be documented in this file.
Format roughly follows [Keep a Changelog](https://keepachangelog.com/);
no SemVer because the upstream we're tracking is a moving feature, not
a library we ship.

## [Unreleased]

- Dual RTX 3090 (3090 Ti + 3090 asymmetric pair) bench row — *in flight*
- Pre-built `linux-cuda-12.6-x86_64.tar.gz` binary in the v0.2.0 release
- GitHub Actions CI for `scripts/apply_patches.sh` (Ubuntu + macOS)

## [0.2.0] — 2026-05-16

The "PR #1744 merged upstream" pivot. Reframed from *"we package an
unmerged PR"* to *"reproducible bench harness for the now-merged feature,
with a swap-controller integration that lives nowhere else."*

### Changed

- **README pivot**: removed the *"if PR #1744 merges, we'll archive"*
  language; replaced with the three-reasons-to-still-use-this section.
- **Status table**: PR #1744 row now shows `merged 2026-05-10` (verified
  via the GitHub API at commit time).
- **Defensive patches `0002` and `0003`**: clarified that these auto-skip
  on a clean `main` and exist for regression-bisect against the pinned
  base SHA `9895026`.
- **CONTRIBUTING.md added** — documents the hardware-row JSON schema and
  the PR template for adding a new bench row.
- **Issue template** — `.github/ISSUE_TEMPLATE/bench-report.yml` collects
  hardware/model/draft-max/baseline/with-mtp/acceptance into a structured
  form so the README table can be updated mechanically.
- **Topic tags** — added `gemma`, `gemma4`, `llama-cpp`, `ik-llama-cpp`,
  `speculative-decoding`, `mtp`, `llm-inference`.

### Removed

- Empty *"archive this repo on upstream merge"* clause (the upstream
  merged; we kept the repo because the use case shifted but didn't disappear).

### Verified

- `gh api repos/ikawrakow/ik_llama.cpp/pulls/1744` returns
  `state: closed, merged: true, merged_at: 2026-05-10T04:44:20Z`.

## [0.1.0] — 2026-05-08

Initial release. Packaged ik_llama.cpp PR #1744 as a build harness; the
PR was open at the time, so the repo's value-prop was *"the friction-free
way to try the unmerged MTP support today, lossless."*

### Added

- `patches/0001-PR-1744-gemma4-mtp.patch` — SamuelOliveirads' PR squashed
  into a single deterministic patch (1193+/154-, 22 files)
- `patches/0002-fix-segfault-params-use-gemma4-external-mtp.patch` —
  pestopoppa's chicken-and-egg fix originally posted as a PR comment
- `patches/0003-silence-mtp-tensor-name-warnings.patch` — pestopoppa's
  cosmetic fix that silences four `Oops: tensor with strange name mtp_*.weight`
  warnings
- `scripts/apply_patches.sh` — idempotent clone of `ik_llama.cpp` at base
  SHA `9895026` and patch application
- `scripts/build_cuda_linux.sh` / `build_cuda_windows.ps1` / `build_cpu.sh`
- `scripts/run_bench.sh` — runs the official PR #1744 benchmark and emits
  a results JSON
- `scripts/swap_controller_integration.py` — adds a `gemma-mtp` slot to
  a `llm_swap_controller.py`-style controller
- Reference benchmarks: `pr-1744-author.json`, `pestopoppa-epyc.json`
- `docs/algorithm.md` (114 lines), `docs/architecture.md` (140 lines),
  `docs/troubleshooting.md` (148 lines)

### Verified

- `apply_patches.sh` validated on Mac (clones + patches in <10s)
- All three patches apply cleanly to ik_llama.cpp HEAD as of base SHA `9895026`
