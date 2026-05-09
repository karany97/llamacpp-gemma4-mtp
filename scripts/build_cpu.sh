#!/usr/bin/env bash
# CPU-only build of ik_llama.cpp + Gemma 4 MTP patches.
#
# Use this when you have no NVIDIA GPU but still want to benefit from MTP on
# pure CPU (PR 1744 author measured 2.98x on EPYC 7K83). On Ryzen / desktop
# CPUs you should still see ~1.6-2.2x — IO-bound prompts benefit the most.
#
# Pre-req: gcc 11+ (or clang 14+), cmake 3.18+
#
# Usage:
#   ./scripts/build_cpu.sh                 # auto-detect, CMAKE_BUILD_TYPE=Release
#   JOBS=8 ./scripts/build_cpu.sh          # cap parallelism (e.g. on shared box)
#   ./scripts/build_cpu.sh /path/to/dir    # use that source tree

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-${REPO_ROOT}/build/ik_llama.cpp}"

log() { printf '\033[1;34m[build_cpu]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[build_cpu] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ -d "$SRC" ]] || err "Source tree not found at $SRC. Run scripts/apply_patches.sh first."

command -v cmake >/dev/null || err "cmake not found. apt install cmake / brew install cmake"
log "cmake: $(cmake --version | head -1)"

# Detect arch — pass GGML_NATIVE so cpu-feature flags (AVX2/AVX512/F16C/etc.) auto-light up
ARCH="$(uname -m)"
log "host arch: $ARCH"

cd "$SRC"
mkdir -p build && cd build

log "configuring (Release, CPU, server)"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DGGML_LTO=ON 2>&1 | tail -8

log "compiling"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"
cmake --build . -j"$JOBS" --target llama-server llama-cli llama-bench 2>&1 | tail -20

log "build complete:"
ls -la bin/llama-server bin/llama-cli bin/llama-bench 2>&1 | sed 's/^/  /'

log "smoke test: llama-server --version"
./bin/llama-server --version | head -3

log "done. CPU run example:"
log "  ./bin/llama-server -m <gemma-4-31B-Q8_0.gguf> --spec-type mtp -md <gemma-4-31B-it-assistant-Q8_0.gguf> --threads $(nproc 2>/dev/null || sysctl -n hw.ncpu) --draft-max 3 --port 8005"
