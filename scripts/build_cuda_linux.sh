#!/usr/bin/env bash
# Build ik_llama.cpp + Gemma 4 MTP patches with CUDA on Linux / WSL2.
#
# Pre-req: CUDA Toolkit 12.x, gcc 11+, cmake 3.18+, ccache (optional but speeds rebuild)
#
# Usage:
#   ./scripts/build_cuda_linux.sh                 # auto-detect arch (sm_86 default for 3090)
#   CUDA_ARCH=86 ./scripts/build_cuda_linux.sh    # explicit Ampere 3090
#   CUDA_ARCH=89 ./scripts/build_cuda_linux.sh    # Ada 4090
#   CUDA_ARCH=80 ./scripts/build_cuda_linux.sh    # Ampere A100
#   CUDA_ARCH=120 ./scripts/build_cuda_linux.sh   # Blackwell B200/RTX PRO 6000
#   CUDA_ARCH="86;89" ./scripts/build_cuda_linux.sh   # multi-arch fat binary

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-${REPO_ROOT}/build/ik_llama.cpp}"
CUDA_ARCH="${CUDA_ARCH:-86}"  # default sm_86 covers RTX 3090 / 3090 Ti

log() { printf '\033[1;34m[build_cuda_linux]\033[0m %s\n' "$*"; }

[[ -d "$SRC" ]] || { echo "Source tree not found at $SRC. Run scripts/apply_patches.sh first."; exit 1; }

# Pre-flight
command -v nvcc >/dev/null || { echo "nvcc not found in PATH. Install CUDA Toolkit 12.x."; exit 1; }
command -v cmake >/dev/null || { echo "cmake not found. apt install cmake"; exit 1; }
log "nvcc: $(nvcc --version | grep release | head -1)"
log "cmake: $(cmake --version | head -1)"
log "target arch(s): $CUDA_ARCH"

cd "$SRC"
mkdir -p build && cd build

log "configuring (release, CUDA, server, full kernels)"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_F16=ON \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DGGML_CCACHE=ON 2>&1 | tail -5

log "compiling (parallel)"
JOBS="${JOBS:-$(nproc)}"
cmake --build . -j"$JOBS" --target llama-server llama-cli llama-bench 2>&1 | tail -20

log "build complete — binaries:"
ls -la bin/llama-server bin/llama-cli bin/llama-bench 2>&1 | sed 's/^/  /'

log "smoke test: llama-server --version"
./bin/llama-server --version | head -3

log "done. Try:"
log "  ./bin/llama-server -m <gemma-4-31B-Q8_0.gguf> --spec-type mtp -md <gemma-4-31B-it-assistant-Q8_0.gguf> -ngl 99 -ngld 99 --draft-max 3 --port 8005"
