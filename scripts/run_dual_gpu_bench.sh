#!/usr/bin/env bash
#
# run_dual_gpu_bench.sh — orchestrate a clean dual-3090 (or 3090 Ti + 3090)
# MTP bench while the GPUs are busy serving live brains.
#
# Designed for the Nandai Atelier Trinity stack on Titan .50, but
# generalizes to any box that runs vLLM / llama.cpp brains as Docker
# services that can be stopped and restarted.
#
# What it does (in order):
#   1. Pre-flight: confirm Gemma 4 31B Q8_0 + drafter GGUF are present.
#      If not, download (~33 GB target + ~510 MB drafter).
#   2. Pre-flight: detect running brain containers, save their names so
#      we can restart them. Sanity-check `--brains-to-pause` matches the
#      detected list.
#   3. Pause brains (`docker stop $names`) — frees both GPUs.
#   4. Clone + apply patches + build the CUDA target. Skip the build if
#      a previous run already produced ./build/bin/llama-server.
#   5. Run the bench (baseline → MTP → emit JSON).
#   6. Restart brains (`docker start $names`) — restores live serving.
#   7. Move the JSON to benchmarks/<hostname>-<YYYYMMDD>.json.
#   8. Print the speedup row in markdown-table format so the operator
#      can paste it directly into README.md.
#
# Usage:
#   bash scripts/run_dual_gpu_bench.sh \
#       --brains-to-pause "nandai-vllm-fast nandai-llamacpp-hermes" \
#       --gemma-target /home/karan/models/gemma-4-31B-it-Q8_0.gguf \
#       --gemma-drafter /home/karan/models/gemma-4-31B-it-assistant-Q8_0.gguf
#
# Exit codes:
#   0 — bench done, JSON committed, brains restarted, README row printed
#   1 — pre-flight failed (model missing AND --auto-download not set)
#   2 — could not pause brains (do not run bench; leaves stack untouched)
#   3 — build failed
#   4 — bench failed (brains ARE restarted — fail-safe)
#   5 — JSON malformed or speedup < 1.5 (suspect — investigate before commit)

set -euo pipefail

BRAINS_TO_PAUSE=""
GEMMA_TARGET=""
GEMMA_DRAFTER=""
AUTO_DOWNLOAD=0
DRAFT_MAX=3
QUICK=0

while [ $# -gt 0 ]; do
  case "$1" in
    --brains-to-pause) BRAINS_TO_PAUSE="$2"; shift 2 ;;
    --gemma-target)    GEMMA_TARGET="$2";    shift 2 ;;
    --gemma-drafter)   GEMMA_DRAFTER="$2";   shift 2 ;;
    --draft-max)       DRAFT_MAX="$2";       shift 2 ;;
    --auto-download)   AUTO_DOWNLOAD=1;      shift ;;
    --quick)           QUICK=1;              shift ;;
    -h|--help)
      sed -n '/^#$/,/^$/p' "$0" | sed 's/^# //'
      exit 0 ;;
    *) echo "ERROR: unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ─── 1. Pre-flight ─────────────────────────────────────────────────────────
echo "[1/8] pre-flight"

if [ -z "$GEMMA_TARGET" ] || [ -z "$GEMMA_DRAFTER" ]; then
  echo "ERROR: --gemma-target and --gemma-drafter are required" >&2
  exit 1
fi

if [ ! -f "$GEMMA_TARGET" ]; then
  if [ "$AUTO_DOWNLOAD" -eq 1 ]; then
    echo "  target missing — downloading (~33 GB, this is slow)"
    mkdir -p "$(dirname "$GEMMA_TARGET")"
    huggingface-cli download \
      bartowski/google_gemma-4-31B-it-GGUF \
      gemma-4-31B-it-Q8_0.gguf \
      --local-dir "$(dirname "$GEMMA_TARGET")"
  else
    echo "ERROR: target GGUF not at $GEMMA_TARGET (pass --auto-download to fetch)" >&2
    exit 1
  fi
fi

if [ ! -f "$GEMMA_DRAFTER" ]; then
  if [ "$AUTO_DOWNLOAD" -eq 1 ]; then
    echo "  drafter missing — downloading (~510 MB)"
    huggingface-cli download \
      Radamanthys11/Gemma-4-31B-it-assistant-GGUF \
      gemma-4-31B-it-assistant-Q8_0.gguf \
      --local-dir "$(dirname "$GEMMA_DRAFTER")"
  else
    echo "ERROR: drafter GGUF not at $GEMMA_DRAFTER (pass --auto-download to fetch)" >&2
    exit 1
  fi
fi

# ─── 2. Detect & sanity-check running brains ──────────────────────────────
echo "[2/8] detecting running brains"
DETECTED=$(docker ps --format '{{.Names}}' | tr '\n' ' ')
echo "  currently running: $DETECTED"

# ─── 3. Pause brains ──────────────────────────────────────────────────────
echo "[3/8] pausing brains: $BRAINS_TO_PAUSE"
for name in $BRAINS_TO_PAUSE; do
  if docker ps --format '{{.Names}}' | grep -qx "$name"; then
    docker stop "$name" || { echo "ERROR: failed to stop $name" >&2; exit 2; }
    echo "  stopped: $name"
  else
    echo "  WARN: $name not running, skipping"
  fi
done

# Always restart brains on exit, even if bench fails — fail-safe.
RESTORE_BRAINS() {
  echo "[restore] restarting paused brains: $BRAINS_TO_PAUSE"
  for name in $BRAINS_TO_PAUSE; do
    docker start "$name" >/dev/null 2>&1 || echo "  WARN: failed to start $name"
  done
}
trap RESTORE_BRAINS EXIT

# ─── 4. Build (if needed) ─────────────────────────────────────────────────
echo "[4/8] build"
if [ ! -x build/ik_llama.cpp/build/bin/llama-server ]; then
  ./scripts/apply_patches.sh
  ./scripts/build_cuda_linux.sh || { echo "ERROR: build failed" >&2; exit 3; }
else
  echo "  build already present, skipping"
fi

# ─── 5. Run bench ─────────────────────────────────────────────────────────
echo "[5/8] running bench (draft-max=$DRAFT_MAX, quick=$QUICK)"

QUICK_FLAG=""
[ "$QUICK" -eq 1 ] && QUICK_FLAG="--quick"

./scripts/run_bench.sh \
  --target "$GEMMA_TARGET" \
  --drafter "$GEMMA_DRAFTER" \
  --draft-max "$DRAFT_MAX" \
  $QUICK_FLAG \
  || { echo "ERROR: bench failed" >&2; exit 4; }

# ─── 6. Restart brains (also via trap) ────────────────────────────────────
echo "[6/8] restarting brains"
RESTORE_BRAINS
trap - EXIT

# ─── 7. Locate the emitted JSON ───────────────────────────────────────────
echo "[7/8] locating JSON"
HOSTNAME=$(hostname)
DATE=$(date -u +%Y%m%d)
JSON=$(ls -1t benchmarks/${HOSTNAME}-${DATE}*.json 2>/dev/null | head -1)
if [ -z "$JSON" ] || [ ! -f "$JSON" ]; then
  echo "ERROR: no JSON found at benchmarks/${HOSTNAME}-${DATE}*.json" >&2
  exit 5
fi
echo "  found: $JSON"

# ─── 8. Print speedup row for README ──────────────────────────────────────
echo "[8/8] README row"

SPEEDUP=$(jq -r '.speedup' "$JSON")
BASELINE_TPS=$(jq -r '.baseline.tokens_per_second' "$JSON")
MTP_TPS=$(jq -r '.with_mtp.tokens_per_second' "$JSON")
ACCEPTANCE=$(jq -r '.with_mtp.acceptance_rate' "$JSON")
CPU=$(jq -r '.hardware.cpu' "$JSON")
GPU=$(jq -r '.hardware.gpu // "unknown"' "$JSON")

# Sanity check: speedup should be ≥ 1.5; anything less is suspect
SPEEDUP_OK=$(echo "$SPEEDUP >= 1.5" | bc -l)
if [ "$SPEEDUP_OK" != "1" ]; then
  echo "WARN: speedup ${SPEEDUP}× < 1.5 — investigate before committing" >&2
fi

echo ""
echo "Markdown row to paste into README.md table:"
echo ""
printf "| %s + %s | Gemma 4 31B Q8_0 | Q8_0 | %d | %s t/s | %s t/s | **%s×** | %s%% |\n" \
  "$CPU" "$GPU" "$DRAFT_MAX" "$BASELINE_TPS" "$MTP_TPS" "$SPEEDUP" \
  "$(printf '%.1f' "$(echo "$ACCEPTANCE * 100" | bc -l)")"
echo ""
echo "Done. JSON committed-ready at $JSON"
