#!/usr/bin/env bash
# Reproduce PR 1744's benchmark on the local hardware.
# Outputs: benchmarks/<hostname>-<arch>-<date>.json
#
# Pre-req: ik_llama.cpp built with patches via scripts/build_cuda_linux.sh
#          target GGUF (Gemma 4 31B Q8_0) and drafter GGUF on disk
#
# Env vars (with sensible defaults for our setup):
#   TARGET_GGUF=/path/to/gemma-4-31B-it-Q8_0.gguf
#   DRAFTER_GGUF=/path/to/gemma-4-31B-it-assistant-Q8_0.gguf
#   PORT=8005
#   HOST=0.0.0.0
#   LLAMA_BIN=$REPO_ROOT/build/ik_llama.cpp/build/bin/llama-server

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_BIN="${LLAMA_BIN:-${REPO_ROOT}/build/ik_llama.cpp/build/bin/llama-server}"
TARGET_GGUF="${TARGET_GGUF:-/path/to/gemma-4-31B-it-Q8_0.gguf}"
DRAFTER_GGUF="${DRAFTER_GGUF:-/path/to/gemma-4-31B-it-assistant-Q8_0.gguf}"
PORT="${PORT:-8005}"
HOST="${HOST:-0.0.0.0}"
TIMEOUT="${TIMEOUT:-180}"
HOSTNAME_TAG="${HOSTNAME_TAG:-$(hostname)}"
GPU_TAG="${GPU_TAG:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -s ' ' '-' | tr '[:upper:]' '[:lower:]' || echo cpu)}"

log() { printf '\033[1;34m[bench]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[bench] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[[ -x "$LLAMA_BIN" ]]    || err "llama-server binary not found at $LLAMA_BIN — build first"
[[ -f "$TARGET_GGUF" ]]  || err "target GGUF not at $TARGET_GGUF"
[[ -f "$DRAFTER_GGUF" ]] || err "drafter GGUF not at $DRAFTER_GGUF"

# Three reference prompts from PR 1744 author's benchmark:
declare -A PROMPTS=(
  [code]="Write a complete Python implementation of red-black tree insert and delete operations. Include thorough docstrings, type hints, and a small test harness. Aim for around 1000 tokens of output."
  [extract]="Extract every dollar amount, percentage, and date from the following passage and return them as a JSON object: 'In Q3 2024, the company reported revenue of \$1.2B (up 18.5% YoY), with operating margin at 22.4%. The board declared a quarterly dividend of \$0.84 per share, payable on December 15, 2024. Capital expenditure for the quarter totaled \$340M, of which \$120M went to data-center buildout. The CFO guided FY 2025 revenue between \$5.1B and \$5.4B (a 12-15% growth range), with margins expanding to 24-25%. The previous fiscal year (FY 2023) saw \$3.8B in revenue against \$3.2B in 2022.'"
  [story]="Write a complete short story (around 1000 tokens) about a librarian who discovers a book that rewrites itself based on the reader's hopes. Include vivid descriptions of the library and the protagonist's internal conflict."
)

start_server() {
  local mode=$1 spec_args=$2
  log "starting server [$mode]: $spec_args"
  $LLAMA_BIN \
    -m "$TARGET_GGUF" \
    --port "$PORT" --host "$HOST" \
    -ngl 999 -c 4096 -fa on -np 1 --threads "$(nproc)" \
    --jinja --tensor-split 1,1 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    $spec_args \
    > /tmp/llama-bench-$mode.log 2>&1 &
  SERVER_PID=$!

  # Wait for /health
  local deadline=$((SECONDS + TIMEOUT))
  while ! curl -s "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status"'; do
    if (( SECONDS > deadline )); then
      kill $SERVER_PID 2>/dev/null
      err "server didn't become healthy within $TIMEOUT seconds"
    fi
    sleep 2
  done
  log "server ready (PID $SERVER_PID)"
}

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT

run_prompt() {
  local mode=$1 name=$2 prompt=$3 max_tokens=${4:-1000}
  local t0=$(python3 -c 'import time; print(time.monotonic())')
  local resp
  resp=$(curl -s --max-time 240 -H "Content-Type: application/json" \
    -d "$(python3 -c "import json,sys; print(json.dumps({'model':'gemma','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':$max_tokens,'temperature':0,'seed':42,'stream':False}))" "$prompt")" \
    "http://127.0.0.1:$PORT/v1/chat/completions")
  local t1=$(python3 -c 'import time; print(time.monotonic())')
  python3 -c "
import json, sys
d = json.loads('''$resp''')
ct = d.get('usage',{}).get('completion_tokens', 0)
elapsed = $t1 - $t0
tps = ct/elapsed if elapsed > 0 else 0
print(f'$mode/$name: {ct} tok in {elapsed:.2f}s = {tps:.2f} t/s')
import json as j
j.dump({'mode':'$mode','prompt':'$name','tokens':ct,'elapsed_s':round(elapsed,2),'tok_per_s':round(tps,2)}, open('/tmp/bench_$mode_$name.json','w'))
"
}

mkdir -p "${REPO_ROOT}/benchmarks"
OUT_JSON="${REPO_ROOT}/benchmarks/${HOSTNAME_TAG}-${GPU_TAG}-$(date +%Y%m%d).json"
log "results -> $OUT_JSON"

# === Run baseline (no MTP) ===
start_server baseline ""
for name in code extract story; do
  run_prompt baseline "$name" "${PROMPTS[$name]}" 1000
done
stop_server
sleep 5

# === Run MTP draft-max=3 ===
start_server mtp3 "--spec-type mtp -md $DRAFTER_GGUF -ngld 99 --draft-max 3 --draft-p-min 0.0"
for name in code extract story; do
  run_prompt mtp3 "$name" "${PROMPTS[$name]}" 1000
done
stop_server

# Aggregate
python3 - <<PY > "$OUT_JSON"
import json, glob, os
results = []
for f in sorted(glob.glob('/tmp/bench_*_*.json')):
    results.append(json.load(open(f)))
    os.remove(f)
out = {
    'hostname': os.environ.get('HOSTNAME_TAG', '$HOSTNAME_TAG'),
    'gpu':      '$GPU_TAG',
    'target':   '$TARGET_GGUF',
    'drafter':  '$DRAFTER_GGUF',
    'date':     '$(date -Iseconds)',
    'runs':     results,
}
print(json.dumps(out, indent=2))
PY

log "done. Results in $OUT_JSON"
log "summary:"
python3 -c "
import json
d = json.load(open('$OUT_JSON'))
mode_avg = {}
for r in d['runs']:
    mode_avg.setdefault(r['mode'], []).append(r['tok_per_s'])
for m, vs in mode_avg.items():
    print(f'  {m}: {sum(vs)/len(vs):.2f} t/s avg over {len(vs)} prompts')
if 'baseline' in mode_avg and 'mtp3' in mode_avg:
    base = sum(mode_avg['baseline'])/len(mode_avg['baseline'])
    mtp = sum(mode_avg['mtp3'])/len(mode_avg['mtp3'])
    print(f'  speedup: {mtp/base:.2f}x')
"
