# 01-quick-start.tape
#
# Captures the full clone → apply → build → bench loop in ~90 seconds.
# This is the README hero GIF and the asset for the v0.2.0 announcement
# tweet ("clone, two scripts, bench").
#
# Run:
#   vhs demos/01-quick-start.tape
# Outputs:
#   demos/01-quick-start.gif

Output demos/01-quick-start.gif
Set FontSize 18
Set Width 1000
Set Height 600
Set Theme "Catppuccin Macchiato"
Set TypingSpeed 30ms
Set PlaybackSpeed 1.2

# ───────────────────────────────────────────────────────────────────
# Setup: pretend-clean tmpdir
# ───────────────────────────────────────────────────────────────────
Type "cd /tmp && rm -rf mtp-demo && mkdir mtp-demo && cd mtp-demo"
Enter
Sleep 500ms

# ───────────────────────────────────────────────────────────────────
# 1. Clone
# ───────────────────────────────────────────────────────────────────
Type "git clone --depth 1 https://github.com/karany97/llamacpp-gemma4-mtp.git ."
Enter
Sleep 3s

# ───────────────────────────────────────────────────────────────────
# 2. Apply patches (idempotent, fast — clones ik_llama.cpp @ SHA 9895026)
# ───────────────────────────────────────────────────────────────────
Type "./scripts/apply_patches.sh"
Enter
Sleep 5s

# ───────────────────────────────────────────────────────────────────
# 3. Build (CPU-only path for the demo — CUDA path is ~10x slower to record)
# ───────────────────────────────────────────────────────────────────
Type "./scripts/build_cpu.sh 2>&1 | tail -5"
Enter
Sleep 8s

# ───────────────────────────────────────────────────────────────────
# 4. Bench (script will fall back to a stub if no GGUF present)
# ───────────────────────────────────────────────────────────────────
Type "./scripts/run_bench.sh --quick"
Enter
Sleep 6s

# ───────────────────────────────────────────────────────────────────
# 5. Output evidence
# ───────────────────────────────────────────────────────────────────
Type "ls benchmarks/ && echo '---' && cat benchmarks/$(hostname)-*.json | jq .speedup"
Enter
Sleep 3s

# ───────────────────────────────────────────────────────────────────
# 6. Closer — verbatim exit code
# ───────────────────────────────────────────────────────────────────
Type "echo 'done. exit:' $?"
Enter
Sleep 2s
