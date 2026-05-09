# Troubleshooting

Things that have actually broken for us or for users on PR #1744's discussion thread, with the fix that worked. Sorted roughly by frequency.

---

## Build

### `nvcc fatal: Unsupported gpu architecture 'compute_120'`

**Symptom**: build fails on a Hopper / Blackwell box because you exported `CUDA_ARCH=120` but you're on CUDA Toolkit 12.4 or earlier.

**Fix**: upgrade to CUDA Toolkit 12.6+, or set `CUDA_ARCH=89` (Ada — works on H100/H200 too via PTX JIT, just slower).

### `error C2039: 'string_view': is not a member of 'std'` on Windows

**Symptom**: MSVC under VS 2019 or older can't find C++17 stdlib bits.

**Fix**: install Visual Studio 2022 Build Tools with the "Desktop development with C++" workload. The `cmake -G "Visual Studio 17 2022"` line in `build_cuda_windows.ps1` requires VS 2022.

### `cmake: command not found` after fresh CUDA install

**Symptom**: nvcc is on PATH, cmake isn't.

**Fix**:
- Linux: `apt install cmake` or `dnf install cmake`
- macOS: `brew install cmake`
- Windows: `winget install Kitware.CMake`

### `git apply` fails on `0001-PR-1744-gemma4-mtp.patch`

**Symptom**: `error: patch failed: src/llama.cpp:NNNN`

**Cause**: the upstream `ik_llama.cpp` HEAD has moved past the verified base SHA (`9895026`) and a hunk no longer applies cleanly.

**Fix**: the apply script pins the base SHA explicitly. If you ran `git pull` inside `build/ik_llama.cpp` after `apply_patches.sh`, you're off the pinned base. Re-run `scripts/apply_patches.sh` — it does `git checkout $BASE_SHA` and `git clean -fdx` to reset.

If you *want* to track upstream HEAD: `cd build/ik_llama.cpp && git fetch origin pull/1744/head:pr-1744 && git merge pr-1744`. Conflicts here are not our problem to solve — they belong to the PR author. Open an issue on this repo with the conflict output and we'll re-pin.

---

## Runtime

### Segfault on the first request, hangs on second

**Symptom**: `llama-server` accepts a connection, returns nothing, dies. You see a stack trace ending in `params_use_gemma4_external_mtp`.

**Cause**: the chicken-and-egg precondition that pestopoppa identified — `params_use_gemma4_external_mtp` was checking `params_base.speculative.type == COMMON_SPECULATIVE_TYPE_MTP` *before* speculative.type had been set by the loader.

**Fix**: this is what `patches/0002-fix-segfault-params-use-gemma4-external-mtp.patch` removes. If you somehow ran `apply_patches.sh` and skipped patch 2, segfault returns. Re-run the script — it's idempotent.

### `Oops: tensor with strange name mtp_pre_proj.weight`

**Symptom**: noisy warning on every model load, four lines per drafter.

**Cause**: `src/llama.cpp`'s per-layer tensor audit doesn't know about the four `mtp_*.weight` tensors introduced by the drafter (see `docs/architecture.md` §2). They're top-level, not per-layer, so the loop emits a "strange name" warning.

**Fix**: applied automatically by `patches/0003-silence-mtp-tensor-name-warnings.patch`. If you see these warnings even after the patch chain, your `src/llama.cpp` may have shifted — re-run `scripts/apply_patches.sh`.

### Speedup is < 1.5× on a workload where we say it should be 2.6×

**Diagnostic checklist**:

1. **Are you actually running MTP?** `curl localhost:8005/props | jq .speculative` should return non-null. If it's null, `--spec-type mtp` didn't take.
2. **Is `-ngld` set?** Without `-ngld 99` the drafter runs on CPU even when the target is on GPU. That kills the speedup.
3. **Is `-np 1`?** Multi-slot serving (`-np 4` or more) saturates HBM bandwidth and erases MTP's advantage. We document this in `docs/algorithm.md` §5.
4. **Is `temperature` ≤ 0.5?** Higher temperatures drop the drafter's acceptance rate — see §5 again.
5. **Is your prompt > 30 tokens of expected output?** Very short replies don't amortise the K=3 lookahead.
6. **Is `--draft-max ≥ 3`?** We default to 3 in `run_bench.sh`; lower values trade speedup for safety.

If all six check out and you still see < 1.5×, please open an issue with `nvidia-smi`, your launch flags, and the prompt — we'll add it to the bench suite.

### Crashes on `--cache-type-k q4_0`

**Symptom**: assertion failure inside `ggml_cuda_op_dequantize_block` during a draft step.

**Cause**: q4 KV cache is incompatible with the K-token parallel forward pass — the dequantise kernel expects a contiguous block per position and the K-token suffix breaks that contiguity.

**Fix**: use `--cache-type-k q8_0 --cache-type-v q8_0` (what `run_bench.sh` does). q8 is what PR #1744 was tested with and what we recommend.

### "Drafter loaded successfully, server does not respond"

**Symptom**: model loads, you see `model loaded` in logs, then nothing.

**Cause**: the drafter's `output_norm` is missing-by-design (it shares the target's), but ik_llama.cpp's older `model_load` would print `output_norm.weight = nullptr` and stall waiting on it.

**Fix**: PR #1744 itself includes the fix in `src/llama.cpp` (`if (name == "output_norm.weight") { continue; }`). If you see this stall, your patch chain didn't apply — re-run `scripts/apply_patches.sh`.

---

## Quality / sampling

### Different output between baseline and MTP at `temperature=0`

**Symptom**: same prompt + seed produces different tokens with vs. without MTP.

**Cause**: this should not happen if you have `--draft-p-min 0.0`. If you set `--draft-p-min 0.05` or higher, the drafter short-circuits any token with `p_d < 0.05`, which biases the residual distribution.

**Fix**: keep `--draft-p-min 0.0` for byte-identical greedy decoding. The 3% speedup gained from `0.05` is not worth the bit-exact regression.

### Tokenisation mismatch between `tokens` field and `completion_tokens` count

**Symptom**: you parsed `usage.completion_tokens = 250` but the streamed `data` blocks gave you 247 tokens.

**Cause**: MTP draft tokens that get rejected are *included in the underlying llama.cpp counter* but not emitted to the client. `usage.completion_tokens` over-counts by the rejection rate (~22%).

**Fix**: count tokens client-side from the streamed deltas, or use `chat/completions` instead of `completions` (which corrects for rejections in newer ik_llama.cpp builds). This is a known upstream cosmetic issue; we don't patch it because it would diverge from upstream API.

---

## Drafter availability

### "Where do I get `gemma-4-31B-it-assistant-Q8_0.gguf`?"

The official Google checkpoints (HF id: `google/gemma-4-31B-it-assistant`) are released only as PyTorch tensors. Conversion to GGUF is one command after `apply_patches.sh`:

```
python3 build/ik_llama.cpp/convert_hf_to_gguf.py \
    --outtype q8_0 \
    /path/to/local/clone/of/google/gemma-4-31B-it-assistant
```

Output: `gemma-4-31B-it-assistant-Q8_0.gguf` in the same directory (~514 MB).

Community member `Radamanthys11` re-uploads the converted GGUFs at <https://huggingface.co/Radamanthys11/gemma-4-mtp-drafters-gguf> if you'd rather not run the convert step. Subject to Google's Gemma Terms of Use either way.

### "Can I use the 9B target with the 31B drafter?"

No. The drafter projection matrices (`mtp_pre_proj`, `mtp_post_proj`) are sized to the target's `hidden_dim`. 31B is 5120, 9B is 2048 — they're not interchangeable. Use `gemma-4-9B-it-assistant` for the 9B target.

### "What about quantising the drafter to q4?"

We don't recommend it. The drafter is already only 1.6B params; q4_0 saves ~250 MB and drops acceptance rate by ~5% (so speedup falls from 2.6× to ~2.3×). Net savings are negligible because the drafter never dominates VRAM. Keep it Q8.

---

## When all else fails

1. Run `scripts/apply_patches.sh` from a clean clone — it's the most common cause.
2. Compare your `git log --oneline -5` inside `build/ik_llama.cpp` against the expected output:
   ```
   abc1234 Silence Oops: tensor with strange name mtp_*.weight warnings (pestopoppa)
   def5678 Fix params_use_gemma4_external_mtp chicken-and-egg (pestopoppa, PR 1744 comment)
   bc7eb61f Apply ik_llama.cpp PR #1744 (Gemma 4 MTP)
   c14d10d (the verified base SHA's last commit message — see scripts/apply_patches.sh)
   ...
   ```
3. Open an issue on this repo with: full `apply_patches.sh` output, full `cmake` output, `nvidia-smi`, `cat /etc/os-release`, and your launch command. We aim to triage within 48h.
