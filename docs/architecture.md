# Gemma4Assistant Architecture & GGUF Layout

This document explains *what files the patch in this repo introduces*, *what the new tensors are*, and *how the drafter shares state with the target*. Read this before opening `src/graphs/build_gemma4.cpp` so the code makes sense.

---

## 1. Two model archs, one weight file family

PR #1744 introduces a new HuggingFace architecture id, **`Gemma4Assistant`**, which the `convert_hf_to_gguf.py` patch maps to GGUF arch **`gemma4_mtp`**. So you actually deal with three model checkpoints:

| Role     | HF id                                     | HF arch          | GGUF arch        |
| -------- | ----------------------------------------- | ---------------- | ---------------- |
| target   | `google/gemma-4-31B-it`                   | `Gemma4ForCausalLM` | `gemma4`         |
| drafter  | `google/gemma-4-31B-it-assistant`         | `Gemma4Assistant`   | `gemma4_mtp`     |
| (also)   | `google/gemma-4-9B-it-assistant`          | `Gemma4Assistant`   | `gemma4_mtp`     |

The drafter is *not* a freestanding LM — it has no embedding table of its own, no LM head of its own. At runtime ik_llama.cpp loads it as a sidecar to the target, which means:

1. The target's `token_embd.weight` is shared with the drafter (saves 800 MB on disk if you let the loader dedupe).
2. The drafter's `output_norm.weight` is *missing* by design — it re-uses the target's. The patches/0003 change in this repo silences the previously-noisy "Oops: tensor with strange name" warnings that flagged this.
3. The drafter's KV cache is populated by *the target's forward pass*. The drafter only consumes the last hidden state `h_n`.

This sharing is what makes MTP cheap — see `docs/algorithm.md` §2 for the cost analysis.

---

## 2. New tensors introduced by the drafter

Beyond the standard `blk.0..N.{attn_*,ffn_*,attn_norm,ffn_norm}` per-layer tensors, the drafter contributes four top-level (non-`blk.*`) tensors. These are the four names that previously triggered "strange name" warnings on load:

| Tensor name              | Shape (Gemma 4 31B drafter) | Purpose                                                                   |
| ------------------------ | --------------------------- | ------------------------------------------------------------------------- |
| `mtp_pre_proj.weight`    | `[hidden_dim, hidden_dim]`  | projects target's `h_n` into the drafter's residual stream                |
| `mtp_post_proj.weight`   | `[hidden_dim, hidden_dim]`  | projects drafter's residual back to target embedding space before lm_head |
| `mtp_centroids.weight`   | `[K, hidden_dim]`           | learned position embeddings for the K future tokens (K=3 for 31B)         |
| `mtp_token_ordering.weight` | `[K, K]`                 | causal mask / ordering matrix for the K-head decode (lower-triangular)    |

`hidden_dim` for Gemma 4 31B is 5120; K = 3 gives a centroids tensor of `[3, 5120] = 15360 fp16 = 30 KB`. The pre/post proj are the bulk of the new params (~26M each at 5120²).

---

## 3. Files added or substantially changed

The 22 files PR 1744 touches break down like this:

### New files (created)

```
src/graphs/build_gemma4.cpp        +282      The MTP-aware build_gemma4 graph
include/graphs/build_gemma4.hpp    +18       Header for above
common/mtp_config.hpp              +25       Runtime config struct passed through
```

### Significantly modified

```
convert_hf_to_gguf.py              +261/-12  Adds Gemma4Assistant -> gemma4_mtp converter
                                              + handles the 4 new tensors above
src/llama.cpp                      +130/-8   Tensor accounting (skip mtp_*.weight from
                                              per-layer audit), arch-dispatch for
                                              create_gemma4_mtp_tensors
examples/server/server-context.cpp +200/-15  --spec-type mtp parsing, drafter loader,
                                              forward orchestration with target
common/speculative.cpp             +62/-5    Rejection-sampling loop variant for MTP
common/common.h / common.cpp       +24       New CLI flags (--spec-type, -md, -ngld,
                                              --draft-max, --draft-p-min)
ggml/src/ggml-cuda/*.cu             +30      Minor kernel tweaks for the K-token
                                              parallel attention call
src/llama-arch.cpp                  +18      LLM_ARCH_GEMMA4_MTP enum entry
src/llama-hparams.cpp               +12      load_hparams_gemma4_mtp
```

The remaining 12 files are 5-30-line additions wiring the new arch through GGUF metadata, model loaders, sampler config, and `examples/server/`'s OpenAI-compat layer.

---

## 4. Forward-pass diagram

When `--spec-type mtp` is on, one user prompt step does this:

```
                                 ┌──────────────────────────────────┐
                                 │  Target Gemma 4 31B (loaded once) │
                                 │  KV cache: target's, single copy  │
                                 └──────────────┬───────────────────┘
   prefix tokens                                │
   (n already in KV cache)                      ▼
                                       run forward(prefix)
                                                │
                                       hidden state h_n
                                                │
                                  ┌─────────────┼─────────────┐
                                  │             │             │
                                  ▼             ▼             ▼
                           lm_head(h_n)   mtp_pre_proj(h_n)   ...
                                  │             │
                                  ▼             ▼
                           token n+1     drafter block 1 (2-layer xformer)
                           (greedy/RNG)         │
                                                ▼
                                         mtp_post_proj
                                                │
                                                ▼
                                          lm_head (shared) → token n+2_draft
                                                │
                                                ▼
                                         drafter block 2
                                                ▼
                                          lm_head        → token n+3_draft
                                                ▼
                                         drafter block 3
                                                ▼
                                          lm_head        → token n+4_draft
```

The target then runs **one** more forward pass over `[token n+2_draft, n+3_draft, n+4_draft]` to produce the *true* `p_target` for each, and rejection-sampling decides how many to accept. On accept all of them, the next user step starts at position n+5 — five tokens emitted for the cost of two target forward passes plus three drafter blocks, where each drafter block is ~10% the cost of a target block.

---

## 5. KV cache notes

This is the part most often gotten wrong in third-party MTP implementations:

- **The drafter does not own a separate KV cache.** It reads `h_n` from the target's last layer at the current position, projects it via `mtp_pre_proj`, and runs its 2-block transformer. There is no per-position drafter state to persist.
- **Rejected drafts roll back the target's KV cache.** When draft `d_3` is rejected, the target's KV positions `[n+2, n+3]` were already written by the K-token parallel forward — they are *truncated* (`llama_kv_cache_seq_rm`) before the residual sample is written. The patch in `examples/server/server-context.cpp` handles this — look for `kv_cache_seq_rm` calls inside `slot_loop`.
- **`--cache-type-k q8_0 --cache-type-v q8_0` works fine** with MTP. We tested it; the rejection ratio is unaffected because the drafter's `p_d` is computed in fp16 inside the drafter blocks (only target KV is quantised).

---

## 6. References for code readers

If you're going to read the implementation, read in this order:

1. `convert_hf_to_gguf.py` — understand the GGUF tensor schema first
2. `src/llama-arch.cpp` + `src/llama-hparams.cpp` — see how the arch flag flows
3. `src/graphs/build_gemma4.cpp` — the actual graph builder (this is the meat)
4. `examples/server/server-context.cpp` — runtime orchestration and the rejection loop
5. `common/speculative.cpp` — the math from `docs/algorithm.md` §3 in code form

The two pestopoppa fixes (patches/0002 and 0003) make sense after step 4 — the segfault was a race between drafter-load and slot-init in `server-context.cpp`, and the warning silence is in step 2's tensor accounting.
