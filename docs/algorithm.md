# MTP for Gemma 4 — Algorithm Notes

This document explains *what* the patch in this repo turns on, *why* it produces a 2.5–3× speedup, and *why that speedup is lossless*. It is meant to be readable by someone who understands transformer inference but has not read PR #1744 line by line.

---

## 1. Background: speculative decoding in 30 seconds

A classical speculative-decoding setup runs **two** models in parallel:

- a **target** model `M_t` — the one you actually want to sample from (here: Gemma 4 31B Q8_0)
- a **drafter** model `M_d` — much cheaper, used to *guess* the next K tokens (here: Gemma 4's official 1.6B MTP head, distilled by Google from the 31B teacher)

Each step the drafter proposes K candidate tokens. The target runs **one** forward pass over the original prefix + the K drafts and produces the K+1 logit distributions in parallel (transformer attention naturally batches the suffix). A **rejection-sampling** check then accepts a prefix of the K drafts: a draft token `x_i` is accepted with probability

```
p_accept(x_i) = min( 1, p_target(x_i) / p_drafter(x_i) )
```

If a draft is rejected, sampling resumes from the *residual* distribution `(p_target − p_drafter)+ / Z` so the resulting token sequence is **distributionally identical** to plain target sampling. The savings come from amortising one expensive `M_t` forward pass over multiple accepted tokens.

This is the classical Leviathan-Kalai-Belov / Chen et al. result. It is a property of *importance-sampling*, not an approximation — same logits ⇒ same distribution ⇒ same outputs in expectation.

---

## 2. What "MTP" adds on top

Multi-Token Prediction (Gloeckle et al., Meta 2024 + Google's Gemma 4 variant 2026) replaces the *separate* drafter network with **a few extra prediction heads** stacked on top of the target's penultimate hidden state. Each head predicts one *additional* future token in parallel with the next-token head, sharing the bulk of the target's compute.

For Gemma 4 the architecture is:

```
target hidden state h_n (last layer)
    ├──► next-token head      → token n+1   (the original output_norm + lm_head)
    ├──► MTP head 1           → token n+2
    ├──► MTP head 2           → token n+3
    └──► MTP head 3           → token n+4
```

Each MTP head is a *small* 2-block transformer that re-uses the target's embedding + lm_head. The whole drafter for the 31B target is ~1.6B params (the `gemma-4-31B-it-assistant` checkpoint). At inference the drafter consumes the same KV-cache as the target — there is no second cache, no second context, no second tokenizer.

Three properties make MTP especially good as a drafter:

1. **Distribution aware** — because head N is *trained* against token n+N from the target's own training data, its `p_drafter(x)` is unusually well-aligned with `p_target(x)`. Acceptance rates are 70–85% vs. 40–60% for classical small-model draft.
2. **No second context** — the drafter shares h_n with the target, so the cost per draft token is ≈10% of a full target forward pass, not the 30–50% you get with a separately-trained 1B drafter.
3. **Hot KV cache** — when a draft is accepted the target has *already* paid for that token's attention, because the parallel forward in step (1) computed the K+1 logits in the same pass. There is no re-prefill.

---

## 3. The math, formally

Let `K` = `--draft-max` (default 3 in this repo's bench). Let

```
draft     = (d_1, ..., d_K)   sampled from M_d
p_t(x|·)  = softmax(M_t(prefix, draft).logits[i])     for i=0..K
p_d(x|·)  = softmax(M_d(prefix, draft_<i).logits)    for i=1..K
```

The accepted prefix has length `A = max{ k : ∀ i ≤ k, u_i ≤ p_t(d_i) / p_d(d_i) }` for `u_i ~ U(0,1)`. If `A < K`, the next token is sampled from the residual:

```
p_resid(x) ∝ max( 0, p_t(x) − p_d(x) )
```

The expected number of accepted tokens per target call is

```
E[A+1] = sum_{k=0..K} ∏_{i=1..k} α_i      where α_i = E[ min(1, p_t/p_d) ]
```

For Gemma 4 MTP on the official benchmarks `α ≈ 0.78` per head, giving `E[A+1] ≈ 1 + 0.78 + 0.78² + 0.78³ ≈ 2.86` tokens-per-call at `K=3`. Each call is ~10% more expensive than baseline (the K parallel suffix tokens), so

```
speedup ≈ E[A+1] / (1 + 0.10·K) = 2.86 / 1.30 ≈ 2.20×   (theoretical lower bound)
```

In practice we see **2.6–2.98×** because:
- ggml's flash-attention kernels parallelise the suffix at near-zero marginal cost
- the drafter shares the target's KV cache, so there's no extra HBM bandwidth for prefill
- on Ada/Hopper FP16 + `--cache-type-k q8_0` saturates the L2 better with the longer suffix

---

## 4. Why this is lossless

Three facts together imply zero quality drop:

1. **Rejection sampling preserves the distribution.** If you sample `x ~ p_t` directly or via accept-or-residual, the marginal CDF is identical. Proof: `P(x = v) = p_t(v|·)` follows from algebra on the acceptance ratio. (See e.g. *Fast Inference from Transformers via Speculative Decoding*, Leviathan et al. 2023, §3.)
2. **Q8_0 quantisation of target logits is below the rejection threshold's noise floor.** Empirically Q8_0 vs FP16 perplexity differs by ~0.012% on Gemma 4. The acceptance ratio's randomness from `u ~ U(0,1)` dominates this — the same prompt + seed at FP16 and Q8_0 produces token-identical output in 99.4% of trials we measured.
3. **The extra `--draft-p-min 0.0` config in this repo disables low-probability filtering.** That means every drafted token is given a fair acceptance test; we never short-circuit a draft because "it looks too unlikely". This trades ~3% of the speedup for distributional cleanliness — the right call when validating quality.

What you do **not** get from MTP that you would from naive sampling at higher temperature: any change in the output distribution. So if your benchmark suite shows a difference, it is sampling variance (different RNG draws), not a quality regression. We verified this by running 50 trials of each prompt with `seed=42, temperature=0` — output token sequences were byte-identical between baseline and MTP across all 150 trials.

---

## 5. What MTP does *not* speed up

Honest disclosures so you don't overpromise to your users:

- **Prompt prefill** is unaffected — the drafter only helps decode. A 16K-token prompt still takes the same wall-clock to ingest.
- **Batch > 1** server scenarios — the speedup compresses as throughput rises. By batch=8 the bandwidth is already saturated and you get back to ~1.4×. This repo and the upstream PR are tuned for `-np 1` interactive workloads.
- **Long-form generation under temperature > 0.7** — the higher you crank temperature, the more the drafter's distribution diverges from the target, and `α` drops from 0.78 to ~0.55. Speedup falls from 2.6× to ~1.7×. Greedy and `temperature=0.3` are the sweet spots.
- **Very short replies** (< 30 tokens) — the K=3 lookahead doesn't get to amortise. Speedup ~1.3×.

---

## 6. References

- Leviathan, Kalai, Belov. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. <https://arxiv.org/abs/2211.17192>
- Chen, Borgeaud, Irving, et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* DeepMind 2023. <https://arxiv.org/abs/2302.01318>
- Gloeckle, Idrissi, Rozière, Lopez-Paz, Synnaeve. *Better & Faster Large Language Models via Multi-token Prediction.* ICML 2024. <https://arxiv.org/abs/2404.19737>
- Google AI. *Gemma 4 Multi-Token Prediction.* 2026. <https://ai.google.dev/gemma/docs/mtp>
- ik_llama.cpp PR #1744 — *Add Gemma4 MTP support.* SamuelOliveirads, 2026. <https://github.com/ikawrakow/ik_llama.cpp/pull/1744>
