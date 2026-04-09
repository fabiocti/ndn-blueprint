# Node Card: High-Precision Regulated Text (HPRT S32)

## Identity

| Field | Value |
|---|---|
| **Node ID** | `hprt_s32` |
| **Domain** | `HPRT` |
| **Subdomain** | `null` |
| **Regime** | `S32` |
| **Checkpoint** | `reg_s32` |
| **Version** | `v1` |
| **Status** | `champion` |
| **Date trained** | `2026-04-07` |
| **Training hardware** | `1x H100 80GB` |

## Purpose

Compresses and reconstructs high-precision regulated text including policy documents, compliance procedures, formal directives, and text with conditional overrides. Trained on synthetic regulated text, this node is optimized for domains where factual precision, override detection, and binding integrity are operationally critical.

## Best Tasks

- Policy documents and compliance text
- Formal procedures and regulatory directives
- Override detection in conditional rule systems

## Failed Tasks

- Complex multi-hop conditionals — no probe exceeds 75%, indicating difficulty with deeply nested logic
- Entity tracking through long chains — TRACK probe at 20% shows poor long-range entity following

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | `gpt2` |
| Sequence length | `128` |
| Num latents (K) | `32` |
| Latent groups | `16,8,8` |
| Learning rate | `1e-4` |
| Warmup ratio | `0.15` |
| Batch size | `32` |
| Epochs | `10` |
| Precision | `bf16` |
| Seed | `137` |
| Dataset | `synthetic_regulated_v1` |
| Dataset size | `200K train / 2K eval` |
| Special loss weights | `N/A` |

## Internal Evaluation (Final Epoch)

| Metric | Value |
|---|---|
| train_loss | — |
| val_loss | `0.0000` |
| first_token_accuracy | `100%` |
| exact_match | `60%` |
| ablation_gap | — |
| shuffled_gap | `2.35` |
| c25 / c50 / c75 | `0.017 / 0.071 / 0.302` |
| numeric_token_acc | — |

## Domain-Native Benchmark Results

| Benchmark | Score | Notes |
|---|---|---|
| REG-CONTROL | 0% cost | Zero reconstruction cost on control set |
| REG-NIAH | 22.2% | Needle-in-a-haystack entity retrieval |
| REG-FACT1 | 45.0% | Single-hop factual binding |
| REG-FACT2 | 50% | Multi-hop binding chain |
| REG-OVERRIDE | 70% | Override detection — most operationally critical probe |
| REG-TRACK | 20% | Entity tracking through passage |
| REG-POS (end) | 70% | Positional bias: end-of-passage accuracy |
| REG-EXACT | 40% all / 76% avg recall | Exact match (full) and average recall |

## Public Benchmark Results (if any)

_No public benchmarks applied._

## Real-World A/B Results (if any)

_No A/B tests conducted._

## Known Weaknesses

1. Hardest domain tested — no individual probe exceeds 75%
2. Complex conditionals are challenging — nested logic and multi-clause rules degrade reconstruction
3. Entity tracking is weak (20%) — long-range entity following through regulated text is unreliable

## Champion / Proxy Status

- [x] Champion for this domain/subdomain
- [ ] Proxy tested on other domains
- [ ] Superseded by newer version
- [ ] Deprecated

## Champion Rationale

S32 chosen over S64 because it scores higher on the two most practically important probes: **FACT1 binding** (45% vs 17.5%) and **OVERRIDE detection** (70% vs 45%). S64 scores higher on FACT2 chains and TRACK, but override detection was judged more operationally critical for regulated text use cases.

## Comparison to Prior Versions (if applicable)

| Version | Key Metric | Value | Notes |
|---|---|---|---|
| S64 candidate | REG-FACT1 | 17.5% | Lost on most critical probe |
| S64 candidate | REG-OVERRIDE | 45% | Lost on override detection |
| **S32 (champion)** | REG-FACT1 | 45.0% | 2.6x improvement over S64 |
| **S32 (champion)** | REG-OVERRIDE | 70% | 1.56x improvement over S64 |

## Provenance

| Item | Details |
|---|---|
| Training script | — |
| Training log | — |
| Checkpoint location | `checkpoints/reg_s32/` |
| model.pt | **Yes (316MB)** |
| Dataset composition | 100% synthetic (synthetic_regulated_v1) |
| Held-out policy | — |
| Data leakage notes | — |

## Notes

_HPRT is the hardest domain tested in the NDN system. Despite low absolute probe scores, S32 was selected as champion over S64 based on operationally critical metrics (FACT1 and OVERRIDE)._

**Retrained 9 Apr 2026 on Verda H100.** Metrics match original: val_loss=0.0000, exact=60%, 1st_tok=100%.
