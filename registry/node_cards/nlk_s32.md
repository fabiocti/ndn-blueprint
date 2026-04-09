# Node Card: Natural Language Knowledge (NLK S32)

## Identity

| Field | Value |
|---|---|
| **Node ID** | `nlk_s32` |
| **Domain** | `NLK` |
| **Subdomain** | `null` |
| **Regime** | `S32` |
| **Checkpoint** | `wiki_s32` |
| **Version** | `v1` |
| **Status** | `champion` |
| **Date trained** | `2026-04-07` |
| **Training hardware** | `1x H100 80GB` |

## Purpose

Compresses and reconstructs encyclopedic factual prose. Trained on English Wikipedia, this node captures structured knowledge articles including entity descriptions, historical facts, scientific summaries, and reference material. Optimized for high entity recall and factual binding in well-structured informational text.

## Best Tasks

- Encyclopedic factual prose
- Reference material and documentation
- Structured knowledge articles with entity-dense content

## Failed Tasks

- Rare proper noun recovery — minor degradation on low-frequency named entities
- High corruption sensitivity — cosine distance degrades steeply at higher corruption levels (c75=12.10)

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
| Dataset | `Wikipedia 20231101.en` |
| Dataset size | `200K train / 2K eval` |
| Special loss weights | `N/A` |

## Internal Evaluation (Final Epoch)

| Metric | Value |
|---|---|
| train_loss | — |
| val_loss | `0.085` |
| first_token_accuracy | `97.0%` |
| exact_match | `0%` |
| ablation_gap | `13.354` |
| shuffled_gap | `18.750` |
| c25 / c50 / c75 | `2.75 / 6.76 / 12.10` |
| numeric_token_acc | — |

## Domain-Native Benchmark Results

| Benchmark | Score | Notes |
|---|---|---|
| WIKI-NIAH | 95% entity recall | Needle-in-a-haystack entity retrieval |
| WIKI-FACT1 | 80% binding | Single-hop factual binding |
| WIKI-FACT2 | 82.5% chain intact | Two-hop factual binding |
| WIKI-EXACT | 94.2% | Exact reconstruction quality |

## Public Benchmark Results (if any)

_No public benchmarks applied._

## Real-World A/B Results (if any)

_No A/B tests conducted._

## Known Weaknesses

1. Minor degradation on rare proper nouns — low-frequency named entities may not reconstruct faithfully
2. High corruption sensitivity — c75 cosine distance of 12.10 indicates steep quality falloff under latent corruption
3. Exact match rate is 0% — reconstructions are semantically faithful but not token-identical

## Champion / Proxy Status

- [x] Champion for this domain/subdomain
- [ ] Proxy tested on other domains
- [ ] Superseded by newer version
- [ ] Deprecated

## Comparison to Prior Versions (if applicable)

_First version. No prior versions._

## Provenance

| Item | Details |
|---|---|
| Training script | — |
| Training log | — |
| Checkpoint location | `checkpoints/wiki_s32/` |
| model.pt | Yes (316 MB) |
| Dataset composition | 100% real (English Wikipedia 20231101) |
| Held-out policy | — |
| Data leakage notes | — |

## Notes

_NLK is the foundational domain for encyclopedic text. This node shows strong entity recall (95%) and factual binding (80%) on domain-native probes, making it the reference champion for well-structured factual prose. The high shuffled_gap (18.750) is consistent with strong latent dependence on token ordering on that evaluation._
