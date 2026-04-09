# Node Card: Formal Technical Artifacts (FTA S64)

## Identity

| Field | Value |
|---|---|
| **Node ID** | `fta_s64` |
| **Domain** | `FTA` |
| **Subdomain** | `null` |
| **Regime** | `S64` |
| **Checkpoint** | `code_s64` |
| **Version** | `v1` |
| **Status** | `champion` |
| **Date trained** | `2026-04-07` |
| **Training hardware** | `1x H100 80GB` |

## Purpose

Compresses and reconstructs formal technical artifacts, specifically Python source code at the function level. Trained on CodeSearchNet (Python split), this node captures variable bindings, syntactic structure, and fine-grained code semantics. Optimized for preserving identifier relationships and structural integrity in code.

## Best Tasks

- Python function-level code compression and reconstruction
- Fine-grained variable binding preservation
- Syntax preservation across code blocks

## Failed Tasks

- Non-Python languages — trained exclusively on Python, no cross-language generalization
- String literal recovery — degrades on long or complex string constants
- Variable name substitution — occasional swaps between similarly scoped identifiers

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | `gpt2` |
| Sequence length | `128` |
| Num latents (K) | `64` |
| Latent groups | `32,16,16` |
| Learning rate | `1e-4` |
| Warmup ratio | `0.05` |
| Batch size | `32` |
| Epochs | `10` |
| Precision | `bf16` |
| Seed | `137` |
| Dataset | `code_search_net (Python split)` |
| Dataset size | `200K train / 2K eval` |
| Special loss weights | `N/A` |

## Internal Evaluation (Final Epoch)

| Metric | Value |
|---|---|
| train_loss | — |
| val_loss | `0.225` |
| first_token_accuracy | `100%` |
| exact_match | `0%` |
| ablation_gap | `8.34` |
| shuffled_gap | — |
| c25 / c50 / c75 | — |
| numeric_token_acc | — |

## Domain-Native Benchmark Results

| Benchmark | Score | Notes |
|---|---|---|
| CODE-NIAH | 91% key recall | Needle-in-a-haystack key identifier retrieval |
| CODE-FACT1 | 55% binding | Single-hop variable binding accuracy |
| CODE-FACT2 | 85% chain intact | Multi-hop binding chain integrity |
| CODE-EXACT | 68.3% | Exact reconstruction quality |

## Public Benchmark Results (if any)

_No public benchmarks applied._

## Real-World A/B Results (if any)

_No A/B tests conducted._

## Known Weaknesses

1. Occasional variable name substitution — similarly scoped identifiers may be swapped during reconstruction
2. String literal recovery degrades — long or complex string constants are not reliably preserved
3. Python only — no training on or generalization to other programming languages

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
| Checkpoint location | `checkpoints/code_s64/` |
| model.pt | Yes (316 MB) |
| Dataset composition | 100% real (CodeSearchNet Python) |
| Held-out policy | — |
| Data leakage notes | — |

## Notes

_FTA is the domain for structured code and formal artifacts. The S64 regime provides sufficient latent capacity for fine-grained variable binding. CODE-FACT2 chain integrity (85%) is notably higher than single-hop FACT1 (55%), suggesting the model captures structural relationships better than isolated bindings._
