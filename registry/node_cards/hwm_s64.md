# Node Card: Human Working Memory Text (HWM S64)

## Identity

| Field | Value |
|---|---|
| **Node ID** | `hwm_s64` |
| **Domain** | `HWM` |
| **Subdomain** | `null` |
| **Regime** | `S64` |
| **Checkpoint** | `hwm_s64` |
| **Version** | `v1` |
| **Status** | `champion` |
| **Date trained** | `2026-04-07` |
| **Training hardware** | `1x H100 80GB` |

## Purpose

Compresses and reconstructs fragmented, unstructured human working memory text such as personal notes, reminders, to-do items, and loosely organized thoughts. Trained on synthetic HWM data, this node handles text that lacks formal structure and may contain abbreviations, incomplete sentences, and context-dependent references.

## Best Tasks

- Fragmented personal notes and reminders
- Unstructured text with loose organization
- Short-form working memory artifacts (to-do lists, scratch notes)

## Failed Tasks

- Agent operational journals — 24% fact recovery when tested as proxy for AOJ subdomain; failed via domain-prior projection (model projects its HWM prior onto other domains)
- Fine-grained detail preservation — specific details are often lost during reconstruction

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
| Dataset | `synthetic_hwm_v1` |
| Dataset size | `200K train / 2K eval` |
| Special loss weights | `N/A` |

## Internal Evaluation (Final Epoch)

| Metric | Value |
|---|---|
| train_loss | — |
| val_loss | `0.0000` |
| first_token_accuracy | `100%` |
| exact_match | `0%` |
| ablation_gap | — |
| shuffled_gap | `3.87` |
| c25 / c50 / c75 | `0.012 / 0.108 / 0.568` |
| numeric_token_acc | — |

## Domain-Native Benchmark Results

| Benchmark | Score | Notes |
|---|---|---|
| HWM-EXACT | 62.5% | Exact reconstruction quality |
| HWM domain-native probes | — | Domain-native probe suite applied |

## Public Benchmark Results (if any)

_No public benchmarks applied._

## Real-World A/B Results (if any)

_No A/B tests conducted._

## Known Weaknesses

1. Lowest metrics overall — weakest performing champion across all domains
2. Strong domain-prior projection — when applied to out-of-domain text, reconstructions are projected onto HWM-like patterns
3. Fine-grained details often lost — specific facts, numbers, and precise wording degrade during reconstruction

## Champion / Proxy Status

- [x] Champion for this domain/subdomain
- [x] Proxy tested on other domains — failed as proxy for AOJ (agent operational journals): 24% fact recovery via domain-prior projection
- [ ] Superseded by newer version
- [ ] Deprecated

## Comparison to Prior Versions (if applicable)

_First version. No prior versions._

## Provenance

| Item | Details |
|---|---|
| Training script | — |
| Training log | — |
| Checkpoint location | `checkpoints/hwm_s64/` |
| model.pt | Yes (316 MB) |
| Dataset composition | 100% synthetic (synthetic_hwm_v1) |
| Held-out policy | — |
| Data leakage notes | — |

## Notes

_HWM is the weakest champion across all domains, reflecting the inherent difficulty of compressing unstructured, fragmented text. The strong domain-prior projection effect is notable: when this node is applied to non-HWM text (e.g., agent journals), it reconstructs HWM-like output rather than faithfully preserving the input. This behavior was observed during AOJ proxy testing (24% fact recovery) and contributed to the justification for domain-specific nodes._
