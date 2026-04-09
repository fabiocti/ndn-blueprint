# Node Card: Conversational Memory (CONV S64 v2)

## Identity

| Field | Value |
|---|---|
| **Node ID** | `conv_s64_v2` |
| **Domain** | `CONV` |
| **Subdomain** | `null` |
| **Regime** | `S64` |
| **Checkpoint** | `conv_s64_v2` |
| **Version** | `v2` |
| **Status** | `champion` |
| **Date trained** | `2026-04-08` |
| **Training hardware** | `1x H100 80GB` |

## Purpose

Compresses and reconstructs multi-turn conversational text including chat histories, dialogue sessions, and cross-session context. Trained on a mix of real dialogue corpora (DialogSum, Blended Skill Talk, OpenAssistant) augmented with ~35K entity-rich synthetic examples. Features overlapping chunking, numeric-aware loss, and entity-rich augmentation to handle the unique challenges of conversational memory.

## Best Tasks

- Multi-turn conversation compression and reconstruction
- Chat history preservation across sessions
- Cross-session context retention

## Failed Tasks

- Specific digit preservation — numeric values degrade during reconstruction
- Entity name fidelity in dense passages — names blur when many entities appear in close proximity
- Isolated local facts — single facts mentioned once without reinforcement can be lost

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
| Dataset | `DialogSum + Blended Skill Talk + OpenAssistant + ~35K entity-rich synthetic` |
| Dataset size | `mixed (real + synthetic augmentation)` |
| Special loss weights | `digit_weight=3.0` |
| Chunking strategy | Overlapping chunking with stride=96 |
| Augmentation | Entity-rich synthetic examples, numeric-aware loss |

## Internal Evaluation (Final Epoch)

| Metric | Value |
|---|---|
| train_loss | — |
| val_loss | `1.821` |
| first_token_accuracy | `20.9%` |
| exact_match | `0%` |
| ablation_gap | `12.00` |
| shuffled_gap | `13.30` |
| c25 / c50 / c75 | `2.29 / 3.12 / 5.80` |
| numeric_token_acc | `86.3%` |

## Domain-Native Benchmark Results

_Domain-native probes not separately reported. See public benchmark below._

## Public Benchmark Results (if any)

| Benchmark | Metric | Value | Baseline | Retention |
|---|---|---|---|---|
| LongMemEval (500 questions) | F1 retention | 94.9% | — | — |
| LongMemEval (500 questions) | KW retention | 93.6% | — | — |
| LongMemEval (500 questions) | Containment retention | 85.1% | — | — |
| LongMemEval (500 questions) | Single-session-user | 113.6% | baseline | Exceeds baseline |
| LongMemEval (500 questions) | Multi-session | 102.5% | baseline | Exceeds baseline |

## Real-World A/B Results (if any)

_No A/B tests conducted._

## Known Weaknesses

1. Specific digits degrade — numeric values are not reliably preserved despite digit_weight=3.0 and 86.3% numeric token accuracy
2. Entity names blur in dense passages — when many entities appear in close proximity, names can be confused or merged
3. Isolated local facts can be lost — single mentions without contextual reinforcement are vulnerable
4. Absolute F1 is low (0.074) — retention percentages are high relative to baseline but absolute quality is modest

## Champion / Proxy Status

- [x] Champion for this domain/subdomain
- [ ] Proxy tested on other domains
- [ ] Superseded by newer version
- [ ] Deprecated

## Comparison to Prior Versions (if applicable)

_v2 is the first champion version for CONV. Prior attempts with synthetic-only templates caused catastrophic overfitting._

## Provenance

| Item | Details |
|---|---|
| Training script | — |
| Training log | — |
| Checkpoint location | `checkpoints/conv_s64_v2/` |
| model.pt | **Yes (316MB)** |
| Dataset composition | Mixed: real corpora (DialogSum, Blended Skill Talk, OpenAssistant) + ~35K entity-rich synthetic |
| Held-out policy | — |
| Data leakage notes | — |

## Notes

**Retrained 9 Apr 2026 on Verda H100.** val_loss=0.1648, gap=9.54, num_tok_acc=97.5%.

_CONV has the highest latent dependence of any domain tested (shuffled_gap=13.30, ablation_gap=12.00), consistent with conversational structure being heavily encoded in the latent. Real data was required for this domain — early attempts with purely synthetic dialogue templates caused catastrophic overfitting, producing nodes that memorized template patterns rather than learning conversational dynamics. LongMemEval shows high F1 retention relative to baseline on that benchmark (94.9% F1; two categories exceed baseline), but absolute F1 remains low at 0.074, and this domain has not been validated on a real-world A/B against markdown._
