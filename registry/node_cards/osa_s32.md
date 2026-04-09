# Node Card: Operational State Artifacts (OSA S32)

## Identity

| Field | Value |
|---|---|
| **Node ID** | `osa_s32` |
| **Domain** | `OSA` |
| **Subdomain** | `null` |
| **Regime** | `S32` |
| **Checkpoint** | `state_s32` |
| **Version** | `v1` |
| **Status** | `champion` |
| **Date trained** | `2026-04-07` |
| **Training hardware** | `1x H100 80GB` |

## Purpose

Compresses and reconstructs timestamped operational state traces. Trained on synthetic state trace data, this node captures sequential state updates, entity tracking through transitions, and override detection. Optimized for structured, temporally ordered event logs with explicit key-value state changes.

## Best Tasks

- Timestamped event traces
- State updates and key-value mutations
- Override detection in sequential logs

## Failed Tasks

- Agent operational journals — 14% fact recovery when tested as proxy for AOJ subdomain; justified creation of dedicated AOJ node
- Long update chains (>20 steps) — accumulates drift over extended sequences
- Unusual entity names — out-of-vocabulary entities reduce accuracy

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
| Dataset | `synthetic_state_traces_v1` |
| Dataset size | `200K train / 2K eval` |
| Special loss weights | `N/A` |

## Internal Evaluation (Final Epoch)

| Metric | Value |
|---|---|
| train_loss | — |
| val_loss | `0.0000` |
| first_token_accuracy | `100%` |
| exact_match | `10%` |
| ablation_gap | `1.905` |
| shuffled_gap | `4.261` |
| c25 / c50 / c75 | `0.106 / 0.363 / 0.999` |
| numeric_token_acc | — |

## Domain-Native Benchmark Results

| Benchmark | Score | Notes |
|---|---|---|
| STATE-NIAH | 99.4% (combined) | All state probes combined |
| STATE-FACT1 | 99.4% (combined) | Included in combined score |
| STATE-FACT2 | 99.4% (combined) | Included in combined score |
| STATE-TRACK | 99.4% (combined) | Included in combined score |
| STATE-OVERRIDE | 99.4% (combined) | Included in combined score |

## Public Benchmark Results (if any)

_No public benchmarks applied._

## Real-World A/B Results (if any)

_No A/B tests conducted._

## Known Weaknesses

1. Long chains (>20 updates) show drift — accumulated errors over extended state sequences
2. Unusual entity names — out-of-vocabulary or rare entity identifiers reduce reconstruction accuracy
3. Expects structured format — performance degrades on unstructured or free-form state descriptions

## Champion / Proxy Status

- [x] Champion for this domain/subdomain
- [x] Proxy tested on other domains — failed as proxy for AOJ (agent operational journals): 14% fact recovery
- [ ] Superseded by newer version
- [ ] Deprecated

## Comparison to Prior Versions (if applicable)

_First version. No prior versions._

## Provenance

| Item | Details |
|---|---|
| Training script | — |
| Training log | — |
| Checkpoint location | `checkpoints/state_s32/` |
| model.pt | Yes (315 MB) |
| Dataset composition | 100% synthetic (synthetic_state_traces_v1) |
| Held-out policy | — |
| Data leakage notes | — |

## Notes

_OSA achieves a very high combined score (99.4%) on in-distribution synthetic state-trace probes, with extremely low cosine distances (c75=0.999). That is not universal fidelity — the same checkpoint failed as a proxy for agent operational journals (14% fact recovery), which justified the AOJ subdomain. The gap between structured state traces and free-form agent journals shows that domain specificity matters even within the same operational category._
