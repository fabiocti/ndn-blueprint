# Node Card: [NODE NAME]

## Identity

| Field | Value |
|---|---|
| **Node ID** | `[e.g., nlk_s32]` |
| **Domain** | `[e.g., NLK]` |
| **Subdomain** | `[e.g., null or AOJ]` |
| **Regime** | `[S16 / S32 / S64]` |
| **Checkpoint** | `[checkpoint filename/path]` |
| **Version** | `[e.g., v2]` |
| **Status** | `[experimental / benchmarked / champion / failed / superseded / deprecated / archived]` |
| **Date trained** | `[YYYY-MM-DD]` |
| **Training hardware** | `[e.g., 1x H100 80GB]` |

## Purpose

_What this node is designed to compress and reconstruct. One paragraph._

## Best Tasks

_List the specific text types and tasks this node handles well._

- [task 1]
- [task 2]
- [task 3]

## Failed Tasks

_List text types and tasks this node was tested on and failed._

- [task 1] — [brief reason for failure]
- [task 2] — [brief reason for failure]

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | `[e.g., gpt2]` |
| Sequence length | `[e.g., 128]` |
| Num latents (K) | `[e.g., 32]` |
| Latent groups | `[e.g., 16,8,8]` |
| Learning rate | `[e.g., 1e-4]` |
| Warmup ratio | `[e.g., 0.15]` |
| Batch size | `[e.g., 32]` |
| Epochs | `[e.g., 10]` |
| Precision | `[e.g., bf16]` |
| Seed | `[e.g., 137]` |
| Dataset | `[dataset name and composition]` |
| Dataset size | `[train / eval sample counts]` |
| Special loss weights | `[e.g., digit_weight=3.0, entity_weight=N/A]` |

## Internal Evaluation (Final Epoch)

| Metric | Value |
|---|---|
| train_loss | |
| val_loss | |
| first_token_accuracy | |
| exact_match | |
| ablation_gap | |
| shuffled_gap | |
| c25 / c50 / c75 | |
| numeric_token_acc | |

## Domain-Native Benchmark Results

| Benchmark | Score | Notes |
|---|---|---|
| [DOMAIN]-NIAH | | |
| [DOMAIN]-FACT1 | | |
| [DOMAIN]-FACT2 | | |
| [DOMAIN]-EXACT | | |
| [other probes] | | |

## Public Benchmark Results (if any)

| Benchmark | Metric | Value | Baseline | Retention |
|---|---|---|---|---|
| [e.g., LongMemEval] | [e.g., F1] | | | |

## Real-World A/B Results (if any)

| Test | Fact Recovery | Compression | Continuity | Verdict |
|---|---|---|---|---|
| [e.g., OpenClaw A/B] | | | | |

## Known Weaknesses

_List specific failure modes with examples where possible._

1. [weakness 1]
2. [weakness 2]
3. [weakness 3]

## Champion / Proxy Status

- [ ] Champion for this domain/subdomain
- [ ] Proxy tested on other domains (list results)
- [ ] Superseded by newer version
- [ ] Deprecated

## Comparison to Prior Versions (if applicable)

| Version | Key Metric | Value | Notes |
|---|---|---|---|
| [prior version] | | | |
| **[this version]** | | | |

## Provenance

| Item | Details |
|---|---|
| Training script | `[path]` |
| Training log | `[path]` |
| Checkpoint location | `[local path or download URL]` |
| Dataset composition | `[% synthetic, % real, sources]` |
| Held-out policy | `[how train/eval are separated]` |
| Data leakage notes | `[any overlap between training and evaluation data]` |

## Notes

_Any additional context, lessons learned, or caveats._
