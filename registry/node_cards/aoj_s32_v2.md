# Node Card: Agent Operational Journals (AOJ S32 v2)

## Identity

| Field | Value |
|---|---|
| **Node ID** | `aoj_s32_v2` |
| **Domain** | `OSA` |
| **Subdomain** | `AOJ` |
| **Regime** | `S32` |
| **Checkpoint** | `aoj_s32_v2` |
| **Version** | `v2` |
| **Status** | `champion` |
| **Date trained** | `2026-04-09` |
| **Training hardware** | `1x H100 80GB` |

## Purpose

Compresses and reconstructs agent operational journals — structured logs from autonomous agent sessions including reconnaissance summaries, tool outputs, multi-target entries, and operational notes. A subdomain of OSA created after the parent OSA node (state_s32) failed as a proxy for agent journals (14% fact recovery). Trained on a mix of 190K synthetic and 10K real journal entries from five OpenClaw targets.

## Best Tasks

- Agent operational journals and session logs
- Reconnaissance summaries and tool output compression
- Multi-target operational entries
- Structured agent action traces

## Failed Tasks

- Exact numeric counts — 58% of missed facts are precise numeric values
- OOV domain names — 25% of missed facts are out-of-vocabulary domain/host names
- Error/status strings — 17% of missed facts are exact error messages or status codes

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
| Dataset | `synthetic_aoj_v2 — 190K synthetic + 10K real (5% real journal mix from 5 OpenClaw targets)` |
| Dataset size | `200K train (190K synthetic + 10K real) / eval` |
| Special loss weights | `digit_weight=3.0, entity_weight=2.5 (0 tokens matched — regex bug)` |
| Domain coverage | 204 SLDs, 52 TLDs |

## Internal Evaluation (Final Epoch)

| Metric | Value |
|---|---|
| train_loss | — |
| val_loss | `0.0016` |
| first_token_accuracy | `100%` |
| exact_match | `100%` |
| ablation_gap | — |
| shuffled_gap | `4.31` |
| c25 / c50 / c75 | — |
| numeric_token_acc | `99.98%` |

## Domain-Native Benchmark Results

_Domain-native probes not separately reported. See Real-World A/B below._

## Public Benchmark Results (if any)

_No public benchmarks applied._

## Real-World A/B Results (if any)

| Test | Fact Recovery | Compression | Continuity | Verdict |
|---|---|---|---|---|
| OpenClaw A/B (real journals) | 73% | 1.69x | 0.92 | MD 4/6, NDN 2/6 — Markdown still wins |

### Missed Fact Taxonomy

| Category | Percentage |
|---|---|
| Exact numeric counts | 58% |
| OOV domain names | 25% |
| Error/status strings | 17% |

## Known Weaknesses

1. Exact numeric counts — the dominant failure mode (58% of missed facts); precise numbers like port counts, IP addresses, and enumeration totals are unreliable
2. OOV domain names — out-of-vocabulary hostnames and domain strings (25% of missed facts) are not reliably preserved
3. Error/status strings — exact error messages and status codes (17% of missed facts) are lossy
4. entity_weight=2.5 was configured but matched 0 tokens due to a regex bug — entity weighting was effectively disabled

## Champion / Proxy Status

- [x] Champion for this subdomain (AOJ under OSA)
- [ ] Proxy tested on other domains
- [ ] Superseded by newer version
- [ ] Deprecated

## Comparison to Prior Versions (if applicable)

| Version | Key Metric | Value | Notes |
|---|---|---|---|
| OSA proxy (baseline) | Fact recovery | 14% | OSA parent node used as proxy — failed on agent journals |
| AOJ v1 | Fact recovery | 54% | First dedicated AOJ training; 3.9x over OSA proxy |
| **AOJ v2 (champion)** | Fact recovery | 73% | 1.35x over v1 (73/54), 5.2x over OSA proxy (73/14) |
| AOJ v3 (attempted) | Fact recovery | 66% | Regressed with stronger loss weighting; v2 remains champion |

## Provenance

| Item | Details |
|---|---|
| Training script | `train_aoj_s32_v2.sh` (reproducible) |
| Training log | — |
| Checkpoint location | `checkpoints/aoj_s32_v2/` |
| model.pt | **Yes (316MB)** |
| Dataset composition | 95% synthetic (190K entries) + 5% real (10K entries from 5 OpenClaw targets) |
| Held-out policy | — |
| Data leakage notes | **A/B test journals were in training data (5%). Model saw test data and still only achieved 73% fact recovery.** |

## Notes

**Retrained 9 Apr 2026 on H100.** val_loss=0.0014, exact=100%, num_tok_acc=99.98%, shuffled_gap=2.36. Entity weight=2.5 still matched 0 tokens (regex bug), so this is effectively digit_weight=3.0 only.

_AOJ was created as a subdomain of OSA after two proxy failures: OSA parent (14% fact recovery) and HWM (24% fact recovery, domain-prior projection). The v2 node achieves 73% fact recovery with 1.69x compression and 0.92 continuity on real agent journals, but Markdown still outperforms NDN compression in head-to-head A/B (MD wins 4/6 vs NDN 2/6)._

_The entity_weight=2.5 parameter was configured but matched zero tokens due to a regex bug, meaning entity loss weighting was effectively disabled during training. Despite this, v2 outperforms v3 (which attempted stronger loss weighting but regressed to 66%)._

_Provenance caveat: The 5% real journal data used in training overlaps with A/B test journals. The model saw its test data during training and still only achieved 73% fact recovery, setting a ceiling for this architecture on this data distribution._
