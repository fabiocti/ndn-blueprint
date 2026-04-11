# Project Status

**Date**: 11 April 2026
**Reporting standard**: Claims are conservative; limitations, failures, and negative results are stated explicitly.

---

## Proven

These claims have empirical evidence behind them and have survived held-out or real-world evaluation.

### Domain-specific training works
Models trained on domain-specific data consistently outperform cross-domain usage. This has been measured via shuffled_gap (range 1.14–18.75 across domains), ablation_gap, and held-out benchmarks. A model trained on the wrong domain reconstructs text in its own style, not the input style. This is reproducible across all six domains.

### Six top-level domains are validated
NLK, FTA, OSA, HWM, HPRT, and CONV have all been trained, evaluated with internal metrics, and tested with domain-native benchmark suites. Each domain shows distinct characteristics:

| Domain | Champion | val_loss | shuf_gap | Key Benchmark |
|---|---|---|---|---|
| NLK | wiki_s32 | 0.085 | 18.75 | NIAH 95%, EXACT 94.2% |
| FTA | code_s64 | 0.225 | 1.14 | NIAH 91%, FACT2 85% |
| OSA | state_s32 | 0.000 | 4.26 | All probes 99.4% |
| HWM | hwm_s64 | 0.000 | 3.87 | EXACT 62.5% |
| HPRT | reg_s32 | 0.000 | 2.35 | OVERRIDE 70%, FACT1 45% |
| CONV | conv_s64_v2 | 1.821 | 13.30 | LongMemEval 94.9% F1 ret |

### AOJ is a validated subdomain
Agent Operational Journals under OSA. Proxy nodes failed (14-24% fact recovery). Dedicated training achieved 54% (v1), 73% (v2) (test data was in training corpus; see evidence/aoj_subdomain_case.md for caveats), and 99% (v4, FROZEN). AOJ v4 closes the raw fact recovery gap with markdown. The progression from v1 through v4 survives real-world A/B testing.

### Wrong-node failures are diagnostic
When a model trained on domain A reconstructs domain B text, the result reveals domain-specific priors. HWM-trained models reconstruct everything as notes. Wiki-trained models reconstruct code as prose. This phenomenon is consistent and informative.

### Conversation memory improvements were real
CONV-S64 v2 achieved 94.9% F1 retention on LongMemEval (500 questions) vs uncompressed baseline. Three targeted interventions (overlapping chunking, numeric-aware loss, entity-rich augmentation) produced 60% per-question win rate over v1.

### Real conversational data outperformed synthetic templates (organic domains)
Formulaic synthetic templates caused catastrophic overfitting on conversation. Switching to real conversational data (DialogSum + Blended Skill Talk + OpenAssistant) immediately enabled healthy learning. Critical lesson applied across all subsequent domain work.

### Compression works at meaningful ratios
2x compression (S64) and 4x compression (S32) both produce usable reconstructions across all tested domains. Quality varies by domain and content type, but the architecture fundamentally works.

---

## Promising

These results are encouraging but require further validation before strong claims.

### AOJ v4 closes the raw fact recovery gap
AOJ v4 achieves 99% raw fact recovery at 1.78x compression (FROZEN). This closes the gap with markdown's 100%. The remaining bottleneck is retrieval — selecting the right compressed packets for a given query — not reconstruction quality. Historical: v2 achieved 73% (test data was in training corpus, 1.69x compression); v4 supersedes it.

### Repeated-work signals reduced (OpenClaw A/B)
In that test, NDN-compressed memory showed 97.8% fewer repeated-work signals than the markdown arm. That is one metric on one harness; it suggests less redundant-looking context under compression, not a general guarantee, and fact recovery on the same test remained below markdown.

### The architecture scales to many nodes cheaply
Each node is ~70M parameters (~280MB). Seven champion nodes together use ~2GB VRAM. The architecture is designed for many specialized nodes running concurrently, and the memory overhead is trivial compared to the reader LLM. But this has not been tested at scale (>6 nodes).

### Latent dependence varies meaningfully across domains
Conversation shows ablation_gap +12.00 and shuffled_gap +13.30 — far higher than any other domain. This means the latent representation is doing more representational work for conversation than for structured text. The practical implications are still being explored.

### TDR (Top-Down Retrieval) — flagship evaluation
TDR achieves 90% hits, 93% facts, 89x compression. With v4 engine: 17/20 (85%), 91% facts, 86x compression. TDR is the current flagship evaluation harness for measuring end-to-end retrieval + reconstruction quality at high compression ratios.

### RWJ (Real-World Journals) — blooming
RWJ v3 + v4 engine achieves 95% facts on dev set. Held-out: 77.5% facts. The dev/held-out gap indicates room for generalization improvement, but dev performance confirms the v4 engine's reconstruction quality on diverse journal data.

### WS — parked
WS reached a 14/20 ceiling and is PARKED. Retrieval limitations prevent further progress; revisit after retrieval quality improvements.

**Active evaluation status**: 1 blooming (RWJ), 1 parked (WS). TDR is flagship.

---

## Unresolved

### Retrieval is the primary bottleneck
AOJ v4 achieves 99% raw fact recovery at 1.78x compression, effectively closing the reconstruction gap with markdown (100%). The remaining bottleneck is retrieval: recency-based packet selection does not surface the right context for a given query. Historical: AOJ v2 achieved 73% (test data was in training corpus); markdown won on raw retention at that stage.

### Rare entity preservation (largely addressed by v4)
AOJ v4 achieves 99% fact recovery, indicating that rare entity preservation is largely solved at the reconstruction level. The v2-era miss taxonomy (58% exact numeric counts, 25% OOV domain names, 17% error/status strings) no longer represents the primary gap. Remaining misses are dominated by retrieval failures, not reconstruction failures.

### Loss weighting is not the solution to entity preservation
AOJ v3 tested token-ID-based entity weighting (122 tokens at 3.0x) and stronger digit weight (5.0x). Internal metrics improved. Real A/B performance regressed from 73% to 66%. The intervention over-corrected.

### Routing is basic
The current router is rule-based with no calibration, no confidence scoring, and no learned boundaries. Misrouting has been shown to catastrophically degrade quality. A better router is needed before production use.

### No production deployment exists
The OpenClaw integration was a test harness using synthetic session hooks. No live agent is currently running with NDN memory.

### Scaling governance is undefined
The taxonomy has 6 domains and 1 subdomain. How to govern taxonomy growth at scale (dozens of domains, hundreds of subdomains) is an open organizational and technical problem.

---

## Failed / Falsified

### AOJ v3: Token-level loss weighting regressed real-world performance
Token-ID entity weighting (3.0x on 122 entity tokens) + digit weight (5.0x) produced better internal metrics (lower val_loss, earlier exact match, higher shuffled_gap) but regressed real A/B fact recovery from 73% (test data was in training corpus) to 66%. The intervention over-corrected, optimizing for entity token reproduction at the expense of broader contextual recall. This disproves "harder entity pressure = better real-world recall."

### Entity weight via regex: complete failure
In AOJ v2, entity weighting was set to 2.5x but identified 0 matching tokens because regex patterns don't match individual sub-word tokens from the tokenizer. The mechanism was completely ineffective. Corrected in v3 via token-ID approach (which then over-corrected in a different way).

### One universal model does not work
Tested implicitly across all proxy experiments. No single model works well on all text types. Cross-domain usage produces domain-prior projection, not generic degradation.

---

## Current Best Practical Checkpoints

| Domain | Checkpoint | Regime | Status | Best Metric |
|---|---|---|---|---|
| NLK | wiki_s32 | S32 | Champion | val_loss 0.085, NIAH 95%, EXACT 94.2% |
| FTA | code_s64 | S64 | Champion | val_loss 0.225, NIAH 91%, FACT2 85% |
| OSA | state_s32 | S32 | Champion | val_loss 0.000, all probes 99.4% |
| HWM | hwm_s64 | S64 | Champion (internal only) | val_loss 0.000, EXACT 62.5% |
| HPRT | reg_s32 | S32 | Champion (internal only) | val_loss 0.000, OVERRIDE 70%, FACT1 45% |
| CONV | conv_s64_v2 | S64 | Champion | val_loss 1.821, LongMemEval 94.9% F1 ret |
| AOJ | aoj_s32_v4 | S32 | Champion (FROZEN) | 99% fact recovery, 1.78x compression |

Note: AOJ v4 is FROZEN as champion (99% fact recovery, 1.78x compression). AOJ v2 model.pt recovered (retrained 9 Apr 2026; `model.pt` present locally); superseded by v4. AOJ v3 model.pt preserved locally but performed worse (66%).

---

## Next Experiments

Priority order, not all required immediately:

1. **Retrieval quality** — relevance-based packet selection, cross-domain retrieval, intelligent context budget allocation. This is the primary remaining bottleneck now that raw fact recovery is competitive (AOJ v4 = 99%).
2. **Router improvement** — learned routing with calibrated confidence. Misrouting remains a major failure mode.
3. **Additional domain A/B tests** — test CONV and other domains on real-world tasks, not just AOJ on OpenClaw.
4. **Multi-domain integration test** — test the full NDN (multiple nodes active, router selecting, fusion assembling) on a realistic agent workflow.
5. **Entity preservation refinement** — copy/pointer mechanisms or retrieval-augmented reconstruction for edge cases not yet covered by v4.
