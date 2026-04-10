# Project Status

**Date**: 10 April 2026 (updated from 9 April)
**Reporting standard**: Claims are conservative; limitations, failures, and negative results are stated explicitly.

---

## Proven

These claims have empirical evidence behind them and have survived held-out or real-world evaluation.

### Domain-specific training works
Models trained on domain-specific data consistently outperform cross-domain usage. This has been measured via shuffled_gap (range 0.88–13.30 across domains), ablation_gap, and held-out benchmarks. A model trained on the wrong domain reconstructs text in its own style, not the input style. This is reproducible across all six domains.

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

### AOJ is a validated subdomain with leaf-level pipelines
Agent Operational Journals under OSA. Proxy nodes failed (14–24% fact recovery). Dedicated training achieved 54% (v1) and 73% (v2). The improvement survives real-world A/B testing. The first flagship leaf (TDR) achieves 93–94% fact recovery at 89–105x compression on 100 HackerOne reports, validated with held-out queries and 5-corpus transfer (114/120 = 95%). Two additional bloomings (WS, RWJ) have been established with documented failure modes and progressive improvement.

### Entity side-channel is a proven mechanism
Regex-based entity extraction at compression time + structured append at reconstruction time raises fact recovery from 14% to 100% on controlled OOV test. Entity extractor expansion is a repeatable, high-leverage intervention: WS gained +31pp (35%→66.6%), RWJ gained +25pp (59%→84%). The entity layer is NDN's primary tuning knob — same pipeline, same model, different patterns per blooming.

### Isolation architecture is validated
Blending multiple retrieved sessions destroys fact specificity (15–25% fact recovery). Isolated per-session reconstruction produces 64–94% avg. This is a cross-leaf architectural finding: retrieve → isolate → reconstruct → rank → select is the correct pipeline.

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

### AOJ v2's fact recovery improvement is real but not sufficient
73% fact recovery at 1.69x compression is a large improvement over proxy nodes (14-24%) and v1 (54%). But markdown still wins at 100%. The gap (27pp) is characterized but not closed.

### Repeated-work signals reduced (OpenClaw A/B)
In that test, NDN-compressed memory showed 97.8% fewer repeated-work signals than the markdown arm. That is one metric on one harness; it suggests less redundant-looking context under compression, not a general guarantee, and fact recovery on the same test remained below markdown.

### The architecture scales to many nodes cheaply
Each node is ~70M parameters (~280MB). Six nodes together use <2GB VRAM. The architecture is designed for many specialized nodes running concurrently, and the memory overhead is trivial compared to the reader LLM. But this has not been tested at scale (>6 nodes).

### Latent dependence varies meaningfully across domains
Conversation shows ablation_gap +12.00 and shuffled_gap +13.30 — far higher than any other domain. This means the latent representation is doing more representational work for conversation than for structured text. The practical implications are still being explored.

---

## Unresolved

### Markdown still wins the OpenClaw AOJ journal A/B
On the original OpenClaw AOJ A/B test, raw markdown achieves 100% fact recovery vs NDN's 73%. Leaf-specific pipelines with entity side-channel reach higher fact recovery (TDR 94%, RWJ 84%) on their respective scale benchmarks, but no single pipeline yet matches markdown across all operational text types.

### Rare entity preservation at the model level
The encoder-decoder reconstructs correct structure but substitutes out-of-vocabulary entities with training priors. The entity side-channel is a pragmatic workaround (not a learned solution). A learned copy/pointer mechanism remains the open architectural goal.

### Loss weighting is not the solution to entity preservation
AOJ v3 tested token-ID-based entity weighting (122 tokens at 3.0x) and stronger digit weight (5.0x). Internal metrics improved. Real A/B performance regressed from 73% to 66%. The intervention over-corrected.

### Routing is basic
The current router is rule-based with no calibration, no confidence scoring, and no learned boundaries. Misrouting has been shown to catastrophically degrade quality. A better router is needed before production use.

### No production deployment exists
The OpenClaw integration was a test harness using synthetic session hooks. No live agent is currently running with NDN memory.

### Scaling governance is undefined
The taxonomy has 6 domains, 1 subdomain, and 3 leaf-level nodes (1 flagship, 2 blooming). How to govern taxonomy growth at scale (dozens of domains, hundreds of subdomains) is an open organizational and technical problem. The node maturity pipeline (Seed → Blooming → Baseline Leaf → Validated Leaf → Flagship Leaf) provides a framework but has only been exercised on TDR so far.

---

## Failed / Falsified

### AOJ v3: Token-level loss weighting regressed real-world performance
Token-ID entity weighting (3.0x on 122 entity tokens) + digit weight (5.0x) produced better internal metrics (lower val_loss, earlier exact match, higher shuffled_gap) but regressed real A/B fact recovery from 73% to 66%. The intervention over-corrected, optimizing for entity token reproduction at the expense of broader contextual recall. This disproves "harder entity pressure = better real-world recall."

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
| AOJ | aoj_s32_v2 | S32 | Champion | 73% fact recovery (real A/B) |

Note: AOJ v2 model.pt has been recovered (retrained 9 Apr 2026; `model.pt` present locally). AOJ v3 model.pt is preserved locally but performed worse.

---

## Next Experiments

Priority order, not all required immediately:

1. **Architectural entity preservation** — copy/pointer mechanisms, entity-aware attention, or retrieval augmentation for rare identifiers. This addresses the primary remaining bottleneck.
2. **Router improvement** — learned routing with calibrated confidence. Misrouting is currently the largest failure mode after entity preservation.
3. **AOJ v2 checkpoint** — recovered via retrain (9 Apr 2026); `model.pt` is available locally.
4. **Additional domain A/B tests** — test CONV and other domains on real-world tasks, not just AOJ on OpenClaw.
5. **Multi-domain integration test** — test the full NDN (multiple nodes active, router selecting, fusion assembling) on a realistic agent workflow.
