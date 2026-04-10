# NDN Node Registry v0

**Date**: 8 Apr 2026
**Status**: Initial layout — 4 top-level domain nodes
**Architecture**: Neural Domain Network (see NDN_ARCHITECTURE.md)

---

## Node 1: Natural Language Knowledge (NLK)

### Purpose
Compress and reconstruct encyclopedic, factual, expository prose. This is the "what do I know about X" memory — stable factual knowledge that rarely changes.

### Current Maturity
**High**. Validated across multiple compression regimes, multiple seeds, extensive benchmarking. This is the most thoroughly tested domain in the project.

### Best Known K Regime
**S32** (4x compression). S32 consistently wins on exact factual recovery, entity binding, and compression efficiency for clean factual prose.

### Key Strengths
- Near-lossless reconstruction of factual content at 4x compression
- Strong entity-fact binding (person → attribute)
- Positional uniformity (facts at any position in text are preserved)
- Low val_loss (0.72 for S64, best across all domains)

### Key Failure Modes
- Minor degradation on very rare proper nouns
- Slight positional bias on extremely long passages (>1000 tokens)
- Not suitable for text that requires exact wording (use HPRT node instead)

### Best Benchmarks
- WIKI-NIAH (needle-in-a-haystack for factual prose)
- WIKI-FACT1/FACT2 (single-hop and two-hop fact binding)
- WIKI-EXACT (exact value recovery)
- Private eval: ablation_gap +7.27, shuffled_gap +1.46

### Likely Practical Use Inside an Agent
- **Long-term factual knowledge store**: compress research findings, documentation summaries, reference material
- **Context window extension**: offload background knowledge from the active context
- **Knowledge persistence across sessions**: facts learned in session N available in session N+1

---

## Node 2: Formal Technical Artifacts (FTA)

### Purpose
Compress and reconstruct source code, function definitions, technical specifications. This is the "what does this code do" memory — preserving syntax, structure, and semantic relationships in technical artifacts.

### Current Maturity
**High**. Validated with Python function-level code. Both S64 and S32 are viable; S64 is the default for code due to its advantage on fine-grained variable binding.

### Best Known K Regime
**S64** (2x compression). S64 wins on fine-grained binding and syntax preservation. S32 is viable and wins on exact recovery, but the safety margin of S64 is preferred for code where a single wrong token can break semantics.

### Key Strengths
- Preserves function structure, variable names, control flow
- Strong fine-grained binding (variable → value, function → return type)
- Handles Python idioms, decorators, nested structures
- Highest ablation gap (8.34) among non-conversation domains

### Key Failure Modes
- Occasional variable name substitution in complex nested scopes
- Exact string literal recovery can degrade for long literals
- Domain-specific: trained on Python, may not generalize to other languages without retraining

### Best Benchmarks
- CODE-NIAH, CODE-FACT1/FACT2 (code-native probes)
- CODE-EXACT (exact code recovery)
- CODE-TRACK (state tracking through code execution)
- Private eval: ablation_gap +8.34, shuffled_gap +3.02

### Likely Practical Use Inside an Agent
- **Codebase memory**: compress function definitions, class structures, API signatures the agent has seen
- **Session-to-session code context**: remember what the agent worked on across sessions
- **Code search augmentation**: reconstruct relevant code snippets from compressed memory instead of re-reading files

---

## Node 3: Operational State Artifacts (OSA)

### Purpose
Compress and reconstruct structured event traces, state updates, workflow logs. This is the "what happened and what is the current state" memory — tracking sequential updates, overrides, and final states.

### Current Maturity
**High**. Validated with synthetic operational state traces. S32 is the clear winner for this domain — structured, sequential data compresses extremely well at 4x.

### Best Known K Regime
**S32** (4x compression). State traces are highly structured and sequential, which S32 handles near-losslessly. S64 provides no meaningful advantage here.

### Key Strengths
- Near-perfect state tracking (final state recovery after multiple updates)
- Strong override detection (later update supersedes earlier)
- Temporal ordering preserved
- Highest raw metrics of any domain (val_loss 0.18)
- Corruption sensitivity is well-behaved (graceful degradation)

### Key Failure Modes
- Very long chains (>20 sequential updates) can show minor drift
- Unusual entity names in state keys can degrade
- Synthetic training data means the model expects structured format — unstructured state descriptions may not compress well

### Best Benchmarks
- STATE-NIAH, STATE-FACT1/FACT2 (state-native probes)
- STATE-TRACK (multi-update final state recovery)
- STATE-OVERRIDE (later-update-wins verification)
- Private eval: ablation_gap +8.86, shuffled_gap +3.64

### Likely Practical Use Inside an Agent
- **Workflow state memory**: compress the agent's task state, completed steps, pending actions
- **Session state persistence**: remember file changes, tool outputs, error states across sessions
- **Undo/audit trail**: compressed log of what the agent did, recoverable for debugging or rollback

---

## Node 4: Conversational Memory (CONV)

### Purpose
Compress and reconstruct multi-turn dialogue, conversation transcripts, chat history. This is the "what did we talk about" memory — preserving speaker roles, topic flow, facts mentioned in conversation, and temporal structure.

### Current Maturity
**Medium-High**. Validated with real conversation data (DialogSum + Blended Skill Talk + OpenAssistant). Benchmarked on LongMemEval (500 questions). v2 improvement pass completed and validated.

### Best Known K Regime
**S64** (2x compression). Conversation is the highest latent-dependence domain (ablation_gap +12.00, shuffled_gap +13.30). The extra capacity of S64 is needed to encode speaker roles, topic shifts, and conversational references.

### Key Strengths
- Preserves conversation structure and speaker turns
- Temporal ordering is strong (98.3% F1 retention on temporal-reasoning)
- General topics and gist are well-preserved
- Numeric details improved to 86.3% token accuracy (v2)
- Exceeds uncompressed baseline on some LongMemEval categories

### Key Failure Modes
- Specific digits still degrade (phone numbers, exact dates)
- Entity names in dense passages can blur
- Isolated local facts buried in long conversations can be lost
- Exact wording for subjective preferences not preserved
- Absolute benchmark scores are low (0.074 F1) — partial fidelity, not lossless

### Best Benchmarks
- LongMemEval (500 questions, public benchmark): 94.9% F1 retention, 93.6% KW retention
- Private eval: ablation_gap +12.00, shuffled_gap +13.30

### Likely Practical Use Inside an Agent
- **Conversation history compression**: replace raw chat transcript storage with compressed memory packets
- **Cross-session context**: remember what was discussed in previous sessions without storing full transcripts
- **Agent personality/user model**: compressed memory of user preferences, past interactions, established context

---

## Node 3b: OSA / Agent Operational Journals (AOJ) — VALIDATED

**Short ID**: AOJ
**Canonical name**: OSA / Agent Operational Journals
**Best checkpoint**: `aoj_s32_v2` (local: `checkpoints/aoj_s32_v2/model.pt` — 316MB, recovered via retrain 9 Apr 2026)
**Latest checkpoint**: `aoj_s32_v3` (local: `checkpoints/aoj_s32_v3/model.pt` — 316MB, regressed on A/B)

### Purpose
Compress and reconstruct structured agent workflow journals: recon summaries, tool outputs, host counts, findings lists, phase/status transitions, blockers, and operational notes. This is the "what did the agent do, find, and decide" memory — tracking operational progress across sessions in markdown prose format.

### Why this is an OSA subdomain
Same semantic domain as OSA (operational state tracking) but a different surface format: markdown-formatted agent journals vs compact timestamped key-value traces. The A/B test proved that existing OSA-S32 (trained on `[T=N] entity key=value`) projects journal text into its training format, losing all domain-specific facts.

### Evidence of need
Real A/B test on OpenClaw bounty agent journals:
- OSA-S32 proxy: 14% fact recovery (catastrophic)
- HWM-S64 proxy: 24% fact recovery (worse domain fit)
- **AOJ-S32 dedicated: 54% fact recovery** (validated, remaining bottleneck is entity preservation)
- Markdown baseline: 100% fact recovery

### Target Text Patterns
1. Phase/state transitions: `Phase 1A: COMPLETED`, `BLOCKED`, `IN PROGRESS`
2. Tool-result-count triples: `- subfinder: 555 hosts`, `- httpx: 169 live`
3. Target/domain names: `vfsglobal.com`, `expressvpn-bug-bounty-program`
4. Aggregate counts: `Total merged: 673 hosts`, `Live hosts: 169`
5. Host classification: `Admin hosts: 12, Auth hosts: 8, API hosts: 36`
6. Error/blocker notes: `502 Bad Gateway`, `REPAIR_NEEDED`
7. Repeated recon iterations with updated counts
8. Operational next-step notes
9. Markdown structure: headers, bullet lists, status tags
10. Multi-target interleaved entries

### Planned K Regime
**S32** (4x compression). OSA parent established that S32 is optimal for structured operational text. Markdown formatting adds moderate complexity but the content is fundamentally structured.

### Training Data
Synthetic corpus (`aoj`) covering all 10 content patterns above, supplemented with real OpenClaw journal data and potentially HackerOne disclosed reports for vocabulary diversity.

### Success Criteria
- Fact recovery >70% on A/B retest (vs current 14%)
- Preserve tool-result-count triples, domain names, aggregate counts, phase markers
- Compression >1.5x with majority fact retention
- Internal metrics: val_loss <0.01 by epoch 5, 100% first_tok, shuffled_gap >3.0

### Trained Metrics

| Version | val_loss | 1st_tok | exact | num_tok | shuffled_gap |
|---|---|---|---|---|---|
| v1 (synthetic_aoj_v1) | 0.0002 | 100% | 10% | 100% | 4.02 |
| **v2 (synthetic_aoj_v2)** | **0.0016** | **100%** | **100%** | **100%** | **4.31** |
| v3 (synthetic_aoj_v3) | 0.0001 | 100% | 100% | 100% | 4.37 |

v2 changes: 204 SLDs (was 42), 52 TLDs (was 16), 5% real data mix, digit_weight=3.0
v3 changes: token-ID entity weighting (122 tokens, 3.0x), digit_weight=5.0, OOV SLDs added. **Internal metrics improved but real A/B regressed — v3 is a failed experiment.**

### A/B Test Results (real OpenClaw data, 5 slices)

| Model | Fact Recovery | Compression | Continuity |
|---|---|---|---|
| HWM-S64 (proxy) | 24% | 1.26x | 0.86 |
| OSA-S32 (proxy) | 14% | 2.00x | 0.76 |
| AOJ v1 | 54% | 1.56x | 0.96 |
| **AOJ v2** | **73%** | **1.69x** | **0.92** |
| AOJ v3 | 66% (-7pp regression) | 1.66x | 0.92 |
| Markdown baseline | 100% | 1.00x | 1.00 |

### Remaining Bottleneck (v2, taxonomized)
- **Exact numeric counts**: 58% of misses (e.g. `1623`, `1946`, `769`, `27 live hosts`)
- **OOV domain names**: 25% of misses (e.g. `bostonacoustics.com`, `xvtest.net`, `jbl.com.br`)
- **Error/status strings**: 17% of misses (e.g. `502 Bad Gateway`, `REPAIR_NEEDED`)

Entity weight mechanism failed in v2 (regex found 0 sub-word entity tokens). v3 used correct token-ID-based approach (122 tokens matched, 3.0x weight) + stronger digit weight (5.0x), but **regressed on real A/B** from 73% to 66%. Internal metrics improved (lower val_loss, earlier exact match, higher shuffled_gap), but real-world fact recovery went down. This rules out loss-level engineering as the path forward.

### What's Next
The entity preservation problem was addressed by the **hybrid packet format** (entity side-channel): regex-extracted entities stored alongside latent blobs, appended on reconstruction. This Tier 1 intervention raised fact recovery from 14% to 100% on controlled test without retraining.

### Status
**v2 LOCKED as best — v3 failed (regression) — entity side-channel proven — first leaf validated (TDR)**

---

## Node 3b-i: OSA / AOJ / Technical Disclosure Reports (TDR) — FIRST VALIDATED LEAF

**Short ID**: TDR
**Canonical name**: OSA / AOJ / Technical Disclosure Reports
**Parent**: AOJ (aoj_s32_v2)
**Compression checkpoint**: `aoj_s32_v2` (same as parent — compression node shared, pipeline differs)
**Date validated**: 2026-04-10

### Purpose
Full end-to-end pipeline for storing, retrieving, and reconstructing facts from an accumulated archive of 100+ technical disclosure reports (vulnerability write-ups, security advisories, bug bounty reports). This is the "search your past findings" memory — not a single session, but a growing archive that must be searched selectively at scale where raw markdown is impossible.

### What Makes TDR Distinct from AOJ

AOJ proves the compression node works on individual sessions. TDR proves the full retrieval-reconstruction pipeline works at scale. The key differences:

1. **Retrieval is the bottleneck** — at 100+ reports / 1.15M tokens, you must find the right report
2. **Isolation is mandatory** — blending entities from multiple reports destroys specificity (21% → 77% when isolated)
3. **Metadata quality matters more than ranking intelligence** — title injection outperformed a 3B LLM reranker
4. **Hybrid packets are required** — latent-only reconstruction hallucinates entities; the entity side-channel preserves exact facts

### Champion Pipeline
```
query → FTS5 top-k → isolate each → reconstruct independently → heuristic rank → select best → output
```

### Benchmark Results (40-query combined: dev + held-out)

| Bucket | Queries | Hits | Hit Rate |
|---|---|---|---|
| A-technical | 8 | 8/8 | 100% |
| B-domain/ver | 8 | 8/8 | 100% |
| C-CVE/vuln | 8 | 7/8 | 88% |
| D-ambiguous | 8 | 5/8 | 63% |
| E-sparse/NL | 8 | 8/8 | 100% |
| **Total** | **40** | **36/40** | **90%** |

Avg fact recovery: **~94%** | Compression: **89–105x** | On correct session: **100% (30/30)**

### Failure Mode
Near-identical titles only. All 4 misses across 40 queries are caused by reports with the same or near-identical titles. Every query with a unique title succeeds.

### Status
**VALIDATED — FIRST LEAF NODE IN NDN TREE**

---

## Nodes NOT Included in v0 (Trained But Not Promoted)

### Human Working Memory Text (HWM)
- Trained and validated, but overlaps significantly with OSA and CONV in practical use
- Best for: fragmented notes, reminders, partial plans
- May be promoted to v1 if agent use cases specifically need messy-note compression

### High-Precision Regulated Text (HPRT)
- Trained and validated, strong on exactness-sensitive content
- Best for: policy documents, compliance text, formal procedures
- May be promoted to v1 for enterprise/legal agent use cases

---

## Cross-Node Summary

| Node | K | Compression | val_loss | shuf_gap | Key Benchmark |
|---|---|---|---|---|---|
| NLK (wiki_s32) | 32 | 4x | 0.085 | 18.75 | NIAH 95%, EXACT 94.2% |
| FTA (code_s64) | 64 | 2x | 0.225 | 1.14 | NIAH 91%, FACT2 85% |
| OSA (state_s32) | 32 | 4x | 0.000 | 4.26 | All probes 99.4% |
| HWM (hwm_s64) | 64 | 2x | 0.000 | 3.87 | EXACT 62.5% |
| HPRT (reg_s32) | 32 | 4x | 0.000 | 2.35 | OVERRIDE 70%, FACT1 45% |
| CONV (conv_s64_v2) | 64 | 2x | 1.821 | 13.30 | LongMemEval 94.9% F1 ret |
| AOJ (aoj_s32_v2) | 32 | 4x | 0.002 | 4.31 | Real A/B 73% fact recovery |
| **AOJ/TDR** | 32 | **89–105x** | — | — | **40-query: 90% hit, 94% facts (first validated leaf)** |
| **AOJ/WS** | 32 | **182x** | — | — | **20-query: 70% hit, 64% facts (baseline leaf)** |

Total params (7 nodes): ~490M. Active at any time: 1-2 nodes (~70-140M).

Note: AOJ/TDR and AOJ/WS use the same compression model (aoj_s32_v2). The high compression ratios are at the pipeline level (retrieval + isolation), not the model level. WS uses an extended entity extractor with 8 additional patterns for operational state (paths, UUIDs, API keys, status keywords, percentages, KV pairs, SSH keys, checkpoints).
