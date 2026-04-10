# NDN Domain Taxonomy

## Taxonomy Principles

Domains are defined by empirical evidence, not by intuition or theoretical categories. A domain exists in the NDN when:

1. A model trained on data matching its structural properties produces measurably better reconstruction than models trained on other domains
2. This improvement is consistent across multiple evaluation methods
3. The domain represents a meaningfully distinct information shape, not a superficial formatting difference

The taxonomy is a living structure. New domains and subdomains are added when evidence justifies them. Existing domains may be split or merged based on results.

---

## Top-Level Domains

### Natural Language Knowledge (NLK)

**Purpose**: Compress and reconstruct encyclopedic, factual, expository prose. The "what do I know about X" memory.

**Typical artifacts**:
- Wikipedia-style encyclopedia entries
- Documentation summaries
- Research findings and reference material
- Factual descriptions and explanations

**Best known regime**: S32 (4x compression)

**Strengths**:
- Strong factual reconstruction at 4x compression on internal and domain-native evals (not bit-exact or lossless storage)
- Strong entity-fact binding (person → attribute)
- Positional uniformity (facts at any position are preserved)
- Champion val_loss 0.085 at S32 with shuffled_gap 18.75 (highest latent dependence of any domain)

**Weaknesses**:
- Minor degradation on very rare proper nouns
- Not suitable for text requiring exact wording (use HPRT)

**Maturity**: High. Most thoroughly tested domain. Multiple compression regimes validated. Domain-native benchmark suite complete.

---

### Formal Technical Artifacts (FTA)

**Purpose**: Compress and reconstruct source code, function definitions, technical specifications. The "what does this code do" memory.

**Typical artifacts**:
- Function and class definitions
- API signatures
- Code review context
- Technical specifications

**Best known regime**: S64 (2x compression)

**Strengths**:
- Preserves function structure, variable names, control flow
- Strong fine-grained binding (variable → value, function → return type)
- Handles Python idioms, decorators, nested structures
- Highest ablation gap (+8.34) among non-conversation domains

**Weaknesses**:
- Occasional variable name substitution in complex nested scopes
- Exact string literal recovery degrades for long literals
- Trained on Python only — other languages untested

**Maturity**: High. Both S32 and S64 validated. S64 preferred for safety margin on fine-grained binding.

---

### Operational State Artifacts (OSA)

**Purpose**: Compress and reconstruct structured event traces, state updates, workflow logs. The "what happened and what is the current state" memory.

**Typical artifacts**:
- Timestamped event traces (`[T=N] entity key=value`)
- State update sequences
- Workflow progress markers
- System status records

**Best known regime**: S32 (4x compression)

**Strengths**:
- Strong state tracking on tested sequences (high final-state recovery after sequential updates; long chains can still drift)
- Strong override detection (later update supersedes earlier)
- Temporal ordering preserved
- Highest raw metrics of any domain (val_loss 0.18)

**Weaknesses**:
- Very long chains (>20 sequential updates) can show minor drift
- Unusual entity names in state keys can degrade
- Synthetic training data means the model expects structured format — unstructured state descriptions compress poorly

**Maturity**: High. The cleanest-metrics domain tested.

---

### Human Working Memory Text (HWM)

**Purpose**: Compress and reconstruct fragmented notes, reminders, partial plans, stream-of-consciousness text. The "messy scratchpad" memory.

**Typical artifacts**:
- Handwritten-style notes
- Incomplete plans and drafts
- Reminder lists with context
- Brainstorm fragments

**Best known regime**: S64 (2x compression)

**Strengths**:
- Handles unstructured, irregular text
- Preserves gist even when surface form is chaotic

**Weaknesses**:
- Lowest metrics overall — the domain is inherently hard
- Fine-grained details in messy text are often lost
- Domain prior is so strong it projects onto other domains (reconstructs everything as notes)

**Maturity**: Medium. Trained and evaluated, but not benchmarked as extensively as NLK or FTA.

---

### High-Precision Regulated Text (HPRT)

**Purpose**: Compress and reconstruct policy documents, compliance text, formal procedures. The "exact rules and thresholds" memory.

**Typical artifacts**:
- Regulatory compliance documents
- Policy procedures with exact dates and thresholds
- Formal specifications with conditional logic
- Legal/contractual text

**Best known regime**: S32 (4x compression)

**Strengths**:
- Preserves exact dates, numeric thresholds, role assignments
- Handles conditional logic structures

**Weaknesses**:
- Complex multi-layered conditional structures can degrade
- Hardest domain tested for exactness requirements

**Maturity**: Medium. Trained and evaluated. Benchmark suite complete. The most demanding domain for precision.

---

### Conversational Memory (CONV)

**Purpose**: Compress and reconstruct multi-turn dialogue, conversation transcripts, chat history. The "what did we talk about" memory.

**Typical artifacts**:
- Multi-turn chat conversations
- Interview transcripts
- Discussion threads
- Agent-user interaction history

**Best known regime**: S64 (2x compression)

**Strengths**:
- Preserves conversation structure and speaker turns
- Strong temporal ordering (98.3% F1 retention on temporal-reasoning)
- General topics and gist well-preserved
- Highest latent dependence of any domain (ablation_gap +12.00, shuffled_gap +13.30)

**Weaknesses**:
- Specific digits still degrade (phone numbers, exact dates)
- Entity names in dense passages can blur
- Isolated local facts buried in long conversations can be lost
- Absolute benchmark scores are low — partial fidelity, not lossless

**Maturity**: Medium-High. Validated with real conversation data. Benchmarked on LongMemEval (500 questions). Two improvement passes completed (v1 → v2). Public benchmark result: 94.9% F1 retention vs uncompressed baseline.

---

## Known Subdomains

### OSA / Agent Operational Journals (AOJ)

**Parent domain**: Operational State Artifacts (OSA)

**Why it exists**: The parent OSA node was trained on timestamped key-value state traces. When tested on real agent operational journals (markdown-formatted recon summaries, tool outputs, findings lists), the OSA-S32 model projected journal text into its training format, losing domain-specific facts. OSA-S32 recovered only 14% of facts. HWM-S64 (wrong domain entirely) recovered only 24%. Both proxy nodes failed.

**What problem it solves**: Agent operational journals have a distinct structure that overlaps with OSA semantically (operational tracking) but differs in surface format (markdown prose vs compact key-value traces). The AOJ subdomain provides a node trained on data matching this specific structure.

**Target artifacts**:
- Per-target recon journals (`Phase 1A: COMPLETED`, tool outputs, host counts)
- Findings lists (vulnerability details, error states, blockers)
- Phase/status transitions
- Multi-target interleaved operational entries
- Repeated recon iterations with updated counts

**Best known regime**: S32 (4x compression)

**Current status**: Validated. AOJ v2 is the current champion checkpoint (73% fact recovery on real OpenClaw A/B, 1.69x compression). AOJ v3 attempted stronger loss weighting but regressed to 66%. The remaining gap to markdown (27pp) is characterized as exact numeric counts (58%), OOV domain names (25%), and error/status strings (17%).

**Progression**:

| Version | Fact Recovery | Compression | Status |
|---|---|---|---|
| OSA-S32 proxy | 14% | 2.00x | Failed — domain-prior mismatch |
| HWM-S64 proxy | 24% | 1.26x | Failed — wrong domain entirely |
| AOJ v1 | 54% | 1.56x | Superseded |
| **AOJ v2** | **73%** | **1.69x** | **Current champion** |
| AOJ v3 | 66% | 1.66x | Failed — over-corrected loss weighting |
| Markdown baseline | 100% | 1.00x | Wins on OpenClaw AOJ A/B (single-session compression test) |

---

## Validated Leaves (under AOJ)

### OSA / AOJ / Technical Disclosure Reports (TDR) — FLAGSHIP LEAF

**Why it exists**: AOJ handles single-session compression, but agent memory often requires retrieval across 100+ accumulated reports. TDR provides the full pipeline: store, retrieve, isolate, reconstruct, rank, select.

**Corpus**: Vulnerability disclosure reports (HackerOne, CIRCL, GitHub Advisory, APT campaigns, threat intelligence). Validated across 5 distinct corpora, 500+ reports, 1.15M+ tokens.

**Champion pipeline**: Title-aware FTS5 retrieval → per-session isolated reconstruction → heuristic scoring (FTS5 rank + entity overlap + title term overlap). Hybrid packets: latent narrative scaffold + exact entity payload.

**Results**: Dev: 18/20 (90%), 93% facts, 89x compression. Held-out: 18/20 (90%), 94% facts, 105x compression. Combined: 36/40 (90%). Multi-corpus transfer: 114/120 (95%) across 5 corpora with zero code changes.

**Failure mode**: Near-identical titles (lexically overlapping report titles pick the same wrong session). Only failure mode across all 40 queries.

---

### OSA / AOJ / Workflow State (WS) — BLOOMING

**Why it exists**: Operational memory for infrastructure state — what machine is active, where credentials are, what expired, what the current blocker is. Motivated by live agent memory failures (Cursor/Opus losing operational continuity across sessions).

**Corpus**: 49 sessions of auto-generated daemon logs, recon journals, and infrastructure state from OpenClaw deployment. 430K tokens.

**Results**: All scoring variants converge to 14/20 (70%) hits, ~64% fact recovery, 182x compression. Ceiling confirmed structural across 3 independent scoring approaches.

**Failure modes**: Session imbalance (39 daemon chunks flood FTS5), sibling-session overlap (two infra sessions both contain credentials), fact-level temporal reasoning (need "latest value" within a session, not just session-level recency).

**Specialist engine needed**: Temporal override logic, session-count balancing, source-aware retrieval. Not yet built.

---

### OSA / AOJ / Recon Workflow Journals (RWJ) — BLOOMING (approaching Baseline Leaf)

**Why it exists**: Campaign narrative memory — what did we attack, what did we find, what pivot did we make, what was the progression. Distinct from WS (infrastructure state) and TDR (disclosure reports). Identified when 257 pentesting operational files were initially misclassified as WS but corpus profiling revealed 247/257 are campaign narrative.

**Corpus**: 257 real human-authored pentesting files (952K tokens): 125 submissions, 71 recon files, 27 operation journals, 17 operational docs. Multi-document-per-target structure (e.g., T-Mobile has journal + recon + submissions).

**Champion pipeline**: Shared AOJ pipeline + embedding-based doc-type classifier (`all-MiniLM-L6-v2` sentence embeddings + prototype matching for journal/recon/submission) + expanded entity extractor (11 RWJ-specific categories: dollar amounts, comma numbers, K/M suffixes, CWEs, CVEs, GHSAs, MITRE T-numbers, env vars, hex hashes, 60+ tool names, bounty count nouns, code identifiers, shell commands).

**Results progression**:
- v1 borrowed pipeline: 14/20 (70%), same ceiling as WS
- v2 keyword classifier: Dev 20/20, Held-out 13/20, Combined 33/40 (82%) — overfit
- v3 embedding classifier: Dev 19/20, Held-out 16/20, Combined 35/40 (88%) — less overfit
- v3 + entity expansion: **84% facts, 50x compression, -16pp oracle gap, 100% retrieval (40/40)**
- Dev/held-out consistency: 84% vs 83% facts — not overfitting

**Failure modes**: E-temporal bucket (~67% vs 100% oracle) — temporal facts embedded in narrative progression need more than entity extraction. Doc-type classification semantic edge cases (queries about findings-in-journals mapped to submission/recon prototypes).

**Path to 🌿 Baseline Leaf**: Pipeline freeze, E-temporal improvement, formal baseline declaration.

---

## Cross-Leaf Architectural Findings

| Property | TDR | WS | RWJ |
|----------|-----|-----|-----|
| **Corpus type** | Disclosure reports | Infra state / daemon logs | Campaign journals / recon / submissions |
| **Entity vocabulary** | CVEs, domains, host counts, ports | Paths, UUIDs, API keys, status keywords | Dollars, counts, tools, security IDs, code identifiers |
| **Retrieval ceiling (borrowed pipeline)** | 90% | 70% | 70% |
| **Best result (specialist)** | 90% (validated, flagship) | 70% (no specialist yet) | 84% facts, 50x compression (entity expansion) |
| **Fact recovery** | 93–94% | 64% | 84% (was 59% before entity expansion) |
| **Oracle gap** | +2pp | N/A | -16pp (was -41pp) |
| **Primary failure mode** | Near-identical titles | Session imbalance + temporal reasoning | E-temporal + doc-type semantic edges |
| **Specialist lever** | Title-aware metadata | Temporal override (not yet built) | Embedding classifier + entity extractor |

The 14/20 (70%) ceiling appears on both WS and RWJ when using the shared pipeline. This is a validated cross-leaf architectural property: heuristic-only ranking on multi-document-per-target data hits a natural boundary. Breaking past it requires leaf-specific logic — different logic for each leaf. Entity extractor expansion is a repeatable intervention: WS gained +31pp, RWJ gained +25pp. The entity layer is NDN's primary tuning knob.

---

## Domains Under Consideration (Not Yet Validated)

The following are plausible future domains that have not been empirically tested:

- **Multimodal Artifact Descriptions** — text descriptions of images, diagrams, UI states
- **Temporal Event Sequences** — calendar events, meeting notes, time-ordered records
- **Scientific/Mathematical Text** — formulas, proofs, quantitative reasoning
- **Structured Data Artifacts** — JSON/YAML configs, database schemas, API responses

These are listed for completeness. None should be added to the taxonomy until proxy-failure evidence exists.
