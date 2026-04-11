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
- Champion val_loss 0.085 at S32 with shuffled_gap 18.750 (highest shuffled_gap of any domain)

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
- Highest raw metrics of any domain (val_loss 0.000)

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
- Highest ablation_gap of any domain (+12.00), with shuffled_gap +13.30

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

**Current status**: Validated. AOJ v4 is the current champion checkpoint (99% fact recovery on real OpenClaw A/B, 1.78x compression, FROZEN). Test data was in the training corpus — data-leakage caveat applies. v2 (73%) is superseded; v3 attempted stronger loss weighting but regressed to 66%. The raw compression gap to markdown is effectively closed by v4; the remaining bottleneck is retrieval, not compression. Downstream evaluation: TDR = 90% (hits flagship quality bar), RWJ = 95% dev facts with v4, WS = PARKED 14/20.

**Progression**:

| Version | Fact Recovery | Compression | Status |
|---|---|---|---|
| OSA-S32 proxy | 14% | 2.00x | Failed — domain-prior mismatch |
| HWM-S64 proxy | 24% | 1.26x | Failed — wrong domain entirely |
| AOJ v1 | 54% | 1.56x | Superseded |
| AOJ v2 | 73% | 1.69x | Superseded |
| AOJ v3 | 66% | 1.66x | Failed — over-corrected loss weighting |
| **AOJ v4** | **99%** | **1.78x** | **Current champion (FROZEN)** |
| Markdown baseline | 100% | 1.00x | Near-parity reached by v4 |

---

## Domains Under Consideration (Not Yet Validated)

The following are plausible future domains that have not been empirically tested:

- **Multimodal Artifact Descriptions** — text descriptions of images, diagrams, UI states
- **Temporal Event Sequences** — calendar events, meeting notes, time-ordered records
- **Scientific/Mathematical Text** — formulas, proofs, quantitative reasoning
- **Structured Data Artifacts** — JSON/YAML configs, database schemas, API responses

These are listed for completeness. None should be added to the taxonomy until proxy-failure evidence exists.
