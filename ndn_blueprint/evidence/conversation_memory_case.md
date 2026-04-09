# Case Study: Conversation Memory Domain (CONV)

## Summary

The conversation memory domain demonstrated that domain-specific training is essential for organic, unstructured text, and that targeted interventions based on error analysis can produce measurable improvements on a public benchmark.

---

## What Failed at First

### Synthetic conversation templates caused catastrophic overfitting

Initial attempts to train a conversation model used synthetic conversation templates — formulaic multi-turn dialogues generated from templates. The model quickly memorized the templates and produced zero generalization. This was a hard failure that required a complete corpus redesign.

### HWM-S64 as proxy

Before training a dedicated CONV model, the HWM-S64 model (trained on fragmented notes) was tested as a proxy. It reconstructed conversations as messy note-style text, preserving some gist but destroying speaker turns, dialogue structure, and temporal flow. This was the domain-prior projection phenomenon: the model's learned prior (messy notes) dominated the reconstruction.

---

## What Worked: Real Data

Switching to real conversational data immediately fixed the training problem:

- **DialogSum**: real dialogue summaries
- **Blended Skill Talk**: multi-turn task-oriented conversations
- **OpenAssistant**: real user-assistant interactions

The model began learning conversation structure within the first epoch. A key lesson from Phase 12: **real data beats synthetic for organic domains** in this setup.

---

## CONV-S64 v1 Results

Trained on real conversation data. Benchmarked on LongMemEval (500 questions):

| Category | Baseline F1 | CNDX v1 F1 | Retention |
|---|---|---|---|
| knowledge-update | 0.079 | 0.067 | 84.8% |
| multi-session | 0.043 | 0.041 | 95.3% |
| single-session-assistant | 0.103 | 0.062 | 60.2% |
| single-session-preference | 0.161 | 0.133 | 82.6% |
| single-session-user | 0.097 | 0.076 | 78.4% |
| temporal-reasoning | 0.073 | 0.072 | 98.6% |
| **Overall** | **0.078** | **0.066** | **84.6%** |

The model retained temporal reasoning with very high measured retention on that category but lost significant quality on single-session-assistant (the category requiring recall of specific assistant statements).

---

## Error Analysis

Focused analysis on the three highest-loss categories revealed a ranked failure taxonomy:

| Failure Type | Count | Fix Direction |
|---|---|---|
| Numeric detail loss | 12 | Numeric-aware training |
| Entity/name loss | 10 | Entity-rich augmentation |
| Dropped local detail | 8 | Overlapping chunking |
| Truncation artifacts | 6 | Overlapping chunking |
| Paraphrase metric artifact | 4 | Not fixable at compression level |
| Reader-model failure | 3 | Reader prompt engineering |

---

## Three Targeted Interventions (v2)

Based on the error taxonomy, three surgical interventions were applied:

### 1. Overlapping chunking (stride=96)
Changed from non-overlapping 128-token blocks to overlapping chunks with stride=96. This reduced information loss at chunk boundaries.

**Impact**: KW +0.205 on single-session-assistant (the worst category).

### 2. Numeric-aware loss (digit_weight=3.0)
Identified 14 digit-containing tokens in the tokenizer. Applied 3x weight multiplier in cross-entropy loss.

**Impact**: Continuity +0.103 on knowledge-update.

### 3. Entity-rich augmentation (~35k samples)
Added entity-dense synthetic conversations (names, handles, URLs, products, places) to the training corpus.

**Impact**: F1 +0.035 on single-session-user.

---

## CONV-S64 v2 Results

| Category | BL F1 | v1 F1 | v2 F1 | v2 Retention |
|---|---|---|---|---|
| knowledge-update | 0.079 | 0.067 | 0.075 | 94.9% |
| multi-session | 0.043 | 0.041 | 0.044 | **102.5%** |
| single-session-assistant | 0.103 | 0.062 | 0.089 | 86.4% |
| single-session-preference | 0.161 | 0.133 | 0.147 | 91.3% |
| single-session-user | 0.097 | 0.076 | 0.110 | **113.6%** |
| temporal-reasoning | 0.073 | 0.072 | 0.072 | 98.3% |
| **Overall** | **0.078** | **0.066** | **0.074** | **94.9%** |

Two categories now exceed the uncompressed baseline (single-session-user and multi-session). Per-question: 226 wins vs 151 losses vs 123 ties (60% win rate over v1).

---

## What This Taught Us

1. **Real data is non-negotiable for organic domains.** Synthetic templates fail. Real conversation data enables immediate healthy learning.

2. **Error-analysis-driven interventions work.** Each of the three targeted fixes addressed a specific diagnosed failure mode and produced measurable improvement.

3. **Domain-prior projection is real and informative.** The HWM model reconstructing conversations as notes proved that the latent space is domain-shaped.

4. **Conversation has the highest latent dependence.** Ablation gap +12.00 and shuffled gap +13.30 are far above all other domains. The latent is doing more representational work here.

5. **94.9% F1 retention is meaningful but not lossless.** The architecture competes on a public benchmark, but absolute F1 is still 0.074. Partial fidelity, not lossless memory.

6. **Do not squeeze further on this benchmark.** The lessons are learned. Further LongMemEval optimization gives diminishing returns. The next valuable test is a different benchmark or real-world integration.
