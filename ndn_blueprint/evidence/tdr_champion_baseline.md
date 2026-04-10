# Champion Baseline: OSA / AOJ / Technical Disclosure Reports

**Frozen**: 10 April 2026
**Status**: Do not modify. All future work must beat this baseline on the same evaluation before replacing it.

---

## Pipeline Specification

### Ingestion

```
raw report text → prepend "# {title}\n\n" → SessionContent
    → router classifies chunks (conv / findings / workflow)
    → entity_extractor.extract_entities(text)
    → compressor.compress(chunk) → MemoryPacket with latent_blob + entity_payload
    → store.save(packets)
    → store.index_session(session_id, title, search_text, entity_values, domains)
```

### Retrieval + Reconstruction

```
query → FTS5 MATCH (OR-joined terms, BM25 ranked) → top-k session_ids
    → for each session_id:
        → retrieve all packets for that session
        → reconstruct independently (decode latent + merge entity payload)
    → score each candidate:
        fts_bonus  = (k - rank_position) × 3
        ent_score  = Σ(5 for each query entity value found in session entity_values)
        title_score = |query_terms ∩ title_terms| × 4
        term_ent   = Σ(2 for each query term found in session entity_values)
        text_score = Σ(1 for each query term found in session search_text)
        total = fts_bonus + ent_score + title_score + term_ent + text_score
    → select highest-scoring candidate
    → output: reconstructed text for that single session
```

### Key Architecture Decisions

| Decision | Choice | Why |
|---|---|---|
| Blending | **Disabled** | Cross-report contamination destroys specificity (21% → 77% fact recovery when disabled) |
| Retrieval | **FTS5 with BM25** | Beat naive recency (0/5 → 4/5 target hits) |
| Reconstruction | **Per-session isolated** | Each candidate reconstructed independently, then ranked |
| Ranking | **Heuristic** | Outperformed 3B LLM reranker (Qwen2.5-3B-Instruct): 87% vs 70% fact recovery |
| Entity preservation | **Regex side-channel** | Entities extracted at compression time, stored alongside latent blob, appended on reconstruction |
| Session indexing | **Title + text + entities** | Title injection was the highest-leverage fix (2/5 → 4/5 on heuristic) |

---

## Frozen Parameters

| Parameter | Value |
|---|---|
| Compression model | `aoj_s32_v2` (CNDX native encoder-decoder, K=32, seq=128) |
| Retrieval top-k | 5 |
| FTS5 query | `OR`-joined terms from query string |
| Scoring weights | fts_bonus: 3/rank, entity: 5, title: 4, term_ent: 2, text: 1 |
| Isolation | mandatory per-session |
| LLM reranker | not used |
| Embedding reranker | not used |

---

## Benchmark Results (frozen)

### Dev Set (20 queries)

| Metric | Value |
|---|---|
| Hits | 18/20 (90%) |
| Fact recovery | 93% |
| Compression | 89x |

### Held-Out Set (20 queries, independent)

| Metric | Value |
|---|---|
| Hits | 18/20 (90%) |
| Fact recovery | 94% |
| Compression | 105x |

### Combined (40 queries)

| Metric | Value |
|---|---|
| Hits | 36/40 (90%) |
| Fact recovery | ~94% |
| On correct session | 100% (30/30 every time) |

### Per-Bucket (combined)

| Bucket | Hits | Rate |
|---|---|---|
| A-technical | 8/8 | 100% |
| B-domain/version | 8/8 | 100% |
| C-CVE/vuln | 7/8 | 88% |
| D-ambiguous | 5/8 | 63% |
| E-sparse/NL | 8/8 | 100% |

---

## Multi-Corpus Transfer (frozen pipeline, zero code changes)

| Corpus | Queries | Hits | Hit Rate | Fact Recovery |
|---|---|---|---|---|
| HackerOne (primary) | 40 | 36 | 90% | 94% |
| CIRCL/vulnerability | 20 | 19 | 95% | 96.7% |
| GitHub Advisory 2023 | 20 | 20 | 100% | 96.7% |
| APT campaign reports | 20 | 19 | 95% | 73.8% |
| Structured threat intel | 20 | 20 | 100% | 87.0% |
| **Total** | **120** | **114** | **95%** | **~87%** |

All 6 misses share the same root cause: ambiguous or garbage titles.

---

## Known Failure Mode

**Near-identical titles.** All 4 misses across 40 queries are caused by multiple reports with the same or near-identical titles. The system cannot disambiguate:
- `[H1-2006 2020] CTF Writeup` (multiple reports)
- `stack overflow #N in libsass` (N=2,3,5,6)

This is data-level ambiguity. Every query with a unique title succeeds.

---

## What Must Beat This

Any replacement baseline must:

1. Run on the same 100-report HackerOne corpus
2. Use both the 20-query dev set and 20-query held-out set
3. Report per-bucket hit rate and fact recovery
4. Not cherry-pick queries
5. Show improvement on at least one metric without regression on others

---

## Files

| File | Purpose |
|---|---|
| `server_backup/cndx_backup.tar.gz` | Full checkpoint + code archive (1.23GB) |
| `server_backup/scale_test_h1.py` | Benchmark script (frozen queries embedded) |
| `server_backup/openclaw_memory/` | Runtime module |
| `ndn_blueprint/registry/node_cards/aoj_tdr_leaf.md` | Leaf node card |
| `ndn_blueprint/evidence/tdr_champion_baseline.md` | This file |
