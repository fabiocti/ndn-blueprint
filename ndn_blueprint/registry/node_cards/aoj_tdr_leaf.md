# Leaf Node Card: Technical Disclosure Reports (TDR)

## Identity

| Field | Value |
|---|---|
| **Node ID** | `aoj_tdr_v1` |
| **Domain** | `OSA` |
| **Subdomain** | `AOJ` |
| **Leaf** | `Technical Disclosure Reports` |
| **Regime** | `S32` |
| **Checkpoint** | `aoj_s32_v2` |
| **Version** | `v1 (first validated leaf)` |
| **Status** | `champion` |
| **Date validated** | `2026-04-10` |
| **Validation hardware** | `1x A100 80GB (Verda)` |

## Purpose

Compresses, stores, retrieves, and reconstructs accumulated technical disclosure reports — vulnerability write-ups, security advisories, bug bounty reports, and CVE-level technical narratives. This is the "what did we find across 100+ past engagements" memory — not a single-session operational journal, but a growing archive of structured technical findings that must be searched and recalled selectively.

## What Makes TDR Distinct from Generic AOJ

Generic AOJ handles single-session operational journals — one agent, one target, one workflow. TDR handles accumulated multi-session archives where:

1. **Retrieval is the bottleneck, not compression.** With 100+ reports and 1M+ tokens, you cannot paste everything into context. The system must find the right report.
2. **Reports have near-identical structural patterns.** Phase headers, tool outputs, finding lists, CVE references — these recur across every report, making blended reconstruction catastrophically wrong.
3. **Titles and metadata are the primary retrieval anchors.** The actual report body is often indistinguishable at the token level from neighboring reports. Without title-aware indexing, retrieval breaks.
4. **Isolation is mandatory.** Blending entities from multiple retrieved reports destroys specificity. Each candidate must be reconstructed independently before ranking.

Generic AOJ proved the compression node works. TDR proves the full retrieval-reconstruction pipeline works at scale.

## Champion Pipeline

```
ingest → extract entities → compress (AOJ S32 v2) → store hybrid packets + session index
                                    ↓
query → FTS5 session search → top-k candidates → isolate → reconstruct each → heuristic rank → select best → output
```

### Pipeline Components

| Stage | Implementation | Key Detail |
|---|---|---|
| **Compression** | AOJ S32 v2 (CNDX encoder) | Latent blob + entity side-channel per packet |
| **Entity extraction** | Regex-based (`entity_extractor.py`) | Domains, counts, ports, versions, IPs, HTTP statuses, CVEs |
| **Storage** | SQLite + FTS5 | `memory_packets` table + `session_index` + `session_fts` virtual table |
| **Session indexing** | Title + first 1000 chars + entity values | Indexed at ingestion time via `hooks.on_session_end` |
| **Retrieval** | FTS5 MATCH with BM25 ranking | Query terms OR-joined, top-k sessions returned |
| **Reconstruction** | Per-session isolated | Each candidate reconstructed independently — no blending |
| **Ranking** | Heuristic scorer | FTS5 rank prior + entity overlap + title term overlap |
| **Entity injection** | Append `[PRESERVED ENTITIES]` section | Regex-extracted entities appended to decoded latent text |

### Frozen Configuration

| Parameter | Value |
|---|---|
| Retrieval top-k | 5 |
| Isolation mode | per-session (mandatory) |
| Scoring | `fts_bonus = (k - rank_pos) * 3` + `entity_overlap * 5` + `title_overlap * 4` + `term_entity * 2` + `text_score * 1` |
| Blending | **disabled** — isolation only |
| LLM reranker | **not used** — heuristic outperformed 3B LLM |

## Benchmark Results

### Corpus

- **Source**: `Hacker0x01/hackerone_disclosed_reports` (public, Hugging Face)
- **Size**: 100 reports, 1,147,822 tokens (avg 11,478/report)
- **Full markdown context**: 1,150,297 tokens

### Dev Set (20 queries, 5 buckets × 4)

| Metric | Value |
|---|---|
| Target hits | **18/20 (90%)** |
| Avg fact recovery | **93%** |
| Compression | **89x** |
| Iso vs oracle | +4% |

### Held-Out Set (20 queries, independent, locked before running)

| Metric | Value |
|---|---|
| Target hits | **18/20 (90%)** |
| Avg fact recovery | **94%** |
| Compression | **105x** |
| Iso vs oracle | -3% |

### Combined 40-Query Result

| Bucket | Queries | Hits | Hit Rate | Fact Recovery |
|---|---|---|---|---|
| A-technical | 8 | **8/8** | **100%** | 100% |
| B-domain/version | 8 | **8/8** | **100%** | 100% |
| C-CVE/vuln | 8 | **7/8** | **88%** | ~96% |
| D-ambiguous | 8 | **5/8** | **63%** | ~74% |
| E-sparse/NL | 8 | **8/8** | **100%** | ~99% |
| **TOTAL** | **40** | **36/40** | **90%** | **~94%** |

### Headline Numbers

| Metric | Value |
|---|---|
| Compression vs full markdown | **89–105x** |
| Fact recovery (avg across 40 queries) | **~94%** |
| Fact recovery (correct session isolated) | **100% (30/30 every time)** |
| Session selection accuracy | **90% (36/40)** |

## Failure Mode

**Single identified failure mode: near-identical titles.**

All 4 misses across 40 queries share the same root cause: multiple reports exist with nearly identical titles.

| Miss | Reason |
|---|---|
| `[H1-2006 2020] CTF Writeup` × 2 | Multiple reports with identical titles |
| `Hackyholidays [h1-ctf] stop the grinch` | Near-identical to other CTF writeups |
| `stack overflow #6 in libsass` | 4 reports titled `stack overflow #N in libsass` (N=2,3,5,6) |

**Every query with a unique title — including extremely sparse 2-word titles — succeeds.**

This is data-level ambiguity, not a system failure. The system cannot distinguish reports that humans also cannot distinguish by title alone.

## Known Weaknesses

1. **Near-identical titles defeat the heuristic ranker** — 63% hit rate on the ambiguous bucket vs 100% on all others
2. **Regex entity extraction is brittle** — hardcoded patterns, no semantic understanding of entity importance
3. **Entity payload is uncompressed text** — dilutes the compression advantage (still 89–105x overall)
4. **Single-corpus validation** — tested on HackerOne reports only, no cross-corpus transfer evidence yet
5. **AOJ v2 training data overlaps with A/B test data** — 5% real journal mix was from the same targets used in early A/B tests (not from HackerOne)

## Comparison: Why Not Just Use Markdown?

| Approach | Context size (100 reports) | Fact recovery | Practical |
|---|---|---|---|
| Full markdown paste | 1,150,297 tokens | ~90% | **Impossible** — exceeds all context windows |
| Naive RAG (chunk retrieval) | ~5,000–20,000 tokens | Varies | Loses report structure, cross-chunk coherence |
| **NDN TDR pipeline** | **~11,000 tokens** | **~94%** | **Yes** — fits in any modern context window |

The entire point of this leaf is that markdown *cannot* work at scale. At 100+ reports and 1M+ tokens, you need selective retrieval and compression. NDN's TDR pipeline delivers 89–105x compression with 94% fact recovery.

## Provenance

| Item | Details |
|---|---|
| Compression checkpoint | `aoj_s32_v2` (316MB, trained 2026-04-09 on Verda H100) |
| Pipeline code | `openclaw_memory/` module (store.py, hooks.py, compressor.py, entity_extractor.py, router.py, fusion.py) |
| Benchmark script | `scale_test_h1.py` |
| Benchmark dataset | `Hacker0x01/hackerone_disclosed_reports` (public) |
| Dev query indices | 0,3,10,11,14,22,27,30,40,49,50,60,62,68,70,78,86,90,94,98 |
| Held-out query indices | Separate 20 from remaining 80 (no overlap with dev) |
| All files backed up | `server_backup/cndx_backup.tar.gz` (1.23GB, includes all checkpoints) |

## Status

**VALIDATED — FIRST LEAF NODE IN NDN TREE**

This is the first node that has been validated end-to-end: from ingestion through compression, storage, retrieval, isolated reconstruction, ranking, and output — on a public corpus, with held-out confirmation, at a scale where raw markdown is impossible.
