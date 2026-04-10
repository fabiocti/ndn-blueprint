# Case Study: How the First Flagship Leaf Emerged

**OSA → AOJ → Technical Disclosure Reports (TDR)**

---

## The Problem

An AI agent running long-term security operations accumulates hundreds of technical reports over time. Each report contains structured findings — vulnerability types, affected domains, tool outputs, CVE identifiers, version strings, HTTP status codes, host counts.

After 100 reports, the accumulated raw markdown exceeds 1.1 million tokens. No context window can hold it. The agent needs memory.

The question: can NDN compress this archive, retrieve the right report for a given query, and reconstruct its facts accurately?

---

## Why Broad Proxies Failed

### OSA-S32: Right Semantic Domain, Wrong Surface Format

The Operational State Artifacts (OSA) node was trained on compact timestamped key-value traces:

```
[T=3] server status=active
[T=7] database connection=timeout
[T=12] server status=recovered
```

When given a real agent journal:

```markdown
## Phase 1A: Subdomain Collection — IN PROGRESS
- subfinder: 555 hosts found
- httpx: 169 live (30.4%)
- Total merged: 673 unique subdomains
```

OSA projected the markdown into its training format. Headers became key-value pairs. Bullet lists collapsed. Counts vanished. Domain names were hallucinated from training vocabulary.

**Result: 14% fact recovery.** The architecture was correct (operational tracking), but the surface format was catastrophically wrong.

### HWM-S64: Wrong Domain Entirely

The Human Working Memory node was trained on messy, fragmented notes. It reconstructed operational journals as stream-of-consciousness text. Some gist survived (tool names, phase labels), but all operational structure — the things that matter in a security report — was lost.

**Result: 24% fact recovery.** Better than OSA because it at least preserved some free-form content, but it was the wrong domain. Recon journals are not notes.

### The Diagnostic Signal

Both failures showed **domain-prior projection**: the model reconstructed in its own training style, not the input style. This is the NDN signal for "you need a new node." The text type has a distinct structure that no existing node learned.

---

## Why AOJ Emerged

The Agent Operational Journals (AOJ) subdomain was created under OSA because:

1. **Same semantic domain** — it is operational state tracking (what happened, what was found, what is the current status)
2. **Different surface format** — markdown-formatted journals vs compact timestamped traces
3. **Documented proxy failure** — two existing nodes failed measurably (14% and 24%)

AOJ v1 was trained on synthetic journals covering 10 structural patterns found in real agent output: phase transitions, tool-result-count triples, domain references, aggregate counts, host classifications, error notes, recon iterations, next-step notes, markdown structure, and multi-target entries.

**Result: 54% fact recovery.** A 3.8x improvement over OSA proxy. The model learned the journal structure but still hallucinated entity names not in its training vocabulary.

AOJ v2 expanded the entity vocabulary (42 → 204 SLDs, 16 → 52 TLDs) and mixed in 5% real journal data.

**Result: 73% fact recovery.** The remaining misses were concentrated and taxonomizable: exact numeric counts (58%), OOV domain names (25%), error/status strings (17%).

AOJ v3 tried to fix these via stronger loss weighting (digit_weight 5.0x, entity_weight 3.0x on 122 identified tokens). Internal metrics improved across the board. Real A/B **regressed to 66%**.

**Lesson: loss weighting is the wrong lever.** Internal metrics can mislead. The champion stayed at v2.

---

## Why This Leaf Emerged Under AOJ

AOJ proved the compression node works on individual sessions. But a real agent does not have one session — it has hundreds. The next question was: does the system work at scale?

The answer was not obvious, and required solving three problems that did not exist at the single-session level:

### Problem 1: Retrieval is Broken at Scale

At 100 reports and 1.15M tokens, you cannot paste everything. You must retrieve. The initial retrieval path used naive recency — always returning the 5 most recent sessions regardless of query content.

**Result: 0/5 target hits.** The system was retrieving the wrong reports every time.

**Fix: FTS5-based session search.** A session-level index (title, first 1000 chars, entity values) with BM25 ranking replaced recency.

**Result: 4/5 target hits.** The retriever now actually found the right reports.

### Problem 2: Blending Destroys Specificity

Even with the correct report in the top-5, fact recovery stayed at 21%. The entity side-channel was merging entities from all 5 retrieved sessions — injecting domain names, counts, and CVEs from unrelated reports into the output.

**Fix: Per-session isolated reconstruction.** Each of the top-5 candidates is reconstructed independently, then scored against the query. Only the best candidate is returned.

**Result: 77% fact recovery.** When isolation picks the correct session: **100% fact recovery (30/30 facts every time).** The 23% drag came entirely from wrong-session picks, not reconstruction failures.

### Problem 3: Metadata Quality Matters More Than Ranking Intelligence

Multiple reranking strategies were tested: entity-only scoring, hybrid FTS+entity, two-stage rerankers, and a 3B-parameter LLM reranker (Qwen2.5-3B-Instruct).

The LLM reranker performed **worse** than the heuristic (70% vs 87% fact recovery). The breakthrough was not a smarter ranker. It was a one-line fix: **prepending the actual report title to the session text before indexing.**

The system had been extracting titles from the first line of the report body — which was often a generic header like "## Summary" — instead of the actual report title. With correct titles indexed, the simple heuristic outperformed everything.

**Lesson: the retriever needs the right anchors, not more intelligence.**

---

## Why Hybrid Packets Mattered

The CNDX encoder-decoder compresses text into latent vectors. It captures narrative structure — phase progression, tool usage patterns, operational flow — but hallucinates specific entities. "subfinder found 555 hosts" becomes "nmap found 312 targets."

The entity side-channel stores exact entities (domains, counts, ports, versions, IPs, HTTP statuses, CVEs) as regex-extracted JSON alongside the latent blob. At reconstruction time, the decoded text provides the narrative; the entity payload provides the truth.

**Before hybrid packets**: 14% fact recovery (on controlled test)
**After hybrid packets**: 100% fact recovery (on controlled test)

At scale, the hybrid format means each report's critical facts survive compression intact. The model does the hard work of structural compression; the regex extractor preserves what the model cannot.

---

## Why Isolation Mattered

This was the most important architectural discovery of the project.

When you retrieve 5 candidate sessions and blend their packets together, you get:
- Entity payloads from 5 different reports merged into one bag
- Domain names from Report A mixed with counts from Report C
- Reconstructed text that is a confused amalgam of unrelated findings

**Blended fact recovery: 21%.**

When you reconstruct each candidate independently and select the best:
- Each candidate's entities are internally consistent
- The reconstructed text for each session is coherent
- The scoring step picks the session that best matches the query

**Isolated fact recovery: 77% average, 100% on correct session.**

The difference is 56 percentage points. Isolation is not an optimization — it is a requirement.

---

## Current Strengths

1. **89–105x compression** on a 1.15M-token archive — makes 100+ reports fit in a standard context window
2. **94% fact recovery** across 40 queries (dev + held-out)
3. **100% fact recovery** when the correct session is isolated — the reconstruction pipeline is not the bottleneck
4. **Simple architecture** — FTS5, regex extraction, heuristic scoring. No embeddings, no neural reranker, no LLM in the retrieval loop
5. **Held-out validated** — dev and held-out sets produce nearly identical metrics (93% vs 94%), confirming no overfitting
6. **Single clear failure mode** — near-identical titles. All other query types achieve 100%

---

## Current Weaknesses

1. **Single corpus** — validated on HackerOne reports only. Generalization to other technical disclosure formats (CVE advisories, pentest reports, compliance findings) is untested
2. **Near-identical titles** remain unsolved — 63% hit rate on ambiguous bucket vs 100% on all others
3. **Regex extraction is brittle** — hardcoded patterns that will miss novel entity formats
4. **No late fusion tested** — for queries that genuinely span multiple reports, there is no multi-session fusion path yet
5. **AOJ v2 training data leakage** — the 5% real journal mix overlaps with early A/B test targets (not with HackerOne, but the caveat exists)
6. **One compression model for all report types** — no evidence yet on whether different technical disclosure subformats need different compression nodes

---

## What This Leaf Proves

1. **NDN can work as a real memory backend at scale** — not just a lab abstraction
2. **The tree structure is real** — OSA → AOJ → TDR is a genuine hierarchy where each level solves a problem the level above cannot
3. **Architecture matters more than model quality** — the biggest gains came from pipeline changes (isolation, metadata, FTS5), not model improvements
4. **The entity side-channel is the right Tier 1 intervention** — it works without retraining and handles the model's weakest failure mode
5. **Held-out evaluation works** — dev and held-out metrics converge, so the system is not gaming a single benchmark

---

## Tree Position

```
NDN
└── OSA (Operational State Artifacts)
    └── AOJ (Agent Operational Journals)
        └── 🟢 TDR (Technical Disclosure Reports) ← FLAGSHIP LEAF
```

Next candidate siblings under AOJ:
- **Workflow State** — agent task state, completed steps, pending actions (different retrieval pattern: latest-state, not historical search)
- **Recon Workflow Journals** — single-target recon narratives (different from accumulated archive: sequential, not searchable)
