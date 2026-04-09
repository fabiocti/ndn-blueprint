# Routing and Fusion

## Overview

The routing and fusion layers are the connective tissue of the NDN. Routing determines which node handles each piece of incoming text. Fusion assembles reconstructed text from multiple domains into coherent context for the agent LLM.

Both layers are currently simple. Improving them is an open problem.

---

## Routing

### What Routing Does

Routing examines incoming text and assigns it to a domain node. This happens at two points:

1. **Compression time**: incoming text is classified to determine which node encodes it
2. **Retrieval time**: a query or context need is classified to determine which domain(s) to search

### What Routing Sees

The router receives:
- The raw text to be classified
- Optional metadata (source type, session context, user-provided labels)

The router produces:
- A primary domain label
- A confidence score
- Optionally, a secondary domain label (for text that spans domains)

### Current Router: Rule-Based (v0)

The v0 router uses heuristic pattern matching:
- Code markers (function definitions, import statements, indentation patterns) → FTA
- Conversation markers (speaker turns, "User:", "Assistant:") → CONV
- Structured state markers (timestamps, key=value, status updates) → OSA
- Section headers + bullet lists + tool outputs → OSA/AOJ
- Policy/regulatory language → HPRT
- Default: NLK

This is deliberately simple. It works for the current prototype but has no calibration, no learned boundaries, and no confidence scoring.

### How Multiple Nodes Can Be Selected

Some text artifacts span domains. A code review that includes conversation about the code touches both FTA and CONV. A journal entry that embeds code snippets touches both OSA/AOJ and FTA.

The current approach: router picks the single best-fit domain. The text is encoded by one node only.

Future approach: the router could segment the text and route different segments to different nodes, or encode the full text with the primary node while tagging segments with secondary domain labels for retrieval purposes.

### Packet Provenance

Every packet records:
- Which domain node encoded it (`domain`, `subdomain`)
- Which checkpoint was used (`checkpoint`)
- The router's confidence for this classification (`confidence`)
- How the packet was retrieved (`retrieval_path`)

This provenance is preserved through reconstruction and fusion. The agent LLM (or a diagnostic tool) can trace any reconstructed text back to its source domain and checkpoint.

### Confidence and Completeness

Each packet carries two quality signals:

- **confidence**: the router's belief that this text belongs in this domain (0-1)
- **quality_est**: the node's estimated reconstruction quality (0-1), derived from internal metrics

These signals can be used during fusion to prioritize higher-confidence, higher-quality reconstructions when context budget is limited.

Currently, confidence is not calibrated. It is a placeholder for future improvement.

---

## Retrieval

### How Retrieval Works

When the agent needs memory, the retrieval layer:

1. **Determines which domain(s) to query** — based on the current task/query classification
2. **Selects packets within each domain** — using one or more strategies:
   - **Recency**: most recent packets first
   - **Session-based**: packets from specific sessions
   - **Relevance**: latent-space similarity or metadata matching
3. **Allocates context budget** — total available tokens distributed across domains
4. **Loads packets to GPU** — selected packets are moved from disk to GPU for decoding

### Context Budget Arbitration

When multiple domains have relevant memories:

```
total_budget = max_context_tokens - system_prompt - current_conversation
per_domain_budget = allocate(total_budget, relevance_scores, domain_priorities)
```

Each domain reconstructs up to its allocated budget. The allocation can be:
- Equal split across active domains
- Weighted by relevance scores
- Priority-based (domain with highest relevance gets the most budget)

The current implementation uses recency-based retrieval with a fixed token budget. More sophisticated strategies are future work.

---

## Fusion

### How Outputs Are Fused

Reconstructed text from multiple domains is assembled into a single context block:

```
[MEMORY — NLK]
<reconstructed encyclopedic text>

[MEMORY — CONV]
<reconstructed conversation history>

[MEMORY — OSA/AOJ]
<reconstructed operational journal>
```

The domain labels are preserved as section headers. The agent LLM reads this fused context and integrates information across domains.

Key fusion principles:
- **Domain labels are explicit** — the agent can see which domain each piece of memory came from
- **Order is controlled** — typically by relevance or recency, not random
- **Budget is respected** — total fused context does not exceed the allocated token budget
- **Provenance is traceable** — each section can be traced back to specific packets

### Post-Reconstruction Processing

After decoding, domain-specific post-processing may apply:
- **Overlapping chunk reassembly**: for chunked text encoded with overlap, take the first `stride` tokens from each chunk (except the last) to eliminate duplication
- **Deduplication**: if multiple packets cover the same time period or topic, deduplicate at fusion time
- **Formatting**: domain-specific formatting cleanup (e.g., ensuring markdown structure is preserved)

---

## Failure Modes

### Wrong-Node Projection

The most diagnostic failure mode. When text is routed to the wrong domain node, the reconstruction reflects the node's training domain, not the input domain.

**Example**: An agent operational journal routed to HWM-S64 (messy notes model) is reconstructed as fragmented note-style text, losing all structured journal content.

**What it tells you**: The text needs a different domain or subdomain. This failure is the primary trigger for taxonomy expansion.

**Severity**: High — fact recovery drops catastrophically (14-24% in tested cases).

### Packet Mixing

When packets from different domains are retrieved together and fused without clear domain boundaries, the agent LLM may confuse facts from one domain with another.

**Example**: A fact from a conversation ("the user mentioned 3 servers") appears in the same context as a journal entry ("27 live hosts"), and the agent conflates them.

**Mitigation**: Explicit domain labels in fused context. Clear section boundaries. Provenance tracking.

### Retrieval Miss

Relevant packets exist in the store but are not retrieved because:
- The relevance scoring failed
- The recency window was too narrow
- The domain classification at retrieval time was wrong

**Mitigation**: Broader retrieval windows. Multi-domain retrieval. Fallback to full-domain scan for critical queries.

### Fusion Hallucination

The agent LLM reads reconstructed text and generates responses that go beyond what was actually in memory. This is not an NDN failure per se — it is a reader LLM behavior — but NDN can exacerbate it by providing plausible-looking but degraded reconstructions.

**Mitigation**: Quality estimation signals. Confidence thresholds. Explicit "reconstructed from compressed memory" markers.

### Subdomain Confusion

Two related subdomains (e.g., OSA and AOJ) produce similar reconstructions but with different quality characteristics. The router assigns text to the parent when the subdomain would have been better, or vice versa.

**Mitigation**: Clear subdomain routing rules. Benchmark comparison between parent and subdomain on the target text type.

### Over-Fragmentation

Text is split across too many chunks or too many domains, losing cross-chunk and cross-domain coherence. Each individual reconstruction is acceptable, but the assembled context is disjointed.

**Mitigation**: Overlapping chunking. Domain-aware segmentation. Minimum chunk size thresholds.
