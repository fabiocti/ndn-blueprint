# NDN Architecture

## Conceptual Model

The Neural Domain Network is organized as a tree of specialized encoder-decoder nodes, connected by a routing layer, backed by a persistent packet store, and producing reconstructed text that is fused into the agent's context window.

```
Agent LLM
    ↑ reads fused context
    |
[Fusion Layer]
    ↑ assembles reconstructed text from multiple domains
    |
[Reconstruction]
    ↑ domain node decodes latent packets to text
    |
[Retrieval]
    ↑ selects relevant packets by domain, recency, relevance
    |
[Packet Store]
    ↑ persists compressed packets with metadata
    |
[Compression]
    ↑ domain node encodes text to latent packets
    |
[Router]
    ↑ classifies incoming text to appropriate domain node
    |
Raw text (incoming memory artifact)
```

---

## Core Abstractions

### Domain

A top-level category of information that shares structural and statistical properties. Domains are defined empirically: a domain exists when a model trained on its data produces measurably better reconstruction than models trained on other domains.

Current validated domains:

| Domain | Short ID | Description |
|---|---|---|
| Natural Language Knowledge | NLK | Encyclopedic, factual, expository prose |
| Formal Technical Artifacts | FTA | Source code, function definitions, technical specifications |
| Operational State Artifacts | OSA | Structured event traces, state updates, workflow logs |
| Human Working Memory Text | HWM | Fragmented notes, reminders, partial plans |
| High-Precision Regulated Text | HPRT | Policy documents, compliance text, formal procedures |
| Conversational Memory | CONV | Multi-turn dialogue, chat history |

### Subdomain

A specialization within a domain. A subdomain is justified only when:

1. Existing proxy nodes from the parent domain demonstrably fail on the target text type
2. The failure is domain-prior mismatch (reconstructing in the wrong style), not general quality degradation
3. Dedicated training on the target text type measurably improves results
4. The improvement survives held-out evaluation

Current validated subdomains:

| Subdomain | Parent | Short ID | Description |
|---|---|---|---|
| Agent Operational Journals | OSA | AOJ | Markdown-formatted agent workflow logs, recon summaries, findings |

### Node

A self-contained CNDX encoder-decoder pair. Each node:

- Has its own trained weights (~70M parameters based on GPT-2 architecture)
- Has its own latent configuration (K regime)
- Operates independently — no weight sharing between nodes
- Can be loaded/unloaded from GPU independently
- Has its own quality profile: strengths, failure modes, benchmark results

A node is the atomic unit of the NDN. It is not subdivided at runtime.

### Regime

The compression configuration, defined by the number of latent vectors K:

| Regime | Latent Count | Latent Groups | Compression Ratio | Typical Use |
|---|---|---|---|---|
| S16 | 16 | 8,4,4 | ~8x | Experimental only. High fragility. |
| S32 | 32 | 16,8,8 | ~4x | Structured, regular content. Empirically the usual tradeoff between compression and quality for most domains. |
| S64 | 64 | 32,16,16 | ~2x | Complex, irregular content. Higher fidelity, lower compression. |

The regime is not arbitrary — it is determined by empirical testing. Structured operational traces work well at S32 (4x). Organic conversation requires S64 (2x). The choice is per-domain, not global.

### Checkpoint

A trained model state for a specific node at a specific version. Checkpoints are versioned and carry their full training configuration and evaluation results.

Naming convention: `{domain}_{regime}_v{version}`
Example: `aoj_s32_v2` — AOJ subdomain, S32 regime, version 2

### Champion vs Experimental

- **Champion**: the best-performing checkpoint for a given domain/subdomain, validated by held-out evaluation and ideally real-world A/B testing
- **Experimental**: a checkpoint that is being tested but has not yet been promoted

Promotion requires surviving evaluation outside the training distribution. Internal training metrics alone are never sufficient.

### Packet

The unit of storage in the NDN:

```
MemoryPacket {
    domain:         string              — which node produced this
    subdomain:      string | null       — subdomain if applicable
    latent:         Tensor[K, D]        — compressed latent representation
    metadata: {
        source_tokens:    int           — original text length
        timestamp:        datetime      — when created
        session_id:       string        — originating session
        compression:      float         — compression ratio
        quality_est:      float         — estimated reconstruction quality (0-1)
        confidence:       float         — router confidence for this domain classification
    }
    provenance: {
        checkpoint:       string        — which checkpoint encoded this
        chunk_index:      int           — position in multi-chunk sequence
        total_chunks:     int           — total chunks for this source
        retrieval_path:   string        — how this packet was retrieved (recency, relevance, etc.)
    }
}
```

Packets are opaque outside their originating node. Only the node that compressed a packet can reconstruct it. Cross-domain decoding produces garbage.

### Active Parameters vs Total Parameters

| Configuration | Active Params | GPU Memory |
|---|---|---|
| 1 node loaded | ~70M | ~280MB |
| 2 nodes loaded | ~140M | ~560MB |
| 4 nodes loaded | ~280M | ~1.1GB |
| All 6 nodes loaded | ~420M | ~1.7GB |

NDN nodes are tiny relative to the reader LLM. Even loading all trained nodes simultaneously uses <2GB VRAM. The practical constraint is not memory but inference latency per encode/decode pass.

For agent use: keep the 2-3 most frequently used nodes GPU-resident. Swap others from disk as needed (~100ms cold load).

---

## Full Lifecycle

### 1. Ingest — Raw artifact enters

An agent produces or receives a text artifact: a conversation turn, a code file, a journal entry, a state update. This raw text is the input to the NDN.

### 2. Route — Router classifies and ranks

The routing layer examines the incoming text and assigns it to a domain node:

- Input: raw text
- Output: domain label, confidence score, optional secondary domain

The v0 router is rule-based (heuristic patterns: code markers, conversation markers, structured data markers). Future versions may use embedding-based or LLM-based classification.

Misrouting degrades reconstruction quality (domain-prior mismatch) but does not cause catastrophic data loss. The text is still stored and reconstructable — just at lower quality.

### 3. Chunk — Artifact is segmented

The text is split into chunks compatible with the node's sequence length (128 tokens for current models). Chunking is domain-aware:

- Overlapping chunking (stride < chunk_size) reduces information loss at boundaries
- Chunk boundaries respect natural breakpoints where possible (paragraph breaks, section headers)

### 4. Compress — Node encodes chunks to packets

Each chunk is processed by the domain node's encoder:

- Input: tokenized chunk (128 tokens)
- Output: latent tensor [K, D] where K is the regime (16, 32, or 64) and D is the latent dimension

Each chunk becomes one memory packet with full metadata and provenance.

### 5. Store — Packets are persisted

Packets are written to the packet store, indexed by:

- Domain and subdomain
- Session ID
- Timestamp
- Chunk index within source

Storage is persistent (disk-backed SQLite in the current implementation) with optional GPU-resident cache for active domains. No cross-domain merging — packets from different domains coexist but do not blend.

### 6. Retrieve — Relevant packets selected

When the agent needs memory, the retrieval layer:

1. Determines which domain(s) to query (router-assisted or explicit)
2. Selects relevant packets within each domain (recency-based, session-based, or relevance-scored)
3. Allocates context budget across domains
4. Loads selected packets to GPU

### 7. Reconstruct — Node decodes packets to text

The domain node's decoder converts latent tensors back to text:

- Input: latent tensor [K, D]
- Output: reconstructed text tokens

Only the originating node can decode a packet. The decoder is autoregressive: it generates tokens conditioned on the latent representation.

### 8. Fuse — Assemble usable context

Reconstructed text from multiple domains is assembled into a coherent context block:

- Domain-labeled sections (provenance preserved)
- Budget-constrained (total reconstruction fits within allocated context tokens)
- Ordered by relevance or recency

The agent LLM reads this fused context as if it were original text. The NDN does not reason — the agent LLM handles cross-domain integration.

---

## Design Principles

### Nodes Do Not Merge

The NDN explicitly does not merge domains into a single model. This is a deliberate architectural choice based on empirical evidence:

1. **No shared latent space** — each node has its own latent geometry
2. **No joint training** — nodes train independently on domain-specific corpora
3. **Reconstruction is text** — the only interface between domains is reconstructed plaintext
4. **The agent LLM is the integrator** — it reads multi-domain reconstructed text and reasons across it
5. **Domain boundaries are hard** — a packet belongs to exactly one domain

### When a Subdomain Is Justified

A subdomain is justified when:

1. Proxy nodes from the parent domain fail on the target text type (measured by held-out evaluation or real-world A/B)
2. The failure mode is domain-prior projection (wrong style), not generic quality degradation
3. Training a dedicated node on the target text type measurably improves results
4. The improvement is reproducible and survives held-out evaluation

Example: OSA-S32 (trained on timestamped key-value traces) recovered only 14% of facts from markdown-formatted agent journals. Dedicated AOJ-S32 training recovered 54% (v1) and 73% (v2). The failure was domain-prior mismatch, not architecture limitation.

### When a Node Should Be Deprecated

A node should be deprecated when:

1. A better node for the same domain exists and is validated
2. The domain is absorbed into a broader domain without quality loss
3. The node's training data is discovered to be contaminated or unrepresentative
4. The node consistently fails held-out evaluation despite retraining attempts

Deprecated nodes are archived, not deleted. Their benchmark results remain in the record.

---

## What This Architecture Does NOT Include (Yet)

- **Memory consolidation** — merging old memories, summarizing, forgetting
- **Hierarchical compression** — compress-then-compress for very old memories
- **Cross-domain retrieval** — finding memories by content similarity across domains
- **Online learning** — updating node weights at runtime with new data
- **Memory importance scoring** — deciding what to keep and what to discard
- **Copy/pointer mechanisms** — exact entity preservation for rare identifiers

These are all potential future extensions. The current NDN is: route → compress → store → retrieve → reconstruct → fuse. Nothing more.
