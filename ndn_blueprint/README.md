# Neural Domain Network (NDN) — Blueprint

**Version**: 0.1 (pre-release draft)
**Date**: 9 April 2026
**Status**: Local design package — not published

---

## What This Is

This is the canonical design package for the Neural Domain Network (NDN). It contains the architecture specification, domain taxonomy, benchmark standards, evidence from completed experiments, and templates for future contribution.

This is not marketing material. It is a technical blueprint intended to serve as the foundation for:

- Open-sourcing the NDN project
- Explaining the architecture to collaborators
- Justifying the domain taxonomy with evidence
- Onboarding future contributors
- Defining benchmark discipline
- Documenting what has been proven, what is promising, and what remains unresolved

---

## What NDN Is

The Neural Domain Network is a modular memory architecture for AI agents. Instead of one monolithic model compressing all memory into a single latent space, the NDN uses a network of specialized encoder-decoder nodes, each trained on a specific domain of knowledge.

The core insight: **different kinds of information have fundamentally different shapes**, and compressing them with a single model produces mediocre results everywhere. A model trained on encyclopedic prose learns to reconstruct everything as encyclopedic prose. A model trained on messy notes reconstructs everything as messy notes. This is not a bug — it is evidence that the latent space encodes domain-specific representational priors.

The NDN makes this explicit: each domain gets its own node, its own training corpus, its own compression regime, and its own benchmark profile. A router classifies incoming memory artifacts and sends them to the appropriate node. Reconstruction is domain-native.

---

## Why Monolithic Memory Is Insufficient

Most current agent memory systems use one of these approaches:

1. **Raw text storage** — store everything as markdown or plaintext. Simple, lossless, but context windows are finite. As history accumulates, either everything gets stuffed into context (slow, expensive, noisy) or old memory is dropped.

2. **Vector retrieval** — embed text chunks, store embeddings, retrieve by similarity. Good for search, but the retrieved chunks are still raw text. No compression, no reconstruction, no domain awareness.

3. **Generic summarization** — LLM summarizes old context. Lossy, non-invertible, and the summarization quality depends on the LLM's own biases. Domain-specific details (exact counts, entity names, code syntax) are systematically lost.

4. **One universal compression model** — train a single encoder-decoder to compress everything. The result is a compromise: acceptable at nothing, good at nothing. Wrong-domain priors actively corrupt reconstruction.

NDN addresses this by treating memory as **plural, not monolithic**. Different information types get different treatment. The system can grow by adding new domain nodes without retraining existing ones.

---

## Core Concepts

### Domain
A top-level category of information that shares structural properties. Examples: encyclopedic knowledge, source code, operational state, conversation. Domains are defined by evidence, not intuition.

### Subdomain
A narrower specialization within a domain. Example: within Operational State Artifacts (OSA), Agent Operational Journals (AOJ) is a subdomain for markdown-formatted agent workflow logs. Subdomains are justified only when proxy nodes from the parent domain demonstrably fail on the target text type.

### Node
A self-contained CNDX encoder-decoder pair trained on a specific domain or subdomain. Each node has its own weights (~70M parameters), its own latent configuration, and its own quality profile. Nodes are the atomic units of the NDN.

### Regime
The compression configuration of a node, defined by the number of latent vectors (K):
- **S16**: 16 latents, ~8x compression. Frontier/experimental. High fragility.
- **S32**: 32 latents, ~4x compression. Best for structured, regular content.
- **S64**: 64 latents, ~2x compression. Best for complex, irregular content.

### Packet
The unit of storage. A memory packet contains a compressed latent tensor, metadata (source domain, timestamp, session, compression ratio, quality estimate), and provenance information. Packets are opaque outside their originating node.

### Router
The classification layer that directs incoming text to the appropriate domain node during compression, and selects relevant domains during retrieval.

### Reconstruction
Decoding a latent packet back into text using the originating domain node. Only the domain that compressed the packet can reconstruct it. Cross-domain decoding is meaningless.

### Fusion
Assembling reconstructed text from multiple domains/packets into a coherent context for the agent's LLM. The agent LLM is the integrator — it reads reconstructed text from multiple domains and reasons across them.

---

## What Has Been Shown

The following have empirical evidence behind them:

1. **Domain-specific training consistently beats cross-domain use.** A wiki-trained model on code text, or a notes-trained model on conversation, produces systematically worse reconstruction. Measured via shuffled_gap across all tested domains.

2. **Six top-level domains validated.** Natural Language Knowledge, Formal Technical Artifacts, Operational State Artifacts, Human Working Memory Text, High-Precision Regulated Text, and Conversational Memory — all trained, evaluated with internal metrics and domain-native benchmarks.

3. **One subdomain validated.** Agent Operational Journals (AOJ) under OSA. Proxy nodes from other domains recovered 14-24% of facts on real OpenClaw agent data. Dedicated AOJ training achieved 99% (v4, trained on 50% real GitHub journals + 45% synthetic + 5% OpenClaw real).

4. **Public benchmark result.** CONV-S64 v2 achieves 94.9% F1 retention on LongMemEval (500 questions) compared to uncompressed baseline.

5. **Compression works at meaningful ratios.** 2x-4x compression with substantial information retention across all domains.

6. **Wrong-node failures are diagnostic.** When a model trained on domain A reconstructs domain B text, the reconstruction reveals the learned domain prior. This is scientifically informative and guides taxonomy refinement.

---

## What Remains Unresolved

1. **Single-session fact recovery is essentially solved.** AOJ v4 achieves 99% fact recovery at 1.78x compression on the same 5-slice OpenClaw A/B test where v2 scored 73%. Markdown (100%, 1.0x) retains a 1pp edge on raw facts but no longer dominates. The remaining gap is in scale retrieval: TDR scale benchmark with v4 scores 17/20 (85%), 91% fact recovery, 86x compression — slightly below v2's 18/20 (93%, 89x). Retrieval, not compression, is now the bottleneck.

2. **Rare entity preservation.** The model reconstructs the correct structure but substitutes out-of-vocabulary entities (domain names, exact counts, error strings) with training priors.

3. **Loss-level interventions have limits.** Token-level entity weighting (AOJ v3) improved internal metrics but regressed real-world A/B performance. The entity problem likely requires architectural changes (copy/pointer mechanisms, retrieval augmentation), not more loss engineering.

4. **Routing is basic.** Current router is rule-based. No calibration, no confidence scoring, no graceful multi-domain handling.

5. **No production integration yet.** The OpenClaw integration was a test harness, not a deployed system.

6. **Scaling to many domains/subdomains is untested.** The taxonomy has 6 domains and 1 subdomain. The architecture is designed to scale, but governance and benchmark discipline at scale are open problems.

---

## Blueprint Contents

```
ndn_blueprint/
├── README.md                          ← this file
├── core_docs/
│   ├── 00_OVERVIEW.md                 System overview
│   ├── 01_MANIFESTO.md                Founding philosophy
│   ├── 02_ARCHITECTURE.md             Full architecture specification
│   ├── 03_TAXONOMY.md                 Domain and subdomain definitions
│   ├── 04_NODE_LIFECYCLE.md           Node maturity stages and promotion
│   ├── 05_ROUTING_AND_FUSION.md       Routing, retrieval, and context fusion
│   ├── 06_BENCHMARK_PHILOSOPHY.md     Evaluation discipline and standards
│   ├── 07_PROJECT_STATUS.md           Honest current status
│   ├── 08_OPEN_PROBLEMS.md            Unsolved problems
│   ├── 09_WHAT_NDN_IS_NOT.md          Explicit scope boundaries
│   └── 10_GLOSSARY.md                 Term definitions
├── registry/
│   ├── domains.yaml                   Top-level domain registry
│   ├── subdomains.yaml                Subdomain registry
│   ├── nodes.yaml                     Node and checkpoint registry
│   ├── regimes.yaml                   Compression regime definitions
│   └── benchmark_families.yaml        Benchmark family registry
├── evidence/
│   ├── conversation_memory_case.md    CONV domain case study
│   ├── openclaw_ab_case.md            Real A/B test case study
│   ├── aoj_subdomain_case.md          AOJ subdomain justification
│   ├── proxy_failure_case.md          Wrong-node failure evidence
│   └── training_vs_real_eval_case.md  Internal metrics vs real A/B
├── launch_prep/
│   ├── repo_split_plan.md             Future repository structure
│   ├── public_release_order.md        Release sequencing
│   ├── contributor_model.md           Contribution standards
│   ├── governance_notes.md            Taxonomy governance
│   └── risks_of_open_sourcing.md      Known risks
├── templates/
│   ├── node_card_template.md          Standard node documentation
│   ├── subdomain_proposal_template.md New subdomain proposals
│   ├── benchmark_report_template.md   Benchmark writeup format
│   ├── failure_analysis_template.md   Failure documentation
│   └── ab_test_report_template.md     A/B test reporting
└── diagrams/
    ├── diagram_01_system_overview.md  NDN system overview
    ├── diagram_02_node_tree.md        Domain taxonomy tree
    ├── diagram_03_packet_flow.md      Compress → store → retrieve → reconstruct
    ├── diagram_04_routing_and_fusion.md  Router and context assembly
    ├── diagram_05_node_lifecycle.md   Node maturity progression
    └── diagram_06_openclaw_integration.md  Agent integration example
```
