# Glossary

## Core Architecture Terms

**Domain**
A top-level category of information that shares structural and statistical properties. Defined empirically: a domain exists when a model trained on its data produces measurably better reconstruction than models trained on other domains. Current domains: NLK, FTA, OSA, HWM, HPRT, CONV.

**Subdomain**
A specialization within a domain, justified by evidence that proxy nodes from the parent domain demonstrably fail on the target text type. Example: AOJ (Agent Operational Journals) under OSA.

**Node**
A self-contained CNDX encoder-decoder pair trained on a specific domain or subdomain. Each node has its own weights (~70M parameters), latent configuration, and quality profile. Nodes are the atomic units of the NDN.

**Packet**
The unit of storage in the NDN. Contains a compressed latent tensor, metadata (domain, timestamp, session, compression ratio, quality estimate), and provenance (checkpoint, chunk index, retrieval path). Packets are opaque outside their originating node.

**Regime**
The compression configuration of a node, defined by the number of latent vectors (K). S16 = 16 latents (~8x compression), S32 = 32 latents (~4x compression), S64 = 64 latents (~2x compression). The regime is determined empirically per domain.

**Routing**
The process of classifying incoming text to the appropriate domain node during compression, and selecting relevant domains during retrieval. The current router is rule-based.

**Fusion**
The process of assembling reconstructed text from multiple domains into a coherent context block for the agent LLM. Includes domain labeling, budget allocation, and ordering.

**Memory Lattice**
The long-term goal: a growing network of benchmarked, domain-specialized compression nodes connected by a calibrated router. Not yet achieved — this is the aspirational end state.

---

## Node Lifecycle Terms

**Champion**
The best-performing checkpoint for a given domain or subdomain, validated by held-out evaluation and ideally real-world A/B testing. "Champion" means best NDN checkpoint, not necessarily better than markdown.

**Proxy**
A node used on text from a domain it was not trained on. Proxy usage is diagnostic: if the proxy fails with domain-prior projection, a dedicated node is justified. Proxies are never promoted to champion for a domain they were not trained on.

## Leaf Maturity Terms

**Leaf**
A specialized pipeline under a subdomain that targets a specific text type. A leaf combines a champion compression checkpoint with retrieval, ranking, entity extraction, and evaluation logic. Examples: TDR (disclosure reports), WS (workflow state), RWJ (recon workflow journals) under AOJ.

**Seed**
A hypothesized leaf that has not been tested. A text type has been identified as potentially distinct, but no benchmarks have been run.

**Blooming**
A leaf with first evidence of being a real branch. Initial benchmarks run, failure modes emerging, pipeline not yet frozen. Examples: WS (70% hits), RWJ (84% facts).

**Baseline Leaf**
A leaf with a frozen pipeline, dev + held-out benchmarks, documented failure modes, and a formal baseline declaration. Distinct from parent and siblings. Not yet proven at the rigor threshold.

**Validated Leaf**
A leaf proven robust on held-out data at or above the rigor threshold. Transfer evidence preferred but not required.

**Flagship Leaf**
The strongest evidence tier: validated + cross-corpus transfer + frozen champion pipeline. Example: TDR (90% hits, 93–94% facts, 89–105x compression, 5-corpus transfer at 95%).

**Entity Side-Channel**
A mechanism where regex-based extraction at compression time stores exact entities (domains, counts, ports, versions, etc.) alongside the latent blob. At reconstruction, decoded text is augmented with the preserved entity payload. The entity extractor is the primary tuning knob across leaves — same model, different patterns per domain.

**Active Parameters**
The number of model parameters currently loaded on GPU. A single node is ~70M. Multiple nodes can be loaded simultaneously. Active params are a small fraction of total params when only a few nodes are resident.

**Total Parameters**
The sum of all trained node parameters in the NDN, whether loaded or on disk. With 7 nodes: ~490M total. Not all are active simultaneously.

---

## Evaluation Terms

**Fact Recovery**
The percentage of known ground-truth facts that appear in reconstructed text. Measured during A/B testing by checking for specific entities, counts, and details in the reconstruction. The primary real-world quality metric.

**Continuity**
A measure of temporal and logical coherence in reconstructed memory. Scored 0-1. High continuity means the reconstructed context preserves the order and flow of the original information.

**Noise**
A measure of irrelevant, garbled, or confusing content in reconstructed text. Lower is better. Noise includes hallucinated entities, format corruption, and domain-prior leakage.

**Repeated-Work Signal**
An indication that the agent would repeat work it has already done, caused by missing or garbled memory. Measured by counting actions in reconstructed context that would be redundant if the agent remembered correctly.

**Shuffled Gap**
The difference in loss between using the correct latent representation and a shuffled (random) latent representation from the same domain. Measures how much the latent encodes about the specific input vs generic domain structure. Higher shuffled gap = more input-specific information in the latent.

**Ablation Gap**
The difference in loss between using the latent representation and using no latent (zero vector). Measures the total information contribution of the latent to reconstruction. Higher ablation gap = more information in the latent.

**Latent Dependence**
How much the decoder relies on the latent representation to produce good output, measured by ablation_gap and shuffled_gap. Conversation has the highest latent dependence (ablation_gap +12.00, shuffled_gap +13.30), meaning the latent is doing the most representational work.

**Held-Out**
Evaluation on data the model did not see during training. For synthetic corpora: eval samples from a different seed. For real data: a clean train/test split. Held-out evaluation is more trustworthy than training eval but less trustworthy than real-world A/B.

**Benchmark Family**
A set of related evaluation probes designed to test specific capabilities within a domain. Examples: NIAH (needle-in-haystack), FACT1/FACT2 (fact binding), EXACT (exact value recovery), TRACK (state tracking).

---

## Failure Mode Terms

**Entity Projection**
The phenomenon where a model reconstructs the correct structure but substitutes specific entities (domain names, counts, identifiers) with entities from its training distribution. The primary remaining failure mode in NDN.

**Wrong-Node Failure**
What happens when text is routed to a node trained on a different domain. The reconstruction reflects the node's training domain, not the input domain. Example: HWM-trained model reconstructs agent journals as messy notes. Diagnostic for taxonomy gaps.

**Domain-Prior Projection**
A broader term for wrong-node failure. The model's learned domain prior (the statistical patterns of its training domain) dominates reconstruction, overriding the actual input content.

---

## Infrastructure Terms

**CNDX**
The underlying encoder-decoder architecture used by NDN nodes. Based on GPT-2 with a latent bottleneck layer. Each CNDX model can encode text into a fixed number of latent vectors and decode those vectors back into text.

**Packet Store**
The persistent storage layer for memory packets. Currently implemented as SQLite. Indexed by domain, session, timestamp, and chunk index.

**Context Budget**
The maximum number of tokens available for injecting reconstructed memory into the agent's context window. Calculated as: max_context - system_prompt - current_conversation. Distributed across domains during fusion.
