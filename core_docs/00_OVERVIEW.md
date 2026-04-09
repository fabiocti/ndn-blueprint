# NDN Overview

## What NDN Is

The Neural Domain Network is a modular memory system for AI agents. It replaces monolithic memory with a network of specialized compression nodes, each trained to handle a specific type of information.

The system has five phases (simplified from the full eight-step lifecycle detailed in 02_ARCHITECTURE.md):

1. **Route** — classify incoming text to the appropriate domain node
2. **Compress** — encode text into a compact latent representation using the domain-specific encoder
3. **Store** — persist the compressed packets with metadata and provenance
4. **Retrieve** — select relevant packets when the agent needs memory
5. **Reconstruct & Fuse** — decode packets back to text and assemble usable context for the agent

Each domain node is a self-contained encoder-decoder (~70M parameters) trained on domain-specific data. Nodes do not share weights. The agent's LLM (GPT, Claude, Qwen, etc.) reads the reconstructed text — the NDN does not reason, it stores and reconstructs.

## Why It Exists

Current agent memory approaches have fundamental limitations:

**Raw text storage** works perfectly but does not scale. Context windows are finite. As agent history grows, either the context becomes prohibitively large or old information is silently dropped.

**Vector retrieval** finds relevant chunks by similarity but stores and returns raw text. No compression, no domain awareness, no structural preservation guarantees.

**Generic summarization** loses domain-specific details systematically. An LLM summarizing a code review session will drop variable names. An LLM summarizing a recon journal will substitute domain names.

**One universal compression model** produces a compromise that is mediocre everywhere. When a single model is trained on mixed data, it learns a blended prior that actively corrupts domain-specific reconstruction.

NDN addresses this with specialization: different information types get different compression models, each optimized for its domain's structural properties.

## How It Differs From Existing Approaches

| Approach | Compression | Domain-aware | Preserves structure | Scales with history |
|---|---|---|---|---|
| Raw markdown | None | No | Perfect | No — context grows unbounded |
| Vector retrieval | None | No | Chunk-level only | Partially — retrieval is selective |
| LLM summarization | Lossy, non-invertible | No | No — LLM biases dominate | Yes — but quality degrades |
| Universal encoder | Fixed ratio | No | Compromised | Yes — but mediocre everywhere |
| **NDN** | **Domain-specific ratio** | **Yes** | **Domain-optimized** | **Yes — per-domain scaling** |

## Core Building Blocks

- **Domain**: a top-level category of information (e.g., encyclopedic knowledge, source code, operational state)
- **Subdomain**: a specialization within a domain, justified by evidence that the parent node fails on the target text type
- **Node**: a trained encoder-decoder pair for a specific domain or subdomain
- **Regime**: the compression level (S16=8x, S32=4x, S64=2x)
- **Packet**: a compressed latent tensor with metadata, the unit of storage
- **Router**: classifies incoming text to the appropriate node
- **Fusion**: assembles reconstructed text from multiple domains into agent context

## What Has Been Validated

Proven with empirical evidence:

1. Domain-specific training consistently outperforms cross-domain use (measured by shuffled_gap across all 6 domains)
2. Six top-level domains trained and evaluated with internal metrics and domain-native benchmarks
3. One subdomain (AOJ) validated with real-world A/B testing
4. Public benchmark result: 94.9% F1 retention on LongMemEval (500 questions, conversation domain)
5. 2x–4x compression at the champion node level with substantial information retention (S16 at ~8x exists but is experimental-only)
6. Wrong-node failures are diagnostic and guide taxonomy refinement

## What Still Needs Proof

1. NDN has not yet beaten raw markdown on a real-world agent A/B test (best: 73% vs 100% fact recovery) (test data was included in training corpus; see evidence/aoj_subdomain_case.md for caveats)
2. Rare entity preservation remains unsolved at the architecture level
3. Routing calibration and multi-domain retrieval are basic
4. No production deployment exists
5. Scaling governance for many domains/subdomains is untested
6. Integration with live agents is at prototype stage only
