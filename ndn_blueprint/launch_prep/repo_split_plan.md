# Repository Split Plan

## Overview

When the NDN project is ready for open-source release, the code and documentation should be split across multiple focused repositories. This prevents a monolithic repo that is hard to navigate and harder to contribute to.

---

## Proposed Repositories

### ndn-core

**Purpose**: The core architecture — CNDX encoder-decoder model, latent bottleneck, training loop, evaluation framework.

**Contents**:
- `cndx/` — model architecture (`native_model.py`), data loading (`data.py`), training (`native_train.py`)
- Core evaluation scripts (ablation gap, shuffled gap, corruption sensitivity)
- Base configuration schemas
- Unit tests for the core model

**What belongs here**:
- Anything needed to train a new CNDX node from scratch
- The encoder-decoder architecture
- The latent bottleneck mechanism
- Standard training and evaluation code

**What does NOT belong here**:
- Pre-trained checkpoints (too large)
- Domain-specific data generators (belong in ndn-nodes)
- Runtime/agent integration code (belong in ndn-runtime)
- Benchmark probe suites (belong in ndn-benchmarks)

---

### ndn-nodes

**Purpose**: Domain-specific node definitions, training scripts, data generators, and node cards.

**Contents**:
- Per-domain directories (`nlk/`, `fta/`, `osa/`, `hwm/`, `hprt/`, `conv/`, `osa-aoj/`)
- Each directory contains:
  - Training data configuration / generator
  - Training script
  - Node card (documentation of the trained checkpoint)
  - Training logs and evaluation results (summary, not full logs)
- Pre-trained checkpoint download links (hosted externally, not in repo)

**What belongs here**:
- Everything specific to a particular domain or subdomain
- Synthetic data generators for each domain
- Domain-specific data processing scripts
- Node card templates (filled in)

**What does NOT belong here**:
- The core model code (in ndn-core)
- Runtime/integration code (in ndn-runtime)
- Model weight files directly (use external hosting + download scripts)

---

### ndn-benchmarks

**Purpose**: Benchmark probe suites, evaluation scripts, result aggregation, and comparison tools.

**Contents**:
- Domain-native benchmark families (NIAH, FACT, EXACT, TRACK, OVERRIDE, POS, CONTROL)
- External benchmark adapters (LongMemEval adapter, etc.)
- A/B test framework (`run_real_ab_v3.py` and similar)
- Result formatting and comparison tables
- Failure analysis tools
- Benchmark report templates

**What belongs here**:
- All evaluation code that is not part of core training
- Benchmark probe generators
- Result comparison and visualization
- A/B test harnesses

**What does NOT belong here**:
- Training code (in ndn-core)
- Domain-specific data (in ndn-nodes)
- Agent integration (in ndn-runtime)

---

### ndn-runtime

**Purpose**: The runtime system for using NDN in an agent — router, compressor, memory store, retrieval, fusion, session hooks.

**Contents**:
- `openclaw_memory/` module (MemoryStore, Router, Compressor, CompressorConfig, SessionHooks, FusedContext, DomainLabel)
- Router implementations (rule-based v0, future learned routers)
- Memory store (SQLite-backed packet storage)
- Retrieval strategies (recency, session-based, relevance)
- Fusion layer (domain-labeled context assembly)
- Configuration schemas for runtime deployment

**What belongs here**:
- Everything needed to use NDN in a running agent
- All runtime components (route, compress, store, retrieve, reconstruct, fuse)
- Agent integration hooks

**What does NOT belong here**:
- Training code (in ndn-core)
- Benchmark code (in ndn-benchmarks)
- Domain-specific training data (in ndn-nodes)

---

### ndn-integrations

**Purpose**: Integration examples and adapters for specific agent frameworks.

**Contents**:
- OpenClaw integration example
- Generic agent memory interface adapter
- Example configurations for different agent architectures
- Integration test harnesses

**What belongs here**:
- Agent-framework-specific adapter code
- Integration examples and tutorials
- Configuration templates for specific agent setups

**What does NOT belong here**:
- The core runtime (in ndn-runtime)
- Anything that is not agent-framework-specific

---

## Dependency Graph

```
ndn-core          ← foundational, no NDN dependencies
    ↑
ndn-nodes         ← depends on ndn-core for training
    ↑
ndn-benchmarks    ← depends on ndn-core for model loading, ndn-nodes for checkpoints
    ↑
ndn-runtime       ← depends on ndn-core for model inference
    ↑
ndn-integrations  ← depends on ndn-runtime
```

---

## Size Estimates

| Repo | Code Size | Data Size | Notes |
|---|---|---|---|
| ndn-core | ~500 LOC | None | Pure architecture |
| ndn-nodes | ~2K LOC | Checkpoints hosted externally (~300MB each) | Data generators + node cards |
| ndn-benchmarks | ~3K LOC | Small (probe configurations) | Evaluation framework |
| ndn-runtime | ~1K LOC | None | Runtime components |
| ndn-integrations | ~500 LOC | None | Adapter code |

---

## Migration Notes

The current monolithic workspace (`Codename - CNDX`) contains all of the above mixed together. Migration steps:

1. Extract `cndx/` core model code → ndn-core
2. Extract domain-specific generators and training scripts → ndn-nodes
3. Extract benchmark code and A/B test framework → ndn-benchmarks
4. Extract `openclaw_memory/` → ndn-runtime
5. Extract integration examples → ndn-integrations
6. Blueprint docs → ndn-core or separate ndn-docs repo

Checkpoints should be hosted externally (HuggingFace Hub, GitHub Releases, or similar) with download scripts in ndn-nodes.
