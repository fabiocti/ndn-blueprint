# Neural Domain Network (NDN)

**An architecture for domain-native memory in AI agents.**

---

## What is this?

NDN is a modular memory architecture. Instead of compressing all agent memory into one model or stuffing raw text into ever-larger context windows, NDN uses a network of specialized encoder-decoder nodes — each trained on a specific domain of knowledge.

The core idea: different kinds of information have fundamentally different shapes, and a single model trained on everything reconstructs nothing faithfully. NDN makes domain specialization explicit.

Each domain gets its own node, its own training data, its own compression regime, and its own benchmark profile. A router classifies incoming artifacts and sends them to the appropriate node. Reconstruction is domain-native.

## What is this repository?

This is the **architecture blueprint** for NDN. It contains:

- The full architecture specification
- A domain taxonomy with 6 validated top-level domains and 1 validated subdomain
- Registry files for domains, nodes, regimes, and benchmarks
- Evidence and case studies from real experiments
- Templates for contributing new nodes, subdomains, and benchmarks
- Honest documentation of what works, what doesn't, and what remains open

This is not a model release, not a product launch, and not a claim of universal superiority. It is the design foundation for the project.

## Where to start

| If you want to… | Start here |
|---|---|
| Understand the whole system | [core_docs/00_OVERVIEW.md](core_docs/00_OVERVIEW.md) |
| Read the architecture spec | [core_docs/02_ARCHITECTURE.md](core_docs/02_ARCHITECTURE.md) |
| See the domain taxonomy | [core_docs/03_TAXONOMY.md](core_docs/03_TAXONOMY.md) |
| Understand the founding philosophy | [core_docs/01_MANIFESTO.md](core_docs/01_MANIFESTO.md) |
| Check honest current status | [core_docs/07_PROJECT_STATUS.md](core_docs/07_PROJECT_STATUS.md) |
| See what NDN is *not* | [core_docs/09_WHAT_NDN_IS_NOT.md](core_docs/09_WHAT_NDN_IS_NOT.md) |
| Browse node registry | [registry/nodes.yaml](registry/nodes.yaml) |
| Read the evidence | [evidence/](evidence/) |
| Understand the release posture | [RELEASE_POSTURE.md](RELEASE_POSTURE.md) |

## Current status

NDN has 7 champion nodes across 6 domains and 1 subdomain. All nodes use a ~70M parameter CNDX v2 encoder-decoder architecture with compression ratios of 2x–4x.

**Validated:**
- Domain-specific training consistently beats cross-domain use
- Wrong-node failures are diagnostic (they reveal learned priors)
- AOJ subdomain achieves 73% fact recovery vs 14–24% from proxy nodes (test data was in training corpus; see [evidence/aoj_subdomain_case.md](evidence/aoj_subdomain_case.md) for caveats)
- CONV-S64 v2 achieves 94.9% F1 retention on LongMemEval

**Honest limitations:**
- On full real-world A/B against raw markdown, markdown still achieves 100% fact recovery vs NDN's best of 73%
- Rare entity preservation (OOV names, exact counts) remains unsolved
- Internal training metrics can be misleading — real-world A/B is the only trustworthy signal
- Router is basic and rule-based, not calibrated

See [core_docs/07_PROJECT_STATUS.md](core_docs/07_PROJECT_STATUS.md) for the full breakdown.

## What is included

```
├── core_docs/          Architecture specification and design documents (11 files)
├── registry/           Domain, node, regime, and benchmark registry (5 YAMLs + 7 node cards)
├── evidence/           Case studies from real experiments (5 files)
├── templates/          Contribution templates (5 files)
├── diagrams/           Architecture diagram specifications (6 files)
├── launch_prep/        Release planning and governance (5 files)
├── CHANGELOG.md        Version history
├── LICENSE             Apache 2.0
└── RELEASE_POSTURE.md  What this release is and is not
```

## What is intentionally excluded

- **Training code** — the CNDX encoder-decoder training pipeline exists but is not part of this release. Blueprint first; code later.
- **Model checkpoints** — binary weights are not distributed in this repository.
- **Internal experiment logs** — the raw experiment journal is private for now; the evidence docs contain the relevant findings.
- **Agent integration code** — the OpenClaw integration was a test harness, not a public API.

These may be released separately in the future.

## License

[Apache License 2.0](LICENSE)
