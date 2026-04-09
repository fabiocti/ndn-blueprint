# Release Posture

This document defines the scope and intent of this release.

---

## What this release is

This is an **architecture blueprint release**. It defines:

- The NDN architecture: domains, nodes, regimes, routing, fusion
- A domain taxonomy with evidence behind each branch
- Benchmark discipline: what counts as validation, what doesn't
- Templates and standards for future contributions
- Honest documentation of failures and negative results alongside successes

The goal is to define the architecture, taxonomy, and benchmark discipline clearly enough that others can evaluate it, critique it, build on it, or reject it with specifics.

## What this release is not

- **Not a model release.** No weights are distributed.
- **Not a product launch.** There is no API, no service, no pricing.
- **Not a claim of universal dominance.** NDN does not yet beat raw markdown on every real-world memory task. The evidence docs say exactly where it wins and where it loses.
- **Not AGI.** NDN is a memory architecture, not an intelligence architecture.

## Status of nodes

There are 7 champion nodes across 6 domains and 1 subdomain. "Champion" means it is the best-performing checkpoint for its domain, with benchmark evidence. It does not mean production-ready or universally better than alternatives.

| Status | Description |
|---|---|
| Validated | Domain-specific training beats cross-domain use across all tested domains. AOJ subdomain justified by proxy failure evidence. |
| Promising | CONV achieves 94.9% F1 retention on LongMemEval at 2x compression. |
| Honest | On full real-world A/B testing, raw markdown still achieves higher fact recovery than NDN on real agent sessions. |
| Failed | AOJ v3 (stronger entity weighting) regressed on real A/B despite better internal metrics. Documented as a negative result. |

## Allowed claims

- NDN is a modular, domain-native memory architecture
- Domain-specific compression outperforms cross-domain compression on internal benchmarks
- AOJ is a validated subdomain with measurable real-world improvement over proxy nodes
- Benchmark discipline is a core design principle
- Wrong-node failures are informative and diagnostic

## Claims that are not supported

- NDN is universally better than markdown memory
- NDN is production-ready
- Low training loss means good real-world performance
- More entity weighting always helps
- The taxonomy is complete
- This architecture is required for agent memory (it is one approach)

## Why blueprint first

Releasing the architecture specification before the code serves several purposes:

1. **Reviewability.** The design can be evaluated independently of implementation details.
2. **Lower attack surface.** No one can claim the code doesn't work if the code isn't the claim.
3. **Honest framing.** The blueprint documents both successes and failures, establishing credibility before any code release.
4. **Contribution clarity.** Contributors know the standards before they start.

Training code, model checkpoints, and integration examples may follow in future releases.
