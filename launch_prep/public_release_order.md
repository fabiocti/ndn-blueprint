# Public Release Order

## Principle

Release from most-stable to least-stable. Documentation and standards first. Code and checkpoints after validation. Integrations last.

---

## Phase 1: Blueprint and Core Documentation

**What**: This NDN blueprint package — architecture spec, taxonomy, manifesto, benchmark philosophy, glossary, project status, open problems.

**Why first**: Sets the framing before any code or checkpoints are public. Establishes benchmark discipline, honest scope, and contribution standards. Prevents the narrative from being shaped by others.

**Prerequisites**: Blueprint review and cleanup. Ensure all evidence references are accurate.

**Risk level**: Low. Documentation alone cannot be misused. Sets the right tone.

---

## Phase 2: Taxonomy and Benchmark Standards

**What**: Domain registry, subdomain registry, regime definitions, benchmark family definitions, node lifecycle spec, contribution templates.

**Why second**: Establishes the structural framework that all future contributions must follow. Makes clear what counts as evidence and what does not.

**Prerequisites**: Phase 1 published. Registry YAML files reviewed for accuracy.

**Risk level**: Low. Standards without code are safe.

---

## Phase 3: Reference Node Summaries and Evidence

**What**: Node cards for all champion checkpoints. Case studies (conversation memory, OpenClaw A/B, AOJ subdomain, proxy failures, training vs real eval). Node registry with honest maturity levels.

**Why third**: Shows what has been done, with evidence. Demonstrates the methodology in practice. Includes negative results (AOJ v3, proxy failures).

**Prerequisites**: Phase 2 published. Case studies reviewed for accuracy and honesty.

**Risk level**: Low-medium. Evidence could be cherry-picked by external commenters. Mitigate by being aggressively honest about limitations.

---

## Phase 4: Core Architecture Code

**What**: ndn-core repository — the NDN encoder-decoder model, training loop, evaluation framework.

**Why fourth**: The code is the primary artifact for reproducing training and inference. Releasing it allows others to reproduce and extend the work. But it should come after documentation so the context is established.

**Prerequisites**: Phases 1-3 published. Code cleaned and tested. License chosen.

**Risk level**: Medium. Others can train their own nodes. This is the point — but it also means losing exclusive control over the narrative.

---

## Phase 5: Domain Nodes and Checkpoints

**What**: ndn-nodes repository — training scripts, data generators, pre-trained checkpoints (hosted externally), node cards.

**Why fifth**: Checkpoints are the practical contribution. They let others use NDN immediately without training from scratch. But they should come after the architecture and standards are established.

**Prerequisites**: Phase 4 published. Checkpoints hosted and download scripts tested. Node cards complete.

**Risk level**: Medium. Checkpoints can be benchmarked by others, potentially revealing weaknesses not yet documented. Mitigate by documenting known weaknesses first (Phase 3).

---

## Phase 6: Benchmarks and Evaluation Framework

**What**: ndn-benchmarks repository — probe suites, A/B test framework, result comparison tools.

**Why sixth**: Enables others to evaluate NDN nodes rigorously. Essential for community trust, but also opens the door to unfavorable comparisons.

**Prerequisites**: Phase 4-5 published. Benchmark code tested. Example results included.

**Risk level**: Medium-high. External benchmarking may reveal new failure modes. This is valuable but potentially embarrassing. Mitigate by being first to document weaknesses.

---

## Phase 7: Runtime and Integrations

**What**: ndn-runtime and ndn-integrations repositories — the components needed to use NDN in a real agent.

**Why last**: The runtime is the least mature component. The router is basic, fusion is simple concatenation, retrieval is recency-only. Releasing it prematurely invites criticism of the integration quality rather than the core research.

**Prerequisites**: All prior phases. Runtime code cleaned and documented. Integration examples tested.

**Risk level**: Highest. A bad runtime experience can color perception of the entire project. Only release when the runtime is solid enough to demo.

---

## Timeline Estimate

No dates. Release when ready, not when pressured.

| Phase | Estimated Effort | Dependency |
|---|---|---|
| Phase 1 | 1-2 days (review/cleanup) | None |
| Phase 2 | 1 day | Phase 1 |
| Phase 3 | 2-3 days | Phase 2 |
| Phase 4 | 3-5 days (code cleanup, tests) | Phase 3 |
| Phase 5 | 2-3 days (hosting, download scripts) | Phase 4 |
| Phase 6 | 2-3 days | Phase 4-5 |
| Phase 7 | 5+ days (runtime cleanup) | Phase 6 |

Total: approximately 2-3 weeks of focused work, not counting review cycles.
