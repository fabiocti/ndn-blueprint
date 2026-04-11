# Open Problems

Problems that are currently unsolved in the NDN project. Listed in approximate priority order based on impact on practical usefulness.

---

## 1. Retrieval and Fusion Quality

**The problem**: Retrieval is currently recency-based. There is no relevance scoring, no cross-domain retrieval, and no intelligent context budget allocation. Fusion is simple concatenation with domain labels.

**Why it matters**: With raw fact recovery now competitive (AOJ v4 = 99%), retrieval is the primary bottleneck. Even with perfect reconstruction, bad retrieval (selecting irrelevant packets) or bad fusion (assembling incoherent context) degrades the agent's behavior. TDR and WS evaluations confirm that retrieval limitations cap end-to-end performance.

**Current state**: The OpenClaw prototype uses recency-based retrieval with fixed budget. This works for the test harness but would not scale to real agent use with diverse memory. WS is PARKED at 14/20 ceiling due to retrieval limitations.

**Likely directions**: Relevance-based packet selection, cross-domain retrieval, intelligent context budget allocation, learned retrieval with query-dependent scoring.

---

## 2. Routing Calibration

**The problem**: The current router is rule-based with no learned boundaries, no confidence scoring, and no calibration. Misrouting causes catastrophic quality degradation via domain-prior projection.

**Why it matters**: Misrouting is a high-impact failure mode. A correctly routed text to a mediocre node produces better results than a misrouted text to a strong node.

**Current state**: Rule-based heuristics work for clearly domain-typed text but fail on ambiguous or mixed-domain text. No mechanism to express "I'm not confident about this classification."

**Likely directions**: Embedding-based classification with learned domain prototypes. Calibrated confidence scores. Multi-domain soft routing for ambiguous cases.

---

## 3. Exact Rare-Token Preservation (largely addressed by v4)

**The problem**: The model reconstructs the correct structure and format of text but substitutes out-of-vocabulary or rare entities with training priors. A journal entry about `northwind-audio.com` (anonymized) gets reconstructed with a different domain name. An exact count of `1623` gets reconstructed as a different number.

**Status update**: AOJ v4 achieves 99% fact recovery, indicating this problem is largely solved at the reconstruction level. The v2-era miss taxonomy below is historical context; the primary gap is now retrieval (#1), not reconstruction.

**Historical evidence (v2 era)**: In the AOJ v2 A/B test, the remaining 24 misses across 5 slices broke down as:
- 58% exact numeric counts
- 25% OOV domain names
- 17% error/status strings

The same facts missed repeatedly: `1623` (missed 5/5 times), `northwind-audio.com` (4/5, anonymized), `27 live hosts` (4/5).

**What was tried and failed (v2/v3 era)**:
- Digit-aware loss weighting (3.0x in v2, 5.0x in v3): improved numeric token accuracy on eval but did not fix specific count recovery on real A/B
- Token-ID entity weighting (3.0x on 122 tokens): improved internal metrics but regressed real A/B from 73% to 66%

Note: in AOJ v3, entity weighting (3.0x) and digit weighting (5.0x) were applied simultaneously; the regression cannot be attributed to either alone.
- Expanded entity vocabulary in synthetic corpus: helped for in-vocabulary entities but cannot cover all possible OOV entities

**Remaining directions**: Copy/pointer mechanisms, entity-aware attention, or retrieval-augmented reconstruction for edge cases not yet covered by v4.

---

## 4. OOV Domain Names (largely addressed by v4)

**The problem**: Domain names, URLs, and identifiers that were not in the training vocabulary are systematically replaced during reconstruction. The model has learned a distribution over entity names and draws from it when reconstructing.

**Status update**: AOJ v4's 99% fact recovery suggests OOV domain name substitution is largely resolved. Remaining failures are retrieval-dominated.

**Why it matters**: In bug-bounty / security / operational contexts, the exact target name is critical. Substituting `northwind-audio.com` (anonymized) with a different domain is not a minor error.

**Relationship to #3**: This is a specific case of the rare-token preservation problem, but it has a distinct character: domain names are compositional (SLD + TLD), can be arbitrarily novel, and appear in specific syntactic positions.

**Historical mitigation**: Expanding the SLD vocabulary in training. This helped for names similar to training vocabulary but did not solve the fundamental OOV problem. V4's improvements suggest the architectural changes were more effective.

---

## 5. Exact Numeric Counts (largely addressed by v4)

**The problem**: Specific numbers (e.g., `1623 subdomains`, `27 live hosts`, `769 hosts`) are reconstructed as plausible but incorrect numbers. The model has learned that numeric counts appear in certain positions but has not memorized specific values.

**Status update**: AOJ v4's 99% fact recovery indicates exact numeric count preservation is largely solved at the reconstruction level.

**Why it matters**: For operational journals, exact counts determine decisions. "We found 1623 subdomains" vs "we found 1500 subdomains" changes the agent's assessment of scope.

**Why loss weighting didn't help (v2/v3 era)**: Digit-aware loss weighting teaches the model to reproduce digit tokens accurately in general. But the problem is not digit-token accuracy — it is specific value recall. The model can produce digit tokens perfectly; it just produces the wrong specific digits for a given context.

---

## 6. Error and Status String Preservation (largely addressed by v4)

**The problem**: Specific error messages (`502 Bad Gateway`), status keywords (`REPAIR_NEEDED`), and protocol-level strings are sometimes dropped or substituted during reconstruction.

**Status update**: AOJ v4's 99% fact recovery indicates error and status string preservation is largely solved at the reconstruction level.

**Why it matters**: Error states and status markers are often the most operationally critical pieces of information in a journal.

**Relationship to #3**: Another case of the rare-token problem, but status strings have limited vocabulary (unlike domain names). This should be more tractable.

---

## 7. Benchmark Generalization

**The problem**: Current benchmarks are primarily domain-native probes and one public benchmark (LongMemEval for CONV). Performance on a wider range of established benchmarks has not been measured.

**Why it matters**: Internal and domain-native benchmarks are designed by the project team. They may systematically under-test certain capabilities or failure modes.

**Likely directions**: Test on additional public benchmarks as they become available. Participate in shared evaluation campaigns if any exist for memory compression.

---

## 8. Contribution Governance

**The problem**: The project has no formal governance model for taxonomy decisions. Who can propose a new domain? Who approves it? How are conflicts resolved?

**Why it matters**: Without governance, the taxonomy will either stagnate (no growth) or explode (taxonomy spam). Both outcomes are bad.

**Current state**: All taxonomy decisions are made by the project founder. This works for a small project but does not scale.

---

## 9. Scaling to Many Domains and Subdomains

**The problem**: The current NDN has 6 domains and 1 subdomain. The architecture is designed to scale, but the practical challenges of managing many nodes — training, benchmarking, versioning, deprecation, routing across many domains — are untested.

**Why it matters**: The NDN's value proposition is specialization. If specialization creates an unmanageable number of nodes, the system becomes impractical.

**Likely directions**: Hierarchical routing, shared backbones with domain-specific heads, automated benchmark pipelines, clear deprecation policies.

---

## 10. Active-Node Selection at Scale

**The problem**: With many nodes available, how does the system decide which nodes to load and activate for a given agent task? Current: load the 2-3 most relevant. Future: this becomes a resource allocation problem.

**Why it matters**: GPU memory and inference latency both scale with the number of active nodes. A system with 50 domains cannot keep all nodes resident.

**Likely directions**: Lightweight node selection layer. Predictive loading based on session type. Node clustering for similar domains.

---

## 11. Internal Metrics vs Real-World Performance Divergence

**The problem**: Internal eval metrics can improve while real-world performance regresses. This has been documented (AOJ v3) but not systematically studied.

**Why it matters**: If internal metrics do not predict real-world performance, every change requires expensive real-world A/B testing. That constraint slows iteration.

**Likely directions**: Better synthetic eval that correlates with real-world performance. Calibrated quality estimates. Understanding when and why the divergence occurs.
