# Subdomain Proposal: [PROPOSED NAME]

## Metadata

| Field | Value |
|---|---|
| **Proposed subdomain** | `[name]` |
| **Parent domain** | `[e.g., OSA]` |
| **Short ID** | `[e.g., AOJ]` |
| **Proposed regime** | `[S16 / S32 / S64]` |
| **Proposer** | `[name / handle]` |
| **Date** | `[YYYY-MM-DD]` |

## Problem Statement

_What text type needs its own subdomain? Why can't existing nodes handle it?_

[Write 2-3 paragraphs explaining the target text type, where it occurs in practice, and why it matters for agent memory.]

## Why Existing Nodes Fail

_Required: at least 2 existing nodes tested as proxies, with quantified results._

### Proxy 1: [NODE ID]

| Metric | Value |
|---|---|
| Fact recovery | |
| Failure type | `[domain-prior projection / generic degradation / other]` |
| Description | |

### Proxy 2: [NODE ID]

| Metric | Value |
|---|---|
| Fact recovery | |
| Failure type | `[domain-prior projection / generic degradation / other]` |
| Description | |

### Summary of proxy failures

_Is the failure domain-prior projection (wrong style) or generic quality degradation? If it is generic degradation, a subdomain may not be the right solution — better training data for the parent might suffice._

## Proposed Subdomain

### Artifact types

_What specific kinds of text does this subdomain target?_

1. [type 1]
2. [type 2]
3. [type 3]

### Training data plan

_What data will be used for training? Synthetic, real, or mixed? How much?_

### Expected success criteria

_What results would justify this subdomain? Be specific._

- Fact recovery: [target] (vs [proxy best])
- Compression: [target]
- Held-out eval: [specific metrics]

## Why This Is Not Taxonomy Spam

_Required: explicit argument for why this is a real subdomain and not unnecessary fragmentation._

Address each of these:

1. **The target text type has a structurally distinct format** from the parent domain because: [explanation]
2. **The parent domain's learned prior** projects incorrectly onto this text type because: [explanation]
3. **A dedicated node trained on this text type** would learn different representational structure because: [explanation]
4. **This is not just a formatting variant** of an existing domain because: [explanation]

## Evaluation Plan

_How will the new subdomain be evaluated?_

1. Internal evaluation: [metrics and methodology]
2. Held-out evaluation: [data source and split]
3. Domain-native benchmarks: [what probes will be created]
4. Real-world evaluation (if applicable): [A/B test plan or other]

## Risks

_What could go wrong? What are the most likely failure modes?_

1. [risk 1]
2. [risk 2]

## References

_Link to relevant prior work, proxy failure evidence, or related experiments._
