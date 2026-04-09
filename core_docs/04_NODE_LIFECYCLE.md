# Node Lifecycle

## Maturity Stages

Every node in the NDN progresses through a defined lifecycle. The stage determines what claims can be made about the node and how it can be used.

### 1. Idea

A domain or subdomain has been proposed but no work has been done. The proposal exists in discussion or documentation only.

**Requirements to enter**: A written proposal identifying the target text type and a hypothesis about why existing nodes would fail on it.

**What you can claim**: Nothing. This is a hypothesis.

### 2. Candidate

Training data has been identified or generated. The proposal has been reviewed for taxonomy fit (it does not obviously overlap with an existing domain). Initial proxy-failure testing may have been done.

**Requirements to enter**:
- Identified training corpus (real or synthetic)
- Documented rationale for why existing nodes are expected to fail
- At least one proxy-failure measurement (test an existing node on the target text type)

**What you can claim**: "Existing nodes appear to fail on this text type. A dedicated node may help."

### 3. Experimental

A model has been trained and internal evaluation is complete. Training curves, val_loss, ablation_gap, shuffled_gap, and corruption sensitivity are documented. The node has not yet been tested on held-out or real-world data.

**Requirements to enter**:
- Trained checkpoint
- Full internal evaluation metrics (all standard metrics)
- Training log and configuration preserved

**What you can claim**: "A model has been trained. Internal metrics look [healthy/concerning/strong]."

**What you cannot claim**: That the node works in practice. Internal metrics are necessary but not sufficient.

### 4. Benchmarked

The node has been tested on held-out evaluation data and/or a domain-native benchmark suite. Results are documented with honest assessment.

**Requirements to enter**:
- Held-out evaluation (data the model did not see during training)
- At least one domain-native benchmark probe (NIAH, FACT, EXACT, TRACK, etc.)
- Documented strengths and failure modes

**What you can claim**: "The node performs [X] on held-out evaluation with these specific strengths and weaknesses."

### 5. Champion

The node is the best-performing checkpoint for its domain/subdomain. It has survived the most rigorous available evaluation, which ideally includes real-world A/B testing.

**Requirements to enter**:
- All "Benchmarked" requirements
- Real-world A/B testing against markdown baseline (strongly preferred)
- Documented comparison with any prior versions or proxy nodes
- No known regressions from prior champion

**What you can claim**: "This is the current best checkpoint for [domain]. Here is the evidence."

**Important**: "Champion" does not mean "beats markdown." It means "best available NDN checkpoint for this domain." The honest comparison to markdown must always be documented.

### 6. Deprecated

A node that was once champion or benchmarked but has been superseded by a better version, or whose training data / evaluation methodology has been found to be flawed.

**Requirements to enter**:
- A newer champion exists for the same domain, OR
- A flaw in training/evaluation has been identified

**What happens**: The checkpoint is preserved in the archive. Benchmark results remain in the record. The node is removed from active use but not deleted.

### 7. Archived

A node that is no longer maintained or actively referenced. Historical record only.

---

## Promotion Rules

### Experimental → Benchmarked

Requires:
- Held-out evaluation with clear results
- At least one domain-native benchmark probe
- Honest documentation of failure modes

Does NOT require:
- Real-world A/B testing
- Beating markdown
- Perfect metrics

### Benchmarked → Champion

Requires:
- All benchmarked requirements
- Real-world A/B testing (strongly preferred) or equivalent held-out testing on real data
- Documented comparison with prior champion (if one exists)
- No unexplained regressions on key metrics

Does NOT require:
- Beating markdown overall (the champion is the best NDN checkpoint, not necessarily better than all alternatives)

### Champion → Deprecated

Triggered by:
- A new champion is validated for the same domain
- A methodological flaw is discovered
- The domain is restructured (merged, split, or redefined)

---

## What Makes a Node a "Proxy Only"

A node used as a proxy for a domain it was not trained on. Proxy usage is diagnostic:

- If the proxy performs well, the target text may not need its own domain
- If the proxy fails with domain-prior projection, a dedicated node is justified

Example: HWM-S64 used as proxy for agent operational journals recovered 24% of facts and reconstructed everything as messy notes. This proxy failure justified the AOJ subdomain.

Proxies are never promoted to champion for a domain they were not trained on. They can only be documented as "tested and found insufficient."

---

## What Justifies a New Subdomain

A subdomain is justified when all of the following are true:

1. **Proxy failure is documented** — existing nodes (including the parent domain node) demonstrably fail on the target text type
2. **The failure is domain-prior mismatch** — the node reconstructs in its own style, not in the target style
3. **Dedicated training measurably improves results** — a model trained on the target text type outperforms all proxies
4. **The improvement survives held-out evaluation** — not just training metrics

If the failure is generic quality degradation (low scores across the board, not style mismatch), the answer may be better training data for the parent node, not a new subdomain.

---

## When a Node Gets Retired

A node is retired (deprecated → archived) when:

1. It has been superseded and no one is actively using or referencing it
2. Its domain has been absorbed into a broader domain
3. The project moves past the experimental phase where the node was relevant

Retired nodes are never deleted. Their training configuration, evaluation results, and failure analyses are preserved as historical record.
