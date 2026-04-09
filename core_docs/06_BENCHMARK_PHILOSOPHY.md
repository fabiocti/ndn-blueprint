# Benchmark Philosophy

## Core Principle

A node is not validated until it survives evaluation outside its training distribution. Training curves are necessary for monitoring convergence, but they are never sufficient evidence for claims about real-world usefulness.

This is not theoretical caution. The project has a documented case (AOJ v3) where internal metrics improved across the board while real-world A/B performance regressed.

---

## Evaluation Hierarchy

### Level 1: Internal Training Metrics

**What it is**: Standard metrics computed during training — train_loss, val_loss, first_token_accuracy, exact_match, ablation_gap, shuffled_gap, corruption sensitivity.

**What it tells you**: Whether the model is converging, whether the latent space is encoding meaningful structure, whether domain-specific priors are forming.

**What it does NOT tell you**: Whether the model will perform well on real data, whether it will recover specific facts, whether it will beat simpler baselines.

**Required for**: All nodes at all stages. These are table stakes.

### Level 2: Held-Out Evaluation

**What it is**: Evaluation on data the model did not see during training. For synthetic corpora, this means eval samples generated with a different seed. For real data, this means a clean train/test split.

**What it tells you**: Whether the model generalizes within its data distribution.

**Caveat**: Held-out evaluation on synthetic data can still be misleading if the synthetic distribution does not match real-world data. AOJ v2's 100% exact match on synthetic eval did not translate to 100% fact recovery on real journals.

**Required for**: All nodes at "Benchmarked" stage or above.

### Level 3: Domain-Native Benchmarks

**What it is**: Structured probe suites designed to test specific capabilities within a domain.

Current benchmark families:

| Family | Domain | What It Tests |
|---|---|---|
| NIAH (Needle-in-a-Haystack) | All | Can the model preserve a specific fact embedded in a longer passage? |
| FACT1 / FACT2 | All | Single-hop and two-hop fact binding (entity → attribute) |
| EXACT | All | Exact value recovery (numbers, names, codes) |
| TRACK | OSA, HPRT | State tracking through sequential updates / overrides |
| OVERRIDE | OSA | Does the model respect later-update-wins semantics? |
| POS | NLK, HWM | Positional uniformity — are facts at different positions preserved equally? |
| CONTROL | HPRT | Baseline control for regulated text accuracy |

**What it tells you**: Whether the model handles specific domain-critical tasks.

**Required for**: "Benchmarked" stage and above.

### Level 4: External / Public Benchmarks

**What it is**: Evaluation on established public benchmarks that the project did not create.

Current results:
- **LongMemEval** (500 questions): CONV-S64 v2 achieves 94.9% F1 retention, 93.6% keyword recall retention, 85.1% containment retention vs uncompressed baseline

**What it tells you**: How the model performs on a task designed by someone else, with evaluation criteria the project did not choose.

**Why it matters**: Public benchmarks are harder to game than internal probes. They test capabilities the project may not have explicitly optimized for.

**Required for**: Not strictly required for any stage, but strongly preferred for champion nodes.

### Level 5: Real-World A/B Testing

**What it is**: Head-to-head comparison of NDN memory vs raw markdown memory on real agent sessions.

Current test: OpenClaw A/B — 5 real bug-bounty session slices, comparing markdown context vs NDN-compressed context on fact recovery, continuity, noise, and repeated-work signals.

**What it tells you**: Whether the compressed memory actually helps in practice, with all the messiness of real data.

**Why it is treated as decisive for practical claims**: Real A/B testing is the only evaluation that captures:
- Routing accuracy on real text
- Reconstruction quality on out-of-distribution entities
- Retrieval relevance on real queries
- Fusion coherence with real agent context

**Required for**: Champion status (strongly preferred). Any claim about practical usefulness.

---

## Benchmark Discipline Rules

### 1. Low loss is not enough

A node with val_loss approaching zero on synthetic data may still fail on real data. The synthetic distribution is a simplification of reality. Always test on held-out data and ideally real data.

### 2. Pretty internal metrics can lie

AOJ v3 had lower val_loss (0.0001 vs 0.0016), earlier exact match (epoch 6 vs epoch 9), and higher shuffled_gap (4.37 vs 4.31) than v2. It was 7 percentage points worse on the real A/B test.

This is a documented phenomenon, not a theoretical risk.

### 3. When internal metrics and real-world A/B disagree, trust the A/B for practical claims

When internal metrics and real-world performance disagree, real-world performance takes precedence for claims about usefulness. Always reconcile why they diverged.

### 4. Negative results matter

Failed experiments are documented with the same rigor as successes:
- AOJ v3 (loss weighting regression) is in the experiment journal
- Wrong-proxy failures (HWM on journals, OSA on journals) are documented
- The entity weight regex bug (0 tokens matched) is documented

Negative results prevent future repetition of the same mistakes and guide the next intervention.

### 5. A node is not good until it survives

A node is not "good" based on training curves alone. It is good when:
- It survives held-out evaluation
- It survives domain-native benchmark probes
- Ideally, it survives real-world A/B testing
- Its failure modes are characterized and documented

### 6. Markdown is the honest baseline

For any claim about practical usefulness, the comparison to raw markdown is mandatory. Markdown is lossless, simple, and always available. If NDN cannot match or exceed markdown on a specific task, that must be stated clearly.

Current honest comparison on the most demanding test (real OpenClaw A/B):
- Markdown: 100% fact recovery
- NDN (AOJ v2): 73% fact recovery (test data was in training corpus), 1.69x compression

NDN compresses and reduces noise, but fact recovery is not yet competitive.

### 7. Failure taxonomies are required

When a node fails to reach a target, the failure must be taxonomized:
- What specific facts were missed?
- What types of facts were missed? (numeric, entity, structural, etc.)
- How many times did each type of miss occur?
- Is the failure concentrated or distributed?

This enables targeted intervention rather than blind hyperparameter tuning.

---

## What Counts As "Enough" Evidence

### For a new subdomain
- Documented proxy failure on the target text type (at least 2 existing nodes tested)
- Dedicated training showing measurable improvement over all proxies
- Held-out evaluation confirming improvement
- Ideally, real-world A/B showing improvement

### For a champion promotion
- Best performance among all tested checkpoints for this domain
- Held-out evaluation
- Domain-native benchmark results
- Real-world A/B preferred
- No unexplained regressions vs prior champion

### For a public claim
- All of the above
- Clean provenance documentation (data composition, held-out policy, overlap)
- Honest comparison to markdown baseline
- Explicit statement of what remains unresolved
