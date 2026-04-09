# Case Study: Proxy Failures and Wrong-Node Diagnostics

> **Note: Target names have been anonymized.**

## Summary

Wrong-node failures — using a model trained on one domain to reconstruct text from another — are the most informative failure mode in the NDN. They reveal domain-specific representational priors and directly motivate taxonomy expansion.

This case study documents three proxy failures that were diagnostic.

---

## What Domain-Prior Projection Looks Like

When a model trained on domain A reconstructs text from domain B, the result is not random noise. It is a coherent reconstruction **in the style of domain A**. The model's learned prior dominates the reconstruction.

This is consistent with the latent space encoding domain-specific structure rather than acting as generic lossy compression.

---

## Case 1: HWM-S64 on Agent Operational Journals

### Setup
The HWM (Human Working Memory) model was trained on fragmented notes, reminders, and partial plans. It was tested as a proxy for OpenClaw agent operational journals.

### Input
```
## Phase 1A: Subdomain Collection — IN PROGRESS
- subfinder: 555 hosts
- httpx: 169 live
- Total merged: 673 hosts
- Target: corp-alpha.example.com
```

### What happened
The HWM model reconstructed this as messy note-style text. Headers became bullet fragments. Counts were lost or substituted. Target names were dropped. The journal structure was replaced by the model's learned "messy notes" prior.

### Result
- **Fact recovery**: 24%
- **Failure type**: Domain-prior projection (journal → messy notes)
- **What it tells us**: HWM and agent journals share some surface-level features (bullet points, short fragments) but differ fundamentally in structure and information type. HWM's prior is too strong and too different.

---

## Case 2: OSA-S32 on Agent Operational Journals

### Setup
The OSA (Operational State Artifacts) model was trained on timestamped key-value state traces like `[T=3] server status=active`. It was tested as a proxy for the same OpenClaw journals.

### Input
Same journal text as above.

### What happened
The OSA model projected the markdown journal into its training format. Headers and bullet lists were reconstructed as key-value pairs. Domain names were dropped. Phase markers were partially preserved but tool counts and findings were lost.

### Result
- **Fact recovery**: 14% (worse than HWM)
- **Failure type**: Domain-prior projection (journal → key-value traces)
- **What it tells us**: OSA and agent journals share the same semantic domain (operational tracking) but differ in surface format (markdown prose vs compact key-value traces). The OSA model's format prior is so strong that it overrides the actual content.

### Why this mattered
This failure justified the AOJ subdomain. OSA-S32 is semantically close to agent journals — it tracks operational state. But its format prior (key-value traces) is incompatible with the journal format (markdown prose). The subdomain is justified because the failure is format-specific domain-prior projection, not general quality degradation.

---

## Case 3: HWM-S64 on LongMemEval Conversations

### Setup
Before training a dedicated CONV model, HWM-S64 was tested on LongMemEval conversation data.

### What happened
The HWM model reconstructed conversations as task-list-style notes. Speaker turns were collapsed. Dialogue structure was replaced by fragmented note points. Some gist was preserved, but the conversational flow was destroyed.

### Result
- **Reconstruction style**: Notes, not conversation
- **What it tells us**: The HWM latent space encodes a "messy notes" representational prior that dominates reconstruction regardless of input type. This is not a bug — it is evidence that the latent space has learned domain-specific structure.

### Why this mattered
This was a central diagnostic finding: the domain prior is real, measurable, and consistent. It supports treating NDN models as learning domain-shaped representations rather than as domain-agnostic compressors.

---

## What Proxy Failures Teach

### 1. The latent space is domain-shaped
If the latent space were domain-agnostic, wrong-node reconstruction would produce garbled text. Instead, it produces coherent text in the wrong style. This means the latent encodes domain-specific structure.

### 2. Semantic proximity ≠ good proxy
OSA is semantically close to agent journals (both track operational state) but was the worst proxy (14%). HWM is semantically further (messy notes vs structured journals) but was a better proxy (24%). Format prior matters more than semantic category.

### 3. Proxy failure is the trigger for taxonomy expansion
When no existing node can handle a text type and the failure is domain-prior projection, a new domain or subdomain is justified. The AOJ subdomain was proposed and validated on exactly this basis.

### 4. Proxy failure severity indicates domain distance
The worse the proxy performance, the more different the target text type is from existing domains. This can guide priority: the worst proxy failures represent the biggest taxonomy gaps.

---

## How to Use Proxy Failures Diagnostically

When evaluating whether a new domain or subdomain is needed:

1. **Test all existing nodes as proxies.** Run each node on the target text type and measure fact recovery.
2. **Classify the failure.** Is it domain-prior projection (wrong style), general degradation (low quality), or something else?
3. **If projection**: the text type likely needs its own node. The failure mode tells you exactly what the existing nodes learned (and thus what the new node needs to be different about).
4. **If degradation**: the text type may be handleable by the closest existing node with better training data, not a new subdomain.
5. **Train a dedicated node and compare.** If the dedicated node significantly outperforms all proxies, the subdomain is justified.
