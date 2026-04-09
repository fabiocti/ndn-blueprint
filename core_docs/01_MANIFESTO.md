# NDN Manifesto

## Memory Is Plural

There is no single "memory" that an AI agent needs. There are many kinds of memory, and they differ in structure, in what fidelity means, in what compression costs, and in what failure looks like.

Encyclopedic knowledge and messy handwritten notes are not the same kind of information. Source code and multi-turn conversation are not the same kind of information. A recon journal tracking subdomains and a regulated policy document tracking compliance thresholds are not the same kind of information.

Treating them as the same — stuffing them all into one latent space, one compression model, one retrieval strategy — produces a system that is mediocre at everything and excellent at nothing.

The Neural Domain Network takes the opposite position: **memory should be domain-native**. Each kind of information gets its own compression node, trained on data that matches its structure, evaluated on benchmarks that test its specific fidelity requirements.

## Different Shapes Need Different Regimes

Not all information compresses equally. Structured operational traces support 4x compression with strong metrics in testing. Organic conversation requires 2x and still loses some details. High-precision regulatory text with exact dates and thresholds needs careful handling.

A single compression ratio, a single latent dimension, a single training corpus cannot serve all of these. The NDN makes the regime explicit: S16 for aggressive compression of highly structured data, S32 for structured content (empirically the usual default regime in most domains), S64 for complex or irregular content where every token carries meaning.

The regime is not arbitrary — it is determined by empirical testing on each domain.

## Wrong-Node Failures Are Informative

When a model trained on domain A is used to reconstruct domain B text, the result is not random noise. It is a coherent reconstruction in the style of domain A. An HWM-trained model (messy notes) reconstructs recon journals as messy notes. A wiki-trained model reconstructs code as prose paragraphs.

This phenomenon — **domain-prior projection** — is diagnostically informative. It indicates that the latent space encodes domain-specific representational structure. It also signals when the taxonomy needs a new branch: when existing nodes systematically fail on a text type, and the failure is domain-prior mismatch rather than generic quality degradation.

## The Tree Grows by Evidence, Not by Intuition

Adding a new domain or subdomain to the NDN is not free. Each new node requires:

- Training data curation
- Model training
- Internal evaluation (val_loss, ablation_gap, shuffled_gap)
- Domain-native benchmark creation
- Held-out evaluation
- Ideally, real-world A/B testing

A new branch in the taxonomy is justified only when:

1. Existing proxy nodes demonstrably fail on the target text type
2. The failure is attributable to domain-prior mismatch, not general quality
3. Dedicated training on the target text type measurably improves results
4. The improvement survives held-out evaluation, not just training metrics

Proposing a new domain because it "feels different" is not sufficient. The bar is empirical evidence of proxy failure and dedicated improvement.

## Benchmark Discipline Is Mandatory

Every node in the NDN carries a benchmark profile. This profile is not optional.

The project has learned the hard way that:

- Low training loss does not guarantee real-world usefulness
- High exact match on synthetic eval does not guarantee fact recovery on real data
- Internal metrics can improve while real A/B performance regresses (AOJ v3 is the documented example)

Therefore:

- No node is promoted to "champion" without surviving held-out evaluation
- Real-world A/B testing against markdown is the required check for practical claims
- Negative results (regressions, failed experiments) are documented with the same rigor as positive results
- Training curves alone are never sufficient evidence for any claim

## Open Contribution Is Welcome, but Not Taxonomy Spam

The NDN is designed to grow. New domains, subdomains, regimes, and benchmarks can all be contributed. But growth must be disciplined.

Contributions that will be valued:

- New domain proposals with proxy-failure evidence
- Benchmark reports on existing nodes (especially on new text types)
- Failure analyses that reveal taxonomy gaps
- Node cards with honest quality profiles

Contributions that will be rejected:

- Domain proposals without evidence of proxy failure
- Nodes trained and evaluated only on synthetic data
- Claims based solely on training metrics
- Taxonomy branches created "for completeness" without empirical motivation

## The Long-Term Goal

The NDN aims to become a **benchmarked memory lattice** — a growing network of domain-specialized compression nodes, each with a documented quality profile, connected by a calibrated router, and governed by benchmark discipline.

This is a research project with working code, not a finished product. The honest current state is:

- The architecture works
- Domain specialization is validated
- Several domains have strong internal results
- One public benchmark result exists
- Raw markdown still wins on the most demanding real-world test
- The remaining gap is characterized but not closed

The goal is to close that gap through disciplined iteration, not through hype.
