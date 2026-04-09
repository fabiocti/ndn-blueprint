# Case Study: OpenClaw Real A/B Test

## Summary

The first real-world A/B comparison of NDN memory vs raw markdown memory on an actual AI agent's operational data. Markdown won overall on fact recovery. NDN provided compression and noise reduction. The test validated the AOJ subdomain and revealed the exact remaining bottleneck.

---

## Setup

### The Agent
OpenClaw: an automated bug-bounty agent that performs reconnaissance, vulnerability scanning, and findings analysis across multiple targets. Runs on Hetzner VPS. Produces per-target operational journals in markdown format.

### The Data
5 real target journals from OpenClaw campaigns:
- `journal_vfsglobal.md`
- `journal_dailymotion.md`
- `journal_expressvpn.md`
- `journal_harman.md`
- `journal_pinelabs.md`

Plus `MEMORY_server.md` (the agent's persistent memory file) and `bounty_daemon.log` (timeline).

### The Test
5 session slices, each representing a point in the campaign:

| Slice | Description | Journals Available |
|---|---|---|
| S1 | 4 targets recon'd | VFS, Dailymotion, ExpressVPN, Harman |
| S2 | All 5 targets complete | All 5 |
| S3 | Mid-campaign | All 5 |
| S4 | Heavy history (800 turns) | All 5 |
| S5 | Full campaign (1400+ turns) | All 5 |

For each slice:
- **Side A (Markdown)**: raw MEMORY.md + accumulated journals up to cutoff
- **Side B (NDN)**: journals compressed via NDN, reconstructed into context

Measured: fact recovery, continuity, noise, repeated-work signals, context size.

---

## Why Markdown Initially Won

In the corrected A/B test (using AOJ v2 for both WORKFLOW and FINDINGS domains), markdown achieved 100% fact recovery on all slices. NDN achieved 73%.

Markdown wins because:
1. **It is lossless.** Every fact in the journal is present in the context.
2. **It is simple.** No routing, compression, reconstruction, or fusion to go wrong.
3. **Context was within budget.** The journals fit within the context window (~2,600 tokens), so compression was not strictly necessary.

---

## Where NDN Helped

Despite lower fact recovery, NDN showed advantages:

1. **Compression**: 1.69x (1,555 tokens vs 2,626 tokens). This matters more as history grows.
2. **Noise reduction**: NDN avg noise 0.74 vs markdown 0.76. Slightly cleaner context.
3. **Repeated-work elimination**: NDN's structured memory reduces redundant re-scanning signals.

These advantages become more significant as campaign history grows beyond what fits in a context window. For short campaigns (5 targets, <3K tokens), markdown's simplicity wins.

---

## The Corrected A/B: What It Showed

### AOJ v2 Results (Best NDN)

| Slice | MD Facts | NDN Facts | Compression | Continuity |
|---|---|---|---|---|
| S1 (4 targets) | 19/19 | 16/19 (84%) | 1.83x | 0.80 |
| S2 (5 targets) | 15/15 | 9/15 (60%) | 1.63x | 1.00 |
| S3 (mid) | 15/15 | 11/15 (73%) | 1.63x | 1.00 |
| S4 (heavy) | 20/20 | 15/20 (75%) | 1.63x | 0.80 |
| S5 (full) | 21/21 | 15/21 (71%) | 1.63x | 1.00 |
| **Average** | **100%** | **73%** | **1.69x** | **0.92** |

Score: Markdown 4/6, NDN 2/6.

### Progression Across All Tested Models

| Model | Role | Fact Recovery |
|---|---|---|
| HWM-S64 | wrong proxy | 24% |
| OSA-S32 | wrong proxy | 14% |
| AOJ v1 | dedicated | 54% |
| **AOJ v2** | **dedicated + entity-diverse** | **73%** |
| AOJ v3 | loss-weighted | 66% (regression) |
| Markdown | baseline | 100% |

---

## What This Establishes

1. **NDN runs end-to-end on real data in this harness.** Routing, compression, storage, retrieval, reconstruction, and fusion all function correctly.

2. **Domain specialization matters.** Wrong-proxy nodes (HWM, OSA) were catastrophically bad (14-24%). The right subdomain (AOJ) recovered far more facts (73%) but still trailed markdown on the same test.

3. **Progressive improvement is real.** Each AOJ version measurably improved on the prior (14% → 54% → 73%), until v3 over-corrected.

4. **The remaining gap is specific and characterized.** Not random degradation. Exactly: 58% exact numeric counts, 25% OOV domain names, 17% error/status strings.

---

## What This Does NOT Prove

1. **NDN does NOT currently beat markdown on fact recovery.** 73% vs 100%. The gap is real.

2. **This is one test on one agent.** Generalization to other agents, other task types, and other data shapes is unproven.

3. **Compression advantage is marginal at this scale.** With ~2,600 tokens, the 1.69x compression saves ~1,100 tokens. This matters more with much larger histories.

4. **The provenance is not fully clean.** AOJ v2's training included the same 5 journals used in testing (5% of corpus). The model saw this data and still only got 73%, which makes the misses more significant — but the result has a data leakage caveat.

---

## Key Lessons

1. **Real A/B testing is irreplaceable.** Internal metrics (exact match, val_loss) were misleading for predicting real-world performance, especially for AOJ v3.

2. **Markdown is a strong baseline.** Simple, lossless, and hard to beat when context budget allows it. NDN's advantage grows only when context pressure forces compression.

3. **On this test, the bottleneck was preserving specific rare tokens through compression/reconstruction, not integration failures.** The pipeline ran end-to-end; markdown remained ahead on fact recovery.
