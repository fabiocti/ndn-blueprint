# Changelog

All notable changes to the NDN Blueprint package are documented here.

---

## Unreleased

### Planned
- Additional real-world A/B tests for CONV and other domains
- Router improvement (learned boundaries, confidence scoring)
- Architectural entity preservation experiments (copy/pointer mechanisms)
- Retrieval layer improvements (identified as current bottleneck)

---

## [0.2.0] — 2026-04-11

### Promoted
- AOJ v4 (`aoj_s32_v4`) promoted to FROZEN champion: 99% raw fact recovery, 1.78x compression. Effectively closes the gap with markdown (100%).
- AOJ v2 (`aoj_s32_v2`) marked as superseded.

### Evaluated
- **TDR**: 90% hits on flagship evaluation.
- **RWJ**: 95% dev facts confirmed.
- **WS**: PARKED at 14/20 — further work deferred.

### Identified
- **Retrieval is the current bottleneck.** Compression quality (v4 at 99%) is no longer the limiting factor; retrieval-layer improvements are now the priority.

### Updated
- Evidence files, case studies, and diagrams updated to reflect v4 champion status.
- Stale "markdown still wins" claims corrected across documentation.
- Node cards and registry references updated from `aoj_s32_v2` to `aoj_s32_v4`.

---

## [0.1.1] — 2026-04-09

### Recovered
- Retrained HPRT reg_s32: model.pt now present locally (316MB). Metrics match original exactly.
- Retrained AOJ v2 aoj_s32_v2: model.pt now present locally (316MB). val_loss=0.0014 vs original 0.0016.
- Retrained CONV v2 conv_s64_v2: model.pt now present locally (316MB). val_loss=0.1648 (significant divergence from original training val_loss of 1.821 — likely due to retraining conditions).
- All three checkpoints downloaded before server outage.

### Updated
- Node cards for hprt_s32, conv_s64_v2, aoj_s32_v2 updated to reflect model.pt presence
- Checkpoints README updated with current inventory

---

## [0.1.0] — 2026-04-09

### Created
- Initial blueprint package with full directory structure
- 11 core documentation files (`00_OVERVIEW.md` through `10_GLOSSARY.md`)
- 5 registry YAML files (`domains.yaml`, `subdomains.yaml`, `nodes.yaml`, `regimes.yaml`, `benchmark_families.yaml`)
- 7 champion node cards (`nlk_s32`, `fta_s64`, `osa_s32`, `hwm_s64`, `hprt_s32`, `conv_s64_v2`, `aoj_s32_v4`)
- 5 evidence/case study files (conversation memory, OpenClaw A/B, AOJ subdomain, proxy failure, training vs real eval)
- 5 launch preparation files (repo split plan, release order, contributor model, governance, risks)
- 5 templates (node card, subdomain proposal, benchmark report, failure analysis, A/B test report)
- 6 diagram specifications with Mermaid diagrams
- Root `README.md`
- Apache 2.0 license

### Verified
- Cross-referenced all evidence numbers against internal experiment records
- Fixed AOJ v2 per-slice fact counts (S1: 16/19, S2: 9/15 — previously had wrong values from v3 comparison table)
- Fixed compression token count (v2: 1,555 tokens, not 1,579 which was v3's number)
- Fixed cross-domain summary table to use actual champion training metrics instead of S64-only comparison numbers
- Fixed NLK taxonomy entry: removed misleading "0.72 at S64" val_loss (that was from a cross-domain comparison table, not the S32 champion)
- Updated NDN_NODE_REGISTRY cross-node summary with correct champion metrics for all 7 nodes
- Sanitization check: no API keys, passwords, SSH keys, or personal identifiers found

### Known Gaps
- AOJ v2 `model.pt` was recovered 9 Apr 2026 (retrain on H100; 316MB locally). Not bit-identical to original; metrics within noise.
- CONV-S64 v2 `model.pt` was recovered 9 Apr 2026 (retrain; 316MB locally).
- HPRT reg_s32 `model.pt` was recovered 9 Apr 2026 (retrain; 316MB locally). Metrics match original training run.
- HWM and HPRT node cards have internal metrics only — no real-world A/B or public benchmark evidence
- AOJ v2 A/B has data leakage caveat (test journals were 5% of training data)
