# Public Release Checklist

**Purpose**: Gate checklist before any public release of NDN materials. Every item must be explicitly verified.

---

## Safe to Publish

- [x] `ndn_blueprint/core_docs/` — all 11 files
- [x] `ndn_blueprint/registry/` — YAML files and node cards
- [x] `ndn_blueprint/evidence/` — all 5 case studies
- [x] `ndn_blueprint/launch_prep/` — all 5 planning docs
- [x] `ndn_blueprint/templates/` — all 5 templates
- [x] `ndn_blueprint/diagrams/` — all 6 diagram specs
- [x] `ndn_blueprint/README.md`
- [x] `ndn_blueprint/LICENSE` (Apache 2.0)
- [x] `ndn_blueprint/CHANGELOG.md`
- [x] `EXPERIMENT_JOURNAL.md` — contains no secrets (verified)
- [x] `NDN_NODE_REGISTRY_v0.md` — contains no secrets (verified)
- [x] `cndx/` training code — architecture, model, data generation, training loop

## Must NOT Leak

- [ ] API keys (none found in current files — re-verify before release)
- [ ] SSH private keys (`~/.ssh/blackbox_forge` — never committed)
- [ ] Server IP addresses (31.22.104.217 etc. — must be scrubbed from any published logs)
- [ ] Personal file paths (`c:\Users\fabio\...` — must not appear in published code)
- [ ] HuggingFace tokens (if any were used)
- [ ] Verda Cloud account credentials
- [ ] `ab_data/` real agent session logs (contain real bug bounty target data)
- [ ] `openclaw_memory/` integration code (not ready for public, may contain private agent logic)

## Still Weak (Fix Before Publishing)

- [ ] CONV v2 entity-rich augmentation code was reconstructed, not original — metrics close but not bit-identical
- [ ] AOJ v4 has data leakage caveat (test journals were in 5% OpenClaw real training data, same caveat as v2) — must be prominently disclosed
- [ ] HWM and HPRT have no real-world A/B or public benchmark evidence — only internal metrics
- [ ] No live production deployment exists — must not imply production readiness
- [ ] Router is rule-based with no learned calibration — must not overclaim routing quality
- [ ] AOJ v4 achieves 99% fact recovery (vs markdown 100%) on single-session A/B — near-parity, but TDR scale retrieval is slightly below v2. Must frame accurately: compression is solved, retrieval is the new bottleneck

## Allowed Claims

- Domain-specific latent compression works across 6 domains
- Wrong-node failures are diagnostic and reproducible
- AOJ is a validated subdomain (proxy failure → dedicated training → progressive improvement)
- CONV v2 achieves 94.9% F1 retention on LongMemEval (500 questions)
- AOJ v4 achieves 99% fact recovery on real OpenClaw A/B (vs markdown 100%), trained on 50% real GitHub + 45% synthetic + 5% OpenClaw real
- NDN achieves 1.78x compression on agent journals (v4; v2 was 1.69x)
- Repeated-work signals reduced by 97.8% in OpenClaw test
- Each node is ~70M parameters, total NDN is ~490M for 7 champions
- Architecture scales cheaply (active params = 1-2 nodes at inference)

## Forbidden Claims

- Do NOT claim NDN beats markdown on real-world tasks (v4 is at 99% vs 100% — near-parity, not superiority; TDR scale retrieval is slightly below v2)
- Do NOT claim "near-lossless" without qualification
- Do NOT claim AGI, general intelligence, or universal memory
- Do NOT claim production-ready
- Do NOT claim the taxonomy is complete or final
- Do NOT claim training metrics prove real-world usefulness
- Do NOT imply AOJ v3's regression didn't happen
- Do NOT claim bit-identical reproducibility for retrained checkpoints

## Pre-Publish Verification Steps

1. [ ] `rg -i "api.key\|secret\|password\|token\|bearer\|credential" ndn_blueprint/`
2. [ ] `rg "fabio\|blackbox_forge\|31\.22\.\|86\.38\.\|C:\\\\Users" ndn_blueprint/`
3. [ ] `rg "hetzner\|verda" ndn_blueprint/` — verify these are generic references, not credentials
4. [ ] Review all node cards for absolute paths
5. [ ] Review CHANGELOG for server IPs
6. [ ] Choose final repo name and org
7. [ ] Verify LICENSE is correct
8. [ ] Write a public README (different from blueprint README — shorter, focused)
9. [ ] Decide which checkpoints to host (HuggingFace Hub? GitHub Releases?)
10. [ ] Final tone proofread by a second person
