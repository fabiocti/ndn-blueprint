# Risks of Open-Sourcing NDN

## 1. Hype Outrunning Evidence

**Risk**: External commenters frame NDN as "solved memory" or "AGI memory" based on selective reading of results. The project's honest caveats get lost in amplification.

**Likelihood**: High. Any project with "neural" and "memory" in the name will attract hype.

**Mitigation**:
- Lead with limitations in all documentation (`09_WHAT_NDN_IS_NOT.md`)
- Project status page (`07_PROJECT_STATUS.md`) has explicit "Proven / Promising / Unresolved / Failed" sections
- Real-world A/B results (markdown still wins) are prominently documented
- Never use "breakthrough" language in official documentation

---

## 2. Community Fragmentation

**Risk**: Contributors fork the project and build incompatible taxonomy branches, diluting the canonical NDN into competing versions.

**Likelihood**: Medium. Happens to most successful open-source projects eventually.

**Mitigation**:
- Clear governance model with defined authority for taxonomy decisions
- Canonical registry that only the maintainer(s) can modify
- Contribution standards that reject unvalidated branches
- Welcoming stance toward forks (they can experiment freely) while maintaining a strong canonical version

---

## 3. Registry Flooding

**Risk**: Contributors submit many low-quality nodes, filling the registry with untested or poorly evaluated entries. The registry becomes noisy and untrustworthy.

**Likelihood**: Medium-high if the project gets popular.

**Mitigation**:
- Strict evidence requirements for registry entries (proxy failure + dedicated improvement + held-out evaluation)
- Node lifecycle stages (experimental nodes are clearly marked, not mixed with champions)
- Periodic registry review and deprecation
- Registry PRs reviewed by maintainer(s) before merge

---

## 4. Benchmark Gaming

**Risk**: Contributors optimize nodes to score well on specific benchmarks without improving general usefulness. Goodhart's law applied to NDN metrics.

**Likelihood**: Medium. Already seen in the project itself (AOJ v3 improved internal metrics but regressed real A/B).

**Mitigation**:
- Real-world A/B as the gold standard (hard to game)
- Multiple evaluation levels (internal → held-out → domain-native → public → real A/B)
- Failure taxonomies required for all benchmark reports
- The training vs real eval case study (`evidence/training_vs_real_eval_case.md`) serves as a warning

---

## 5. Proxy Nodes Mistaken for Champions

**Risk**: Someone uses a proxy node (e.g., NLK-S32 for code compression) and assumes it works because it produces coherent output. The output looks plausible but has domain-prior corruption.

**Likelihood**: High. Proxy usage is the most natural mistake for new users.

**Mitigation**:
- Node cards clearly state "best for" and "failed for" tasks
- Documentation explains domain-prior projection in detail
- The proxy failure case study provides concrete examples
- Router documentation emphasizes the cost of misrouting

---

## 6. AGI Branding Too Early

**Risk**: External media or community members call NDN "a step toward AGI" or similar, creating expectations the project cannot meet.

**Likelihood**: Medium. Depends on how viral the project becomes.

**Mitigation**:
- `09_WHAT_NDN_IS_NOT.md` explicitly states NDN is not AGI
- No language in official docs that could be interpreted as AGI claims
- Focus framing on "domain-specialized memory compression" not "artificial memory"
- If asked directly: "NDN is infrastructure, not intelligence"

---

## 7. Contribution Noise

**Risk**: Many low-quality PRs, issues, and discussions that consume maintainer time without advancing the project.

**Likelihood**: High if popular. Standard open-source challenge.

**Mitigation**:
- Clear contribution guidelines with explicit evidence requirements
- Issue and PR templates that require structured information
- "What will be rejected" section in contributor model
- Do not promise to review every submission quickly — quality review takes time

---

## 8. Security / Misuse Concerns

**Risk**: NDN nodes could be used to compress and persist sensitive information (PII, credentials, private conversations) in a format that is harder to audit than plaintext.

**Likelihood**: Low for now (tiny user base), but increases with adoption.

**Mitigation**:
- Documentation acknowledges this concern
- Packet store is auditable (packets can be decoded and inspected)
- No encryption or obfuscation of stored packets
- Recommend that users apply the same data governance to NDN packets as to the source text

---

## 9. Competitive Risk

**Risk**: A well-resourced organization takes the architecture, trains better nodes with more data and compute, and releases a competing product without attribution.

**Likelihood**: Low-medium. The architecture is not complex enough to patent. The value is in the research methodology and evidence base.

**Mitigation**:
- Appropriate open-source license (copyleft if wanting to prevent proprietary forks, permissive if wanting maximum adoption)
- The evidence base and benchmark discipline are harder to replicate than the code alone
- Publishing the methodology and results first establishes priority

---

## 10. Premature Release Damage

**Risk**: Releasing before the project is ready creates a negative first impression that is hard to reverse. Especially if the "markdown still wins" result is the headline.

**Likelihood**: Medium. Depends on release timing and framing.

**Mitigation**:
- Release documentation first (Phase 1-3), code later
- Frame honestly: "research project with validated results, not finished product"
- Lead with what works, be upfront about what doesn't
- Time the code release to coincide with a meaningful result (not necessarily "beats markdown" — a well-documented honest result is compelling)
