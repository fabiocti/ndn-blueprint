# Governance Notes

## Taxonomy Governance

### Who can propose a new domain or subdomain
Anyone. Proposals are evaluated on evidence quality, not contributor status.

### Who can approve a new domain or subdomain
Currently: the project maintainer(s). As the project grows, this should transition to a small technical review committee (3-5 people) with domain expertise.

### How decisions are made
Decisions are evidence-based, not consensus-based. A well-evidenced proposal from a single contributor outweighs vague agreement from many.

The review criteria are fixed and documented:
1. Is there documented proxy failure on the target text type?
2. Is the failure attributable to domain-prior mismatch?
3. Does dedicated training measurably improve results?
4. Does the improvement survive held-out evaluation?

If all four criteria are met with clear evidence, the proposal should be approved regardless of reviewer opinion about whether the domain "feels" necessary.

### How conflicts are resolved
If reviewers disagree on the evidence quality:
1. Request additional evaluation (more benchmarks, different held-out data)
2. If still disputed, run a real-world A/B test (the gold standard)
3. If A/B testing is not feasible, the more conservative position wins (do not add the branch)

---

## Node Promotion Governance

### Who can promote a node to champion
The project maintainer(s), after review of the evidence. Champion promotion is a higher bar than initial submission:
- Must demonstrate improvement over current champion
- Must include real-world evaluation (preferred) or extensive held-out testing
- Must document any regressions or trade-offs

### How failed nodes are recorded
Failed experiments are documented in the experiment journal and in the node registry (marked as "experimental" or "deprecated" with a link to the failure analysis). Failed nodes are never deleted — they are preserved as negative results.

---

## Avoiding Taxonomy Explosion

The biggest governance risk is taxonomy spam: well-intentioned contributors proposing many fine-grained domains and subdomains without sufficient evidence.

### Guardrails

1. **Proxy failure is mandatory.** No new branch without documented evidence that existing nodes fail on the target text type.

2. **The burden of proof is on the proposer.** It is not the reviewer's job to prove the branch is unnecessary. It is the proposer's job to prove it is necessary.

3. **"Different formatting" is not a domain.** Two text types that differ only in formatting (e.g., JSON vs YAML for the same semantic content) should not be separate domains. The question is whether the latent representation needs to be different, not whether the surface form is different.

4. **Subdomains before new top-level domains.** If a proposed text type is semantically close to an existing domain, it should be proposed as a subdomain first. New top-level domains require stronger evidence of fundamental structural difference.

5. **Periodic review.** The taxonomy should be reviewed periodically (e.g., every 6 months) to identify branches that are underperforming, redundant, or unjustified.

6. **Deprecation is not failure.** Deprecating a node that is no longer useful is a healthy sign of a maturing taxonomy, not a sign of bad judgment.

---

## Benchmark Discipline as Guardrail

A strong guardrail against taxonomy spam and low-quality contributions is benchmark discipline:

- Every node must carry a benchmark profile
- Claims require held-out evaluation at minimum
- Champion claims require real-world A/B or equivalent
- Negative results are documented and valued

If benchmark discipline is maintained, the taxonomy grows only where evidence supports it.

---

## Future Governance Model

As the project scales beyond a single maintainer:

### Phase 1 (current): Maintainer-led
All decisions by the project founder/maintainer. Works for small projects.

### Phase 2: Review committee
3-5 technical reviewers with domain expertise. Proposals reviewed by at least 2 members. Majority approval required for new domains/subdomains. Maintainer retains veto.

### Phase 3: Community governance
Formal contribution guidelines, RFC process for taxonomy changes, elected or appointed review committee, transparent decision records.

The transition should happen based on project activity level, not on a fixed timeline. Premature governance bureaucracy kills small projects.
