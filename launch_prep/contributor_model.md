# Contributor Model

## Principles

1. Contributions are welcome but must meet evidence standards.
2. Quality over quantity. One well-validated node beats ten untested ones.
3. Negative results are valued. A failed experiment that is documented is a contribution.
4. The taxonomy grows by evidence, not by consensus or enthusiasm.

---

## Contribution Types

### New Node Proposal

**What**: Training a new domain or subdomain node.

**Required evidence**:
1. Proxy failure documentation — test at least 2 existing nodes on the target text type, show they fail via domain-prior projection
2. Training on target data — show that a dedicated node measurably outperforms proxies
3. Held-out evaluation — results on data the model did not see during training
4. Node card — standard documentation including strengths, weaknesses, and failure modes

**Template**: `templates/subdomain_proposal_template.md`

**Review process**: Proposal is reviewed for evidence quality, taxonomy fit, and benchmark rigor. The reviewer checks that proxy failures are real (not staged) and that improvements survive held-out evaluation.

---

### New Subdomain Proposal

**What**: Proposing a new specialization within an existing domain.

**Additional requirements beyond new node**:
1. Clear explanation of why the parent domain node fails on this text type
2. The failure must be domain-prior projection, not generic quality degradation
3. Quantified improvement vs parent domain proxy

**Template**: `templates/subdomain_proposal_template.md`

**Why the bar is higher**: Subdomains add complexity to the taxonomy. Each one must be justified by evidence, not by intuition.

---

### Benchmark Report

**What**: Evaluating an existing node on a new benchmark or dataset.

**Required content**:
1. Benchmark description and methodology
2. Results with comparison to baseline (uncompressed text)
3. Failure analysis for categories with significant degradation
4. Honest assessment of what the benchmark does and does not show

**Template**: `templates/benchmark_report_template.md`

---

### Node Card Update

**What**: Adding benchmarks, failure analyses, or updated documentation to an existing node.

**Required content**:
1. New evaluation results with methodology
2. Comparison to existing documented results
3. Updated strengths / weaknesses if applicable

---

### Failure Analysis

**What**: Documenting a failure mode, regression, or unexpected behavior.

**Required content**:
1. Description of what was expected vs what happened
2. Quantified results
3. Root cause analysis (or honest "unknown")
4. Implications for future work

**Template**: `templates/failure_analysis_template.md`

---

### Champion Promotion Request

**What**: Requesting that a new checkpoint replace the current champion for a domain.

**Required evidence**:
1. All standard node card documentation
2. Direct comparison with current champion on same evaluation
3. No unexplained regressions on any standard metric
4. Real-world A/B testing (strongly preferred)
5. Explanation of what changed and why it helped

Use the A/B test report template (`templates/ab_test_report_template.md`) for documentation.

---

## What Will Be Rejected

- **Node proposals without proxy failure evidence.** "I think X is a different domain" is not sufficient.
- **Nodes evaluated only on synthetic data.** Synthetic eval alone is insufficient for champion promotion.
- **Claims based solely on training metrics.** val_loss and exact_match are not evidence of real-world usefulness.
- **Taxonomy branches "for completeness."** Every branch must be motivated by evidence.
- **Benchmark reports without failure analysis.** Reporting only the good numbers is not acceptable.
- **Champion promotions without comparison to the incumbent.** Must demonstrate improvement, not just good performance.

---

## Contribution Workflow

1. **Open an issue** describing the intended contribution and the evidence you plan to provide
2. **Get preliminary feedback** on whether the proposal fits the taxonomy and evidence standards
3. **Do the work** — train, evaluate, document
4. **Submit a pull request** with all required documentation
5. **Review** — the contribution is reviewed for evidence quality, not just code quality
6. **Iterate or merge** based on review feedback

---

## Recognition

Contributions that are merged are attributed in:
- The node card (contributor credit)
- The experiment journal (entry for the contribution)
- The changelog

Negative-result contributions (documented failures) receive the same attribution as positive results.
