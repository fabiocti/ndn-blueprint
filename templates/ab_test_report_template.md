# A/B Test Report: [TEST NAME]

## Metadata

| Field | Value |
|---|---|
| **Test name** | `[e.g., OpenClaw Real A/B v3]` |
| **Side A** | `[e.g., Raw markdown memory]` |
| **Side B** | `[e.g., NDN-compressed memory (AOJ v2)]` |
| **Data source** | `[e.g., OpenClaw bug-bounty agent sessions]` |
| **Date** | `[YYYY-MM-DD]` |
| **Evaluator** | `[name / handle]` |

## Test Design

_Explain the test setup clearly enough to reproduce._

### Side A (Control)
_What does Side A use for memory? Be specific._

### Side B (NDN)
_What does Side B use? Which nodes, which checkpoints, which routing?_

### Slices
_How is the data divided into test slices?_

| Slice | Description | Data included |
|---|---|---|
| S1 | | |
| S2 | | |
| ... | | |

### Evaluation criteria
_What is measured for each slice?_

- **Fact recovery**: [how specific facts are checked]
- **Continuity**: [how temporal/logical coherence is measured]
- **Noise**: [how irrelevant/garbled content is measured]
- **Repeated-work signals**: [how redundant actions are counted]
- **Context size**: [how token counts are measured]

### Important constraints
- [ ] Each slice uses only information available up to its cutoff (no future leakage)
- [ ] Both sides see the same source data for each slice
- [ ] Evaluation criteria are defined before running the test (not post hoc)

## Results

### Per-Slice

| Slice | Side A Tokens | Side A Facts | Side B Tokens | Side B Facts | Side B Cont | Side B Noise | Compression |
|---|---|---|---|---|---|---|---|
| S1 | | | | | | | |
| S2 | | | | | | | |
| ... | | | | | | | |

### Summary

| Metric | Side A | Side B | Winner |
|---|---|---|---|
| Avg context tokens | | | |
| Fact recovery % | | | |
| Total missed facts | | | |
| Avg continuity | | | |
| Avg noise | | | |
| Repeated-work signals | | | |

**Overall score**: Side A [X]/[N], Side B [Y]/[N]

**Verdict**: [which side wins and why]

## Missed Fact Analysis (Side B)

_Required if Side B missed facts. What specifically was lost?_

| Fact | Times Missed | Failure Type |
|---|---|---|
| [fact 1] | | |
| [fact 2] | | |

### Failure Taxonomy

| Failure Type | Count | % of Total |
|---|---|---|
| [type 1] | | |
| [type 2] | | |

## What This Test Proves

_Be specific._

1. [claim 1]
2. [claim 2]

## What This Test Does NOT Prove

_Be specific._

1. [caveat 1]
2. [caveat 2]

## Data Leakage Assessment

- [ ] Side B training data is fully independent of test data
- [ ] Side B training data has partial overlap with test data (describe below)
- [ ] Leakage assessment is unclear

_If overlap exists, describe it and explain the implications._

## Comparison to Prior A/B Tests (if applicable)

| Test Version | Side B Checkpoint | Fact Recovery | Notes |
|---|---|---|---|
| [prior test] | | | |
| **[this test]** | | | |

## Reproducibility

| Item | Location |
|---|---|
| A/B test script | `[path]` |
| Test data | `[path]` |
| Side B checkpoint | `[path / download]` |
| Full output log | `[path]` |
| Scoring methodology | `[path or inline description]` |
