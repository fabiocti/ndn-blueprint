# Failure Analysis: [SHORT TITLE]

## Metadata

| Field | Value |
|---|---|
| **Node / experiment** | `[node_id or experiment name]` |
| **Type** | `[regression / wrong-node failure / training failure / evaluation surprise / other]` |
| **Severity** | `[critical / significant / minor]` |
| **Date** | `[YYYY-MM-DD]` |
| **Analyst** | `[name / handle]` |

## What Was Expected

_What result was the experiment supposed to produce?_

## What Actually Happened

_What result was observed? Be specific with numbers._

## Quantified Impact

| Metric | Expected | Actual | Delta |
|---|---|---|---|
| [metric 1] | | | |
| [metric 2] | | | |
| [metric 3] | | | |

## Root Cause Analysis

_What caused the failure? If unknown, say so._

### Confirmed causes
- [cause 1, with evidence]

### Suspected causes
- [cause 1, with reasoning but not proof]

### Ruled out
- [thing that was initially suspected but is not the cause, with evidence]

## Internal Metrics vs Real-World (if applicable)

_Did internal metrics predict this failure? If not, why?_

| Metric | Internal | Real-world | Diverged? |
|---|---|---|---|
| [metric 1] | | | |
| [metric 2] | | | |

## What This Failure Teaches

_What general lesson does this failure provide for the project?_

1. [lesson 1]
2. [lesson 2]

## Implications for Future Work

_How should this failure influence future experiments or design decisions?_

- [implication 1]
- [implication 2]

## What Was Tried to Fix It (if applicable)

| Intervention | Result |
|---|---|
| [intervention 1] | [outcome] |
| [intervention 2] | [outcome] |

## Current Status

- [ ] Understood and documented
- [ ] Root cause identified
- [ ] Fix implemented
- [ ] Fix validated
- [ ] Remains open / unresolved

## Artifacts

| Item | Location |
|---|---|
| Experiment log | `[path]` |
| Training script | `[path]` |
| Evaluation results | `[path]` |
| Related journal entry | `[EXPERIMENT_JOURNAL.md section]` |
