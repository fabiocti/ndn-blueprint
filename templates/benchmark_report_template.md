# Benchmark Report: [BENCHMARK NAME] on [NODE ID]

## Metadata

| Field | Value |
|---|---|
| **Benchmark** | `[name]` |
| **Node tested** | `[node_id, version]` |
| **Domain** | `[domain]` |
| **Date** | `[YYYY-MM-DD]` |
| **Evaluator** | `[name / handle]` |

## Benchmark Description

_What does this benchmark test? Who created it? How many questions/probes?_

## Protocol

_Exact evaluation methodology. Someone should be able to reproduce this._

1. [step 1]
2. [step 2]
3. [step 3]

### Configuration

| Parameter | Value |
|---|---|
| Chunk size | `[e.g., 128]` |
| Stride | `[e.g., 96 or N/A]` |
| Reader model | `[e.g., Qwen2.5-7B-Instruct or N/A]` |
| Baseline | `[e.g., uncompressed text]` |
| Number of samples | |

## Results

### Overall

| Metric | Baseline | NDN | Retention | Delta |
|---|---|---|---|---|
| [metric 1] | | | | |
| [metric 2] | | | | |
| [metric 3] | | | | |

### Per-Category (if applicable)

| Category | n | Baseline | NDN | Retention |
|---|---|---|---|---|
| [cat 1] | | | | |
| [cat 2] | | | | |

## Failure Analysis

_Required. What went wrong? Analyze the worst-performing categories._

### Top failure modes

| Failure Type | Count | % of Total | Affected Categories |
|---|---|---|---|
| [type 1] | | | |
| [type 2] | | | |
| [type 3] | | | |

### Example failures

_Show 3-5 specific examples of failures with input, expected output, and actual output._

**Example 1**:
- Input: [brief description]
- Expected: [what the correct answer was]
- NDN produced: [what the reconstruction led to]
- Failure type: [classification]

**Example 2**:
...

## Comparison to Prior Results (if applicable)

| Version | Metric | Value | Notes |
|---|---|---|---|
| [prior] | | | |
| **[current]** | | | |

## What This Benchmark Does Show

_Be specific about what claims the results support._

## What This Benchmark Does NOT Show

_Be specific about what claims the results do NOT support._

## Honest Assessment

_One paragraph summarizing the result, including both strengths and weaknesses._

## Reproducibility

| Item | Location |
|---|---|
| Evaluation script | `[path]` |
| Benchmark data | `[source / path]` |
| Checkpoint used | `[path / download]` |
| Full results log | `[path]` |
