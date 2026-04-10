# Second Leaf Analysis: Candidates Under AOJ

**Date**: 10 April 2026
**Purpose**: Decide which sibling leaf to validate next under OSA/AOJ
**Status**: Historical — both candidates (RWJ, WS) have since been established as bloomings with benchmark results. See `03_TAXONOMY.md` for current status.

---

## Why a Second Leaf Matters

One flagship leaf (TDR) proves the pipeline works for one use case. A second leaf proves the tree structure is real — that AOJ genuinely contains multiple distinct patterns that share a compression node but differ in retrieval, reconstruction, and evaluation.

---

## Candidate A: Recon Workflow Journals (RWJ)

### What It Is

Single-target sequential operational journals recording an agent's recon workflow against one target. Each journal tracks the full Phase 1A → 1B lifecycle for one bug bounty program: subdomain collection → merge → nuclei scans → httpx → host classification.

### Available Data

5 real OpenClaw journals:
- `journal_vfsglobal.md` — 6 recon iterations with evolving counts
- `journal_harman.md` — multi-domain recon (5 subdomains under Harman umbrella)
- `journal_pinelabs.md` — includes repair cycle and SerpAPI quota error
- `journal_dailymotion.md` — large-scale (1345 hosts), multi-phase completion
- `journal_expressvpn.md` — minimal (3 domains, no subdomains found)

Plus a 12,800-line bounty daemon log (`bounty_daemon.log`) with session-level operational state.

### How It Differs from TDR

| Property | TDR | RWJ |
|---|---|---|
| **Retrieval pattern** | Search across 100+ accumulated reports by query | Sequential access to one target's history; latest state matters most |
| **Content structure** | Post-hoc disclosure narrative | Live operational log with repeated iterations |
| **Key facts** | CVEs, vuln classes, affected endpoints | Tool counts, host numbers, phase status, error states |
| **Update pattern** | Static after publication | Evolving — same section rewritten as counts change |
| **Scale challenge** | Many reports, find the right one | One long journal, find the current state |
| **Evaluation** | "Did the system retrieve the right report?" | "Does the system know the latest count/status?" |

### Why It Might Need Its Own Leaf

The vfsglobal journal has 6 repetitions of the same phase with evolving counts (Static hosts: 0 → 55 → 49 → 49 → 49 → 98). A standard retrieval approach would not know which iteration is current. The memory challenge here is **temporal override** — later values supersede earlier ones.

This is closer to the parent OSA's original purpose (state tracking with override detection) but in markdown format.

### Evaluation Design

- Ingest N recon journals as sequential sessions
- Query: "What is the current state of target X?"
- Ground truth: the final/latest values in the journal
- Metrics: fact recovery on latest counts, phase status, error states
- Challenge: does the system return the latest values, not earlier iterations?

### Risk

- Small dataset (5 journals, short text)
- May be too similar to TDR at the compression level (same AOJ S32 v2 model)
- The interesting difference is temporal override, not compression

---

## Candidate B: Workflow State (WS)

### What It Is

Agent operational state at the daemon/orchestrator level. Session start/stop events, task assignments, error counts, backoff states, token usage progression.

### Available Data

`bounty_daemon.log` — 12,800 lines of timestamped daemon events:
- Session boundaries (`--- Turn N (session=bounty-hunt-M) ---`)
- Task types (`BOOTSTRAP`, `CONTINUE`)
- Success/failure with token usage (`in=731870 out=1828 total=45828`)
- Error categorization (`no_output`, timeouts, gateway failures)
- Backoff state management
- Multi-target queue progression

### How It Differs from TDR

| Property | TDR | WS |
|---|---|---|
| **Format** | Markdown narrative | Timestamped log lines |
| **Content** | Technical findings | System state transitions |
| **Retrieval pattern** | Find a specific report | Track cumulative state across sessions |
| **Key facts** | CVEs, domains, vuln details | Session counts, error rates, token costs |
| **Compression challenge** | Structured prose → latent | Repetitive log lines → compressed state |

### Why It Might Need Its Own Leaf

The daemon log is structurally closer to the parent OSA's training data (timestamped events with key-value pairs) than to AOJ's markdown journals. But it's richer — multi-session, with complex state transitions, backoff logic, and cumulative accounting.

The memory challenge is **aggregation** — what is the total token usage? How many failures? What is the current session number?

### Evaluation Design

- Ingest the full daemon log as a series of sessions (e.g., one session per N turns)
- Query: "How many sessions failed?", "What was total token usage?", "What is the current error state?"
- Ground truth: computed from raw log
- Metrics: fact recovery on aggregated values

### Risk

- May actually work better with parent OSA node (closer to timestamped traces)
- Single data source
- Aggregation queries are hard for any compression system

---

## Recommendation

**Start with Recon Workflow Journals (RWJ).**

Reasons:
1. We already have the original AOJ A/B test data on these exact journals (73% fact recovery baseline)
2. The temporal-override challenge is a real distinct problem that TDR does not test
3. The data is ready — no new collection needed
4. It tests whether the same AOJ S32 v2 compression works for a different retrieval pattern (latest-state vs historical-search)
5. If RWJ needs a different pipeline from TDR, that proves the leaf distinction is real

The test can reuse the existing server backup data and the frozen AOJ S32 v2 checkpoint.

---

## Proposed RWJ Validation Plan

### Phase 1: Setup

1. Spin up A100 instance
2. Deploy from `server_backup/cndx_backup.tar.gz`
3. Verify AOJ S32 v2 checkpoint loads

### Phase 2: Ingest

1. Load all 5 recon journals as separate sessions
2. Optionally: chunk the daemon log into session-boundary segments and ingest those too
3. Use the same `SessionHooks` pipeline as TDR

### Phase 3: Build Evaluation

1. Write 10–15 queries targeting latest-state facts:
   - "How many live hosts does vfsglobal have?"
   - "What is the current phase for harman?"
   - "Did the expressvpn scan find any subdomains?"
   - "How many sources reported hosts for pinelabs?"
   - "What is the classified breakdown for dailymotion?"
2. Ground truth: manually extracted from final state of each journal
3. Measure: fact recovery, temporal correctness (latest vs stale values)

### Phase 4: Compare Approaches

1. **TDR pipeline (FTS5 → isolated reconstruction → heuristic)** — does the same pipeline work?
2. **Latest-only retrieval** — retrieve only the most recent session for each target
3. **Override-aware reconstruction** — reconstruct with temporal override logic

### Phase 5: Decide

- If TDR pipeline works for RWJ → same leaf, different evaluation
- If RWJ needs a different retrieval pattern → distinct leaf, document why
- If RWJ fails entirely → document the failure, investigate root cause
