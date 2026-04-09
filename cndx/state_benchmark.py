"""CNDX Operational-State Benchmark Suite — information retrieval through the
latent bottleneck evaluated on Operational State Artifacts (event/update traces).

Domain: sequential updates, state continuity, event ordering, value overwrites,
final-state recovery. Unlike static prose or code — this is dynamic state memory.

Probes:
  1. STATE-NIAH    — recover a specific update buried in distractors
  2. STATE-FACT1   — single-hop state binding (entity -> current value)
  3. STATE-FACT2   — two-hop transition chain (A changed B, then B changed C)
  4. STATE-TRACK   — final state recovery after multiple updates to same field
  5. STATE-OVERRIDE — later updates correctly overwrite earlier ones
  6. STATE-POS     — sensitivity to early/middle/late position of critical update
  7. STATE-EXACT   — exact recovery of final key=value or critical event line

Usage:
    python -m cndx.state_benchmark --checkpoint_dir <path> [--tests niah,fact1,fact2,track,override,pos,exact]
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cndx.native_model import CNDXNativeModel, NativeConfig, build_native_model


# ---------------------------------------------------------------------------
# State-trace building blocks
# ---------------------------------------------------------------------------

ENTITIES = [
    "server-01", "server-02", "server-03", "server-04",
    "node-alpha", "node-beta", "node-gamma", "node-delta",
    "worker-A", "worker-B", "worker-C", "worker-D",
    "sensor-north", "sensor-south", "sensor-east", "sensor-west",
    "pipeline-main", "pipeline-backup", "container-web", "container-db",
]

STATUSES = ["idle", "active", "busy", "degraded", "offline", "standby",
            "draining", "recovering", "pending", "running", "completed", "failed"]

LOCATIONS = ["rack-1", "rack-2", "rack-3", "zone-us-east", "zone-eu-west",
             "zone-ap-south", "bay-A", "bay-B", "bay-C",
             "datacenter-primary", "datacenter-backup"]

FIELDS = ["load", "temperature", "memory_pct", "queue_depth",
          "error_count", "latency_ms", "throughput", "connections", "cpu_pct"]

ACTIONS = ["migrated to", "assigned to", "moved to", "relocated to", "deployed at"]

FILLER_EVENTS = [
    "system health check passed",
    "routine maintenance window opened",
    "backup snapshot completed",
    "log rotation executed",
    "heartbeat acknowledged",
    "configuration reloaded",
    "certificate renewed",
    "DNS cache flushed",
    "connection pool recycled",
    "metrics aggregation cycle finished",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_decode(model, tokenizer, text, device, amp_ctx, seq_len):
    ids = tokenizer(text, truncation=True, max_length=seq_len,
                    padding="max_length", return_tensors="pt")
    input_ids = ids["input_ids"].to(device)
    attn_mask = ids["attention_mask"].to(device)
    with amp_ctx:
        gen = model.generate(input_ids, attn_mask, max_new_tokens=seq_len)
    return tokenizer.decode(gen[0], skip_special_tokens=True)


def _tokenizer_roundtrip(tokenizer, text, seq_len):
    ids = tokenizer(text, truncation=True, max_length=seq_len,
                    padding="max_length", return_tensors="pt")
    return tokenizer.decode(ids["input_ids"][0], skip_special_tokens=True)


def _build_filler_lines(n, t_start):
    """Generate n filler event lines with timestamps."""
    lines = []
    t = t_start
    for _ in range(n):
        lines.append(f"[T={t}] {random.choice(FILLER_EVENTS)}")
        t += random.randint(1, 2)
    return lines, t


# ---------------------------------------------------------------------------
# Test 1: STATE-NIAH — recover specific update among distractors
# ---------------------------------------------------------------------------

def test_state_niah(model, tokenizer, device, amp_ctx, cfg, n_trials=30):
    """Can the model recover a specific entity state update buried in filler events?"""
    print("\n══════════════════════════════════════════")
    print("  STATE-NIAH: State Needle Recovery")
    print("══════════════════════════════════════════")
    model.eval()

    results_by_pos = {}
    for position in ["start", "middle", "end"]:
        pos_results = []
        for _ in range(n_trials):
            entity = random.choice(ENTITIES)
            field = random.choice(FIELDS)
            value = str(random.randint(10, 99))
            needle = f"[T=5] {entity} {field}={value}"

            fillers_before, _ = _build_filler_lines(3, 1)
            fillers_after, _ = _build_filler_lines(3, 8)

            if position == "start":
                lines = [needle] + fillers_before + fillers_after
            elif position == "end":
                lines = fillers_before + fillers_after + [needle]
            else:
                lines = fillers_before + [needle] + fillers_after

            passage = "\n".join(lines)
            decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

            entity_found = entity in decoded
            field_found = field in decoded
            value_found = value in decoded
            full_recovery = entity_found and field_found and value_found

            pos_results.append({
                "entity_found": entity_found,
                "field_found": field_found,
                "value_found": value_found,
                "full_recovery": full_recovery,
            })

        rate = sum(1 for r in pos_results if r["full_recovery"]) / len(pos_results)
        entity_rate = sum(1 for r in pos_results if r["entity_found"]) / len(pos_results)
        value_rate = sum(1 for r in pos_results if r["value_found"]) / len(pos_results)
        results_by_pos[position] = {
            "n_trials": n_trials,
            "entity_rate": round(entity_rate, 3),
            "value_rate": round(value_rate, 3),
            "full_recovery_rate": round(rate, 3),
        }
        print(f"  {position}: entity={entity_rate:.1%}  value={value_rate:.1%}  full={rate:.1%}")

    overall = sum(r["full_recovery_rate"] for r in results_by_pos.values()) / len(results_by_pos)
    print(f"  OVERALL: {overall:.1%}")
    return {"test": "STATE-NIAH", "results_by_position": results_by_pos, "overall": round(overall, 3)}


# ---------------------------------------------------------------------------
# Test 2: STATE-FACT1 — single-hop state binding
# ---------------------------------------------------------------------------

def test_state_fact1(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    """Does the model bind the right value to the right entity?
    Two entities, each with a different field value. Score correct binding."""
    print("\n══════════════════════════════════════════")
    print("  STATE-FACT1: Single-Hop State Binding")
    print("══════════════════════════════════════════")
    model.eval()

    results = []
    for _ in range(n_trials):
        e1, e2 = random.sample(ENTITIES, 2)
        field = random.choice(FIELDS)
        v1 = str(random.randint(10, 49))
        v2 = str(random.randint(50, 99))

        fillers, t = _build_filler_lines(2, 1)
        lines = fillers + [f"[T={t}] {e1} {field}={v1}"]
        t += 2
        fillers2, t = _build_filler_lines(2, t)
        lines += fillers2 + [f"[T={t}] {e2} {field}={v2}"]

        passage = "\n".join(lines)
        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

        e1_present = e1 in decoded
        e2_present = e2 in decoded
        v1_present = v1 in decoded
        v2_present = v2 in decoded
        e1_bound = e1_present and v1_present
        e2_bound = e2_present and v2_present
        no_cross = not (e1 in decoded and v2 in decoded and e2 not in decoded)

        results.append({
            "e1_bound": e1_bound,
            "e2_bound": e2_bound,
            "both_bound": e1_bound and e2_bound,
        })

    both_rate = sum(1 for r in results if r["both_bound"]) / len(results)
    e1_rate = sum(1 for r in results if r["e1_bound"]) / len(results)
    e2_rate = sum(1 for r in results if r["e2_bound"]) / len(results)

    print(f"  E1 correctly bound: {e1_rate:.1%}")
    print(f"  E2 correctly bound: {e2_rate:.1%}")
    print(f"  Both correct: {both_rate:.1%}")

    return {
        "test": "STATE-FACT1",
        "n_trials": n_trials,
        "e1_binding_rate": round(e1_rate, 3),
        "e2_binding_rate": round(e2_rate, 3),
        "both_binding_rate": round(both_rate, 3),
    }


# ---------------------------------------------------------------------------
# Test 3: STATE-FACT2 — two-hop transition chain
# ---------------------------------------------------------------------------

def test_state_fact2(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    """Two-hop chain: entity A's status change triggers B's reassignment.
    A goes active -> B migrates to new location. Does the chain survive?"""
    print("\n══════════════════════════════════════════")
    print("  STATE-FACT2: Two-Hop Transition Chain")
    print("══════════════════════════════════════════")
    model.eval()

    results = []
    for _ in range(n_trials):
        e_a, e_b = random.sample(ENTITIES, 2)
        status = random.choice(["active", "busy", "degraded"])
        location = random.choice(LOCATIONS)

        fillers1, _ = _build_filler_lines(2, 1)
        hop1 = f"[T=4] {e_a} status={status}"
        fillers2, _ = _build_filler_lines(1, 5)
        hop2 = f"[T=7] {e_b} migrated to {location}"
        fillers3, _ = _build_filler_lines(2, 8)

        lines = fillers1 + [hop1] + fillers2 + [hop2] + fillers3
        passage = "\n".join(lines)
        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

        a_status = e_a in decoded and status in decoded
        b_location = e_b in decoded and location in decoded
        chain = a_status and b_location

        results.append({
            "hop1_present": a_status,
            "hop2_present": b_location,
            "chain_intact": chain,
        })

    hop1_rate = sum(1 for r in results if r["hop1_present"]) / len(results)
    hop2_rate = sum(1 for r in results if r["hop2_present"]) / len(results)
    chain_rate = sum(1 for r in results if r["chain_intact"]) / len(results)

    print(f"  Hop 1 (A status): {hop1_rate:.1%}")
    print(f"  Hop 2 (B location): {hop2_rate:.1%}")
    print(f"  Full chain: {chain_rate:.1%}")

    return {
        "test": "STATE-FACT2",
        "n_trials": n_trials,
        "hop1_rate": round(hop1_rate, 3),
        "hop2_rate": round(hop2_rate, 3),
        "chain_intact_rate": round(chain_rate, 3),
    }


# ---------------------------------------------------------------------------
# Test 4: STATE-TRACK — final state after multiple updates
# ---------------------------------------------------------------------------

def test_state_track(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    """Same entity, same field, updated 4-6 times. Does the model recover
    the FINAL value, not an intermediate one?"""
    print("\n══════════════════════════════════════════")
    print("  STATE-TRACK: Final State Recovery")
    print("══════════════════════════════════════════")
    model.eval()

    results = []
    for _ in range(n_trials):
        entity = random.choice(ENTITIES)
        field = random.choice(FIELDS)
        n_updates = random.randint(4, 6)
        values = [str(random.randint(1, 99)) for _ in range(n_updates)]
        final_value = values[-1]

        lines = []
        t = 1
        for v in values:
            lines.append(f"[T={t}] {entity} {field}={v}")
            if random.random() < 0.3:
                lines.append(f"[T={t}] {random.choice(FILLER_EVENTS)}")
            t += random.randint(1, 3)

        passage = "\n".join(lines)
        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

        entity_present = entity in decoded
        final_present = final_value in decoded
        intermediates = [v for v in values[:-1] if v in decoded]

        results.append({
            "entity_present": entity_present,
            "final_value_present": final_present,
            "n_intermediates_present": len(intermediates),
            "n_updates": n_updates,
        })

    entity_rate = sum(1 for r in results if r["entity_present"]) / len(results)
    final_rate = sum(1 for r in results if r["final_value_present"]) / len(results)
    avg_intermediates = sum(r["n_intermediates_present"] for r in results) / len(results)

    print(f"  Entity present: {entity_rate:.1%}")
    print(f"  Final value recovered: {final_rate:.1%}")
    print(f"  Avg intermediates also present: {avg_intermediates:.1f}")

    return {
        "test": "STATE-TRACK",
        "n_trials": n_trials,
        "entity_rate": round(entity_rate, 3),
        "final_value_rate": round(final_rate, 3),
        "avg_intermediates_present": round(avg_intermediates, 2),
    }


# ---------------------------------------------------------------------------
# Test 5: STATE-OVERRIDE — later updates overwrite earlier ones
# ---------------------------------------------------------------------------

def test_state_override(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    """Entity has an early value, then a late overwrite. Does the model
    prefer the later value over the earlier one?"""
    print("\n══════════════════════════════════════════")
    print("  STATE-OVERRIDE: Overwrite Correctness")
    print("══════════════════════════════════════════")
    model.eval()

    results = []
    for _ in range(n_trials):
        entity = random.choice(ENTITIES)
        field = random.choice(FIELDS)
        old_val = str(random.randint(10, 49))
        new_val = str(random.randint(50, 99))

        fillers1, _ = _build_filler_lines(2, 1)
        early_line = f"[T=4] {entity} {field}={old_val}"
        fillers2, _ = _build_filler_lines(3, 5)
        late_line = f"[T=9] {entity} {field}={new_val}"
        fillers3, _ = _build_filler_lines(1, 10)

        lines = fillers1 + [early_line] + fillers2 + [late_line] + fillers3
        passage = "\n".join(lines)
        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

        old_present = old_val in decoded
        new_present = new_val in decoded
        override_correct = new_present
        retained_old = old_present and not new_present

        results.append({
            "old_present": old_present,
            "new_present": new_present,
            "override_correct": override_correct,
            "retained_old_only": retained_old,
        })

    new_rate = sum(1 for r in results if r["new_present"]) / len(results)
    old_rate = sum(1 for r in results if r["old_present"]) / len(results)
    override_rate = sum(1 for r in results if r["override_correct"]) / len(results)
    stale_rate = sum(1 for r in results if r["retained_old_only"]) / len(results)

    print(f"  New (overwritten) value present: {new_rate:.1%}")
    print(f"  Old value still present: {old_rate:.1%}")
    print(f"  Override correct: {override_rate:.1%}")
    print(f"  Stale (only old value): {stale_rate:.1%}")

    return {
        "test": "STATE-OVERRIDE",
        "n_trials": n_trials,
        "new_value_rate": round(new_rate, 3),
        "old_value_rate": round(old_rate, 3),
        "override_correct_rate": round(override_rate, 3),
        "stale_rate": round(stale_rate, 3),
    }


# ---------------------------------------------------------------------------
# Test 6: STATE-POS — positional sensitivity of critical updates
# ---------------------------------------------------------------------------

def test_state_pos(model, tokenizer, device, amp_ctx, cfg, n_trials=20):
    """Does the model preserve a critical state update equally well at
    start, middle, and end of the event stream?"""
    print("\n══════════════════════════════════════════")
    print("  STATE-POS: Positional Sensitivity")
    print("══════════════════════════════════════════")
    model.eval()

    results_by_pos = {}
    for position in ["start", "middle", "end"]:
        pos_results = []
        for _ in range(n_trials):
            entity = random.choice(ENTITIES)
            status = random.choice(STATUSES)
            needle = f"[T=5] {entity} status={status}"

            fillers_a, _ = _build_filler_lines(4, 1)
            fillers_b, _ = _build_filler_lines(4, 8)

            if position == "start":
                lines = [needle] + fillers_a + fillers_b
            elif position == "end":
                lines = fillers_a + fillers_b + [needle]
            else:
                lines = fillers_a + [needle] + fillers_b

            passage = "\n".join(lines)
            decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

            entity_found = entity in decoded
            status_found = status in decoded
            full = entity_found and status_found

            pos_results.append({"entity": entity_found, "status": status_found, "full": full})

        full_rate = sum(1 for r in pos_results if r["full"]) / len(pos_results)
        entity_rate = sum(1 for r in pos_results if r["entity"]) / len(pos_results)
        results_by_pos[position] = {
            "n_trials": n_trials,
            "entity_rate": round(entity_rate, 3),
            "full_recovery_rate": round(full_rate, 3),
        }
        print(f"  {position}: entity={entity_rate:.1%}  full={full_rate:.1%}")

    return {"test": "STATE-POS", "results_by_position": results_by_pos}


# ---------------------------------------------------------------------------
# Test 7: STATE-EXACT — exact recovery of key event line
# ---------------------------------------------------------------------------

def test_state_exact(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    """Can the model reproduce a critical event line character-for-character?"""
    print("\n══════════════════════════════════════════")
    print("  STATE-EXACT: Exact Event Recovery")
    print("══════════════════════════════════════════")
    model.eval()

    results = []
    for _ in range(n_trials):
        entity = random.choice(ENTITIES)
        field = random.choice(FIELDS)
        value = str(random.randint(10, 99))
        exact_line = f"[T=5] {entity} {field}={value}"

        fillers, _ = _build_filler_lines(5, 1)
        mid = len(fillers) // 2
        lines = fillers[:mid] + [exact_line] + fillers[mid:]
        passage = "\n".join(lines)

        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

        exact_match = exact_line in decoded
        words = exact_line.split()
        word_recall = sum(1 for w in words if w in decoded) / len(words)

        trial_result = {
            "exact_match": exact_match,
            "word_recall": round(word_recall, 3),
        }
        if _ < 3:
            trial_result["target"] = exact_line
            trial_result["decoded_preview"] = decoded[:200]
        results.append(trial_result)

    exact_rate = sum(1 for r in results if r["exact_match"]) / len(results)
    avg_recall = sum(r["word_recall"] for r in results) / len(results)

    print(f"  Exact match: {exact_rate:.1%}")
    print(f"  Avg word recall: {avg_recall:.1%}")

    return {
        "test": "STATE-EXACT",
        "n_trials": n_trials,
        "exact_match_rate": round(exact_rate, 3),
        "avg_word_recall": round(avg_recall, 3),
        "samples": [r for r in results if "decoded_preview" in r],
    }


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

def test_state_control(model, tokenizer, device, amp_ctx, cfg, n_trials=30):
    """No-compression control for state traces."""
    print("\n══════════════════════════════════════════")
    print("  STATE-CONTROL: No-Compression Baseline")
    print("══════════════════════════════════════════")
    model.eval()

    raw_scores, comp_scores = [], []
    for _ in range(n_trials):
        entity = random.choice(ENTITIES)
        field = random.choice(FIELDS)
        value = str(random.randint(10, 99))
        needle = f"[T=5] {entity} {field}={value}"

        fillers, _ = _build_filler_lines(4, 1)
        lines = fillers[:2] + [needle] + fillers[2:]
        passage = "\n".join(lines)

        raw = _tokenizer_roundtrip(tokenizer, passage, cfg.seq_len)
        raw_hit = entity in raw and value in raw
        raw_scores.append(raw_hit)

        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)
        comp_hit = entity in decoded and value in decoded
        comp_scores.append(comp_hit)

    raw_rate = sum(raw_scores) / n_trials
    comp_rate = sum(comp_scores) / n_trials

    print(f"  Raw (tokenizer):     {raw_rate:.1%}")
    print(f"  Compressed (latent): {comp_rate:.1%}")
    print(f"  Compression cost:    {raw_rate - comp_rate:+.1%}")

    return {
        "test": "STATE-CONTROL",
        "n_trials": n_trials,
        "raw_recovery_rate": round(raw_rate, 3),
        "compressed_recovery_rate": round(comp_rate, 3),
        "compression_cost": round(raw_rate - comp_rate, 3),
    }


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_dir, device="cuda"):
    results_path = Path(ckpt_dir) / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        cfg_dict = results.get("config", {})
    else:
        cfg_dict = {}

    cfg = NativeConfig(**{k: v for k, v in cfg_dict.items()
                          if k in NativeConfig.__dataclass_fields__})
    model, tokenizer = build_native_model(cfg, device=device)

    ckpt_path = Path(ckpt_dir) / "model.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        raise FileNotFoundError(f"No model.pt in {ckpt_dir}")

    model.eval()
    return model, tokenizer, cfg


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_state_benchmarks(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float32)
    amp_ctx = (torch.autocast("cuda", dtype=dtype) if device.type == "cuda"
               else torch.autocast("cpu", enabled=False))

    print(f"Device: {device}, AMP dtype: {dtype}")
    model, tokenizer, cfg = load_checkpoint(args.checkpoint_dir, device=device)
    print(f"Model: {model.num_params()/1e6:.1f}M params, K={cfg.num_latents}")

    ALL_TESTS = ["control", "niah", "fact1", "fact2", "track", "override", "pos", "exact"]
    tests = args.tests.split(",") if args.tests else ALL_TESTS
    all_results = {
        "checkpoint": args.checkpoint_dir,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_summary": {
            "num_latents": cfg.num_latents,
            "latent_groups": cfg.latent_groups,
            "seq_len": cfg.seq_len,
            "warmup_ratio": cfg.warmup_ratio,
        },
    }

    if "control" in tests:
        all_results["control"] = test_state_control(model, tokenizer, device, amp_ctx, cfg)
    if "niah" in tests:
        all_results["niah"] = test_state_niah(model, tokenizer, device, amp_ctx, cfg)
    if "fact1" in tests:
        all_results["fact1"] = test_state_fact1(model, tokenizer, device, amp_ctx, cfg)
    if "fact2" in tests:
        all_results["fact2"] = test_state_fact2(model, tokenizer, device, amp_ctx, cfg)
    if "track" in tests:
        all_results["track"] = test_state_track(model, tokenizer, device, amp_ctx, cfg)
    if "override" in tests:
        all_results["override"] = test_state_override(model, tokenizer, device, amp_ctx, cfg)
    if "pos" in tests:
        all_results["pos"] = test_state_pos(model, tokenizer, device, amp_ctx, cfg)
    if "exact" in tests:
        all_results["exact"] = test_state_exact(model, tokenizer, device, amp_ctx, cfg)

    out_path = Path(args.checkpoint_dir) / "state_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n══════════════════════════════════════════")
    print(f"  State benchmark results saved to {out_path}")
    print(f"══════════════════════════════════════════")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CNDX Operational-State Benchmark Suite")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--tests", type=str, default=None,
                        help="Comma-separated: control,niah,fact1,fact2,track,override,pos,exact")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    run_state_benchmarks(args)
