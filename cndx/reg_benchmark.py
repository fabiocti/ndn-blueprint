"""CNDX High-Precision Regulated Text Benchmark Suite — information retrieval
through the latent bottleneck on exactness-sensitive compliance/policy text.

Probes:
  1. REG-CONTROL  — no-compression baseline
  2. REG-NIAH     — recover buried clause/threshold/deadline from distractors
  3. REG-FACT1    — single-hop exact binding (role→obligation, threshold→action)
  4. REG-FACT2    — two-hop regulated chain (condition→rule→action)
  5. REG-OVERRIDE — later clause supersedes earlier clause
  6. REG-TRACK    — evolving procedure status across formal updates
  7. REG-POS      — positional sensitivity for critical clauses
  8. REG-EXACT    — exact recovery of dates, thresholds, role names, clause text

Usage:
    python -m cndx.reg_benchmark --checkpoint_dir <path> [--tests ...]
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cndx.native_model import CNDXNativeModel, NativeConfig, build_native_model

ROLES = [
    "Analyst", "Senior Analyst", "Compliance Officer", "Risk Manager",
    "Auditor", "Team Lead", "Director", "VP of Operations",
    "Data Protection Officer", "Incident Commander", "Account Manager",
    "Quality Assurance Lead", "Regional Supervisor", "Chief Risk Officer",
]
DEPARTMENTS = [
    "Risk & Compliance", "Internal Audit", "Operations", "Legal",
    "Information Security", "Human Resources", "Finance",
]
DURATIONS = [
    "7 days", "14 days", "30 days", "60 days", "90 days",
    "120 days", "6 months", "1 year", "3 business days",
    "5 business days", "24 hours", "48 hours",
]
THRESHOLDS = [
    "$5,000", "$10,000", "$25,000", "$50,000", "$100,000",
    "$250,000", "$500,000", "$1,000,000",
]
CATEGORIES = [
    "Category A", "Category B", "Category C", "Tier-1", "Tier-2",
    "Tier-3", "Priority 1", "Priority 2", "Class I", "Class II",
]
DOC_TYPES = [
    "incident report", "exception request", "variance analysis",
    "compliance attestation", "audit trail", "change request",
    "risk assessment", "remediation plan",
]
FLAGS = [
    "confidentiality flag", "PII indicator", "restricted access marker",
    "cross-border flag", "manual override indicator", "high-risk tag",
]
DATES = [
    "1 January 2025", "15 March 2025", "1 April 2025", "30 June 2025",
    "1 September 2025", "31 December 2025", "1 February 2026",
    "15 May 2026", "1 July 2026", "30 September 2026",
]
FILLER_CLAUSES = [
    "All personnel must complete annual compliance training before the deadline.",
    "Records must be stored in an immutable audit-compliant repository.",
    "Quarterly reconciliation reports shall be submitted to the designated authority.",
    "Access to restricted systems requires multi-factor authentication.",
    "Any deviation from standard procedure must be documented and justified.",
    "Communication with external regulators must follow the approved template.",
    "All exceptions must be logged in the central exception register.",
    "Backup procedures must be tested at least once per fiscal quarter.",
]


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


# -----------------------------------------------------------------------
# REG-CONTROL
# -----------------------------------------------------------------------

def test_reg_control(model, tokenizer, device, amp_ctx, cfg, n_trials=30):
    print("\n══════════════════════════════════════════")
    print("  REG-CONTROL: No-Compression Baseline")
    print("══════════════════════════════════════════")
    model.eval()
    raw_scores, comp_scores = [], []
    for _ in range(n_trials):
        role = random.choice(ROLES)
        thresh = random.choice(THRESHOLDS)
        dur = random.choice(DURATIONS)
        needle = f"CLAUSE: If the amount exceeds {thresh}, {role} must escalate within {dur}."
        fillers = random.sample(FILLER_CLAUSES, 3)
        passage = "\n".join(fillers[:1] + [needle] + fillers[1:])

        raw = _tokenizer_roundtrip(tokenizer, passage, cfg.seq_len)
        raw_hit = thresh.lower() in raw.lower() and role.lower() in raw.lower() and dur.lower() in raw.lower()
        raw_scores.append(raw_hit)

        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)
        comp_hit = thresh.lower() in decoded.lower() and role.lower() in decoded.lower() and dur.lower() in decoded.lower()
        comp_scores.append(comp_hit)

    raw_rate = sum(raw_scores) / n_trials
    comp_rate = sum(comp_scores) / n_trials
    print(f"  Raw: {raw_rate:.1%}  Compressed: {comp_rate:.1%}  Cost: {raw_rate - comp_rate:+.1%}")
    return {
        "test": "REG-CONTROL", "n_trials": n_trials,
        "raw_recovery_rate": round(raw_rate, 3),
        "compressed_recovery_rate": round(comp_rate, 3),
        "compression_cost": round(raw_rate - comp_rate, 3),
    }


# -----------------------------------------------------------------------
# REG-NIAH
# -----------------------------------------------------------------------

def test_reg_niah(model, tokenizer, device, amp_ctx, cfg, n_trials=30):
    print("\n══════════════════════════════════════════")
    print("  REG-NIAH: Regulated Needle Recovery")
    print("══════════════════════════════════════════")
    model.eval()
    results_by_pos = {}
    for position in ["start", "middle", "end"]:
        pos_results = []
        for _ in range(n_trials):
            role = random.choice(ROLES)
            thresh = random.choice(THRESHOLDS)
            dur = random.choice(DURATIONS)
            needle = f"REQUIREMENT: {role} must retain all records for {dur} when exposure exceeds {thresh}."

            fillers = random.sample(FILLER_CLAUSES, 4)
            if position == "start":
                lines = [needle] + fillers
            elif position == "end":
                lines = fillers + [needle]
            else:
                lines = fillers[:2] + [needle] + fillers[2:]

            passage = "\n".join(lines)
            decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)
            dl = decoded.lower()

            role_found = role.lower() in dl
            thresh_found = thresh.lower() in dl
            dur_found = dur.lower() in dl
            full = role_found and thresh_found and dur_found

            pos_results.append({"role": role_found, "thresh": thresh_found, "dur": dur_found, "full": full})

        full_rate = sum(1 for r in pos_results if r["full"]) / len(pos_results)
        role_rate = sum(1 for r in pos_results if r["role"]) / len(pos_results)
        results_by_pos[position] = {
            "n_trials": n_trials,
            "role_rate": round(role_rate, 3),
            "full_recovery_rate": round(full_rate, 3),
        }
        print(f"  {position}: role={role_rate:.1%}  full={full_rate:.1%}")

    overall = sum(r["full_recovery_rate"] for r in results_by_pos.values()) / len(results_by_pos)
    print(f"  OVERALL: {overall:.1%}")
    return {"test": "REG-NIAH", "results_by_position": results_by_pos, "overall": round(overall, 3)}


# -----------------------------------------------------------------------
# REG-FACT1
# -----------------------------------------------------------------------

def test_reg_fact1(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    print("\n══════════════════════════════════════════")
    print("  REG-FACT1: Role→Obligation Binding")
    print("══════════════════════════════════════════")
    model.eval()
    results = []
    for _ in range(n_trials):
        r1, r2 = random.sample(ROLES, 2)
        d1, d2 = random.sample(DURATIONS, 2)
        doc1, doc2 = random.sample(DOC_TYPES, 2)

        line1 = f"POLICY: {r1} shall submit {doc1} within {d1}."
        line2 = f"POLICY: {r2} shall submit {doc2} within {d2}."
        fillers = random.sample(FILLER_CLAUSES, 2)
        passage = "\n".join([line1] + fillers + [line2])
        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)
        dl = decoded.lower()

        bind1 = r1.lower() in dl and d1.lower() in dl and doc1.lower() in dl
        bind2 = r2.lower() in dl and d2.lower() in dl and doc2.lower() in dl

        results.append({"bind1": bind1, "bind2": bind2, "both": bind1 and bind2})

    both_rate = sum(1 for r in results if r["both"]) / len(results)
    b1_rate = sum(1 for r in results if r["bind1"]) / len(results)
    b2_rate = sum(1 for r in results if r["bind2"]) / len(results)
    print(f"  Binding 1: {b1_rate:.1%}  Binding 2: {b2_rate:.1%}  Both: {both_rate:.1%}")
    return {
        "test": "REG-FACT1", "n_trials": n_trials,
        "binding1_rate": round(b1_rate, 3),
        "binding2_rate": round(b2_rate, 3),
        "both_binding_rate": round(both_rate, 3),
    }


# -----------------------------------------------------------------------
# REG-FACT2
# -----------------------------------------------------------------------

def test_reg_fact2(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    print("\n══════════════════════════════════════════")
    print("  REG-FACT2: Condition→Rule→Action Chain")
    print("══════════════════════════════════════════")
    model.eval()
    results = []
    for _ in range(n_trials):
        thresh = random.choice(THRESHOLDS)
        cat = random.choice(CATEGORIES)
        role = random.choice(ROLES)
        dur = random.choice(DURATIONS)

        hop1 = f"RULE: When the amount exceeds {thresh}, classify as {cat}."
        filler = random.choice(FILLER_CLAUSES)
        hop2 = f"ESCALATION: All {cat} items require approval from {role} within {dur}."

        passage = "\n".join([hop1, filler, hop2])
        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)
        dl = decoded.lower()

        thresh_found = thresh.lower() in dl
        cat_found = cat.lower() in dl
        role_found = role.lower() in dl
        dur_found = dur.lower() in dl
        chain = thresh_found and cat_found and role_found and dur_found

        results.append({"thresh": thresh_found, "cat": cat_found,
                        "role": role_found, "dur": dur_found, "chain": chain})

    chain_rate = sum(1 for r in results if r["chain"]) / len(results)
    thresh_rate = sum(1 for r in results if r["thresh"]) / len(results)
    role_rate = sum(1 for r in results if r["role"]) / len(results)
    print(f"  Thresh: {thresh_rate:.1%}  Role: {role_rate:.1%}  Chain: {chain_rate:.1%}")
    return {
        "test": "REG-FACT2", "n_trials": n_trials,
        "threshold_rate": round(thresh_rate, 3),
        "role_rate": round(role_rate, 3),
        "chain_intact_rate": round(chain_rate, 3),
    }


# -----------------------------------------------------------------------
# REG-OVERRIDE
# -----------------------------------------------------------------------

def test_reg_override(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    print("\n══════════════════════════════════════════")
    print("  REG-OVERRIDE: Clause Supersession")
    print("══════════════════════════════════════════")
    model.eval()
    results = []
    for _ in range(n_trials):
        doc = random.choice(DOC_TYPES)
        old_dur = random.choice(DURATIONS[:6])
        new_dur = random.choice(DURATIONS[6:])
        date = random.choice(DATES)

        original = f"POLICY: Retain {doc} for {old_dur}."
        filler = random.choice(FILLER_CLAUSES)
        override = f"UPDATE: Effective {date}, {doc} retention changed to {new_dur}. Previous {old_dur} requirement is superseded."

        passage = "\n".join([original, filler, override])
        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)
        dl = decoded.lower()

        old_present = old_dur.lower() in dl
        new_present = new_dur.lower() in dl
        date_present = date.lower() in dl
        doc_present = doc.lower() in dl
        override_ok = new_present and doc_present

        results.append({"old": old_present, "new": new_present,
                        "date": date_present, "override_ok": override_ok})

    new_rate = sum(1 for r in results if r["new"]) / len(results)
    old_rate = sum(1 for r in results if r["old"]) / len(results)
    date_rate = sum(1 for r in results if r["date"]) / len(results)
    override_rate = sum(1 for r in results if r["override_ok"]) / len(results)
    print(f"  New duration: {new_rate:.1%}  Old: {old_rate:.1%}  Date: {date_rate:.1%}")
    print(f"  Override correct: {override_rate:.1%}")
    return {
        "test": "REG-OVERRIDE", "n_trials": n_trials,
        "new_duration_rate": round(new_rate, 3),
        "old_duration_rate": round(old_rate, 3),
        "date_rate": round(date_rate, 3),
        "override_correct_rate": round(override_rate, 3),
    }


# -----------------------------------------------------------------------
# REG-TRACK
# -----------------------------------------------------------------------

def test_reg_track(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    print("\n══════════════════════════════════════════")
    print("  REG-TRACK: Procedure Status Tracking")
    print("══════════════════════════════════════════")
    model.eval()
    statuses = ["Draft", "Under Review", "Approved", "Effective", "Superseded"]
    results = []
    for _ in range(n_trials):
        doc = random.choice(DOC_TYPES)
        n_updates = random.randint(3, 5)
        chosen = [random.choice(statuses) for _ in range(n_updates)]
        final_status = chosen[-1]

        lines = []
        for i, s in enumerate(chosen):
            lines.append(f"STATUS UPDATE: {doc} is now {s}.")
            if random.random() < 0.4:
                lines.append(random.choice(FILLER_CLAUSES))

        passage = "\n".join(lines)
        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)
        dl = decoded.lower()

        doc_found = doc.lower() in dl
        final_found = final_status.lower() in dl

        results.append({"doc_found": doc_found, "final_status_found": final_found})

    doc_rate = sum(1 for r in results if r["doc_found"]) / len(results)
    final_rate = sum(1 for r in results if r["final_status_found"]) / len(results)
    print(f"  Doc present: {doc_rate:.1%}  Final status: {final_rate:.1%}")
    return {
        "test": "REG-TRACK", "n_trials": n_trials,
        "doc_rate": round(doc_rate, 3),
        "final_status_rate": round(final_rate, 3),
    }


# -----------------------------------------------------------------------
# REG-POS
# -----------------------------------------------------------------------

def test_reg_pos(model, tokenizer, device, amp_ctx, cfg, n_trials=20):
    print("\n══════════════════════════════════════════")
    print("  REG-POS: Positional Sensitivity")
    print("══════════════════════════════════════════")
    model.eval()
    results_by_pos = {}
    for position in ["start", "middle", "end"]:
        pos_results = []
        for _ in range(n_trials):
            role = random.choice(ROLES)
            thresh = random.choice(THRESHOLDS)
            dur = random.choice(DURATIONS)
            needle = f"CRITICAL: {role} must act within {dur} when exposure exceeds {thresh}."

            fillers = random.sample(FILLER_CLAUSES, 5)
            if position == "start":
                lines = [needle] + fillers
            elif position == "end":
                lines = fillers + [needle]
            else:
                lines = fillers[:3] + [needle] + fillers[3:]

            passage = "\n".join(lines)
            decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)
            dl = decoded.lower()

            role_found = role.lower() in dl
            thresh_found = thresh.lower() in dl
            dur_found = dur.lower() in dl
            full = role_found and thresh_found and dur_found

            pos_results.append({"role": role_found, "full": full})

        full_rate = sum(1 for r in pos_results if r["full"]) / len(pos_results)
        role_rate = sum(1 for r in pos_results if r["role"]) / len(pos_results)
        results_by_pos[position] = {
            "n_trials": n_trials,
            "role_rate": round(role_rate, 3),
            "full_recovery_rate": round(full_rate, 3),
        }
        print(f"  {position}: role={role_rate:.1%}  full={full_rate:.1%}")

    return {"test": "REG-POS", "results_by_position": results_by_pos}


# -----------------------------------------------------------------------
# REG-EXACT — the most important probe
# -----------------------------------------------------------------------

def test_reg_exact(model, tokenizer, device, amp_ctx, cfg, n_trials=50):
    print("\n══════════════════════════════════════════")
    print("  REG-EXACT: Exact Detail Recovery")
    print("══════════════════════════════════════════")
    model.eval()
    results = []
    for _ in range(n_trials):
        detail_type = random.choice(["threshold_role", "date_dur", "clause_wording"])

        if detail_type == "threshold_role":
            role = random.choice(ROLES)
            thresh = random.choice(THRESHOLDS)
            dur = random.choice(DURATIONS)
            needle = f"RULE: {role} must escalate to {random.choice(DEPARTMENTS)} within {dur} if exposure exceeds {thresh}."
            targets = [role, thresh, dur]
        elif detail_type == "date_dur":
            date = random.choice(DATES)
            dur = random.choice(DURATIONS)
            doc = random.choice(DOC_TYPES)
            needle = f"EFFECTIVE {date}: {doc} retention period is {dur}."
            targets = [date, dur, doc]
        else:
            cat = random.choice(CATEGORIES)
            flag = random.choice(FLAGS)
            role = random.choice(ROLES)
            needle = f"EXCEPTION: {cat} items with {flag} require approval from {role}."
            targets = [cat, flag, role]

        fillers = random.sample(FILLER_CLAUSES, 3)
        passage = "\n".join(fillers[:1] + [needle] + fillers[1:])
        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)
        dl = decoded.lower()

        hits = sum(1 for t in targets if t.lower() in dl)
        exact_all = hits == len(targets)
        word_recall = hits / len(targets)

        results.append({"exact_all": exact_all, "word_recall": round(word_recall, 3)})

    exact_rate = sum(1 for r in results if r["exact_all"]) / len(results)
    avg_recall = sum(r["word_recall"] for r in results) / len(results)
    print(f"  Exact all: {exact_rate:.1%}  Avg recall: {avg_recall:.1%}")
    return {
        "test": "REG-EXACT", "n_trials": n_trials,
        "exact_all_rate": round(exact_rate, 3),
        "avg_target_recall": round(avg_recall, 3),
    }


# -----------------------------------------------------------------------
# Checkpoint loader
# -----------------------------------------------------------------------

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


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def run_reg_benchmarks(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float32)
    amp_ctx = (torch.autocast("cuda", dtype=dtype) if device.type == "cuda"
               else torch.autocast("cpu", enabled=False))

    print(f"Device: {device}, AMP dtype: {dtype}")
    model, tokenizer, cfg = load_checkpoint(args.checkpoint_dir, device=device)
    print(f"Model: {model.num_params()/1e6:.1f}M params, K={cfg.num_latents}")

    ALL_TESTS = ["control", "niah", "fact1", "fact2", "override", "track", "pos", "exact"]
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
        all_results["control"] = test_reg_control(model, tokenizer, device, amp_ctx, cfg)
    if "niah" in tests:
        all_results["niah"] = test_reg_niah(model, tokenizer, device, amp_ctx, cfg)
    if "fact1" in tests:
        all_results["fact1"] = test_reg_fact1(model, tokenizer, device, amp_ctx, cfg)
    if "fact2" in tests:
        all_results["fact2"] = test_reg_fact2(model, tokenizer, device, amp_ctx, cfg)
    if "override" in tests:
        all_results["override"] = test_reg_override(model, tokenizer, device, amp_ctx, cfg)
    if "track" in tests:
        all_results["track"] = test_reg_track(model, tokenizer, device, amp_ctx, cfg)
    if "pos" in tests:
        all_results["pos"] = test_reg_pos(model, tokenizer, device, amp_ctx, cfg)
    if "exact" in tests:
        all_results["exact"] = test_reg_exact(model, tokenizer, device, amp_ctx, cfg)

    out_path = Path(args.checkpoint_dir) / "reg_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n══════════════════════════════════════════")
    print(f"  REG benchmark results saved to {out_path}")
    print(f"══════════════════════════════════════════")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CNDX High-Precision Regulated Text Benchmark Suite")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--tests", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    run_reg_benchmarks(args)
