"""CNDX Human Working-Memory Benchmark Suite — information retrieval through
the latent bottleneck on messy, fragmented, human-like working context.

Domain: fragmented notes, reminders, partial plans, messy task state,
inconsistent wording, corrections, real-ish working context.

Probes:
  1. HWM-NIAH    — recover a specific task/reminder from noise
  2. HWM-FACT1   — single-hop binding (person -> task, time -> place)
  3. HWM-FACT2   — two-hop (person assigned task, task has deadline)
  4. HWM-TRACK   — task status after updates/corrections
  5. HWM-OVERRIDE — correction overwrites original info
  6. HWM-POS     — positional sensitivity for critical fragments
  7. HWM-EXACT   — exact recovery of specific detail (name, time, phone)

Usage:
    python -m cndx.hwm_benchmark --checkpoint_dir <path> [--tests ...]
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from cndx.native_model import CNDXNativeModel, NativeConfig, build_native_model


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

PEOPLE = [
    "Sarah", "Mike", "John", "Lisa", "Dave", "Emma", "Chris", "Amy",
    "Tom", "Rachel", "Ben", "Maria", "Jake", "Nina", "Alex", "Kate",
]
TASKS = [
    "fix the login bug", "review the PR", "update the docs",
    "send the report", "call the vendor", "book the flight",
    "finish the slides", "order new monitors", "cancel the subscription",
    "reschedule the demo", "write the proposal", "clean up the repo",
    "test the deployment", "set up the new hire", "draft the email",
]
TIMES = [
    "2pm", "3:30", "tomorrow morning", "friday", "next week",
    "before lunch", "after the meeting", "end of day", "9am sharp",
    "monday at 10", "tonight", "by thursday",
]
PLACES = [
    "the office", "room 204", "downtown", "the cafe on 5th",
    "building B", "conference room", "the library", "home",
]
PROJECTS = [
    "Q3 budget", "website redesign", "API migration",
    "client onboarding", "security audit", "performance review",
    "product launch", "data cleanup",
]
FILLER_FRAGMENTS = [
    "... need to think about this more",
    "hmm not sure yet",
    "tbd check later",
    "low priority for now",
    "ask about this tomorrow",
    "pending response from team",
    "idk maybe next sprint",
    "look into it when free",
    "revisit after launch",
    "someone mentioned this in standup",
    "might be related to the other thing",
    "parking lot item",
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


def _make_filler(n):
    return [random.choice(FILLER_FRAGMENTS) for _ in range(n)]


# ---------------------------------------------------------------------------
# Test 1: HWM-NIAH — recover specific task/reminder from noise
# ---------------------------------------------------------------------------

def test_hwm_niah(model, tokenizer, device, amp_ctx, cfg, n_trials=30):
    print("\n══════════════════════════════════════════")
    print("  HWM-NIAH: Working Memory Needle Recovery")
    print("══════════════════════════════════════════")
    model.eval()

    results_by_pos = {}
    for position in ["start", "middle", "end"]:
        pos_results = []
        for _ in range(n_trials):
            person = random.choice(PEOPLE)
            task = random.choice(TASKS)
            time = random.choice(TIMES)
            needle = f"TODO: {person} needs to {task} by {time}"

            fillers = _make_filler(4)
            if position == "start":
                lines = [needle] + fillers
            elif position == "end":
                lines = fillers + [needle]
            else:
                lines = fillers[:2] + [needle] + fillers[2:]

            passage = "\n".join(lines)
            decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

            person_found = person.lower() in decoded.lower()
            task_found = task.lower() in decoded.lower()
            time_found = time.lower() in decoded.lower()
            full = person_found and task_found and time_found

            pos_results.append({
                "person": person_found, "task": task_found,
                "time": time_found, "full": full,
            })

        full_rate = sum(1 for r in pos_results if r["full"]) / len(pos_results)
        person_rate = sum(1 for r in pos_results if r["person"]) / len(pos_results)
        task_rate = sum(1 for r in pos_results if r["task"]) / len(pos_results)
        results_by_pos[position] = {
            "n_trials": n_trials,
            "person_rate": round(person_rate, 3),
            "task_rate": round(task_rate, 3),
            "full_recovery_rate": round(full_rate, 3),
        }
        print(f"  {position}: person={person_rate:.1%}  task={task_rate:.1%}  full={full_rate:.1%}")

    overall = sum(r["full_recovery_rate"] for r in results_by_pos.values()) / len(results_by_pos)
    print(f"  OVERALL: {overall:.1%}")
    return {"test": "HWM-NIAH", "results_by_position": results_by_pos, "overall": round(overall, 3)}


# ---------------------------------------------------------------------------
# Test 2: HWM-FACT1 — single-hop binding
# ---------------------------------------------------------------------------

def test_hwm_fact1(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    print("\n══════════════════════════════════════════")
    print("  HWM-FACT1: Person-Task Binding")
    print("══════════════════════════════════════════")
    model.eval()

    results = []
    for _ in range(n_trials):
        p1, p2 = random.sample(PEOPLE, 2)
        t1, t2 = random.sample(TASKS, 2)

        line1 = f"TODO: {p1} - {t1}"
        line2 = f"TODO: {p2} - {t2}"
        fillers = _make_filler(3)

        lines = [line1] + fillers[:2] + [line2] + fillers[2:]
        passage = "\n".join(lines)
        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

        dl = decoded.lower()
        p1_found = p1.lower() in dl
        p2_found = p2.lower() in dl
        t1_found = t1.lower() in dl
        t2_found = t2.lower() in dl
        bind1 = p1_found and t1_found
        bind2 = p2_found and t2_found

        results.append({"bind1": bind1, "bind2": bind2, "both": bind1 and bind2})

    both_rate = sum(1 for r in results if r["both"]) / len(results)
    b1_rate = sum(1 for r in results if r["bind1"]) / len(results)
    b2_rate = sum(1 for r in results if r["bind2"]) / len(results)
    print(f"  Binding 1: {b1_rate:.1%}  Binding 2: {b2_rate:.1%}  Both: {both_rate:.1%}")

    return {
        "test": "HWM-FACT1", "n_trials": n_trials,
        "binding1_rate": round(b1_rate, 3),
        "binding2_rate": round(b2_rate, 3),
        "both_binding_rate": round(both_rate, 3),
    }


# ---------------------------------------------------------------------------
# Test 3: HWM-FACT2 — two-hop (person->task->deadline)
# ---------------------------------------------------------------------------

def test_hwm_fact2(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    print("\n══════════════════════════════════════════")
    print("  HWM-FACT2: Two-Hop Person->Task->Deadline")
    print("══════════════════════════════════════════")
    model.eval()

    results = []
    for _ in range(n_trials):
        person = random.choice(PEOPLE)
        task = random.choice(TASKS)
        deadline = random.choice(TIMES)

        hop1 = f"NOTE: {person} is responsible for {task}"
        fillers = _make_filler(2)
        hop2 = f"DEADLINE: {task} due {deadline}"

        lines = [hop1] + fillers + [hop2] + _make_filler(1)
        passage = "\n".join(lines)
        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

        dl = decoded.lower()
        person_found = person.lower() in dl
        task_found = task.lower() in dl
        deadline_found = deadline.lower() in dl
        chain = person_found and task_found and deadline_found

        results.append({
            "person": person_found, "task": task_found,
            "deadline": deadline_found, "chain": chain,
        })

    person_rate = sum(1 for r in results if r["person"]) / len(results)
    task_rate = sum(1 for r in results if r["task"]) / len(results)
    deadline_rate = sum(1 for r in results if r["deadline"]) / len(results)
    chain_rate = sum(1 for r in results if r["chain"]) / len(results)
    print(f"  Person: {person_rate:.1%}  Task: {task_rate:.1%}  Deadline: {deadline_rate:.1%}")
    print(f"  Full chain: {chain_rate:.1%}")

    return {
        "test": "HWM-FACT2", "n_trials": n_trials,
        "person_rate": round(person_rate, 3),
        "task_rate": round(task_rate, 3),
        "deadline_rate": round(deadline_rate, 3),
        "chain_intact_rate": round(chain_rate, 3),
    }


# ---------------------------------------------------------------------------
# Test 4: HWM-TRACK — task status after updates
# ---------------------------------------------------------------------------

def test_hwm_track(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    print("\n══════════════════════════════════════════")
    print("  HWM-TRACK: Task Status Tracking")
    print("══════════════════════════════════════════")
    model.eval()

    statuses_pool = ["not started", "in progress", "blocked", "done",
                     "waiting", "deprioritized", "almost done"]

    results = []
    for _ in range(n_trials):
        task = random.choice(TASKS)
        n_updates = random.randint(3, 5)
        statuses = [random.choice(statuses_pool) for _ in range(n_updates)]
        final_status = statuses[-1]

        lines = []
        for i, s in enumerate(statuses):
            lines.append(f"status: {task} -> {s}")
            if random.random() < 0.4:
                lines.append(random.choice(FILLER_FRAGMENTS))

        passage = "\n".join(lines)
        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

        dl = decoded.lower()
        task_found = task.lower() in dl
        final_found = final_status.lower() in dl
        earlier = [s for s in statuses[:-1] if s.lower() in dl]

        results.append({
            "task_found": task_found,
            "final_status_found": final_found,
            "n_earlier_present": len(earlier),
        })

    task_rate = sum(1 for r in results if r["task_found"]) / len(results)
    final_rate = sum(1 for r in results if r["final_status_found"]) / len(results)
    print(f"  Task present: {task_rate:.1%}")
    print(f"  Final status recovered: {final_rate:.1%}")

    return {
        "test": "HWM-TRACK", "n_trials": n_trials,
        "task_rate": round(task_rate, 3),
        "final_status_rate": round(final_rate, 3),
    }


# ---------------------------------------------------------------------------
# Test 5: HWM-OVERRIDE — correction overwrites original
# ---------------------------------------------------------------------------

def test_hwm_override(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    print("\n══════════════════════════════════════════")
    print("  HWM-OVERRIDE: Correction Overwrite")
    print("══════════════════════════════════════════")
    model.eval()

    results = []
    for _ in range(n_trials):
        person = random.choice(PEOPLE)
        old_time = random.choice(TIMES[:6])
        new_time = random.choice(TIMES[6:])
        task = random.choice(TASKS)

        original = f"meeting with {person} at {old_time} re: {task}"
        fillers = _make_filler(2)
        correction = f"correction: meeting is {new_time} not {old_time}"

        lines = [original] + fillers + [correction] + _make_filler(1)
        passage = "\n".join(lines)
        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

        dl = decoded.lower()
        old_present = old_time.lower() in dl
        new_present = new_time.lower() in dl
        person_present = person.lower() in dl
        override_correct = new_present and person_present

        results.append({
            "old_present": old_present, "new_present": new_present,
            "person_present": person_present, "override_correct": override_correct,
        })

    new_rate = sum(1 for r in results if r["new_present"]) / len(results)
    old_rate = sum(1 for r in results if r["old_present"]) / len(results)
    override_rate = sum(1 for r in results if r["override_correct"]) / len(results)
    print(f"  New time present: {new_rate:.1%}  Old time: {old_rate:.1%}")
    print(f"  Override correct: {override_rate:.1%}")

    return {
        "test": "HWM-OVERRIDE", "n_trials": n_trials,
        "new_time_rate": round(new_rate, 3),
        "old_time_rate": round(old_rate, 3),
        "override_correct_rate": round(override_rate, 3),
    }


# ---------------------------------------------------------------------------
# Test 6: HWM-POS — positional sensitivity
# ---------------------------------------------------------------------------

def test_hwm_pos(model, tokenizer, device, amp_ctx, cfg, n_trials=20):
    print("\n══════════════════════════════════════════")
    print("  HWM-POS: Positional Sensitivity")
    print("══════════════════════════════════════════")
    model.eval()

    results_by_pos = {}
    for position in ["start", "middle", "end"]:
        pos_results = []
        for _ in range(n_trials):
            person = random.choice(PEOPLE)
            task = random.choice(TASKS)
            needle = f"URGENT: {person} must {task}"

            fillers = _make_filler(5)
            if position == "start":
                lines = [needle] + fillers
            elif position == "end":
                lines = fillers + [needle]
            else:
                lines = fillers[:3] + [needle] + fillers[3:]

            passage = "\n".join(lines)
            decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

            dl = decoded.lower()
            person_found = person.lower() in dl
            task_found = task.lower() in dl
            full = person_found and task_found

            pos_results.append({"person": person_found, "task": task_found, "full": full})

        full_rate = sum(1 for r in pos_results if r["full"]) / len(pos_results)
        person_rate = sum(1 for r in pos_results if r["person"]) / len(pos_results)
        results_by_pos[position] = {
            "n_trials": n_trials,
            "person_rate": round(person_rate, 3),
            "full_recovery_rate": round(full_rate, 3),
        }
        print(f"  {position}: person={person_rate:.1%}  full={full_rate:.1%}")

    return {"test": "HWM-POS", "results_by_position": results_by_pos}


# ---------------------------------------------------------------------------
# Test 7: HWM-EXACT — exact recovery of specific detail
# ---------------------------------------------------------------------------

def test_hwm_exact(model, tokenizer, device, amp_ctx, cfg, n_trials=40):
    print("\n══════════════════════════════════════════")
    print("  HWM-EXACT: Exact Detail Recovery")
    print("══════════════════════════════════════════")
    model.eval()

    results = []
    for _ in range(n_trials):
        detail_type = random.choice(["phone", "time_place", "budget"])

        if detail_type == "phone":
            person = random.choice(PEOPLE)
            phone = f"{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
            needle = f"{person}: {phone}"
            exact_target = phone
        elif detail_type == "time_place":
            person = random.choice(PEOPLE)
            time = random.choice(TIMES)
            place = random.choice(PLACES)
            needle = f"meet {person} at {place} {time}"
            exact_target = f"{place} {time}"
        else:
            project = random.choice(PROJECTS)
            amount = random.randint(10, 99)
            needle = f"{project} budget: ${amount}k"
            exact_target = f"${amount}k"

        fillers = _make_filler(4)
        lines = fillers[:2] + [needle] + fillers[2:]
        passage = "\n".join(lines)
        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)

        exact_match = exact_target.lower() in decoded.lower()
        words = exact_target.split()
        word_recall = sum(1 for w in words if w.lower() in decoded.lower()) / len(words)

        results.append({"exact_match": exact_match, "word_recall": round(word_recall, 3)})

    exact_rate = sum(1 for r in results if r["exact_match"]) / len(results)
    avg_recall = sum(r["word_recall"] for r in results) / len(results)
    print(f"  Exact match: {exact_rate:.1%}  Word recall: {avg_recall:.1%}")

    return {
        "test": "HWM-EXACT", "n_trials": n_trials,
        "exact_match_rate": round(exact_rate, 3),
        "avg_word_recall": round(avg_recall, 3),
    }


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

def test_hwm_control(model, tokenizer, device, amp_ctx, cfg, n_trials=30):
    print("\n══════════════════════════════════════════")
    print("  HWM-CONTROL: No-Compression Baseline")
    print("══════════════════════════════════════════")
    model.eval()

    raw_scores, comp_scores = [], []
    for _ in range(n_trials):
        person = random.choice(PEOPLE)
        task = random.choice(TASKS)
        needle = f"TODO: {person} - {task}"
        fillers = _make_filler(3)
        lines = fillers[:1] + [needle] + fillers[1:]
        passage = "\n".join(lines)

        raw = _tokenizer_roundtrip(tokenizer, passage, cfg.seq_len)
        raw_hit = person.lower() in raw.lower() and task.lower() in raw.lower()
        raw_scores.append(raw_hit)

        decoded = _encode_decode(model, tokenizer, passage, device, amp_ctx, cfg.seq_len)
        comp_hit = person.lower() in decoded.lower() and task.lower() in decoded.lower()
        comp_scores.append(comp_hit)

    raw_rate = sum(raw_scores) / n_trials
    comp_rate = sum(comp_scores) / n_trials
    print(f"  Raw: {raw_rate:.1%}  Compressed: {comp_rate:.1%}  Cost: {raw_rate - comp_rate:+.1%}")

    return {
        "test": "HWM-CONTROL", "n_trials": n_trials,
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
# Main
# ---------------------------------------------------------------------------

def run_hwm_benchmarks(args):
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
        all_results["control"] = test_hwm_control(model, tokenizer, device, amp_ctx, cfg)
    if "niah" in tests:
        all_results["niah"] = test_hwm_niah(model, tokenizer, device, amp_ctx, cfg)
    if "fact1" in tests:
        all_results["fact1"] = test_hwm_fact1(model, tokenizer, device, amp_ctx, cfg)
    if "fact2" in tests:
        all_results["fact2"] = test_hwm_fact2(model, tokenizer, device, amp_ctx, cfg)
    if "track" in tests:
        all_results["track"] = test_hwm_track(model, tokenizer, device, amp_ctx, cfg)
    if "override" in tests:
        all_results["override"] = test_hwm_override(model, tokenizer, device, amp_ctx, cfg)
    if "pos" in tests:
        all_results["pos"] = test_hwm_pos(model, tokenizer, device, amp_ctx, cfg)
    if "exact" in tests:
        all_results["exact"] = test_hwm_exact(model, tokenizer, device, amp_ctx, cfg)

    out_path = Path(args.checkpoint_dir) / "hwm_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n══════════════════════════════════════════")
    print(f"  HWM benchmark results saved to {out_path}")
    print(f"══════════════════════════════════════════")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CNDX Human Working-Memory Benchmark Suite")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--tests", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    run_hwm_benchmarks(args)
