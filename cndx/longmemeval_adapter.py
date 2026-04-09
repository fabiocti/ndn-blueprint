"""CNDX LongMemEval Adapter — test the latent memory substrate on a public
conversational memory benchmark.

Pipeline (compress-then-read):
  1. Load LongMemEval questions + conversation histories (oracle: evidence-only)
  2. Chunk each history into 128-token segments
  3. Encode each chunk through CNDX → latent → decode back
  4. Reassemble decoded chunks as reconstructed history
  5. Feed [reconstructed history + question] to a reader LLM
  6. Collect answers in LongMemEval format
  7. Also run a raw-text baseline (skip step 3) for comparison
  8. Auto-score with token-F1, exact-match, keyword-recall

Usage:
  python -m cndx.longmemeval_adapter \
    --cndx_checkpoint native_K64_S128_hwm_s64 \
    --data_path data/longmemeval_oracle.json \
    --reader_model Qwen/Qwen2.5-7B-Instruct \
    --output_dir results/longmemeval/ \
    [--baseline_only] [--cndx_only] [--max_questions 50]
"""

import argparse
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from cndx.native_model import NativeConfig, build_native_model

# ── Scoring ──────────────────────────────────────────────────────────

def _normalize(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def token_f1(prediction, gold):
    pred_toks = _normalize(prediction).split()
    gold_toks = _normalize(gold).split()
    if not gold_toks:
        return 1.0 if not pred_toks else 0.0
    if not pred_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(pred_toks)
    rec = num_same / len(gold_toks)
    return 2 * prec * rec / (prec + rec)


def exact_match(prediction, gold):
    return float(_normalize(prediction) == _normalize(gold))


def keyword_recall(prediction, gold):
    gold_words = set(_normalize(gold).split())
    if not gold_words:
        return 1.0
    pred_words = set(_normalize(prediction).split())
    return len(gold_words & pred_words) / len(gold_words)


def containment(prediction, gold):
    """Check if the normalized gold answer appears as a substring of prediction."""
    return float(_normalize(gold) in _normalize(prediction))


def score_answers(results):
    """Add per-question scores and return aggregate stats."""
    metrics = ["f1", "em", "kw", "cont"]
    agg = {m: [] for m in metrics}
    by_type = {}
    for r in results:
        f1 = token_f1(r["predicted_answer"], r["gold_answer"])
        em = exact_match(r["predicted_answer"], r["gold_answer"])
        kw = keyword_recall(r["predicted_answer"], r["gold_answer"])
        ct = containment(r["predicted_answer"], r["gold_answer"])
        r["score_f1"] = round(f1, 4)
        r["score_em"] = round(em, 4)
        r["score_kw"] = round(kw, 4)
        r["score_cont"] = round(ct, 4)
        for m, v in zip(metrics, [f1, em, kw, ct]):
            agg[m].append(v)
        qt = r.get("question_type", "unknown")
        if qt not in by_type:
            by_type[qt] = {m: [] for m in metrics}
        for m, v in zip(metrics, [f1, em, kw, ct]):
            by_type[qt][m].append(v)

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0

    summary = {
        "overall": {"n": len(results),
                     **{m: _mean(agg[m]) for m in metrics}},
        "by_type": {},
    }
    for qt, vals in sorted(by_type.items()):
        n = len(vals["f1"])
        summary["by_type"][qt] = {"n": n, **{m: _mean(vals[m]) for m in metrics}}
    return summary


# ── Model Loading ────────────────────────────────────────────────────

def load_cndx_model(ckpt_dir, device="cuda"):
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
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"[CNDX] Loaded {ckpt_path} — K={cfg.num_latents}, seq={cfg.seq_len}")
    return model, tokenizer, cfg


# ── Text Processing ──────────────────────────────────────────────────

def flatten_sessions(sessions, session_dates=None):
    lines = []
    for i, session in enumerate(sessions):
        if session_dates and i < len(session_dates):
            lines.append(f"[Session — {session_dates[i]}]")
        else:
            lines.append(f"[Session {i+1}]")
        for turn in session:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("")
    return "\n".join(lines)


def chunk_text(text, tokenizer, chunk_size=128):
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(ids), chunk_size):
        chunks.append(ids[i:i + chunk_size])
    return chunks


def cndx_roundtrip(text, cndx_model, cndx_tokenizer, cfg, device, amp_ctx,
                    batch_size=16):
    """Encode text through CNDX and decode back, batched for speed."""
    chunks = chunk_text(text, cndx_tokenizer, chunk_size=cfg.seq_len)
    decoded_parts = []

    for b_start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[b_start:b_start + batch_size]
        padded_ids, attn_masks = [], []
        for chunk_ids in batch_chunks:
            pad_len = cfg.seq_len - len(chunk_ids)
            padded_ids.append(chunk_ids + [cndx_tokenizer.pad_token_id or 0] * pad_len)
            attn_masks.append([1] * len(chunk_ids) + [0] * pad_len)

        input_ids = torch.tensor(padded_ids, device=device)
        attn_mask = torch.tensor(attn_masks, device=device)

        with amp_ctx:
            gen = cndx_model.generate(input_ids, attn_mask, max_new_tokens=cfg.seq_len)
        for g in gen:
            decoded_parts.append(
                cndx_tokenizer.decode(g, skip_special_tokens=True))

    return " ".join(decoded_parts)


def build_qa_prompt(context, question, question_date=None):
    date_hint = f" (asked on {question_date})" if question_date else ""
    return (
        f"You are a helpful assistant with access to a user's conversation history. "
        f"Answer the question based ONLY on the conversation history provided.\n\n"
        f"=== CONVERSATION HISTORY ===\n{context}\n"
        f"=== END HISTORY ===\n\n"
        f"Question{date_hint}: {question}\n\n"
        f"Give a SHORT, DIRECT answer (just the key fact, name, date, or phrase). "
        f"Do NOT explain or elaborate. If the answer is not in the history, "
        f"reply exactly: \"Unknown\".\n\n"
        f"Answer:"
    )


def run_reader(prompt, reader_model, reader_tokenizer, device, max_new_tokens=256,
               reader_max_len=24576):
    inputs = reader_tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=reader_max_len).to(device)
    with torch.no_grad():
        outputs = reader_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=reader_tokenizer.eos_token_id,
        )
    answer_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return reader_tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


# ── Main ─────────────────────────────────────────────────────────────

def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float32)
    amp_ctx = (torch.autocast("cuda", dtype=dtype) if device.type == "cuda"
               else torch.autocast("cpu", enabled=False))

    with open(args.data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions from {args.data_path}")

    if args.max_questions:
        data = data[:args.max_questions]
        print(f"  Limiting to {args.max_questions} questions")

    cndx_model, cndx_tokenizer, cfg = None, None, None
    if not args.baseline_only and args.cndx_checkpoint:
        cndx_model, cndx_tokenizer, cfg = load_cndx_model(args.cndx_checkpoint, device)

    print(f"Loading reader model: {args.reader_model}")
    reader_tokenizer = AutoTokenizer.from_pretrained(args.reader_model)
    if reader_tokenizer.pad_token is None:
        reader_tokenizer.pad_token = reader_tokenizer.eos_token
    reader_model = AutoModelForCausalLM.from_pretrained(
        args.reader_model, torch_dtype=dtype, device_map="auto",
    )
    reader_model.eval()
    print(f"  Reader loaded: {sum(p.numel() for p in reader_model.parameters())/1e9:.1f}B params")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_baseline = []
    results_cndx = []
    t_start = time.time()

    for qi, item in enumerate(data):
        qid = item.get("question_id", f"q{qi}")
        question = item["question"]
        question_date = item.get("question_date", None)
        gold_answer = item.get("answer", "")
        qtype = item.get("question_type", "unknown")

        sessions = item.get("haystack_sessions", [])
        session_dates = item.get("haystack_dates", [])
        raw_context = flatten_sessions(sessions, session_dates)

        token_count = len(reader_tokenizer.encode(raw_context))
        elapsed = time.time() - t_start
        rate = (qi / elapsed * 60) if elapsed > 0 and qi > 0 else 0
        print(f"\n[{qi+1}/{len(data)}] {qid} ({qtype}) — "
              f"{token_count} toks — {rate:.1f} q/min")

        if not args.cndx_only:
            prompt_raw = build_qa_prompt(raw_context, question, question_date)
            t0 = time.time()
            answer_raw = run_reader(prompt_raw, reader_model, reader_tokenizer,
                                     device, reader_max_len=args.reader_max_len)
            t_raw = time.time() - t0
            results_baseline.append({
                "question_id": qid,
                "question_type": qtype,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": answer_raw,
                "context_tokens": token_count,
                "time_s": round(t_raw, 2),
            })
            print(f"  BASE: {answer_raw[:120]}")

        if cndx_model and not args.baseline_only:
            t0 = time.time()
            compressed_context = cndx_roundtrip(
                raw_context, cndx_model, cndx_tokenizer, cfg, device, amp_ctx)
            t_compress = time.time() - t0

            prompt_cndx = build_qa_prompt(compressed_context, question, question_date)
            t0 = time.time()
            answer_cndx = run_reader(prompt_cndx, reader_model, reader_tokenizer,
                                      device, reader_max_len=args.reader_max_len)
            t_read = time.time() - t0
            results_cndx.append({
                "question_id": qid,
                "question_type": qtype,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": answer_cndx,
                "context_tokens": token_count,
                "compress_time_s": round(t_compress, 2),
                "read_time_s": round(t_read, 2),
            })
            print(f"  CNDX: {answer_cndx[:120]}")

    # ── Scoring ──────────────────────────────────────────────────────
    def _print_scores(label, scores):
        print(f"\n{'═'*70}")
        print(f"  {label} ({scores['overall']['n']} questions)")
        print(f"{'═'*70}")
        o = scores["overall"]
        print(f"  Overall — F1: {o['f1']:.4f}  EM: {o['em']:.4f}  "
              f"KW: {o['kw']:.4f}  Cont: {o['cont']:.4f}")
        for qt, s in scores["by_type"].items():
            print(f"  {qt:30s} F1={s['f1']:.3f} EM={s['em']:.3f} "
                  f"KW={s['kw']:.3f} Cont={s['cont']:.3f}  (n={s['n']})")

    scores_base, scores_cndx = None, None
    if results_baseline:
        scores_base = score_answers(results_baseline)
        out_path = output_dir / "baseline_answers.json"
        with open(out_path, "w") as f:
            json.dump({"scores": scores_base, "answers": results_baseline}, f, indent=2)
        _print_scores("BASELINE", scores_base)

    if results_cndx:
        scores_cndx = score_answers(results_cndx)
        out_path = output_dir / "cndx_answers.json"
        with open(out_path, "w") as f:
            json.dump({"scores": scores_cndx, "answers": results_cndx}, f, indent=2)
        _print_scores("CNDX", scores_cndx)

    if scores_base and scores_cndx:
        sb, sc = scores_base["overall"], scores_cndx["overall"]
        print(f"\n{'═'*70}")
        print(f"  DELTA (CNDX − Baseline)")
        print(f"{'═'*70}")
        print(f"  Overall — dF1: {sc['f1']-sb['f1']:+.4f}  "
              f"dEM: {sc['em']-sb['em']:+.4f}  "
              f"dKW: {sc['kw']-sb['kw']:+.4f}  "
              f"dCont: {sc['cont']-sb['cont']:+.4f}")
        for qt in scores_base["by_type"]:
            if qt in scores_cndx["by_type"]:
                sb_t, sc_t = scores_base["by_type"][qt], scores_cndx["by_type"][qt]
                print(f"  {qt:30s} dF1={sc_t['f1']-sb_t['f1']:+.3f} "
                      f"dKW={sc_t['kw']-sb_t['kw']:+.3f} "
                      f"dCont={sc_t['cont']-sb_t['cont']:+.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNDX LongMemEval Adapter")
    parser.add_argument("--cndx_checkpoint", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--reader_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="results/longmemeval")
    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--cndx_only", action="store_true")
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--reader_max_len", type=int, default=24576)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    run_evaluation(args)
