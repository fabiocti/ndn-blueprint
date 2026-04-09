"""Adversarial evaluation battery for CNDX v2 native model.

Run post-training against a saved checkpoint to validate whether the model
is genuinely relying on sample-specific latent content, or exploiting
train/eval softness, near-duplicates, or autoregressive priors.

Tests implemented:
  1. Shuffled-latent loss  — wrong sample's latent → should collapse
  2. Partial corruption    — zero 25/50/75% of latent slots
  3. Cross-sample gen      — qualitative: decode sample A with latent from B
  4. Article overlap audit — check wikitext/wiki title intersection
  5. N-gram duplicate scan — high-overlap val paragraphs vs train set
  6. Per-layer ablation    — disable cross-attention at specific decoder layers
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cndx.native_model import CNDXNativeModel, NativeConfig, build_native_model
from cndx.data import build_dataloaders


def load_checkpoint(ckpt_dir, device="cuda"):
    """Load model + config from a results directory."""
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
        # Strip torch.compile _orig_mod. prefix if present
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        raise FileNotFoundError(f"No model.pt in {ckpt_dir}")

    model.eval()
    return model, tokenizer, cfg


# ──────────────────────── Test 1: Shuffled-latent ────────────────────────

@torch.no_grad()
def test_shuffled_latent(model, val_loader, device, amp_ctx, n_trials=5):
    """Average shuffled-latent loss over multiple random permutations."""
    print("\n═══ TEST 1: Shuffled-latent loss ═══")
    model.eval()
    results = []
    normal_total, n = 0.0, 0

    for trial in range(n_trials):
        total_shuf = 0.0
        batch_count = 0
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            if trial == 0:
                with amp_ctx:
                    loss, _, _ = model(ids, mask)
                normal_total += loss.item()
                n += 1
            with amp_ctx:
                shuf_loss = model.shuffled_latent_forward(ids, mask)
            total_shuf += shuf_loss.item()
            batch_count += 1
        results.append(total_shuf / max(batch_count, 1))

    normal_avg = normal_total / max(n, 1)
    shuf_avg = sum(results) / len(results)
    shuf_std = (sum((r - shuf_avg)**2 for r in results) / len(results)) ** 0.5

    print(f"  Normal val loss:       {normal_avg:.4f}")
    print(f"  Shuffled loss (mean):  {shuf_avg:.4f} ± {shuf_std:.4f}")
    print(f"  Shuffled gap:          {shuf_avg - normal_avg:.4f}")
    print(f"  Verdict: {'PASS — shuffled causes collapse' if shuf_avg > normal_avg * 3 else 'WARN — shuffled gap too small, model may be ignoring latent content'}")
    return {"normal_loss": normal_avg, "shuffled_loss_mean": shuf_avg,
            "shuffled_loss_std": shuf_std, "shuffled_gap": shuf_avg - normal_avg,
            "trials": results}


# ──────────────────────── Test 2: Corruption sweep ───────────────────────

@torch.no_grad()
def test_corruption_sweep(model, val_loader, device, amp_ctx):
    """Loss with 10/25/50/75/90/100% of latent slots zeroed."""
    print("\n═══ TEST 2: Partial corruption sweep ═══")
    model.eval()
    fracs = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]
    losses = {f: 0.0 for f in fracs}
    n = 0

    for batch in val_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        n += 1
        for frac in fracs:
            if frac == 0.0:
                with amp_ctx:
                    loss, _, _ = model(ids, mask)
                losses[0.0] += loss.item()
            elif frac == 1.0:
                with amp_ctx:
                    loss = model.ablation_forward(ids, mask)
                losses[1.0] += loss.item()
            else:
                with amp_ctx:
                    loss = model.partial_corrupt_forward(ids, mask, corrupt_frac=frac)
                losses[frac] += loss.item()

    print(f"  {'Corrupt%':>10}  {'Loss':>10}  {'Δ from 0%':>10}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*10}")
    baseline = losses[0.0] / max(n, 1)
    sweep = {}
    for frac in fracs:
        avg = losses[frac] / max(n, 1)
        sweep[f"corrupt_{int(frac*100)}"] = avg
        print(f"  {frac*100:>9.0f}%  {avg:>10.4f}  {avg - baseline:>+10.4f}")

    monotonic = all(sweep[f"corrupt_{int(fracs[i]*100)}"] <=
                    sweep[f"corrupt_{int(fracs[i+1]*100)}"] + 0.1
                    for i in range(len(fracs) - 1))
    print(f"  Monotonic degradation: {'YES ✓' if monotonic else 'NO — non-monotonic is suspicious'}")
    return sweep


# ──────────────────────── Test 3: Cross-sample generation ────────────────

@torch.no_grad()
def test_cross_sample_generation(model, val_loader, tokenizer, device, amp_ctx,
                                  cfg, n_samples=5):
    """Generate with the wrong latent. Should produce garbled or wrong-topic output."""
    print("\n═══ TEST 3: Cross-sample generation (wrong latent) ═══")
    model.eval()
    batch = next(iter(val_loader))
    ids = batch["input_ids"][:n_samples].to(device)
    mask = batch["attention_mask"][:n_samples].to(device)

    with amp_ctx:
        latent_state, _ = model.encode(ids, mask)

    originals = tokenizer.batch_decode(ids, skip_special_tokens=True)
    B = ids.shape[0]

    shifted_latent = torch.roll(latent_state, 1, dims=0)
    with amp_ctx:
        cur = ids.new_full((B, 1), model.bos_token_id)
        for _ in range(cfg.seq_len):
            logits = model.decode_logits(cur, shifted_latent)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
    cross_decoded = tokenizer.batch_decode(cur[:, 1:], skip_special_tokens=True)

    with amp_ctx:
        cur = ids.new_full((B, 1), model.bos_token_id)
        for _ in range(cfg.seq_len):
            logits = model.decode_logits(cur, latent_state)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
    normal_decoded = tokenizer.batch_decode(cur[:, 1:], skip_special_tokens=True)

    samples = []
    for i in range(B):
        print(f"\n  Sample {i}:")
        print(f"    ORIGINAL:   {originals[i][:120]}")
        print(f"    NORMAL_GEN: {normal_decoded[i][:120]}")
        print(f"    CROSS_GEN:  {cross_decoded[i][:120]}")
        print(f"    LATENT_SRC: (sample {(i-1)%B}) {originals[(i-1)%B][:80]}")
        samples.append({
            "original": originals[i].strip()[:200],
            "normal_gen": normal_decoded[i].strip()[:200],
            "cross_gen": cross_decoded[i].strip()[:200],
            "latent_source": originals[(i-1)%B].strip()[:200],
        })
    return samples


# ──────────────────────── Test 4: Article overlap audit ──────────────────

def test_article_overlap(cfg, max_articles=50000):
    """Check whether wikitext-103 articles overlap with wikipedia train articles."""
    print("\n═══ TEST 4: Article overlap audit ═══")
    from datasets import load_dataset

    print("  Loading wikitext-103 validation split ...")
    wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    wt_titles = set()
    for row in wt:
        line = row["text"].strip()
        if line.startswith("= ") and line.endswith(" =") and not line.startswith("= ="):
            title = line.strip("= ").strip()
            if title:
                wt_titles.add(title.lower())

    print(f"  Wikitext-103 val unique article titles: {len(wt_titles)}")

    print("  Loading wikipedia article titles (streaming) ...")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    wiki_titles = set()
    articles_scanned = 0
    min_chars = getattr(cfg, 'min_text_chars', 100)
    usable_paras = 0

    for article in wiki:
        articles_scanned += 1
        title = article.get("title", "").lower()
        wiki_titles.add(title)
        for p in article["text"].split("\n\n"):
            if len(p.strip()) >= min_chars:
                usable_paras += 1
        if usable_paras >= cfg.max_train_samples:
            break
        if articles_scanned >= max_articles:
            break

    print(f"  Wikipedia articles scanned (≈training window): {articles_scanned:,}")
    print(f"  Wikipedia unique titles: {len(wiki_titles):,}")

    overlap = wt_titles & wiki_titles
    print(f"\n  OVERLAP: {len(overlap)} wikitext articles also in wiki train window")
    print(f"  Overlap rate: {len(overlap)/max(len(wt_titles),1)*100:.1f}% of wikitext val")

    if overlap:
        print(f"  Sample overlapping titles: {list(overlap)[:10]}")

    return {
        "wikitext_titles": len(wt_titles),
        "wiki_train_titles": len(wiki_titles),
        "overlap_count": len(overlap),
        "overlap_pct": len(overlap) / max(len(wt_titles), 1) * 100,
        "sample_overlaps": list(overlap)[:20],
    }


# ──────────────────────── Test 5: N-gram duplicate scan ──────────────────

def test_ngram_duplicates(tokenizer, cfg, n=8, top_k=20):
    """Find val paragraphs with suspiciously high n-gram overlap against train."""
    print(f"\n═══ TEST 5: {n}-gram duplicate scan ═══")
    from cndx.data import build_dataloaders as _build

    train_loader, val_loader = _build(tokenizer, cfg)

    print(f"  Building {n}-gram index from train set ...")
    train_ngrams = Counter()
    for batch in train_loader:
        for ids in batch["input_ids"]:
            tokens = ids[ids != tokenizer.pad_token_id].tolist()
            for i in range(len(tokens) - n + 1):
                train_ngrams[tuple(tokens[i:i+n])] += 1

    print(f"  Unique {n}-grams in train: {len(train_ngrams):,}")
    print(f"  Scanning val set for overlaps ...")

    val_overlaps = []
    for batch in val_loader:
        for idx, ids in enumerate(batch["input_ids"]):
            tokens = ids[ids != tokenizer.pad_token_id].tolist()
            total_ng = max(len(tokens) - n + 1, 1)
            hits = sum(1 for i in range(len(tokens) - n + 1)
                      if tuple(tokens[i:i+n]) in train_ngrams)
            overlap_pct = hits / total_ng * 100
            if overlap_pct > 10:
                text = tokenizer.decode(ids[ids != tokenizer.pad_token_id],
                                       skip_special_tokens=True)
                val_overlaps.append({
                    "text_preview": text[:120],
                    "overlap_pct": round(overlap_pct, 1),
                    "hits": hits,
                    "total_ngrams": total_ng,
                })

    val_overlaps.sort(key=lambda x: x["overlap_pct"], reverse=True)
    print(f"  Val paragraphs with >{10}% {n}-gram overlap: {len(val_overlaps)}")

    if val_overlaps:
        print(f"\n  Top {min(top_k, len(val_overlaps))} most suspicious:")
        for i, v in enumerate(val_overlaps[:top_k]):
            print(f"    [{v['overlap_pct']:5.1f}%] {v['text_preview'][:100]}")

    return {
        "ngram_size": n,
        "train_unique_ngrams": len(train_ngrams),
        "val_high_overlap_count": len(val_overlaps),
        "worst_overlaps": val_overlaps[:top_k],
    }


# ──────────────────────── Test 6: Per-layer cross-attn ablation ──────────

@torch.no_grad()
def test_layer_ablation(model, val_loader, device, amp_ctx):
    """Disable cross-attention at individual decoder layers to find which matter most."""
    print("\n═══ TEST 6: Per-layer cross-attention ablation ═══")
    model.eval()
    n_layers = len(model.decoder.layers)
    results = {}

    normal_loss = 0.0
    n = 0
    for batch in val_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with amp_ctx:
            loss, _, _ = model(ids, mask)
        normal_loss += loss.item()
        n += 1
    normal_loss /= max(n, 1)
    results["normal"] = normal_loss

    for layer_idx in range(n_layers):
        original_forward = model.decoder.layers[layer_idx].ca.forward

        def _zeroed_cross_attn(q, kv, **kwargs):
            return torch.zeros_like(q)

        model.decoder.layers[layer_idx].ca.forward = _zeroed_cross_attn

        layer_loss = 0.0
        n = 0
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            with amp_ctx:
                loss, _, _ = model(ids, mask)
            layer_loss += loss.item()
            n += 1
        layer_loss /= max(n, 1)
        results[f"layer_{layer_idx}_disabled"] = layer_loss

        model.decoder.layers[layer_idx].ca.forward = original_forward

    print(f"  {'Config':>20}  {'Loss':>10}  {'Δ':>10}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*10}")
    for k, v in results.items():
        delta = v - normal_loss
        print(f"  {k:>20}  {v:>10.4f}  {delta:>+10.4f}")

    return results


# ──────────────────────── Main orchestrator ──────────────────────────────

def run_adversarial_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    amp_ctx = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else torch.autocast("cpu", enabled=False)

    print(f"Device: {device}, AMP dtype: {dtype}")

    model, tokenizer, cfg = load_checkpoint(args.checkpoint_dir, device=device)
    print(f"Model: {model.num_params()/1e6:.1f}M params")

    _, val_loader = build_dataloaders(tokenizer, cfg)
    print(f"Val batches: {len(val_loader)}")

    all_results = {"checkpoint": args.checkpoint_dir, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    tests = args.tests.split(",") if args.tests else ["shuffle", "corrupt", "cross", "overlap", "ngram", "layer"]

    if "shuffle" in tests:
        all_results["shuffled_latent"] = test_shuffled_latent(model, val_loader, device, amp_ctx)

    if "corrupt" in tests:
        all_results["corruption_sweep"] = test_corruption_sweep(model, val_loader, device, amp_ctx)

    if "cross" in tests:
        all_results["cross_sample"] = test_cross_sample_generation(
            model, val_loader, tokenizer, device, amp_ctx, cfg)

    if "overlap" in tests:
        all_results["article_overlap"] = test_article_overlap(cfg)

    if "ngram" in tests:
        all_results["ngram_duplicates"] = test_ngram_duplicates(tokenizer, cfg)

    if "layer" in tests:
        all_results["layer_ablation"] = test_layer_ablation(model, val_loader, device, amp_ctx)

    out_path = Path(args.checkpoint_dir) / "adversarial_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n═══ All results saved to {out_path} ═══")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial eval battery for CNDX v2")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing model.pt and results.json")
    parser.add_argument("--tests", type=str, default=None,
                        help="Comma-separated tests to run: shuffle,corrupt,cross,overlap,ngram,layer")
    run_adversarial_eval(parser.parse_args())
