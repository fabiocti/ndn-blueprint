"""
CNDX v2 — Training loop for the latent-native encoder-decoder.

Usage:
    python -m cndx.native_train
    python -m cndx.native_train --num_latents 64 --num_epochs 10 --max_train_samples 200000
"""

import argparse
import json
import math
import os
import random
import time

import torch
import numpy as np
from tqdm import tqdm

from cndx.native_model import NativeConfig, build_native_model
from cndx.data import build_dataloaders


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _lr_lambda(warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


@torch.no_grad()
def evaluate(model, val_loader, tokenizer, cfg, device, amp_ctx, digit_ids=None):
    model.eval()

    total_loss, total_abl_loss, total_shuf_loss = 0.0, 0.0, 0.0
    total_corrupt25, total_corrupt50, total_corrupt75 = 0.0, 0.0, 0.0
    correct, total_tok, n = 0, 0, 0
    digit_correct, digit_total = 0, 0

    for batch in val_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        with amp_ctx:
            loss, logits, _ = model(ids, mask)
        total_loss += loss.item()

        with amp_ctx:
            abl_loss = model.ablation_forward(ids, mask)
        total_abl_loss += abl_loss.item()

        with amp_ctx:
            shuf_loss = model.shuffled_latent_forward(ids, mask)
        total_shuf_loss += shuf_loss.item()

        with amp_ctx:
            c25 = model.partial_corrupt_forward(ids, mask, corrupt_frac=0.25)
            c50 = model.partial_corrupt_forward(ids, mask, corrupt_frac=0.50)
            c75 = model.partial_corrupt_forward(ids, mask, corrupt_frac=0.75)
        total_corrupt25 += c25.item()
        total_corrupt50 += c50.item()
        total_corrupt75 += c75.item()

        pred_first = logits[:, 0, :].argmax(dim=-1)
        correct += (pred_first == ids[:, 0]).sum().item()
        total_tok += ids.shape[0]

        if digit_ids:
            preds = logits.argmax(dim=-1)
            target_ids = ids
            digit_mask = torch.zeros_like(target_ids, dtype=torch.bool)
            for d_id in digit_ids:
                digit_mask |= (target_ids == d_id)
            if digit_mask.any():
                digit_correct += (preds[digit_mask] == target_ids[digit_mask]).sum().item()
                digit_total += digit_mask.sum().item()

        n += 1

    avg_loss = total_loss / max(n, 1)
    avg_abl = total_abl_loss / max(n, 1)
    avg_shuf = total_shuf_loss / max(n, 1)
    avg_c25 = total_corrupt25 / max(n, 1)
    avg_c50 = total_corrupt50 / max(n, 1)
    avg_c75 = total_corrupt75 / max(n, 1)
    first_tok_acc = correct / max(total_tok, 1)
    numeric_tok_acc = digit_correct / max(digit_total, 1)

    # Normal reconstruction samples
    batch = next(iter(val_loader))
    ids = batch["input_ids"][:cfg.num_generate_samples].to(device)
    mask = batch["attention_mask"][:cfg.num_generate_samples].to(device)
    with amp_ctx:
        gen = model.generate(ids, mask, max_new_tokens=cfg.seq_len)

    originals = tokenizer.batch_decode(ids, skip_special_tokens=True)
    reconstructed = tokenizer.batch_decode(gen, skip_special_tokens=True)
    exact = sum(1 for o, r in zip(originals, reconstructed)
                if o.strip() == r.strip())

    samples = [{"original": o.strip(), "reconstructed": r.strip()}
               for o, r in zip(originals, reconstructed)]

    # Cross-sample generation: encode sample i, decode with latent from sample (i+1)%B
    with amp_ctx:
        latent_state, _ = model.encode(ids, mask)
        B_s = latent_state.shape[0]
        if B_s >= 2:
            shifted_latent = torch.roll(latent_state, 1, dims=0)
            cur = ids.new_full((B_s, 1), model.bos_token_id)
            for _ in range(cfg.seq_len):
                logits_cs = model.decode_logits(cur, shifted_latent)
                nxt = logits_cs[:, -1, :].argmax(dim=-1, keepdim=True)
                cur = torch.cat([cur, nxt], dim=1)
            cross_gen = cur[:, 1:]
            cross_decoded = tokenizer.batch_decode(cross_gen, skip_special_tokens=True)
            cross_samples = [
                {"original": o.strip(),
                 "latent_from": originals[(i - 1) % B_s].strip()[:80],
                 "cross_reconstructed": c.strip()}
                for i, (o, c) in enumerate(zip(originals, cross_decoded))]
        else:
            cross_samples = []

    return {
        "val_loss": avg_loss,
        "ablation_loss": avg_abl,
        "ablation_gap": avg_abl - avg_loss,
        "shuffled_loss": avg_shuf,
        "shuffled_gap": avg_shuf - avg_loss,
        "corrupt_25_loss": avg_c25,
        "corrupt_50_loss": avg_c50,
        "corrupt_75_loss": avg_c75,
        "first_token_accuracy": first_tok_acc,
        "numeric_token_acc": numeric_tok_acc,
        "exact_match": exact / max(len(originals), 1),
        "samples": samples,
        "cross_samples": cross_samples,
    }


def run_native_train(cfg: NativeConfig, tag: str = ""):
    if cfg.seed >= 0:
        _set_seed(cfg.seed)
        print(f"Seed: {cfg.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 tensor cores: enabled")
        torch.backends.cuda.enable_cudnn_sdp(False)
        print("cuDNN SDP: disabled (Blackwell compat)")

    amp_ctx = (torch.amp.autocast('cuda', dtype=torch.bfloat16)
               if cfg.use_bf16 and device.type == "cuda"
               else torch.amp.autocast('cuda', enabled=False))
    if cfg.use_bf16 and device.type == "cuda":
        print("Mixed precision: bf16")

    print(f"\n{'='*60}")
    print("CNDX v2 — Latent-Native Encoder-Decoder")
    print(f"{'='*60}")
    print(f"Tokenizer: {cfg.tokenizer_name}")
    print(f"d_model={cfg.d_model}  d_ff={cfg.d_ff}  heads={cfg.n_heads}")
    print(f"Encoder: {cfg.enc_token_layers} token + {cfg.enc_cross_layers} "
          f"cross + {cfg.enc_refine_layers} refine")
    print(f"Decoder: {cfg.dec_layers} layers (self-attn + cross-attn + FFN)")
    groups_str = (f"  groups={cfg.latent_groups}" if cfg.is_structured
                  else "  flat")
    print(f"Latents: K={cfg.num_latents}{groups_str}  seq_len={cfg.seq_len}  "
          f"compression={cfg.seq_len / cfg.num_latents:.1f}x")
    print(f"LR={cfg.learning_rate}  warmup={cfg.warmup_ratio}  "
          f"wd={cfg.weight_decay}  clip={cfg.max_grad_norm}")
    print(f"Contrastive weight: {cfg.contrastive_weight}")
    print(f"Epochs: {cfg.num_epochs}  |  Train: {cfg.max_train_samples}  |  "
          f"Val: {cfg.max_eval_samples}")

    model, tokenizer = build_native_model(cfg, device)
    total_p = model.num_params()
    unique_p = total_p - model.embedding.weight.numel()  # shared weight counted once
    print(f"\nTotal params: {total_p:,}  (unique: {unique_p + model.embedding.weight.numel():,})")
    print(f"All trainable: {model.num_trainable():,}")

    # Build token weight vector for digit-aware + entity-aware loss
    digit_ids = set()
    entity_ids = set()

    for tok_id in range(tokenizer.vocab_size):
        tok_str = tokenizer.decode([tok_id])
        if any(c.isdigit() for c in tok_str):
            digit_ids.add(tok_id)

    # Entity tokens: encode known entity strings, collect constituent token IDs.
    # This handles sub-word tokenizers correctly (e.g. ".com" -> [".", "com"]).
    _entity_seeds = [
        # TLDs and domain fragments
        ".com", ".net", ".org", ".io", ".dev", ".app", ".cloud", ".tech",
        ".ai", ".co", ".uk", ".de", ".fr", ".jp", ".br", ".au", ".eu",
        ".biz", ".info", ".xyz", ".me", ".pro", ".in", ".sg", ".hk",
        # Common entity-bearing strings
        "CVE-2024", "CVE-2023", "CVE-2025", "CVE-2022", "CVE-2021",
        "REPAIR_NEEDED", "COMPLETED", "IN PROGRESS", "BLOCKED", "SKIPPED",
        "502 Bad Gateway", "403 Forbidden", "Connection timed out",
        "Rate limited", "DNS resolution failed", "TLS handshake",
        # Real A/B test entities that were missed
        "bostonacoustics", "xvtest", "jbl", "vfsglobal", "dailymotion",
        "expressvpn", "harman", "pinelabs",
        # Protocol/URL markers
        "https://", "http://", "CNAME", "A record", "AAAA",
        # Common company name tokens
        "google", "microsoft", "cloudflare", "coinbase", "shopify",
        "stripe", "paypal", "nordvpn", "surfshark", "protonmail",
        "tesla", "amazon", "netflix", "github", "gitlab",
    ]
    for seed in _entity_seeds:
        tids = tokenizer.encode(seed, add_special_tokens=False)
        entity_ids.update(tids)
    # Remove very common tokens (space, newline, period, comma) to avoid
    # upweighting tokens that appear everywhere
    for common_str in [" ", "\n", ".", ",", "-", ":", "#", "|", "(", ")", "[", "]"]:
        for cid in tokenizer.encode(common_str, add_special_tokens=False):
            entity_ids.discard(cid)
    entity_ids -= digit_ids  # digits already weighted separately

    print(f"Digit tokens identified: {len(digit_ids)}")
    print(f"Entity tokens identified: {len(entity_ids)}")

    need_weights = cfg.digit_weight > 1.0 or cfg.entity_weight > 1.0
    if need_weights:
        tw = torch.ones(tokenizer.vocab_size, device=device)
        if cfg.digit_weight > 1.0:
            for d in digit_ids:
                tw[d] = max(tw[d], cfg.digit_weight)
            print(f"Numeric token weight: {cfg.digit_weight}x on {len(digit_ids)} digit tokens")
        if cfg.entity_weight > 1.0:
            for e in entity_ids:
                tw[e] = max(tw[e], cfg.entity_weight)
            print(f"Entity token weight: {cfg.entity_weight}x on {len(entity_ids)} entity tokens")
        model.token_weights = tw

    if device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: FAILED ({e}), running eager")

    print("\nLoading dataset...")
    train_loader, val_loader = build_dataloaders(tokenizer, cfg)
    print(f"Train: {len(train_loader)} batches  |  Val: {len(val_loader)} batches")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(warmup_steps, total_steps))

    all_epochs = []

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        for step_in_epoch, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            with amp_ctx:
                loss, _, _ = model(ids, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            if (step_in_epoch + 1) % cfg.log_every == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_avg = epoch_loss / len(train_loader)
        train_time = time.time() - t0
        print(f"\nEpoch {epoch+1} - train_loss={train_avg:.4f}  time={train_time:.1f}s")

        print(f"Epoch {epoch+1} - evaluating...")
        ev = evaluate(model, val_loader, tokenizer, cfg, device, amp_ctx,
                      digit_ids=digit_ids)

        print(f"  val_loss={ev['val_loss']:.4f}  "
              f"ablation_loss={ev['ablation_loss']:.4f}  "
              f"gap={ev['ablation_gap']:.4f}  "
              f"1st_tok={ev['first_token_accuracy']:.4f}  "
              f"exact={ev['exact_match']:.4f}  "
              f"num_tok={ev['numeric_token_acc']:.4f}")
        print(f"  ADVERSARIAL >> shuffled={ev['shuffled_loss']:.4f} "
              f"(gap={ev['shuffled_gap']:.4f})  "
              f"corrupt25={ev['corrupt_25_loss']:.4f}  "
              f"corrupt50={ev['corrupt_50_loss']:.4f}  "
              f"corrupt75={ev['corrupt_75_loss']:.4f}")

        for s in ev.get("samples", [])[:3]:
            print(f"  ORIG: {s['original'][:100]}")
            print(f"  RECO: {s['reconstructed'][:100]}")
            print()

        if ev.get("cross_samples"):
            print("  --- CROSS-SAMPLE (wrong latent) ---")
            for cs in ev["cross_samples"][:3]:
                print(f"  TARGET:     {cs['original'][:100]}")
                print(f"  LATENT_SRC: {cs['latent_from'][:80]}")
                print(f"  CROSS_RECO: {cs['cross_reconstructed'][:100]}")
                print()

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(train_avg, 4),
            "val_loss": round(ev["val_loss"], 4),
            "ablation_loss": round(ev["ablation_loss"], 4),
            "ablation_gap": round(ev["ablation_gap"], 4),
            "shuffled_loss": round(ev["shuffled_loss"], 4),
            "shuffled_gap": round(ev["shuffled_gap"], 4),
            "corrupt_25": round(ev["corrupt_25_loss"], 4),
            "corrupt_50": round(ev["corrupt_50_loss"], 4),
            "corrupt_75": round(ev["corrupt_75_loss"], 4),
            "first_token_acc": round(ev["first_token_accuracy"], 4),
            "numeric_token_acc": round(ev["numeric_token_acc"], 4),
            "exact_match": round(ev["exact_match"], 4),
            "train_time_s": round(train_time, 1),
            "samples": ev.get("samples", []),
            "cross_samples": ev.get("cross_samples", []),
        }
        all_epochs.append(epoch_record)

    # Summary
    print(f"\n{'='*60}")
    print("EPOCH SUMMARY")
    print(f"{'='*60}")
    hdr = (f"{'Ep':>3}  {'Train':>8}  {'Val':>8}  {'Ablat':>8}  "
           f"{'Gap':>6}  {'Shuf':>8}  {'ShGap':>6}  "
           f"{'C25':>7}  {'C50':>7}  {'C75':>7}  "
           f"{'1stTok':>7}  {'NumTok':>7}  {'Exact':>6}  {'Time':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in all_epochs:
        print(f"{r['epoch']:3d}  {r['train_loss']:8.4f}  {r['val_loss']:8.4f}  "
              f"{r['ablation_loss']:8.4f}  {r['ablation_gap']:6.3f}  "
              f"{r.get('shuffled_loss',0):8.4f}  {r.get('shuffled_gap',0):6.3f}  "
              f"{r.get('corrupt_25',0):7.4f}  {r.get('corrupt_50',0):7.4f}  "
              f"{r.get('corrupt_75',0):7.4f}  "
              f"{r['first_token_acc']:7.4f}  {r.get('numeric_token_acc',0):7.4f}  "
              f"{r['exact_match']:6.4f}  "
              f"{r['train_time_s']:6.1f}")

    best_loss = min(all_epochs, key=lambda r: r['val_loss'])
    best_acc = max(all_epochs, key=lambda r: r['first_token_acc'])
    print(f"\nBest val_loss:  ep {best_loss['epoch']}  "
          f"val={best_loss['val_loss']:.4f}  gap={best_loss['ablation_gap']:.3f}")
    print(f"Best 1st-tok:   ep {best_acc['epoch']}  "
          f"val={best_acc['val_loss']:.4f}  1st={best_acc['first_token_acc']:.4f}")

    suffix = f"_{tag}" if tag else ""
    out_dir = f"native_K{cfg.num_latents}_S{cfg.seq_len}{suffix}"
    os.makedirs(out_dir, exist_ok=True)

    results_path = os.path.join(out_dir, "results.json")
    payload = {
        "config": {k: v for k, v in cfg.__dict__.items()},
        "epochs": all_epochs,
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    model_path = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model checkpoint saved to {model_path}")


def main():
    p = argparse.ArgumentParser(description="CNDX v2 native training")
    p.add_argument("--tokenizer_name", type=str,
                   default="HuggingFaceTB/SmolLM2-360M")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--enc_token_layers", type=int, default=2)
    p.add_argument("--enc_cross_layers", type=int, default=2)
    p.add_argument("--enc_refine_layers", type=int, default=2)
    p.add_argument("--num_latents", type=int, default=64)
    p.add_argument("--dec_layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--contrastive_weight", type=float, default=0.1)
    p.add_argument("--dataset_name", type=str, default="wikipedia")
    p.add_argument("--eval_dataset_name", type=str, default="wikitext")
    p.add_argument("--max_train_samples", type=int, default=200_000)
    p.add_argument("--max_eval_samples", type=int, default=2_000)
    p.add_argument("--min_text_chars", type=int, default=100)
    p.add_argument("--num_generate_samples", type=int, default=10)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--latent_groups", type=str, default="",
                   help="Comma-separated slot counts per group, e.g. '32,16,16'")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--digit_weight", type=float, default=1.0,
                   help="Loss weight multiplier for digit-containing tokens")
    p.add_argument("--entity_weight", type=float, default=1.0,
                   help="Loss weight multiplier for entity tokens (domains, IPs, CVEs)")
    args = p.parse_args()

    if args.no_bf16:
        args.use_bf16 = False

    cfg = NativeConfig(
        tokenizer_name=args.tokenizer_name,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        enc_token_layers=args.enc_token_layers,
        enc_cross_layers=args.enc_cross_layers,
        enc_refine_layers=args.enc_refine_layers,
        num_latents=args.num_latents,
        dec_layers=args.dec_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        contrastive_weight=args.contrastive_weight,
        dataset_name=args.dataset_name,
        eval_dataset_name=args.eval_dataset_name,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        min_text_chars=args.min_text_chars,
        num_generate_samples=args.num_generate_samples,
        log_every=args.log_every,
        seed=args.seed,
        use_bf16=args.use_bf16,
        latent_groups=args.latent_groups,
        digit_weight=args.digit_weight,
        entity_weight=args.entity_weight,
    )
    run_native_train(cfg, tag=args.tag)


if __name__ == "__main__":
    main()
