"""
CNDX Phase 1 - Compression ratio sweep.

Trains a fresh bottleneck model for each K, evaluates all 3 modes,
saves results to JSON, prints a summary table.

Usage:
    python -m cndx.sweep
    python -m cndx.sweep --seq_len 128 --num_epochs 5
"""

import argparse
import json
import time

import torch

from cndx.config import CNDXConfig
from cndx.data import build_dataloaders
from cndx.eval import run_full_eval
from cndx.model import build_model
from cndx.train_recon import get_device, train_loop

K_RATIOS = [2, 4]  # K=32 and K=16 only for corruption+LoRA experiment


def sweep(base_cfg: CNDXConfig):
    device = get_device()
    print(f"Device: {device}")
    print(f"Seq len: {base_cfg.seq_len}  |  Epochs: {base_cfg.num_epochs}")
    print(f"Train samples: {base_cfg.max_train_samples}  |  "
          f"Val samples: {base_cfg.max_eval_samples}")
    print(f"Ratios to sweep: {K_RATIOS}")
    print(f"Decoder mask rate: {base_cfg.decoder_mask_rate}")
    print(f"LoRA: rank={base_cfg.lora_rank}  layers={base_cfg.lora_layers}")
    print("=" * 60)

    # Build dataloaders once (shared across K values)
    first_cfg = CNDXConfig(
        seq_len=base_cfg.seq_len,
        num_latents=base_cfg.seq_len,  # placeholder
        batch_size=base_cfg.batch_size,
        dataset_name=base_cfg.dataset_name,
        max_train_samples=base_cfg.max_train_samples,
        max_eval_samples=base_cfg.max_eval_samples,
    )
    print("Loading dataset...")
    # Need a tokenizer to build loaders; build a throwaway model for it
    _, tokenizer = build_model(first_cfg, "cpu")
    train_loader, val_loader = build_dataloaders(tokenizer, first_cfg)
    print(f"Train: {len(train_loader)} batches  |  Val: {len(val_loader)} batches\n")

    all_results = []

    for ratio in K_RATIOS:
        k = base_cfg.seq_len // ratio
        cfg = CNDXConfig(
            seq_len=base_cfg.seq_len,
            num_latents=k,
            num_encoder_layers=base_cfg.num_encoder_layers,
            num_encoder_heads=base_cfg.num_encoder_heads,
            encoder_dropout=base_cfg.encoder_dropout,
            decoder_mask_rate=base_cfg.decoder_mask_rate,
            lora_rank=base_cfg.lora_rank,
            lora_layers=base_cfg.lora_layers,
            batch_size=base_cfg.batch_size,
            learning_rate=base_cfg.learning_rate,
            num_epochs=base_cfg.num_epochs,
            max_train_samples=base_cfg.max_train_samples,
            max_eval_samples=base_cfg.max_eval_samples,
            num_generate_samples=base_cfg.num_generate_samples,
            log_every=base_cfg.log_every,
        )

        print(f"--- K={k}  ratio={ratio}x ---")
        model, _ = build_model(cfg, device)
        print(f"Trainable: {model.num_trainable():,}")

        t0 = time.time()
        train_loop(model, train_loader, cfg, device)
        train_time = time.time() - t0

        print("Evaluating ...")
        ev = run_full_eval(model, val_loader, tokenizer, cfg, device)

        row = {
            "K": k,
            "ratio": ratio,
            "val_loss": round(ev["val_loss"], 4),
            "first_token_acc": round(ev["first_token_accuracy"], 4),
            "exact_match": round(ev["exact_match"], 4),
            "train_time_s": round(train_time, 1),
        }
        all_results.append(row)

        # Show a few samples
        for s in ev.get("samples", [])[:2]:
            print(f"  ORIG: {s['original'][:90]}")
            print(f"  RECO: {s['reconstructed'][:90]}")
            print()

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 60)
    print("CNDX COMPRESSION SWEEP RESULTS")
    print("=" * 60)
    header = f"{'K':>4}  {'Ratio':>6}  {'Val Loss':>9}  {'1st-Tok Acc':>11}  {'Exact Match':>11}  {'Time(s)':>7}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['K']:4d}  {r['ratio']:5d}x  {r['val_loss']:9.4f}  "
              f"{r['first_token_acc']:11.4f}  {r['exact_match']:11.4f}  "
              f"{r['train_time_s']:7.1f}")
    print()

    # Save to JSON
    out_path = f"sweep_results_N{base_cfg.seq_len}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")

    return all_results


def parse_args() -> CNDXConfig:
    p = argparse.ArgumentParser(description="CNDX compression sweep")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_samples", type=int, default=10_000)
    p.add_argument("--max_eval_samples", type=int, default=500)
    p.add_argument("--decoder_mask_rate", type=float, default=0.85)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_layers", type=int, default=4)
    p.add_argument("--num_generate_samples", type=int, default=10)
    p.add_argument("--log_every", type=int, default=50)
    args = p.parse_args()
    return CNDXConfig(**vars(args))


if __name__ == "__main__":
    sweep(parse_args())
