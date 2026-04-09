"""
CNDX v0 - Reconstruction training.

Exposes train_loop() for reuse by sweep.py, and a standalone CLI.

Usage:
    python -m cndx.train_recon
    python -m cndx.train_recon --num_latents 16
"""

import argparse
import math
import time

import torch
from tqdm import tqdm

from cndx.config import CNDXConfig


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _lr_lambda(warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


def train_loop(model, train_loader, cfg: CNDXConfig, device) -> float:
    """
    Train encoder + projector for cfg.num_epochs.
    Returns final epoch average loss.
    """
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(warmup_steps, total_steps),
    )

    step = 0
    final_avg = 0.0
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch + 1}/{cfg.num_epochs}")
        for batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            out = model(ids, mask)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += out.loss.item()
            step += 1
            if step % cfg.log_every == 0:
                pbar.set_postfix(loss=f"{out.loss.item():.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")

        final_avg = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch + 1} - avg_loss={final_avg:.4f}  "
              f"time={time.time() - t0:.1f}s")

    return final_avg


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--num_latents", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_train_samples", type=int, default=10_000)
    p.add_argument("--max_eval_samples", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)
    args = p.parse_args()
    cfg = CNDXConfig(**vars(args))

    device = get_device()
    print(f"Device: {device}")
    print(f"Compression: {cfg.compression_ratio:.0f}x  (N={cfg.seq_len}, K={cfg.num_latents})")

    from cndx.model import build_model
    from cndx.data import build_dataloaders
    from cndx.eval import run_full_eval

    model, tokenizer = build_model(cfg, device)
    print(f"Trainable: {model.num_trainable():,}  |  Frozen: {model.num_frozen():,}")

    print("Loading dataset...")
    train_loader, val_loader = build_dataloaders(tokenizer, cfg)
    print(f"Train: {len(train_loader)} batches  |  Val: {len(val_loader)} batches")

    train_loop(model, train_loader, cfg, device)

    print("\nEvaluating...")
    results = run_full_eval(model, val_loader, tokenizer, cfg, device)
    print(f"Val loss:            {results['val_loss']:.4f}")
    print(f"First-token acc:     {results['first_token_accuracy']:.4f}")
    print(f"Exact match:         {results['exact_match']:.4f}")

    if results.get("samples"):
        print("\nSample reconstructions:")
        for s in results["samples"][:3]:
            print(f"  ORIG: {s['original'][:100]}")
            print(f"  RECO: {s['reconstructed'][:100]}")
            print()


if __name__ == "__main__":
    main()
