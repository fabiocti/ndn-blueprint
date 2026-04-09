"""
CNDX overnight run - single K, per-epoch eval with sample logging.

Usage:
    python -m cndx.overnight
    python -m cndx.overnight --num_epochs 10 --max_train_samples 10000
"""

import argparse
import json
import math
import random
import time

import torch
import numpy as np
from tqdm import tqdm

from cndx.config import CNDXConfig
from cndx.data import build_dataloaders
from cndx.eval import run_full_eval
from cndx.model import build_model
from cndx.train_recon import get_device, _lr_lambda


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_overnight(cfg: CNDXConfig, tag: str = ""):
    if cfg.seed >= 0:
        _set_seed(cfg.seed)
        print(f"Seed: {cfg.seed}")
    device = get_device()
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
    print(f"Model: {cfg.model_name}")
    train_labels = {"wikipedia": "wikipedia_full_articles_paragraphized",
                     "code": "code_search_net_python_functions",
                     "security": "hackerone_disclosed_reports_chunked"}
    train_label = train_labels.get(cfg.dataset_name, cfg.dataset_name)
    eval_label = cfg.eval_dataset_name if cfg.eval_dataset_name else cfg.dataset_name
    print(f"Train source: {train_label}")
    print(f"Eval source:  {eval_label}")
    print(f"K={cfg.num_latents}  seq_len={cfg.seq_len}  "
          f"compression={cfg.compression_ratio:.0f}x")
    print(f"Epochs: {cfg.num_epochs}  |  Train: {cfg.max_train_samples}  |  "
          f"Val: {cfg.max_eval_samples}")
    print(f"Decoder mask rate: {cfg.decoder_mask_rate}")
    print(f"Unfreeze top layers: {cfg.unfreeze_top_layers}")
    print(f"LoRA: rank={cfg.lora_rank}  layers={cfg.lora_layers}  "
          f"mlp={cfg.lora_include_mlp}")
    print(f"LR={cfg.learning_rate}  warmup={cfg.warmup_ratio}  "
          f"weight_decay={cfg.weight_decay}  max_grad_norm={cfg.max_grad_norm}")
    print(f"Aux token retention weight: {cfg.aux_loss_weight}")
    print(f"Aux span retention weight: {cfg.span_aux_weight}")
    print(f"Contrastive identity weight: {cfg.contrastive_weight}")
    print(f"Anchor side-channel slots: {cfg.num_anchor_slots}")
    if cfg.structured_slots:
        print(f"Structured slots: roles={cfg.num_roles}  pos_buckets={cfg.num_pos_buckets}")
    else:
        print("Structured slots: off")
    print(f"Encoder refine layers: {cfg.num_refine_layers}")
    if cfg.shallow_struct:
        total_slots = cfg.ss_num_detail + cfg.ss_num_identity + cfg.ss_num_global
        print(f"Shallow structured encoder: {cfg.ss_num_detail} detail + "
              f"{cfg.ss_num_identity} identity + "
              f"{cfg.ss_num_global} global = {total_slots} total (parallel, no cascade)")
        print(f"  cross_per_role={cfg.ss_cross_per_role}  "
              f"shared_refine={cfg.ss_shared_refine}")
    elif cfg.tri_level:
        total_slots = (cfg.num_detail_latents + cfg.num_entity_latents +
                       cfg.num_summary_latents)
        print(f"Tri-level encoder: {cfg.num_detail_latents} detail + "
              f"{cfg.num_entity_latents} entity + "
              f"{cfg.num_summary_latents} summary = {total_slots} total prefix slots")
        print(f"  detail_refine={cfg.detail_refine_layers}  "
              f"entity_refine={cfg.entity_refine_layers}  "
              f"summary_refine={cfg.summary_refine_layers}")
    elif cfg.dual_level:
        total_slots = cfg.num_local_latents + cfg.num_global_latents
        print(f"Dual-level encoder: {cfg.num_local_latents} local + "
              f"{cfg.num_global_latents} global = {total_slots} total prefix slots")
        print(f"  local_refine={cfg.local_refine_layers}  "
              f"global_refine={cfg.global_refine_layers}")
    elif cfg.hierarchical:
        print(f"Hierarchical encoder: intermediate={cfg.num_intermediate_latents}  "
              f"stage1_refine={cfg.stage1_refine_layers}  "
              f"stage2_refine={cfg.stage2_refine_layers}")
    else:
        print("Hierarchical encoder: off")
    print("=" * 60)

    model, tokenizer = build_model(cfg, device)
    print(f"Trainable: {model.num_trainable():,}  |  Frozen: {model.num_frozen():,}")

    if device.type == "cuda" and hasattr(torch, "compile") and not getattr(cfg, '_no_compile', False):
        try:
            model = torch.compile(model)
            print("torch.compile: enabled (Blackwell/Hopper optimized)")
        except Exception as e:
            print(f"torch.compile: FAILED ({e}), running eager mode")

    print("Loading dataset...")
    train_loader, val_loader = build_dataloaders(tokenizer, cfg)
    print(f"Train: {len(train_loader)} batches  |  Val: {len(val_loader)} batches")

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

    all_epochs = []
    step = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        for batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            with amp_ctx:
                out = model(ids, mask)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += out.loss.item()
            step += 1
            if step % cfg.log_every == 0:
                pbar.set_postfix(loss=f"{out.loss.item():.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_avg = epoch_loss / len(train_loader)
        train_time = time.time() - t0
        print(f"Epoch {epoch+1} - train_loss={train_avg:.4f}  time={train_time:.1f}s")

        print(f"Epoch {epoch+1} - evaluating...")
        ev = run_full_eval(model, val_loader, tokenizer, cfg, device,
                           amp_ctx=amp_ctx)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(train_avg, 4),
            "val_loss": round(ev["val_loss"], 4),
            "first_token_acc": round(ev["first_token_accuracy"], 4),
            "exact_match": round(ev["exact_match"], 4),
            "train_time_s": round(train_time, 1),
            "samples": ev.get("samples", []),
        }
        all_epochs.append(epoch_record)

        print(f"  val_loss={ev['val_loss']:.4f}  "
              f"first_tok_acc={ev['first_token_accuracy']:.4f}  "
              f"exact_match={ev['exact_match']:.4f}")

        for s in ev.get("samples", [])[:3]:
            print(f"  ORIG: {s['original'][:100]}")
            print(f"  RECO: {s['reconstructed'][:100]}")
            print()

    print("\n" + "=" * 60)
    print("EPOCH SUMMARY")
    print("=" * 60)
    header = f"{'Ep':>3}  {'Train':>8}  {'Val':>8}  {'1st-Tok':>8}  {'Exact':>8}  {'Time':>6}"
    print(header)
    print("-" * len(header))
    for r in all_epochs:
        print(f"{r['epoch']:3d}  {r['train_loss']:8.4f}  {r['val_loss']:8.4f}  "
              f"{r['first_token_acc']:8.4f}  {r['exact_match']:8.4f}  "
              f"{r['train_time_s']:6.1f}")

    best_loss_ep = min(all_epochs, key=lambda r: r['val_loss'])
    best_acc_ep = max(all_epochs, key=lambda r: r['first_token_acc'])
    print(f"\nBest val_loss:  ep {best_loss_ep['epoch']}  "
          f"val={best_loss_ep['val_loss']:.4f}  "
          f"1st={best_loss_ep['first_token_acc']:.4f}")
    print(f"Best 1st-tok:   ep {best_acc_ep['epoch']}  "
          f"val={best_acc_ep['val_loss']:.4f}  "
          f"1st={best_acc_ep['first_token_acc']:.4f}")

    suffix = f"_{tag}" if tag else ""
    if cfg.shallow_struct:
        k_label = (f"SS{cfg.ss_num_detail}d{cfg.ss_num_identity}i"
                   f"{cfg.ss_num_global}g")
    elif cfg.tri_level:
        k_label = (f"D{cfg.num_detail_latents}E{cfg.num_entity_latents}"
                   f"S{cfg.num_summary_latents}")
    elif cfg.dual_level:
        k_label = f"L{cfg.num_local_latents}G{cfg.num_global_latents}"
    else:
        k_label = f"K{cfg.num_latents}"
    out_path = f"overnight_{k_label}_N{cfg.seq_len}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_epochs, f, indent=2)
    print(f"\nResults saved to {out_path}")


def main():
    p = argparse.ArgumentParser(description="CNDX overnight run")
    p.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-360M")
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--eval_dataset_name", type=str, default="")
    p.add_argument("--min_text_chars", type=int, default=100)
    p.add_argument("--num_latents", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_train_samples", type=int, default=10_000)
    p.add_argument("--max_eval_samples", type=int, default=1_000)
    p.add_argument("--num_generate_samples", type=int, default=10)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--decoder_mask_rate", type=float, default=1.0)
    p.add_argument("--unfreeze_top_layers", type=int, default=0)
    p.add_argument("--lora_rank", type=int, default=0)
    p.add_argument("--lora_layers", type=int, default=0)
    p.add_argument("--lora_include_mlp", action="store_true", default=False)
    p.add_argument("--no_lora_mlp", action="store_true")
    p.add_argument("--aux_loss_weight", type=float, default=0.1)
    p.add_argument("--span_aux_weight", type=float, default=0.0)
    p.add_argument("--contrastive_weight", type=float, default=0.0)
    p.add_argument("--num_anchor_slots", type=int, default=0)
    p.add_argument("--structured_slots", action="store_true", default=False)
    p.add_argument("--num_roles", type=int, default=6)
    p.add_argument("--num_pos_buckets", type=int, default=8)
    p.add_argument("--num_refine_layers", type=int, default=3)
    p.add_argument("--shallow_struct", action="store_true", default=False)
    p.add_argument("--ss_num_detail", type=int, default=24)
    p.add_argument("--ss_num_identity", type=int, default=24)
    p.add_argument("--ss_num_global", type=int, default=16)
    p.add_argument("--ss_cross_per_role", type=int, default=1)
    p.add_argument("--ss_shared_refine", type=int, default=2)
    p.add_argument("--tri_level", action="store_true", default=False)
    p.add_argument("--num_detail_latents", type=int, default=24)
    p.add_argument("--num_entity_latents", type=int, default=16)
    p.add_argument("--num_summary_latents", type=int, default=8)
    p.add_argument("--detail_refine_layers", type=int, default=1)
    p.add_argument("--entity_refine_layers", type=int, default=1)
    p.add_argument("--summary_refine_layers", type=int, default=1)
    p.add_argument("--dual_level", action="store_true", default=False)
    p.add_argument("--num_local_latents", type=int, default=32)
    p.add_argument("--num_global_latents", type=int, default=8)
    p.add_argument("--local_refine_layers", type=int, default=2)
    p.add_argument("--global_refine_layers", type=int, default=1)
    p.add_argument("--hierarchical", action="store_true", default=False)
    p.add_argument("--num_intermediate_latents", type=int, default=48)
    p.add_argument("--stage1_refine_layers", type=int, default=1)
    p.add_argument("--stage2_refine_layers", type=int, default=1)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--no_compile", action="store_true", default=False)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    if args.no_lora_mlp:
        args.lora_include_mlp = False
    if args.no_bf16:
        args.use_bf16 = False

    no_compile = args.no_compile
    cfg = CNDXConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        eval_dataset_name=args.eval_dataset_name,
        min_text_chars=args.min_text_chars,
        unfreeze_top_layers=args.unfreeze_top_layers,
        num_latents=args.num_latents,
        seq_len=args.seq_len,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        num_generate_samples=args.num_generate_samples,
        log_every=args.log_every,
        decoder_mask_rate=args.decoder_mask_rate,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
        lora_include_mlp=args.lora_include_mlp,
        aux_loss_weight=args.aux_loss_weight,
        span_aux_weight=args.span_aux_weight,
        contrastive_weight=args.contrastive_weight,
        num_anchor_slots=args.num_anchor_slots,
        structured_slots=args.structured_slots,
        num_roles=args.num_roles,
        num_pos_buckets=args.num_pos_buckets,
        num_refine_layers=args.num_refine_layers,
        shallow_struct=args.shallow_struct,
        ss_num_detail=args.ss_num_detail,
        ss_num_identity=args.ss_num_identity,
        ss_num_global=args.ss_num_global,
        ss_cross_per_role=args.ss_cross_per_role,
        ss_shared_refine=args.ss_shared_refine,
        tri_level=args.tri_level,
        num_detail_latents=args.num_detail_latents,
        num_entity_latents=args.num_entity_latents,
        num_summary_latents=args.num_summary_latents,
        detail_refine_layers=args.detail_refine_layers,
        entity_refine_layers=args.entity_refine_layers,
        summary_refine_layers=args.summary_refine_layers,
        dual_level=args.dual_level,
        num_local_latents=args.num_local_latents,
        num_global_latents=args.num_global_latents,
        local_refine_layers=args.local_refine_layers,
        global_refine_layers=args.global_refine_layers,
        hierarchical=args.hierarchical,
        num_intermediate_latents=args.num_intermediate_latents,
        stage1_refine_layers=args.stage1_refine_layers,
        stage2_refine_layers=args.stage2_refine_layers,
        seed=args.seed,
        use_bf16=args.use_bf16,
    )
    if no_compile:
        cfg._no_compile = True
    run_overnight(cfg, tag=args.tag)


if __name__ == "__main__":
    main()
