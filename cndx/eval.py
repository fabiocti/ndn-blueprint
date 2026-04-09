"""
CNDX Phase 1 evaluation - three modes.

A. Teacher-forced val loss (does the prefix help prediction?)
B. Prefix-only generation  (can it reconstruct from prefix alone?)
C. First-token accuracy     (does the prefix carry real information?)
"""

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from cndx.config import CNDXConfig


@torch.no_grad()
def eval_teacher_forced(model, val_loader: DataLoader, device,
                        amp_ctx=None) -> dict:
    """Standard teacher-forced reconstruction loss on val set."""
    model.eval()
    ctx = amp_ctx or torch.amp.autocast('cuda', enabled=False)
    total_loss = 0.0
    n = 0
    for batch in val_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with ctx:
            total_loss += model(ids, mask).loss.item()
        n += 1
    avg = total_loss / max(n, 1)
    return {"val_loss": avg}


@torch.no_grad()
def eval_first_token(model, val_loader: DataLoader, device,
                     amp_ctx=None) -> dict:
    """
    From the compressed prefix alone, predict the first original token.
    Uses model._encode_to_prefix to ensure structured slots / anchors
    are included — same encoding path as forward() and generate().
    """
    model.eval()
    ctx = amp_ctx or torch.amp.autocast('cuda', enabled=False)
    correct = 0
    total = 0

    for batch in val_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        with ctx:
            prefix, _, _ = model._encode_to_prefix(ids, mask)
            out = model.base_model(inputs_embeds=prefix)

        pred = out.logits[:, -1, :].argmax(dim=-1)
        actual = ids[:, 0]

        correct += (pred == actual).sum().item()
        total += ids.shape[0]

    acc = correct / max(total, 1)
    return {"first_token_accuracy": acc, "first_token_correct": correct, "first_token_total": total}


@torch.no_grad()
def eval_generation(model, val_loader: DataLoader,
                    tokenizer: PreTrainedTokenizer, cfg: CNDXConfig,
                    device, amp_ctx=None) -> dict:
    """
    Prefix-only greedy generation: encode -> compress -> generate from prefix.
    Measures exact match and collects samples for qualitative inspection.
    """
    model.eval()
    ctx = amp_ctx or torch.amp.autocast('cuda', enabled=False)
    batch = next(iter(val_loader))
    n = min(cfg.num_generate_samples, batch["input_ids"].shape[0])
    ids = batch["input_ids"][:n].to(device)
    mask = batch["attention_mask"][:n].to(device)

    with ctx:
        gen = model.generate(ids, mask, max_new_tokens=cfg.seq_len)

    originals = tokenizer.batch_decode(ids, skip_special_tokens=True)
    reconstructed = tokenizer.batch_decode(gen, skip_special_tokens=True)

    exact = sum(
        1 for o, r in zip(originals, reconstructed) if o.strip() == r.strip()
    )
    exact_rate = exact / max(n, 1)

    samples = [
        {"original": o.strip(), "reconstructed": r.strip()}
        for o, r in zip(originals, reconstructed)
    ]

    return {"exact_match": exact_rate, "exact_match_count": exact, "samples": samples}


def run_full_eval(model, val_loader, tokenizer, cfg, device,
                  amp_ctx=None) -> dict:
    """Run all three eval modes and merge results."""
    results = {}
    results.update(eval_teacher_forced(model, val_loader, device, amp_ctx))
    results.update(eval_first_token(model, val_loader, device, amp_ctx))
    gen = eval_generation(model, val_loader, tokenizer, cfg, device, amp_ctx)
    results["exact_match"] = gen["exact_match"]
    results["samples"] = gen["samples"]
    return results
