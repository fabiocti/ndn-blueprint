"""
CNDX v0 smoke test - validates the core pipeline on a tiny batch.

Success criteria:
  1. Forward pass produces a loss
  2. Backward pass gives gradients to encoder+projector, not base model
  3. Loss decreases over a few steps
  4. Prefix-only greedy generation produces tokens
"""

import torch
from cndx.config import CNDXConfig
from cndx.model import build_model


def smoke_test():
    device = torch.device("cpu")
    cfg = CNDXConfig(seq_len=32, num_latents=16)

    print(f"Building model (K={cfg.num_latents}, N={cfg.seq_len}, "
          f"ratio={cfg.compression_ratio:.0f}x) ...")
    model, tokenizer = build_model(cfg, device)
    print(f"  Trainable: {model.num_trainable():,}  Frozen: {model.num_frozen():,}")

    texts = [
        "Once upon a time there was a little cat named Whiskers.",
        "The sun was shining and the birds were singing in the trees.",
    ]
    tok = tokenizer(texts, truncation=True, max_length=cfg.seq_len,
                    padding="max_length", return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    # --- 1. Forward pass ---
    print("\n1. Forward pass ...")
    outputs = model(input_ids, attention_mask)
    print(f"   Loss: {outputs.loss.item():.4f}")
    assert outputs.loss.item() > 0

    # --- 2. Backward pass + gradient check ---
    print("2. Backward pass ...")
    outputs.loss.backward()

    trainable_grads = 0
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0:
            trainable_grads += 1
    trainable_total = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    print(f"   {trainable_grads}/{trainable_total} trainable params have gradients")
    assert trainable_grads == trainable_total, "Not all trainable params got gradients"

    frozen_grads = [n for n, p in model.base_model.named_parameters() if p.grad is not None]
    assert len(frozen_grads) == 0, "Frozen base model should have no gradients"
    print("   Base model frozen - no gradients (correct)")

    # --- 3. Loss decreases over a few steps ---
    print("3. Training for 20 steps ...")
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=3e-4)
    losses = []
    for step in range(20):
        optimizer.zero_grad()
        out = model(input_ids, attention_mask)
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.item())
        if step % 5 == 0:
            print(f"   step {step}: loss={losses[-1]:.4f}")
    best = min(losses)
    assert best < losses[0], f"Loss never improved: start={losses[0]:.4f} best={best:.4f}"
    print(f"   Loss: {losses[0]:.4f} -> best {best:.4f} (final {losses[-1]:.4f})")

    # --- 4. Prefix-only generation ---
    print("4. Generation from prefix only ...")
    model.eval()
    gen = model.generate(input_ids[:1], attention_mask[:1], max_new_tokens=16)
    print(f"   Shape: {gen.shape}")
    assert gen.shape == (1, 16)
    decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
    print(f"   Original:      {texts[0][:60]}")
    print(f"   Reconstructed: {decoded[:60]}")

    # --- Compression ratio sweep ---
    print("\n5. Compression ratio sweep ...")
    for ratio in [1, 2, 4, 8]:
        k = cfg.seq_len // ratio
        m, _ = build_model(CNDXConfig(seq_len=cfg.seq_len, num_latents=k), device)
        out = m(input_ids, attention_mask)
        print(f"   K={k:3d}  ratio={ratio}x  loss={out.loss.item():.4f}")

    print("\n=== ALL SMOKE TESTS PASSED ===")


if __name__ == "__main__":
    smoke_test()
