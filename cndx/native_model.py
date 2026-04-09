"""
CNDX v2 — Latent-native encoder-decoder for encyclopedic prose.

Architecture:
  Input tokens -> Bidirectional Encoder -> K latent vectors
  K latent vectors -> Causal Decoder (cross-attn every layer) -> Reconstructed tokens

The decoder has dedicated cross-attention to the latent state at every layer,
not prefix-only. This eliminates the attention dilution that caused entity drift,
repetition loops, and number degeneration in the v1 frozen-decoder setup.

~70M params at d=512 (all trainable, no frozen components).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from transformers import AutoTokenizer


@dataclass
class NativeConfig:
    tokenizer_name: str = "HuggingFaceTB/SmolLM2-360M"
    d_model: int = 512
    d_ff: int = 2048
    n_heads: int = 8

    enc_token_layers: int = 2
    enc_cross_layers: int = 2
    enc_refine_layers: int = 2
    num_latents: int = 64

    dec_layers: int = 6

    dropout: float = 0.1
    max_seq_len: int = 256

    # Structured latent groups: comma-separated slot counts per group.
    # Empty string = flat (all slots in one group). E.g. "32,16,16" for
    # 32 detail + 16 identity + 16 global = 64 total.
    latent_groups: str = ""

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    contrastive_weight: float = 0.1

    # Data (fields match CNDXConfig interface for build_dataloaders compat)
    dataset_name: str = "wikipedia"
    eval_dataset_name: str = "wikitext"
    max_train_samples: int = 200_000
    max_eval_samples: int = 2_000
    min_text_chars: int = 100
    seq_len: int = 128

    seed: int = -1
    use_bf16: bool = True
    num_generate_samples: int = 10
    log_every: int = 50
    digit_weight: float = 1.0
    entity_weight: float = 1.0

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def is_structured(self):
        return bool(self.latent_groups.strip())

    @property
    def group_sizes(self):
        if not self.is_structured:
            return [self.num_latents]
        sizes = [int(x) for x in self.latent_groups.split(",")]
        assert sum(sizes) == self.num_latents, (
            f"latent_groups {sizes} must sum to num_latents={self.num_latents}")
        return sizes


# ---------------------------------------------------------------------------
# Pre-norm building blocks
# ---------------------------------------------------------------------------

class PreNormSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        h = self.norm(x)
        out, _ = self.attn(h, h, h, attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask)
        return x + out


class PreNormCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x, context, key_padding_mask=None):
        q = self.norm_q(x)
        kv = self.norm_kv(context)
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        return x + out


class PreNormFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


# ---------------------------------------------------------------------------
# Encoder components
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Bidirectional self-attention + FFN for token contextualization."""
    def __init__(self, d, h, ff, dr):
        super().__init__()
        self.sa = PreNormSelfAttention(d, h, dr)
        self.ffn = PreNormFFN(d, ff, dr)

    def forward(self, x, key_padding_mask=None):
        return self.ffn(self.sa(x, key_padding_mask=key_padding_mask))


class PerceiverBlock(nn.Module):
    """Cross-attention (latents query context) + FFN."""
    def __init__(self, d, h, ff, dr):
        super().__init__()
        self.ca = PreNormCrossAttention(d, h, dr)
        self.ffn = PreNormFFN(d, ff, dr)

    def forward(self, latents, context, key_padding_mask=None):
        return self.ffn(self.ca(latents, context, key_padding_mask))


class RefineBlock(nn.Module):
    """Self-attention + FFN on latent vectors."""
    def __init__(self, d, h, ff, dr):
        super().__init__()
        self.sa = PreNormSelfAttention(d, h, dr)
        self.ffn = PreNormFFN(d, ff, dr)

    def forward(self, latents):
        return self.ffn(self.sa(latents))


class NativeEncoder(nn.Module):
    """
    Bidirectional token encoder -> Perceiver cross-attention -> Latent refinement.
    Compresses N input tokens into K latent vectors.

    Supports flat mode (all K slots in one group) and structured mode
    (parallel groups with separate queries/cross-attention, joint refinement).
    """
    def __init__(self, cfg: NativeConfig):
        super().__init__()
        d, h, ff, dr = cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout

        self.token_layers = nn.ModuleList(
            [EncoderBlock(d, h, ff, dr) for _ in range(cfg.enc_token_layers)])

        self.group_sizes = cfg.group_sizes
        self.n_groups = len(self.group_sizes)

        self.group_queries = nn.ParameterList([
            nn.Parameter(torch.randn(1, gs, d) * 0.02)
            for gs in self.group_sizes
        ])
        self.group_cross = nn.ModuleList([
            nn.ModuleList([PerceiverBlock(d, h, ff, dr)
                           for _ in range(cfg.enc_cross_layers)])
            for _ in self.group_sizes
        ])

        self.refine_layers = nn.ModuleList(
            [RefineBlock(d, h, ff, dr) for _ in range(cfg.enc_refine_layers)])
        self.final_norm = nn.LayerNorm(d)

    def forward(self, token_embeddings, padding_mask=None):
        x = token_embeddings
        for layer in self.token_layers:
            x = layer(x, key_padding_mask=padding_mask)

        group_outputs = []
        for queries, cross_layers in zip(self.group_queries, self.group_cross):
            g = queries.expand(x.shape[0], -1, -1)
            for layer in cross_layers:
                g = layer(g, x, key_padding_mask=padding_mask)
            group_outputs.append(g)

        latents = torch.cat(group_outputs, dim=1)

        for layer in self.refine_layers:
            latents = layer(latents)

        return self.final_norm(latents)


# ---------------------------------------------------------------------------
# Decoder components
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """Causal self-attention -> Cross-attention to latent state -> FFN."""
    def __init__(self, d, h, ff, dr):
        super().__init__()
        self.sa = PreNormSelfAttention(d, h, dr)
        self.ca = PreNormCrossAttention(d, h, dr)
        self.ffn = PreNormFFN(d, ff, dr)

    def forward(self, x, latent_state, causal_mask=None, padding_mask=None):
        x = self.sa(x, attn_mask=causal_mask, key_padding_mask=padding_mask)
        x = self.ca(x, latent_state)
        return self.ffn(x)


class NativeDecoder(nn.Module):
    """Causal decoder with cross-attention to latent state at every layer."""
    def __init__(self, cfg: NativeConfig):
        super().__init__()
        d, h, ff, dr = cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout
        self.layers = nn.ModuleList(
            [DecoderBlock(d, h, ff, dr) for _ in range(cfg.dec_layers)])
        self.final_norm = nn.LayerNorm(d)

    def forward(self, x, latent_state, causal_mask=None, padding_mask=None):
        for layer in self.layers:
            x = layer(x, latent_state, causal_mask, padding_mask)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Contrastive identity loss
# ---------------------------------------------------------------------------

class ContrastiveLatentLoss(nn.Module):
    """InfoNCE: each sample's mean-pooled latent should match its own source embedding."""
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.log_temp = nn.Parameter(torch.tensor(2.66))

    def forward(self, latents, token_embeddings, padding_mask=None):
        B = latents.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=latents.device, requires_grad=True)

        summary = self.proj(latents.mean(dim=1))

        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).float()
            source = (token_embeddings * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            source = token_embeddings.mean(dim=1)

        summary = F.normalize(summary, dim=-1)
        source = F.normalize(source.detach(), dim=-1)

        temp = self.log_temp.exp().clamp(1.0, 100.0)
        logits = (summary @ source.T) * temp
        return F.cross_entropy(logits, torch.arange(B, device=logits.device))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class CNDXNativeModel(nn.Module):
    """
    Latent-native encoder-decoder for compressed reconstruction.

    Encoder compresses input tokens into K latent vectors.
    Decoder reconstructs original tokens via cross-attention to the latent
    state at every layer — no prefix hack, no frozen components.
    """

    def __init__(self, cfg: NativeConfig, vocab_size: int,
                 pad_token_id: int, bos_token_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, cfg.d_model)
        self.enc_pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.dec_pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.embed_drop = nn.Dropout(cfg.dropout)

        self.encoder = NativeEncoder(cfg)
        self.decoder = NativeDecoder(cfg)

        self.output_proj = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight

        self.cl_head = (ContrastiveLatentLoss(cfg.d_model)
                        if cfg.contrastive_weight > 0 else None)
        self.cl_weight = cfg.contrastive_weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "embedding" in name or "pos" in name:
                nn.init.normal_(p, std=0.02)
            elif p.dim() > 1 and "norm" not in name:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _causal_mask(n, device):
        return torch.triu(torch.full((n, n), float('-inf'), device=device), diagonal=1)

    def encode(self, input_ids, attention_mask=None):
        """Encode tokens -> K latent vectors. Returns (latent_state, token_emb)."""
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0)
        tok_emb = self.embed_drop(self.embedding(input_ids) + self.enc_pos(pos))
        pad_mask = ~attention_mask.bool() if attention_mask is not None else None
        return self.encoder(tok_emb, pad_mask), tok_emb

    def decode_logits(self, decoder_ids, latent_state, attention_mask=None):
        """Decode with teacher forcing. Returns logits [B, N, V]."""
        B, N = decoder_ids.shape
        pos = torch.arange(N, device=decoder_ids.device).unsqueeze(0)
        dec_emb = self.embed_drop(self.embedding(decoder_ids) + self.dec_pos(pos))
        cmask = self._causal_mask(N, decoder_ids.device)
        pad_mask = ~attention_mask.bool() if attention_mask is not None else None
        return self.output_proj(
            self.decoder(dec_emb, latent_state, cmask, pad_mask))

    def forward(self, input_ids, attention_mask=None):
        """
        Encode -> bottleneck -> decode -> reconstruction loss.
        Decoder input is input_ids shifted right (BOS prepended).
        """
        B, N = input_ids.shape
        latent_state, tok_emb = self.encode(input_ids, attention_mask)

        bos = input_ids.new_full((B, 1), self.bos_token_id)
        dec_in = torch.cat([bos, input_ids[:, :-1]], dim=1)
        logits = self.decode_logits(dec_in, latent_state, attention_mask)

        _tw = getattr(self, 'token_weights', None)
        recon_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size), input_ids.view(-1),
            weight=_tw, ignore_index=self.pad_token_id)

        loss = recon_loss
        if self.cl_head is not None and self.training:
            pad_mask = ~attention_mask.bool() if attention_mask is not None else None
            loss = loss + self.cl_weight * self.cl_head(
                latent_state, tok_emb.detach(), pad_mask)

        return loss, logits, latent_state

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=None):
        """Autoregressive generation from latent state only."""
        if max_new_tokens is None:
            max_new_tokens = input_ids.shape[1]
        latent_state, _ = self.encode(input_ids, attention_mask)
        B = input_ids.shape[0]
        cur = input_ids.new_full((B, 1), self.bos_token_id)

        for _ in range(max_new_tokens):
            logits = self.decode_logits(cur, latent_state)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)

        return cur[:, 1:]

    @torch.no_grad()
    def ablation_forward(self, input_ids, attention_mask=None):
        """Forward with zeroed-out latent state — measures bottleneck utilization."""
        B, N = input_ids.shape
        latent_state, _ = self.encode(input_ids, attention_mask)
        zeroed = torch.zeros_like(latent_state)

        bos = input_ids.new_full((B, 1), self.bos_token_id)
        dec_in = torch.cat([bos, input_ids[:, :-1]], dim=1)
        logits = self.decode_logits(dec_in, zeroed, attention_mask)

        return F.cross_entropy(
            logits.view(-1, self.vocab_size), input_ids.view(-1),
            ignore_index=self.pad_token_id)

    @torch.no_grad()
    def shuffled_latent_forward(self, input_ids, attention_mask=None):
        """Forward with latent states randomly permuted across the batch.

        If the model genuinely depends on sample-specific latent content,
        feeding the wrong sample's latent should cause near-ablation-level
        collapse.  If performance stays high, the decoder is reconstructing
        from autoregressive priors or memorised templates, not from the
        latent state.
        """
        B, N = input_ids.shape
        if B < 2:
            return self.ablation_forward(input_ids, attention_mask)

        latent_state, _ = self.encode(input_ids, attention_mask)
        perm = torch.randperm(B, device=latent_state.device)
        while (perm == torch.arange(B, device=perm.device)).all():
            perm = torch.randperm(B, device=latent_state.device)
        shuffled = latent_state[perm]

        bos = input_ids.new_full((B, 1), self.bos_token_id)
        dec_in = torch.cat([bos, input_ids[:, :-1]], dim=1)
        logits = self.decode_logits(dec_in, shuffled, attention_mask)

        return F.cross_entropy(
            logits.view(-1, self.vocab_size), input_ids.view(-1),
            ignore_index=self.pad_token_id)

    @torch.no_grad()
    def partial_corrupt_forward(self, input_ids, attention_mask=None,
                                corrupt_frac=0.5):
        """Forward with a random subset of latent slots zeroed out."""
        B, N = input_ids.shape
        latent_state, _ = self.encode(input_ids, attention_mask)
        K = latent_state.shape[1]
        n_corrupt = max(1, int(K * corrupt_frac))
        mask = torch.ones(K, device=latent_state.device)
        idx = torch.randperm(K, device=latent_state.device)[:n_corrupt]
        mask[idx] = 0.0
        corrupted = latent_state * mask.unsqueeze(0).unsqueeze(-1)

        bos = input_ids.new_full((B, 1), self.bos_token_id)
        dec_in = torch.cat([bos, input_ids[:, :-1]], dim=1)
        logits = self.decode_logits(dec_in, corrupted, attention_mask)

        return F.cross_entropy(
            logits.view(-1, self.vocab_size), input_ids.view(-1),
            ignore_index=self.pad_token_id)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_native_model(cfg: NativeConfig, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id
    bos_id = (tokenizer.bos_token_id
              if tokenizer.bos_token_id is not None
              else tokenizer.eos_token_id)

    model = CNDXNativeModel(cfg, vocab_size, pad_id, bos_id).to(device)
    return model, tokenizer
