"""
CNDX v0 - Bottleneck-prefix autoencoder on top of a frozen causal LM.

Flow:
  input_ids -> frozen embeddings -> CNDXEncoder -> K cdx latents
  -> CNDXProjector -> prefix embeddings -> [prefix ; target embeds]
  -> frozen SmolLM -> reconstruct original tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from cndx.config import CNDXConfig


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen nn.Linear."""

    def __init__(self, original: nn.Linear, rank: int = 16):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(original.in_features, rank) * (1.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B)


def inject_lora(base_model, rank: int, num_layers: int, include_mlp: bool = False):
    """Inject LoRA into attention (and optionally MLP) projections of the top layers."""
    layers = base_model.model.layers
    total = len(layers)
    for i in range(total - num_layers, total):
        attn = layers[i].self_attn
        attn.q_proj = LoRALinear(attn.q_proj, rank)
        attn.v_proj = LoRALinear(attn.v_proj, rank)
        if include_mlp:
            mlp = layers[i].mlp
            mlp.gate_proj = LoRALinear(mlp.gate_proj, rank)
            mlp.up_proj = LoRALinear(mlp.up_proj, rank)
            mlp.down_proj = LoRALinear(mlp.down_proj, rank)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CNDXCrossAttentionLayer(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, latents, context, key_padding_mask=None,
                return_weights=False):
        attn_out, attn_weights = self.cross_attn(
            query=latents, key=context, value=context,
            key_padding_mask=key_padding_mask,
            need_weights=return_weights,
            average_attn_weights=True,
        )
        latents = self.norm1(latents + attn_out)
        latents = self.norm2(latents + self.ffn(latents))
        if return_weights:
            return latents, attn_weights
        return latents


class CNDXSelfAttentionLayer(nn.Module):
    """Standard transformer self-attention block for latent refinement."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, latents):
        attn_out, _ = self.self_attn(
            query=latents, key=latents, value=latents,
        )
        latents = self.norm1(latents + attn_out)
        latents = self.norm2(latents + self.ffn(latents))
        return latents


class CNDXEncoder(nn.Module):
    """K learned latent queries cross-attend over N token embeddings,
    then refine via self-attention blocks -> K cdx vectors."""

    def __init__(self, num_latents: int, hidden_size: int,
                 num_heads: int = 8, num_layers: int = 2,
                 num_refine_layers: int = 0, dropout: float = 0.0):
        super().__init__()
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latents, hidden_size) * 0.02
        )
        self.layers = nn.ModuleList([
            CNDXCrossAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.refine_layers = nn.ModuleList([
            CNDXSelfAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(num_refine_layers)
        ])

    def forward(self, token_embeddings, attention_mask=None,
                return_cross_weights=False):
        B = token_embeddings.shape[0]
        latents = self.latent_queries.expand(B, -1, -1)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        cross_weights = None
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            if return_cross_weights and is_last:
                latents, cross_weights = layer(
                    latents, token_embeddings, key_padding_mask,
                    return_weights=True)
            else:
                latents = layer(latents, token_embeddings, key_padding_mask)

        for layer in self.refine_layers:
            latents = layer(latents)

        if return_cross_weights:
            return latents, cross_weights
        return latents


class CNDXHierarchicalEncoder(nn.Module):
    """2-stage compressor: tokens -> intermediate latents -> final latents.
    Stage 1 gathers local info, stage 2 builds global compressed representation."""

    def __init__(self, num_latents: int, num_intermediate: int,
                 hidden_size: int, num_heads: int = 8,
                 stage1_refine: int = 1, stage2_refine: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.stage1_queries = nn.Parameter(
            torch.randn(1, num_intermediate, hidden_size) * 0.02
        )
        self.stage1_cross = CNDXCrossAttentionLayer(hidden_size, num_heads, dropout)
        self.stage1_refine = nn.ModuleList([
            CNDXSelfAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(stage1_refine)
        ])

        self.stage2_queries = nn.Parameter(
            torch.randn(1, num_latents, hidden_size) * 0.02
        )
        self.stage2_cross = CNDXCrossAttentionLayer(hidden_size, num_heads, dropout)
        self.stage2_refine = nn.ModuleList([
            CNDXSelfAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(stage2_refine)
        ])

    def forward(self, token_embeddings, attention_mask=None):
        B = token_embeddings.shape[0]

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        intermediate = self.stage1_queries.expand(B, -1, -1)
        intermediate = self.stage1_cross(intermediate, token_embeddings, key_padding_mask)
        for layer in self.stage1_refine:
            intermediate = layer(intermediate)

        final = self.stage2_queries.expand(B, -1, -1)
        final = self.stage2_cross(final, intermediate)
        for layer in self.stage2_refine:
            final = layer(final)

        return final


class CNDXDualLevelEncoder(nn.Module):
    """2-level latent layout: local slots attend tokens, global slots attend locals.
    Both levels are output as prefix: [global ; local].
    Global-first ordering gives the decoder identity/context before details."""

    def __init__(self, num_local: int, num_global: int,
                 hidden_size: int, num_heads: int = 8,
                 local_cross_layers: int = 2,
                 local_refine: int = 2, global_refine: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.num_local = num_local
        self.num_global = num_global

        self.local_queries = nn.Parameter(
            torch.randn(1, num_local, hidden_size) * 0.02
        )
        self.local_cross = nn.ModuleList([
            CNDXCrossAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(local_cross_layers)
        ])
        self.local_refine = nn.ModuleList([
            CNDXSelfAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(local_refine)
        ])

        self.global_queries = nn.Parameter(
            torch.randn(1, num_global, hidden_size) * 0.02
        )
        self.global_cross = CNDXCrossAttentionLayer(hidden_size, num_heads, dropout)
        self.global_refine = nn.ModuleList([
            CNDXSelfAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(global_refine)
        ])

    def forward(self, token_embeddings, attention_mask=None):
        B = token_embeddings.shape[0]

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        local = self.local_queries.expand(B, -1, -1)
        for layer in self.local_cross:
            local = layer(local, token_embeddings, key_padding_mask)
        for layer in self.local_refine:
            local = layer(local)

        glob = self.global_queries.expand(B, -1, -1)
        glob = self.global_cross(glob, local)
        for layer in self.global_refine:
            glob = layer(glob)

        return torch.cat([glob, local], dim=1)


class CNDXTriLevelEncoder(nn.Module):
    """3-role structured latent layout: detail -> entity -> summary.
    All three levels output as prefix: [summary ; entity ; detail].
    Global-to-local ordering lets the decoder see identity before specifics."""

    def __init__(self, num_detail: int, num_entity: int, num_summary: int,
                 hidden_size: int, num_heads: int = 8,
                 detail_cross_layers: int = 2,
                 detail_refine: int = 1, entity_refine: int = 1,
                 summary_refine: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_detail = num_detail
        self.num_entity = num_entity
        self.num_summary = num_summary

        self.detail_queries = nn.Parameter(
            torch.randn(1, num_detail, hidden_size) * 0.02
        )
        self.detail_cross = nn.ModuleList([
            CNDXCrossAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(detail_cross_layers)
        ])
        self.detail_refine = nn.ModuleList([
            CNDXSelfAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(detail_refine)
        ])

        self.entity_queries = nn.Parameter(
            torch.randn(1, num_entity, hidden_size) * 0.02
        )
        self.entity_cross = CNDXCrossAttentionLayer(hidden_size, num_heads, dropout)
        self.entity_refine = nn.ModuleList([
            CNDXSelfAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(entity_refine)
        ])

        self.summary_queries = nn.Parameter(
            torch.randn(1, num_summary, hidden_size) * 0.02
        )
        self.summary_cross = CNDXCrossAttentionLayer(hidden_size, num_heads, dropout)
        self.summary_refine = nn.ModuleList([
            CNDXSelfAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(summary_refine)
        ])

    def forward(self, token_embeddings, attention_mask=None):
        B = token_embeddings.shape[0]

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        detail = self.detail_queries.expand(B, -1, -1)
        for layer in self.detail_cross:
            detail = layer(detail, token_embeddings, key_padding_mask)
        for layer in self.detail_refine:
            detail = layer(detail)

        entity = self.entity_queries.expand(B, -1, -1)
        entity = self.entity_cross(entity, detail)
        for layer in self.entity_refine:
            entity = layer(entity)

        summary = self.summary_queries.expand(B, -1, -1)
        summary = self.summary_cross(summary, entity)
        for layer in self.summary_refine:
            summary = layer(summary)

        return torch.cat([summary, entity, detail], dim=1)


class CNDXShallowStructuredEncoder(nn.Module):
    """Shallow structured layout: all role groups attend tokens directly (no cascade).
    After parallel extraction, roles merge and refine together via shared self-attention.
    Output: [global ; identity ; detail]."""

    def __init__(self, num_detail: int, num_identity: int, num_global: int,
                 hidden_size: int, num_heads: int = 8,
                 cross_layers_per_role: int = 1,
                 shared_refine: int = 2, dropout: float = 0.0):
        super().__init__()
        self.num_detail = num_detail
        self.num_identity = num_identity
        self.num_global = num_global

        self.detail_queries = nn.Parameter(
            torch.randn(1, num_detail, hidden_size) * 0.02)
        self.detail_cross = nn.ModuleList([
            CNDXCrossAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(cross_layers_per_role)])

        self.identity_queries = nn.Parameter(
            torch.randn(1, num_identity, hidden_size) * 0.02)
        self.identity_cross = nn.ModuleList([
            CNDXCrossAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(cross_layers_per_role)])

        self.global_queries = nn.Parameter(
            torch.randn(1, num_global, hidden_size) * 0.02)
        self.global_cross = nn.ModuleList([
            CNDXCrossAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(cross_layers_per_role)])

        self.shared_refine = nn.ModuleList([
            CNDXSelfAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(shared_refine)])

    def forward(self, token_embeddings, attention_mask=None):
        B = token_embeddings.shape[0]

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        detail = self.detail_queries.expand(B, -1, -1)
        for layer in self.detail_cross:
            detail = layer(detail, token_embeddings, key_padding_mask)

        identity = self.identity_queries.expand(B, -1, -1)
        for layer in self.identity_cross:
            identity = layer(identity, token_embeddings, key_padding_mask)

        glob = self.global_queries.expand(B, -1, -1)
        for layer in self.global_cross:
            glob = layer(glob, token_embeddings, key_padding_mask)

        merged = torch.cat([glob, identity, detail], dim=1)
        for layer in self.shared_refine:
            merged = layer(merged)

        return merged


class CNDXProjector(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, cdx_latents):
        return self.norm(cdx_latents + self.proj(cdx_latents))


# ---------------------------------------------------------------------------
# Auxiliary: source token retention head
# ---------------------------------------------------------------------------

class SourceRetentionHead(nn.Module):
    """Bag-of-source-token retention: predict which source tokens appeared from pooled latents."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, cdx_latents, input_ids, embedding_weight,
                attention_mask=None, special_ids=None):
        B = cdx_latents.shape[0]
        device = cdx_latents.device

        summary = self.proj(cdx_latents.mean(dim=1))  # [B, H]

        valid_mask = attention_mask.bool() if attention_mask is not None else \
            torch.ones_like(input_ids, dtype=torch.bool)

        all_ids = input_ids[valid_mask].unique()
        if special_ids is not None:
            keep = torch.ones(all_ids.shape[0], dtype=torch.bool, device=device)
            for sid in special_ids:
                keep &= (all_ids != sid)
            all_ids = all_ids[keep]

        if all_ids.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        cand_embeds = embedding_weight[all_ids].detach()  # [C, H]
        scores = summary @ cand_embeds.T  # [B, C]

        targets = torch.zeros(B, all_ids.shape[0], device=device)
        for b in range(B):
            src = input_ids[b][valid_mask[b]]
            present = (src.unsqueeze(1) == all_ids.unsqueeze(0)).any(dim=0)
            targets[b] = present.float()

        freq = targets.sum(dim=0).clamp(min=1)
        weights = (1.0 / freq)
        weights = weights / weights.mean()

        return F.binary_cross_entropy_with_logits(
            scores, targets,
            weight=weights.unsqueeze(0).expand_as(scores),
            reduction='mean',
        )


# ---------------------------------------------------------------------------
# Auxiliary: contrastive source-identity loss
# ---------------------------------------------------------------------------

class ContrastiveIdentityLoss(nn.Module):
    """InfoNCE: each compressed summary should be closest to its own source embedding."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.summary_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.log_temperature = nn.Parameter(torch.tensor(2.66))  # ~exp(2.66) = ~14.3

    def forward(self, cdx_latents, token_embeddings, attention_mask=None,
                special_ids=None, input_ids=None):
        B = cdx_latents.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=cdx_latents.device, requires_grad=True)

        summary = self.summary_proj(cdx_latents.mean(dim=1))  # [B, H]

        if attention_mask is not None:
            mask = attention_mask.bool()
            if special_ids is not None and input_ids is not None:
                for sid in special_ids:
                    mask = mask & (input_ids != sid)
            mask_f = mask.unsqueeze(-1).float()  # [B, N, 1]
            source = (token_embeddings * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            source = token_embeddings.mean(dim=1)

        summary = F.normalize(summary, dim=-1)
        source = F.normalize(source.detach(), dim=-1)

        temperature = self.log_temperature.exp().clamp(min=1.0, max=100.0)
        logits = (summary @ source.T) * temperature  # [B, B]
        labels = torch.arange(B, device=logits.device)
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Auxiliary: span (n-gram) retention head (disabled, kept for reference)
# ---------------------------------------------------------------------------

class SpanRetentionHead(nn.Module):
    """Predict which source 2-grams and 3-grams appeared, forcing local structure retention."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _extract_ngrams(self, input_ids, attention_mask, special_ids, n):
        """Return unique n-gram tuples across the batch and per-sample membership."""
        B, N = input_ids.shape
        device = input_ids.device

        special_set = set()
        if special_ids is not None:
            special_set = set(int(s) for s in special_ids)

        all_ngrams = set()
        per_sample = []

        for b in range(B):
            length = int(attention_mask[b].sum()) if attention_mask is not None else N
            ids_b = input_ids[b, :length].tolist()
            sample_ngrams = set()
            for i in range(len(ids_b) - n + 1):
                gram = tuple(ids_b[i:i + n])
                if any(t in special_set for t in gram):
                    continue
                sample_ngrams.add(gram)
            all_ngrams.update(sample_ngrams)
            per_sample.append(sample_ngrams)

        if not all_ngrams:
            return None, None

        gram_list = sorted(all_ngrams)
        gram_to_idx = {g: i for i, g in enumerate(gram_list)}
        gram_ids = torch.tensor(gram_list, dtype=torch.long, device=device)  # [G, n]

        targets = torch.zeros(B, len(gram_list), device=device)
        for b in range(B):
            for g in per_sample[b]:
                targets[b, gram_to_idx[g]] = 1.0

        return gram_ids, targets

    def forward(self, cdx_latents, input_ids, embedding_weight,
                attention_mask=None, special_ids=None):
        device = cdx_latents.device
        B = cdx_latents.shape[0]

        summary = self.proj(cdx_latents.mean(dim=1))  # [B, H]

        all_gram_ids = []
        all_targets = []

        gram_ids, targets = self._extract_ngrams(
            input_ids, attention_mask, special_ids, 2)
        if gram_ids is not None:
            all_gram_ids.append(gram_ids)
            all_targets.append(targets)

        if not all_gram_ids:
            return torch.tensor(0.0, device=device, requires_grad=True)

        scores_list = []
        targets_list = []

        for gram_ids, targets in zip(all_gram_ids, all_targets):
            token_embeds = embedding_weight[gram_ids].detach()  # [G, n, H]
            span_embeds = token_embeds.mean(dim=1)  # [G, H]
            scores = summary @ span_embeds.T  # [B, G]
            scores_list.append(scores)
            targets_list.append(targets)

        all_scores = torch.cat(scores_list, dim=1)
        all_tgt = torch.cat(targets_list, dim=1)

        freq = all_tgt.sum(dim=0).clamp(min=1)
        weights = (1.0 / freq)
        weights = weights / weights.mean()

        return F.binary_cross_entropy_with_logits(
            all_scores, all_tgt,
            weight=weights.unsqueeze(0).expand_as(all_scores),
            reduction='mean',
        )


# ---------------------------------------------------------------------------
# Structured dense slots: role + position-bucket within the same prefix space
# ---------------------------------------------------------------------------

class StructuredSlotLayer(nn.Module):
    """Add role and position-bucket structure to latent slots.

    Each slot gets a soft role assignment and a position-bucket assignment
    derived from cross-attention weights.  The result stays in the same
    dense space — no foreign side-channel.

        slot_out = LayerNorm(latent + role_emb + pos_emb)
    """

    def __init__(self, hidden_size: int, num_roles: int = 6,
                 num_pos_buckets: int = 8):
        super().__init__()
        self.num_roles = num_roles
        self.num_pos_buckets = num_pos_buckets

        self.role_embeddings = nn.Parameter(
            torch.randn(num_roles, hidden_size) * 0.02)
        self.role_head = nn.Linear(hidden_size, num_roles)

        self.pos_embeddings = nn.Parameter(
            torch.randn(num_pos_buckets, hidden_size) * 0.02)

        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, latents, cross_attn_weights):
        """
        latents:             [B, K, H]
        cross_attn_weights:  [B, K, N]  (head-averaged attention from encoder)
        Returns:             [B, K, H]
        """
        # --- role ---
        role_logits = self.role_head(latents)                  # [B, K, R]
        role_probs = F.softmax(role_logits, dim=-1)            # [B, K, R]
        role_emb = role_probs @ self.role_embeddings           # [B, K, H]

        # --- position bucket ---
        N = cross_attn_weights.shape[-1]
        bucket_size = max(1, N // self.num_pos_buckets)
        usable = bucket_size * self.num_pos_buckets

        # reshape [B, K, usable] -> [B, K, buckets, bucket_size] -> sum
        bucket_attn = cross_attn_weights[:, :, :usable].reshape(
            latents.shape[0], latents.shape[1],
            self.num_pos_buckets, bucket_size,
        ).sum(dim=-1)                                          # [B, K, P]

        if usable < N:
            bucket_attn[:, :, -1] += cross_attn_weights[:, :, usable:].sum(
                dim=-1)

        pos_emb = bucket_attn @ self.pos_embeddings            # [B, K, H]

        return self.out_norm(latents + role_emb + pos_emb)


# ---------------------------------------------------------------------------
# Anchor side-channel: explicit rare-token prefix slots
# ---------------------------------------------------------------------------

class AnchorSideChannel(nn.Module):
    """Extract rare/important source tokens and encode them as explicit prefix slots.

    Per sample, selects the `num_slots` rarest tokens (by batch frequency)
    and projects their frozen embeddings into anchor prefix slots that get
    appended after the dense CNDX latents.
    """

    def __init__(self, num_slots: int, hidden_size: int):
        super().__init__()
        self.num_slots = num_slots
        self.slot_pos = nn.Parameter(torch.randn(1, num_slots, hidden_size) * 0.02)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.no_anchor = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input_ids, embedding_weight, attention_mask=None,
                special_ids=None):
        B, N = input_ids.shape
        device = input_ids.device
        V = embedding_weight.shape[0]

        valid = attention_mask.bool() if attention_mask is not None \
            else torch.ones_like(input_ids, dtype=torch.bool)

        presence = torch.zeros(B, V, device=device, dtype=torch.bool)
        for b in range(B):
            presence[b].scatter_(0, input_ids[b][valid[b]], True)

        if special_ids:
            for sid in special_ids:
                if sid < V:
                    presence[:, sid] = False

        batch_freq = presence.float().sum(dim=0)  # [V]
        rarity = torch.zeros_like(batch_freq)
        active = batch_freq > 0
        rarity[active] = 1.0 / batch_freq[active]

        anchor_ids = torch.zeros(B, self.num_slots, dtype=torch.long, device=device)
        anchor_valid = torch.zeros(B, self.num_slots, dtype=torch.bool, device=device)

        for b in range(B):
            cands = presence[b].nonzero(as_tuple=True)[0]
            if cands.numel() == 0:
                continue
            scores = rarity[cands]
            k = min(self.num_slots, cands.numel())
            topk_idx = scores.topk(k).indices
            anchor_ids[b, :k] = cands[topk_idx]
            anchor_valid[b, :k] = True

        embeds = embedding_weight[anchor_ids].detach()  # [B, num_slots, H]

        no_anc = self.no_anchor.unsqueeze(0).unsqueeze(0).expand_as(embeds)
        embeds = torch.where(anchor_valid.unsqueeze(-1), embeds, no_anc)

        embeds = self.proj(embeds)
        embeds = self.norm(embeds + self.slot_pos)
        return embeds


# ---------------------------------------------------------------------------
# Full bottleneck model
# ---------------------------------------------------------------------------

class CNDXBottleneckModel(nn.Module):

    def __init__(self, base_model, encoder, projector,
                 decoder_mask_rate=0.0, aux_loss_weight=0.0,
                 span_aux_weight=0.0, contrastive_weight=0.0,
                 unfreeze_top_layers=0, num_anchor_slots=0,
                 structured_slots=False, num_roles=6, num_pos_buckets=8):
        super().__init__()
        self.base_model = base_model
        self.encoder = encoder
        self.projector = projector
        self.decoder_mask_rate = decoder_mask_rate
        self.aux_loss_weight = aux_loss_weight
        self.span_aux_weight = span_aux_weight
        self.contrastive_weight = contrastive_weight

        hidden_size = base_model.config.hidden_size
        self.mask_embedding = nn.Parameter(torch.randn(hidden_size) * 0.02)

        if structured_slots:
            self.structured_layer = StructuredSlotLayer(
                hidden_size, num_roles, num_pos_buckets)
        else:
            self.structured_layer = None

        if num_anchor_slots > 0:
            self.anchor_channel = AnchorSideChannel(num_anchor_slots, hidden_size)
        else:
            self.anchor_channel = None

        if aux_loss_weight > 0:
            self.retention_head = SourceRetentionHead(hidden_size)
        else:
            self.retention_head = None

        if span_aux_weight > 0:
            self.span_head = SpanRetentionHead(hidden_size)
        else:
            self.span_head = None

        if contrastive_weight > 0:
            self.contrastive_head = ContrastiveIdentityLoss(hidden_size)
        else:
            self.contrastive_head = None

        for p in self.base_model.parameters():
            p.requires_grad = False

        if unfreeze_top_layers > 0:
            layers = self.base_model.model.layers
            total = len(layers)
            for i in range(total - unfreeze_top_layers, total):
                for p in layers[i].parameters():
                    p.requires_grad = True
            for p in self.base_model.model.norm.parameters():
                p.requires_grad = True

    def _special_ids(self):
        special = []
        mcfg = self.base_model.config
        if hasattr(mcfg, 'bos_token_id') and mcfg.bos_token_id is not None:
            special.append(mcfg.bos_token_id)
        if hasattr(mcfg, 'eos_token_id') and mcfg.eos_token_id is not None:
            special.append(mcfg.eos_token_id)
        if hasattr(mcfg, 'pad_token_id') and mcfg.pad_token_id is not None:
            special.append(mcfg.pad_token_id)
        return special

    def _encode_to_prefix(self, input_ids, attention_mask=None):
        """Shared encoding path used by forward, generate, and eval.

        Returns (prefix, cdx_latents, token_embeddings) so callers can
        access whatever they need without reimplementing the pipeline.
        """
        with torch.no_grad():
            token_embeddings = self.base_model.get_input_embeddings()(input_ids)
        token_embeddings = token_embeddings.detach()

        if self.structured_layer is not None:
            cdx_latents, cross_weights = self.encoder(
                token_embeddings, attention_mask, return_cross_weights=True)
            cdx_latents = self.structured_layer(cdx_latents, cross_weights)
        else:
            cdx_latents = self.encoder(token_embeddings, attention_mask)
        prefix = self.projector(cdx_latents)

        if self.anchor_channel is not None:
            emb_w = self.base_model.get_input_embeddings().weight
            anchor_prefix = self.anchor_channel(
                input_ids, emb_w, attention_mask, self._special_ids())
            prefix = torch.cat([prefix, anchor_prefix], dim=1)

        return prefix, cdx_latents, token_embeddings

    def forward(self, input_ids, attention_mask=None):
        B, N = input_ids.shape

        prefix, cdx_latents, token_embeddings = self._encode_to_prefix(
            input_ids, attention_mask)

        K = prefix.shape[1]

        if self.training and self.decoder_mask_rate > 0:
            decoder_embeds = token_embeddings.clone()
            if self.decoder_mask_rate >= 1.0:
                decoder_embeds[:, 1:] = self.mask_embedding
            else:
                corrupt_mask = torch.rand(B, N, device=input_ids.device) < self.decoder_mask_rate
                corrupt_mask[:, 0] = False
                decoder_embeds[corrupt_mask] = self.mask_embedding
        else:
            decoder_embeds = token_embeddings

        combined_embeds = torch.cat([prefix, decoder_embeds], dim=1)

        labels = torch.cat([
            torch.full((B, K), -100, dtype=input_ids.dtype, device=input_ids.device),
            input_ids,
        ], dim=1)

        if attention_mask is not None:
            prefix_mask = torch.ones(B, K, dtype=attention_mask.dtype,
                                     device=attention_mask.device)
            combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        output = self.base_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=labels,
        )

        special = self._special_ids()
        if self.retention_head is not None or self.span_head is not None:
            emb_w = self.base_model.get_input_embeddings().weight

            if self.retention_head is not None:
                aux = self.retention_head(cdx_latents, input_ids, emb_w,
                                           attention_mask, special_ids=special)
                output.loss = output.loss + self.aux_loss_weight * aux

            if self.span_head is not None:
                span_loss = self.span_head(cdx_latents, input_ids, emb_w,
                                            attention_mask, special_ids=special)
                output.loss = output.loss + self.span_aux_weight * span_loss

        if self.contrastive_head is not None and self.training:
            cl = self.contrastive_head(
                cdx_latents, token_embeddings, attention_mask,
                special_ids=special, input_ids=input_ids)
            output.loss = output.loss + self.contrastive_weight * cl

        return output

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = input_ids.shape[1]

        prefix, _, _ = self._encode_to_prefix(input_ids, attention_mask)

        current_embeds = prefix
        generated = []

        for _ in range(max_new_tokens):
            out = self.base_model(inputs_embeds=current_embeds)
            next_tok = out.logits[:, -1, :].argmax(dim=-1)
            generated.append(next_tok)
            next_emb = self.base_model.get_input_embeddings()(next_tok).unsqueeze(1)
            current_embeds = torch.cat([current_embeds, next_emb], dim=1)

        return torch.stack(generated, dim=1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_frozen(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_model(cfg: CNDXConfig, device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, dtype=torch.float32,
    )
    hidden_size = base_model.config.hidden_size

    if cfg.shallow_struct:
        encoder = CNDXShallowStructuredEncoder(
            num_detail=cfg.ss_num_detail,
            num_identity=cfg.ss_num_identity,
            num_global=cfg.ss_num_global,
            hidden_size=hidden_size,
            num_heads=cfg.num_encoder_heads,
            cross_layers_per_role=cfg.ss_cross_per_role,
            shared_refine=cfg.ss_shared_refine,
            dropout=cfg.encoder_dropout,
        )
    elif cfg.tri_level:
        encoder = CNDXTriLevelEncoder(
            num_detail=cfg.num_detail_latents,
            num_entity=cfg.num_entity_latents,
            num_summary=cfg.num_summary_latents,
            hidden_size=hidden_size,
            num_heads=cfg.num_encoder_heads,
            detail_cross_layers=cfg.num_encoder_layers,
            detail_refine=cfg.detail_refine_layers,
            entity_refine=cfg.entity_refine_layers,
            summary_refine=cfg.summary_refine_layers,
            dropout=cfg.encoder_dropout,
        )
    elif cfg.dual_level:
        encoder = CNDXDualLevelEncoder(
            num_local=cfg.num_local_latents,
            num_global=cfg.num_global_latents,
            hidden_size=hidden_size,
            num_heads=cfg.num_encoder_heads,
            local_cross_layers=cfg.num_encoder_layers,
            local_refine=cfg.local_refine_layers,
            global_refine=cfg.global_refine_layers,
            dropout=cfg.encoder_dropout,
        )
    elif cfg.hierarchical:
        encoder = CNDXHierarchicalEncoder(
            num_latents=cfg.num_latents,
            num_intermediate=cfg.num_intermediate_latents,
            hidden_size=hidden_size,
            num_heads=cfg.num_encoder_heads,
            stage1_refine=cfg.stage1_refine_layers,
            stage2_refine=cfg.stage2_refine_layers,
            dropout=cfg.encoder_dropout,
        )
    else:
        encoder = CNDXEncoder(
            num_latents=cfg.num_latents,
            hidden_size=hidden_size,
            num_heads=cfg.num_encoder_heads,
            num_layers=cfg.num_encoder_layers,
            num_refine_layers=cfg.num_refine_layers,
            dropout=cfg.encoder_dropout,
        )
    projector = CNDXProjector(hidden_size)
    model = CNDXBottleneckModel(base_model, encoder, projector,
                                 decoder_mask_rate=cfg.decoder_mask_rate,
                                 aux_loss_weight=cfg.aux_loss_weight,
                                 span_aux_weight=cfg.span_aux_weight,
                                 contrastive_weight=cfg.contrastive_weight,
                                 unfreeze_top_layers=cfg.unfreeze_top_layers,
                                 num_anchor_slots=cfg.num_anchor_slots,
                                 structured_slots=cfg.structured_slots,
                                 num_roles=cfg.num_roles,
                                 num_pos_buckets=cfg.num_pos_buckets)

    if cfg.lora_rank > 0 and cfg.lora_layers > 0:
        inject_lora(model.base_model, cfg.lora_rank, cfg.lora_layers,
                     include_mlp=cfg.lora_include_mlp)

    model = model.to(device)
    return model, tokenizer
