from dataclasses import dataclass
from typing import Optional


@dataclass
class CNDXConfig:
    # --- Verified champion defaults (EXP-40, Apr 2 2026) ---
    model_name: str = "HuggingFaceTB/SmolLM2-360M"
    seq_len: int = 64
    num_latents: int = 64
    num_encoder_layers: int = 2
    num_encoder_heads: int = 8
    num_refine_layers: int = 3
    encoder_dropout: float = 0.0
    decoder_mask_rate: float = 1.0
    aux_loss_weight: float = 0.1

    # --- Shallow structured latent layout ---
    shallow_struct: bool = False
    ss_num_detail: int = 24
    ss_num_identity: int = 24
    ss_num_global: int = 16
    ss_cross_per_role: int = 1
    ss_shared_refine: int = 2

    # --- Dual-level latent layout ---
    dual_level: bool = False
    num_local_latents: int = 32
    num_global_latents: int = 8
    local_refine_layers: int = 2
    global_refine_layers: int = 1

    # --- Tri-level (3-role) latent layout ---
    tri_level: bool = False
    num_detail_latents: int = 24
    num_entity_latents: int = 16
    num_summary_latents: int = 8
    detail_refine_layers: int = 1
    entity_refine_layers: int = 1
    summary_refine_layers: int = 1

    # --- Dead branches (kept for backward compat, do not enable) ---
    hierarchical: bool = False
    num_intermediate_latents: int = 48
    stage1_refine_layers: int = 1
    stage2_refine_layers: int = 1
    unfreeze_top_layers: int = 0
    lora_rank: int = 0
    lora_layers: int = 0
    lora_include_mlp: bool = False
    span_aux_weight: float = 0.0
    contrastive_weight: float = 0.0
    num_anchor_slots: int = 0
    structured_slots: bool = False
    num_roles: int = 6
    num_pos_buckets: int = 8

    # --- Training ---
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # --- Data ---
    dataset_name: str = "wikitext"
    eval_dataset_name: str = ""  # empty = same as dataset_name; "wikitext" for benchmark consistency
    min_text_chars: int = 100
    max_train_samples: int = 10_000
    max_eval_samples: int = 1_000

    seed: int = -1  # -1 = no fixed seed
    use_bf16: bool = True

    num_generate_samples: int = 10
    log_every: int = 50

    @property
    def compression_ratio(self) -> float:
        return self.seq_len / self.num_latents
