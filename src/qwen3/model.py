import torch
import torch.nn as nn

from .transformer_block import TransformerBlock
from .rms_norm import RMSNorm
from .rope import compute_rope_params

# 0.6 billion parameters
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,     # Vocabulary size
    "context_length": 40_960,  # Length originally used during training
    "emb_dim": 1024,           # Embedding dimension
    "n_heads": 16,             # Number of attention heads
    "n_layers": 28,            # Number of layers
    "hidden_dim": 3072,        # Size of intermediate dim in FeedForward
    "head_dim": 128,           # Size of the heads in GQA
    "qk_norm": True,           # Whether to normalize queries & keys in GQA
    "n_kv_groups": 8,          # Key-Value groups for GQA
    "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16,   # Lower-precision dtype to reduce memory
}


def assign(left, right, tensor_name="unknown"):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

    with torch.no_grad():
        if isinstance(right, torch.Tensor):
            left.copy_(right)
        else:
            left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

    return left


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusable utilities
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0  # Track current position in KV cache

    def forward(self, in_idx, cache=None):
        # Forward pass
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        pos_start = self.current_pos if cache is not None else 0
        pos_end = pos_start + num_tokens
        if pos_end > self.cfg["context_length"]:
            raise ValueError(
                f"Sequence length {pos_end} exceeds the model's context length "
                f"of {self.cfg['context_length']} tokens."
            )

        if cache is not None:
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1
            )
        # Prefill (no cache): mask starts as (num_tokens, num_tokens)
        # Cached decoding: mask starts as (num_tokens, prev_k_number_tokens + num_tokens)
        #
        # We add two leading dimensions so the mask becomes
        # (1, 1, num_tokens, num_tokens) during prefill and
        # (1, 1, num_tokens, total_key_tokens) during cached decoding.
        # These extra dimensions let PyTorch broadcast the same mask
        # across all batches and attention heads when applying it to
        # attn_scores of shape (batch, num_heads, num_tokens, total_key_tokens).
        mask = mask[None, None, :, :]  # broadcast mask

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     start_pos=pos_start,
                                     cache=blk_cache)
            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0

    @staticmethod
    def load(model, param_config, params):
        model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

        for l in range(param_config["n_layers"]):  # noqa: E741
            block = model.trf_blocks[l]
            att = block.att

            # Q, K, V projections
            att.W_query.weight = assign(
                att.W_query.weight,
                params[f"model.layers.{l}.self_attn.q_proj.weight"],
                f"model.layers.{l}.self_attn.q_proj.weight"
            )
            att.W_key.weight = assign(
                att.W_key.weight,
                params[f"model.layers.{l}.self_attn.k_proj.weight"],
                f"model.layers.{l}.self_attn.k_proj.weight"
            )
            att.W_value.weight = assign(
                att.W_value.weight,
                params[f"model.layers.{l}.self_attn.v_proj.weight"],
                f"model.layers.{l}.self_attn.v_proj.weight"
            )

            # Output projection
            att.out_proj.weight = assign(
                att.out_proj.weight,
                params[f"model.layers.{l}.self_attn.o_proj.weight"],
                f"model.layers.{l}.self_attn.o_proj.weight"
            )

            # QK norms
            if att.q_norm is not None:
                att.q_norm.scale = assign(
                    att.q_norm.scale,
                    params[f"model.layers.{l}.self_attn.q_norm.weight"],
                    f"model.layers.{l}.self_attn.q_norm.weight"
                )
            if att.k_norm is not None:
                att.k_norm.scale = assign(
                    att.k_norm.scale,
                    params[f"model.layers.{l}.self_attn.k_norm.weight"],
                    f"model.layers.{l}.self_attn.k_norm.weight"
                )

            # Attention layernorm
            block.norm1.scale = assign(
                block.norm1.scale,
                params[f"model.layers.{l}.input_layernorm.weight"],
                f"model.layers.{l}.input_layernorm.weight"
            )

            # Feedforward weights
            block.ff.fc1.weight = assign(
                block.ff.fc1.weight,
                params[f"model.layers.{l}.mlp.gate_proj.weight"],
                f"model.layers.{l}.mlp.gate_proj.weight"
            )
            block.ff.fc2.weight = assign(
                block.ff.fc2.weight,
                params[f"model.layers.{l}.mlp.up_proj.weight"],
                f"model.layers.{l}.mlp.up_proj.weight"
            )
            block.ff.fc3.weight = assign(
                block.ff.fc3.weight,
                params[f"model.layers.{l}.mlp.down_proj.weight"],
                f"model.layers.{l}.mlp.down_proj.weight"
            )

            block.norm2.scale = assign(
                block.norm2.scale,
                params[f"model.layers.{l}.post_attention_layernorm.weight"],
                f"model.layers.{l}.post_attention_layernorm.weight"
            )

        # Final normalization and output head
        model.final_norm.scale = assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")

        if "lm_head.weight" in params:
            model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
        else:
            model.out_head.weight = model.tok_emb.weight
            print("Model uses weight tying.")
