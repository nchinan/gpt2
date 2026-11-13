import torch
import torch.nn as nn

from .transformer_block import TransformerBlock
from .rms_norm import RMSNorm

LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,     # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,         # Embedding dimension
    "n_heads": 32,           # Number of attention heads
    "n_layers": 32,          # Number of layers
    "hidden_dim": 11008,     # NEW: Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
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


def permute(w: torch.Tensor, n_heads, out_dim, in_dim):
    return (w.view(n_heads, out_dim // n_heads // 2, 2, in_dim)
            .transpose(1, 2)  # put axis 2 next to heads
            .reshape(out_dim, in_dim))

class Llama2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        # self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        ################################### NEW ###################################
        # self.final_norm = LayerNorm(cfg["emb_dim"])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        ###########################################################################
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    def forward(self, in_idx):
        # batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds  # + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        # x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    @staticmethod
    def load(model, param_config, params):
        cfg = LLAMA2_CONFIG_7B

        model.tok_emb.weight = assign(model.tok_emb.weight, params["tok_embeddings.weight"])

        for l in range(param_config["n_layers"]):
            # The original Meta/Llama checkpoints store Q and K so that the two numbers
            # that form one complex RoPE pair sit next to each other inside the head dimension ("sliced" layout).
            # Our RoPE implementation, similar to the one in Hugging Face, expects an interleaved layout
            # For example, with n_heads=2 and head_dim = 8
            #                         ┌── pair 0 ──┐      ┌── pair 1 ──┐
            # Meta (sliced):    [ h0:  r0 r1 r2 r3,   h1:  r0 r1 r2 r3  ]
            # Ours & HF (interleaved):  [ h0: r0 r0 r1 r1 r2 r2 r3 r3  , h1: ... ]
            # For more information, please see the discussion in the PR: https://github.com/rasbt/LLMs-from-scratch/pull/747

            # So, below, for q_raw and k_raw, we must re‑order the checkpoint weights using the slices_to_interleave helper

            q_raw = params[f"layers.{l}.attention.wq.weight"]
            model.trf_blocks[l].att.W_query.weight = assign(
                model.trf_blocks[l].att.W_query.weight,
                permute(q_raw, cfg["n_heads"], cfg["emb_dim"], cfg["emb_dim"])
            )
            k_raw = params[f"layers.{l}.attention.wk.weight"]
            model.trf_blocks[l].att.W_key.weight = assign(
                model.trf_blocks[l].att.W_key.weight,
                permute(k_raw, cfg["n_heads"], cfg["emb_dim"], cfg["emb_dim"])
            )
            model.trf_blocks[l].att.W_value.weight = assign(
                model.trf_blocks[l].att.W_value.weight,
                params[f"layers.{l}.attention.wv.weight"]
            )
            model.trf_blocks[l].att.out_proj.weight = assign(
                model.trf_blocks[l].att.out_proj.weight,
                params[f"layers.{l}.attention.wo.weight"]
            )
            model.trf_blocks[l].norm1.weight = assign(
                model.trf_blocks[l].norm1.weight,
                params[f"layers.{l}.attention_norm.weight"]
            )

            # Load FeedForward weights
            model.trf_blocks[l].ff.fc1.weight = assign(
                model.trf_blocks[l].ff.fc1.weight,
                params[f"layers.{l}.feed_forward.w1.weight"]
            )
            # For some reason w2 and w3 are provided in the wrong order in the weights file
            model.trf_blocks[l].ff.fc2.weight = assign(
                model.trf_blocks[l].ff.fc2.weight,
                params[f"layers.{l}.feed_forward.w3.weight"]
            )
            model.trf_blocks[l].ff.fc3.weight = assign(
                model.trf_blocks[l].ff.fc3.weight,
                params[f"layers.{l}.feed_forward.w2.weight"]
            )
            model.trf_blocks[l].norm2.weight = assign(
                model.trf_blocks[l].norm2.weight,
                params[f"layers.{l}.ffn_norm.weight"]
            )

        # Load output layer weights
        model.final_norm.weight = assign(model.final_norm.weight, params["norm.weight"])
        model.out_head.weight = assign(model.out_head.weight, params["output.weight"])
