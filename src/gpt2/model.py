import math

import torch
import torch.nn as nn

from .layer_norm import LayerNorm
from .transformer_block import TransformerBlock

class GPT2Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Weight tying: the output head reuses the token embedding matrix, as in the original GPT-2.
        #
        # Both layers are a (vocab_size, emb_dim) table with one row per token:
        #   - tok_emb looks a row up:     token id -> vector       ("cat" = 9246 -> tok_emb.weight[9246])
        #   - out_head scores every row:  vector   -> one logit per token, logits = x @ out_head.weight.T
        # So the logit for "cat" is the dot product of the final hidden state with "cat"'s row.
        # Tying means "cat" is read in and predicted with the same vector: a hidden state that
        # ends up close to the "cat" embedding gives "cat" a high score. Training updates that
        # row from both ends, which helps rare tokens that seldom appear as inputs.
        #
        # It also saves vocab_size * emb_dim parameters: 50257 * 768 = 38.6M for GPT-2 small,
        # which is why the model is "124M" rather than ~163M.
        #
        # Assigning the Parameter shares the tensor (not a copy), so an update to one is an update to both.
        self.out_head.weight = self.tok_emb.weight

        self._init_weights(cfg["n_layers"])

    def _init_weights(self, n_layers):
        """GPT-2's initialisation. Pretrained weights overwrite all of this when loaded.

        Tying means one initialisation has to suit both roles. nn.Embedding's default N(0, 1) is fine
        for lookups but as an output head it produces huge logits: an untrained model would start at
        a loss of ~477 instead of ~10.8 (= ln(50257), a uniform guess over the vocabulary).
        So GPT-2 starts token embeddings small, N(0, 0.02), and everything else has to match that scale:

        - Position embeddings N(0, 0.01). Left at the default N(0, 1) they are 50x larger than the token
          embeddings, so the first layer sees mostly position and barely which token it is reading.
          Training then stalls at the loss of predicting token frequencies (~7.5 on WikiText-2).
        - Linear weights N(0, 0.02) and biases zero, instead of PyTorch's random-uniform defaults.
        - The last layer of each residual branch (attention output and feed-forward output) is scaled
          down by 1/sqrt(2 * n_layers), because 2 * n_layers branches are added into the residual stream.
        """
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.01)

        residual_std = 0.02 / math.sqrt(2 * n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and module is not self.out_head:  # out_head is tok_emb
                is_residual_out = name.endswith(("attention.proj", "attention.out_proj", "ff.layers.2"))
                nn.init.normal_(module.weight, mean=0.0, std=residual_std if is_residual_out else 0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
