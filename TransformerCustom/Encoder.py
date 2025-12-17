import torch
import torch.nn as nn

from TransformerCustom.TransformerBlock import TransformerBlock

class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.embed_layer = nn.Embedding(src_vocab_size, embed_size)
        self.positional_encoding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_size=embed_size,
                num_heads=heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
            )
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask):
        N, seq_len = src.size()

        position_embedding = torch.arange(0, seq_len, device=src.device).expand(N, seq_len)
        
        out = self.dropout(self.embed_layer(src) + self.positional_encoding(position_embedding))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out