import torch
import torch.nn as nn

from TransformerCustom.DecoderBlock import DecoderBlock

class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
    ):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.embed_layer = nn.Embedding(trg_vocab_size, embed_size)
        self.positional_encoding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(
                embed_size=embed_size,
                num_heads=heads,
                dropout=dropout,
                forward_expansion=forward_expansion,
            )
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size, bias=False)
        self.fc_out.weight = self.embed_layer.weight
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.size()

        position_embedding = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.embed_layer(x) + self.positional_encoding(position_embedding))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.dropout(self.fc_out(x))

        return out
