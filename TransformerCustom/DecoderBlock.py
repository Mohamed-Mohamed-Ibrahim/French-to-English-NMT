import torch
import torch.nn as nn

from TransformerCustom.TransformerBlock import TransformerBlock
from TransformerCustom.SelfAttention import SelfAttention

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion=4, dropout=0.1, activation=nn.ReLU, device='cuda'):
        super(DecoderBlock, self).__init__()

        self.attention = SelfAttention(embed_size, num_heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer = TransformerBlock(
            embed_size,
            num_heads,
            dropout,
            forward_expansion,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, k, v, src_mask, trg_mask):
        att = self.attention(q=x, k=x, v=x, mask=trg_mask)
        q = self.dropout(self.norm(att + x))
        out = self.transformer(q, k, v, src_mask)
        return out